"""Trans O/E from the mcool pixel table, plus the trans tree."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cooler
import h5py
import numpy as np
from threadpoolctl import threadpool_limits

from .clustering import informative_mask
from .config import TRANS_NULL_CHOICES
from .domains import SelectedUnit
from .errors import ClusteringError, McoolError
from .matrix import (
    canonicalize_pc1_sign,
    log2_ratio,
    pcs_from_log_oe,
)
from .trees import (
    GENOME_ROOT_LABEL,
    BranchingTree,
    build_branching_tree,
)

CHUNK_ROWS = 10_000_000


class PairInfo(NamedTuple):
    """Unit offsets and sizes for one chromosome pair."""

    unit_offset_a: int
    unit_offset_b: int
    n_units_a: int
    n_units_b: int


class ScanTask(NamedTuple):
    """One contiguous pixel-row range to scan."""

    start_row: int
    end_row: int


class BlockSum(NamedTuple):
    """One chromosome-pair unit block summed from a scan chunk."""

    chrom_a: int
    chrom_b: int
    block: np.ndarray


class PairSum(NamedTuple):
    """Total contact value for one chromosome pair in a scan chunk."""

    chrom_a: int
    chrom_b: int
    total: float


class ChunkResult(NamedTuple):
    """Unit blocks, grand total, and pair totals from one scan chunk."""

    blocks: list[BlockSum]
    total: float
    pair_sums: list[PairSum]


@dataclass(frozen=True)
class TransInput:
    """Selected units for one chromosome, in genomic order."""

    chrom: str
    units: Sequence[SelectedUnit]


@dataclass(frozen=True)
class PixelMap:
    """Global bin-to-unit mapping and chromosome-pair bookkeeping."""

    bin_to_unit: np.ndarray
    bin_chrom: np.ndarray
    unit_chrom: np.ndarray
    col_sums: np.ndarray
    n_bins_chrom: dict[str, int]
    chrom_bin_ranges: dict[str, tuple[int, int]]
    bin1_offsets: np.ndarray
    pair_info: dict[tuple[int, int], PairInfo]
    pair_counts: dict[tuple[int, int], int]
    n_units: int
    n_chrom: int
    pair_count: int


@dataclass(frozen=True)
class TransInfo:
    """Null-model parameters and pair totals for a trans matrix."""

    null: str
    alpha_mode: str
    grand_mean: float
    alpha: float
    min_bins: float
    total: float
    pair_count: int
    n_chrom: int
    pair_means: dict[tuple[str, str], float]
    pair_alphas: dict[tuple[str, str], float]
    pair_counts: dict[tuple[str, str], int]


@dataclass(frozen=True)
class TransMatrix:
    """A symmetric trans log2 O/E matrix and unit metadata."""

    log_oe: np.ndarray
    info: TransInfo
    unit_chrom: np.ndarray
    col_sums: np.ndarray
    chrom_names: list[str]


@dataclass(frozen=True)
class TransTreeResult:
    """Trans PCs and the tree built from them."""

    pcs: np.ndarray
    eigenvalues: np.ndarray
    variance_fraction: np.ndarray
    tree: BranchingTree
    unit_indices: np.ndarray


def trans_matrix(
    mcool_path: str | Path,
    resolution: int,
    cooler_object: cooler.Cooler,
    inputs: Sequence[TransInput],
    *,
    pseudocount: float,
    clip_min: float,
    clip_max: float,
    balance: bool = True,
    null: str = "pair",
    workers: int = 1,
) -> TransMatrix:
    """Build the genome-wide trans log2 O/E matrix from the pixel table.

    The pair null is not coverage-aware; chromosome territory and coverage
    effects can remain in the trans axes.
    """
    if null not in TRANS_NULL_CHOICES:
        raise ClusteringError(f"unknown trans null: {null!r}")
    if workers < 1:
        raise ClusteringError("workers must be at least 1")

    n_units = sum(len(item.units) for item in inputs)
    if n_units == 0:
        raise ClusteringError("trans clustering has no selected units")

    chrom_names = [item.chrom for item in inputs]
    alpha_mode = "global" if null == "grand" else "per_pair"

    if len(inputs) < 2:
        return TransMatrix(
            log_oe=np.zeros((n_units, n_units), dtype=np.float64),
            info=TransInfo(
                null=null,
                alpha_mode=alpha_mode,
                grand_mean=0.0,
                alpha=0.0,
                min_bins=1.0,
                total=0.0,
                pair_count=0,
                n_chrom=len(inputs),
                pair_means={},
                pair_alphas={},
                pair_counts={},
            ),
            unit_chrom=np.zeros(n_units, dtype=np.int64),
            col_sums=np.zeros(n_units, dtype=np.int64),
            chrom_names=chrom_names,
        )

    try:
        handle = h5py.File(mcool_path, "r")
    except OSError as exc:
        raise McoolError(
            f"cannot open mcool: {mcool_path}",
            path=str(mcool_path),
            resolution=resolution,
        ) from exc

    try:
        group = handle.get(f"resolutions/{resolution}")
        if not isinstance(group, h5py.Group):
            raise McoolError(
                f"{mcool_path}: resolution {resolution} not found",
                path=str(mcool_path),
                resolution=resolution,
            )

        pixel_map = build_pixel_map(cooler_object, group, inputs)
        weight = _read_weights(group) if balance else None
        tasks = _pixel_tasks(pixel_map, inputs)
        if workers > 1:
            chunks = _scan_parallel(
                mcool_path,
                resolution,
                pixel_map,
                weight,
                tasks,
                workers,
            )
        else:
            chunks = [_scan_chunk(group, pixel_map, weight, task) for task in tasks]
    finally:
        handle.close()

    observed_upper = np.zeros(
        (pixel_map.n_units, pixel_map.n_units),
        dtype=np.float64,
    )
    pair_totals: dict[tuple[int, int], float] = {}
    total = 0.0

    for chunk in chunks:
        total += chunk.total
        for pair_sum in chunk.pair_sums:
            key = (pair_sum.chrom_a, pair_sum.chrom_b)
            pair_totals[key] = pair_totals.get(key, 0.0) + pair_sum.total
        for block_sum in chunk.blocks:
            info = pixel_map.pair_info[(block_sum.chrom_a, block_sum.chrom_b)]
            a0 = info.unit_offset_a
            b0 = info.unit_offset_b
            observed_upper[
                a0 : a0 + info.n_units_a,
                b0 : b0 + info.n_units_b,
            ] += block_sum.block

    observed = observed_upper + observed_upper.T
    grand_mean = total / pixel_map.pair_count if pixel_map.pair_count > 0 else 0.0
    positive_bins = pixel_map.col_sums[pixel_map.col_sums > 0]
    min_bins = float(positive_bins.min()) if positive_bins.size else 1.0
    alpha = pseudocount * grand_mean * min_bins * min_bins

    pair_means: dict[tuple[str, str], float] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    for (i, j), count in pixel_map.pair_counts.items():
        pair_total = pair_totals.get((i, j), 0.0)
        mean = pair_total / count if count > 0 else 0.0
        key = (chrom_names[i], chrom_names[j])
        pair_means[key] = mean
        pair_counts[key] = count

    values = np.zeros_like(observed, dtype=np.float64)
    pair_alphas: dict[tuple[str, str], float] = {}
    if pixel_map.pair_count > 0 and total > 0:
        for (i, j), info in pixel_map.pair_info.items():
            key = (chrom_names[i], chrom_names[j])
            pair_mean = pair_means[key]
            mean = grand_mean if null == "grand" else pair_mean
            pair_alpha = (
                alpha
                if null == "grand"
                else pseudocount * pair_mean * min_bins * min_bins
            )
            pair_alphas[key] = pair_alpha
            a0 = info.unit_offset_a
            b0 = info.unit_offset_b
            a1 = a0 + info.n_units_a
            b1 = b0 + info.n_units_b
            col_i = pixel_map.col_sums[a0:a1]
            col_j = pixel_map.col_sums[b0:b1]
            expected = mean * col_i[:, None] * col_j[None, :]
            block = log2_ratio(
                observed[a0:a1, b0:b1],
                expected,
                pair_alpha,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            values[a0:a1, b0:b1] = block
            values[b0:b1, a0:a1] = block.T

    info = TransInfo(
        null=null,
        alpha_mode=alpha_mode,
        grand_mean=grand_mean,
        alpha=alpha,
        min_bins=min_bins,
        total=total,
        pair_count=pixel_map.pair_count,
        n_chrom=pixel_map.n_chrom,
        pair_means=pair_means,
        pair_alphas=pair_alphas,
        pair_counts=pair_counts,
    )
    return TransMatrix(
        log_oe=values,
        info=info,
        unit_chrom=pixel_map.unit_chrom,
        col_sums=pixel_map.col_sums,
        chrom_names=chrom_names,
    )


def build_pixel_map(
    cooler_object: cooler.Cooler,
    group: h5py.Group,
    inputs: Sequence[TransInput],
) -> PixelMap:
    """Map working-resolution bins to global selected units by bin center."""
    chrom_offset = _dataset(group, "indexes/chrom_offset")
    bin1_offset = _dataset(group, "indexes/bin1_offset")
    bins_chrom = _dataset(group, "bins/chrom")
    n_working_bins = int(bins_chrom.shape[0])
    bin_chrom = np.asarray(bins_chrom[:], dtype=np.int64)

    chrom_index = {
        str(name): index for index, name in enumerate(cooler_object.chromnames)
    }
    n_units = sum(len(item.units) for item in inputs)
    bin_to_unit = np.full(n_working_bins, -1, dtype=np.int64)
    unit_chrom = np.full(n_units, -1, dtype=np.int64)
    col_sums = np.zeros(n_units, dtype=np.int64)
    n_bins_chrom: dict[str, int] = {}
    chrom_bin_ranges: dict[str, tuple[int, int]] = {}
    pair_info: dict[tuple[int, int], PairInfo] = {}
    pair_counts: dict[tuple[int, int], int] = {}
    offsets: list[int] = []
    unit_offset = 0

    for input_index, item in enumerate(inputs):
        if not item.units:
            raise ClusteringError(f"{item.chrom}: trans input has no units")
        if item.chrom in n_bins_chrom:
            raise ClusteringError(f"duplicate trans chromosome: {item.chrom}")
        if item.chrom not in chrom_index:
            raise McoolError(f"{item.chrom}: chromosome absent from the mcool")

        chromosome = chrom_index[item.chrom]
        start_bin = int(chrom_offset[chromosome])
        end_bin = int(chrom_offset[chromosome + 1])
        chromosome_bins = cooler_object.bins().fetch(item.chrom)
        centers = (
            np.asarray(chromosome_bins["start"], dtype=np.int64)
            + np.asarray(chromosome_bins["end"], dtype=np.int64)
        ) / 2.0
        if centers.size != end_bin - start_bin:
            raise ClusteringError(
                f"{item.chrom}: bin count does not match the mcool index"
            )

        unit_starts = np.asarray(
            [unit.start for unit in item.units],
            dtype=np.float64,
        )
        unit_ends = np.asarray(
            [unit.end for unit in item.units],
            dtype=np.float64,
        )
        local = np.searchsorted(unit_starts, centers, side="right") - 1
        candidate = np.clip(local, 0, len(item.units) - 1)
        local = np.where(
            (local >= 0) & (centers < unit_ends[candidate]),
            local,
            -1,
        ).astype(np.int64)
        mapped = local >= 0
        bin_to_unit[start_bin:end_bin] = np.where(
            mapped,
            local + unit_offset,
            -1,
        )
        col_sums[unit_offset : unit_offset + len(item.units)] = np.bincount(
            local[mapped],
            minlength=len(item.units),
        )
        unit_chrom[unit_offset : unit_offset + len(item.units)] = input_index

        n_bins_chrom[item.chrom] = end_bin - start_bin
        chrom_bin_ranges[item.chrom] = (start_bin, end_bin)
        for prior in range(input_index):
            prior_offset = offsets[prior]
            pair_info[(prior, input_index)] = PairInfo(
                prior_offset,
                unit_offset,
                len(inputs[prior].units),
                len(item.units),
            )
            pair_counts[(prior, input_index)] = n_bins_chrom[inputs[prior].chrom] * (
                end_bin - start_bin
            )

        offsets.append(unit_offset)
        unit_offset += len(item.units)

    if np.any(unit_chrom < 0):
        raise ClusteringError("trans pixel map left some units unassigned")

    return PixelMap(
        bin_to_unit=bin_to_unit,
        bin_chrom=bin_chrom,
        unit_chrom=unit_chrom,
        col_sums=col_sums,
        n_bins_chrom=n_bins_chrom,
        chrom_bin_ranges=chrom_bin_ranges,
        bin1_offsets=np.asarray(bin1_offset[:], dtype=np.int64),
        pair_info=pair_info,
        pair_counts=pair_counts,
        n_units=n_units,
        n_chrom=len(inputs),
        pair_count=sum(pair_counts.values()),
    )


def build_trans_tree(
    trans: TransMatrix,
    unit_bins: np.ndarray,
    *,
    n_pcs: int,
    branch_k: int,
    seed: int,
) -> TransTreeResult:
    """Compute trans PCs and build the trans tree."""
    bins = np.asarray(unit_bins, dtype=np.int64)
    n_units = trans.log_oe.shape[0]
    if bins.shape[0] != n_units:
        raise ClusteringError("trans matrix and unit bin counts have different lengths")

    pcs, eigenvalues, fractions = pcs_from_log_oe(
        trans.log_oe,
        n_pcs,
        label="trans",
    )

    pcs = canonicalize_pc1_sign(pcs)
    indices = np.flatnonzero(informative_mask(pcs, n_pcs))
    if indices.size == 0:
        if n_units == 1:
            indices = np.arange(1, dtype=np.int64)
        else:
            raise ClusteringError("trans tree has no informative units")

    tree = build_branching_tree(
        pcs[indices],
        bins[indices],
        n_pcs=n_pcs,
        seed=seed,
        branch_k=branch_k,
        root_label=GENOME_ROOT_LABEL,
    )
    return TransTreeResult(
        pcs=pcs,
        eigenvalues=eigenvalues,
        variance_fraction=fractions,
        tree=tree,
        unit_indices=indices,
    )


_SCAN_FILE = None
_SCAN_GROUP = None
_SCAN_MAP = None
_SCAN_WEIGHT = None
_SCAN_LIMITER = None


def _scan_parallel(
    mcool_path: str | Path,
    resolution: int,
    pixel_map: PixelMap,
    weight: np.ndarray | None,
    tasks: list[ScanTask],
    workers: int,
) -> list[ChunkResult]:
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_init_scan_worker,
        initargs=(mcool_path, resolution, pixel_map, weight),
    ) as pool:
        return list(
            pool.imap(
                _scan_worker,
                tasks,
                chunksize=1,
            )
        )


def _init_scan_worker(
    mcool_path: str | Path,
    resolution: int,
    pixel_map: PixelMap,
    weight: np.ndarray | None,
) -> None:
    global _SCAN_FILE, _SCAN_GROUP, _SCAN_MAP, _SCAN_WEIGHT, _SCAN_LIMITER

    _SCAN_LIMITER = threadpool_limits(limits=1)
    _SCAN_FILE = h5py.File(mcool_path, "r")
    group = _SCAN_FILE.get(f"resolutions/{resolution}")
    if not isinstance(group, h5py.Group):
        raise McoolError(
            f"{mcool_path}: resolution {resolution} not found",
            path=str(mcool_path),
            resolution=resolution,
        )
    _SCAN_GROUP = group
    _SCAN_MAP = pixel_map
    _SCAN_WEIGHT = weight


def _scan_worker(
    task: ScanTask,
) -> ChunkResult:
    assert _SCAN_GROUP is not None
    assert _SCAN_MAP is not None
    return _scan_chunk(
        _SCAN_GROUP,
        _SCAN_MAP,
        _SCAN_WEIGHT,
        task,
    )


def _read_weights(group: h5py.Group) -> np.ndarray:
    dataset = _dataset(group, "bins/weight")
    return np.nan_to_num(
        np.asarray(dataset[:], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _pixel_tasks(
    pixel_map: PixelMap,
    inputs: Sequence[TransInput],
) -> list[ScanTask]:
    tasks: list[ScanTask] = []
    for item in inputs:
        start_bin, end_bin = pixel_map.chrom_bin_ranges[item.chrom]
        start_row = int(pixel_map.bin1_offsets[start_bin])
        end_row = int(pixel_map.bin1_offsets[end_bin])
        for start in range(start_row, end_row, CHUNK_ROWS):
            tasks.append(ScanTask(start, min(start + CHUNK_ROWS, end_row)))
    return tasks


def _scan_chunk(
    group: h5py.Group,
    pixel_map: PixelMap,
    weight: np.ndarray | None,
    task: ScanTask,
) -> ChunkResult:
    start, end = task
    bin1 = _dataset(group, "pixels/bin1_id")[start:end]
    bin2 = _dataset(group, "pixels/bin2_id")[start:end]
    counts = np.asarray(
        _dataset(group, "pixels/count")[start:end],
        dtype=np.float64,
    )

    chrom1 = pixel_map.bin_chrom[bin1]
    chrom2 = pixel_map.bin_chrom[bin2]
    cross = chrom1 != chrom2
    if not cross.any():
        return ChunkResult([], 0.0, [])

    bin1 = bin1[cross]
    bin2 = bin2[cross]
    counts = counts[cross]
    chrom1 = chrom1[cross]
    chrom2 = chrom2[cross]
    unit1 = pixel_map.bin_to_unit[bin1]
    unit2 = pixel_map.bin_to_unit[bin2]

    values = counts
    if weight is not None:
        values = values * weight[bin1] * weight[bin2]

    swap = chrom1 > chrom2
    unit1_sorted = np.where(swap, unit2, unit1)
    unit2_sorted = np.where(swap, unit1, unit2)
    chrom1_sorted = np.where(swap, chrom2, chrom1)
    chrom2_sorted = np.where(swap, chrom1, chrom2)

    keys = chrom1_sorted * pixel_map.n_chrom + chrom2_sorted
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    unit1_sorted = unit1_sorted[order]
    unit2_sorted = unit2_sorted[order]
    values = values[order]

    cuts = np.flatnonzero(np.diff(keys)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(keys)]
    blocks: list[BlockSum] = []
    pair_sums: list[PairSum] = []
    total = 0.0

    for block_start, block_end in zip(starts, ends):
        key = int(keys[block_start])
        i = key // pixel_map.n_chrom
        j = key % pixel_map.n_chrom
        info = pixel_map.pair_info[(i, j)]
        block_units1 = unit1_sorted[block_start:block_end]
        block_units2 = unit2_sorted[block_start:block_end]
        block_values = values[block_start:block_end]

        # The null is fitted on every cross-chromosome pixel, including bins
        # outside selected units. Only mapped bins enter the observed blocks.
        observed = (block_units1 >= 0) & (block_units2 >= 0)
        if observed.any():
            local = (block_units1[observed] - info.unit_offset_a) * info.n_units_b + (
                block_units2[observed] - info.unit_offset_b
            )
            block = np.bincount(
                local,
                weights=block_values[observed],
                minlength=info.n_units_a * info.n_units_b,
            ).reshape(info.n_units_a, info.n_units_b)
            blocks.append(BlockSum(i, j, block))

        pair_sum = float(block_values.sum())
        pair_sums.append(PairSum(i, j, pair_sum))
        total += pair_sum

    return ChunkResult(blocks, total, pair_sums)


def _dataset(group: h5py.Group, path: str) -> h5py.Dataset:
    dataset = group.get(path)
    if not isinstance(dataset, h5py.Dataset):
        raise McoolError(f"mcool group has no dataset: {path}")
    return dataset
