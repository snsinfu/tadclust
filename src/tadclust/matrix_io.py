"""Unit-level cis matrix HDF5 product."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .cohesion import CohesionContext
from .domains import SelectedUnit
from .errors import ClusteringError
from .matrix import (
    UnitMatrices,
    distance_bin_edges,
    double_centered_log_frequency,
)

MATRIX_FILE = "cis_matrices.h5"
MATRIX_SCHEMA_VERSION = "2.1"
MATRIX_DTYPE = np.float32
MATRIX_COMPRESSION = "gzip"
MATRIX_COMPRESSION_LEVEL = 1
MATRIX_DATASETS = (
    "unit_id",
    "tad_id",
    "unit_start",
    "unit_end",
    "unit_center",
    "n_working_bins",
    "log_oe",
    "observed",
    "expected",
    "dc_log_freq",
    "log_oe_excess",
    "log_oe_baseline",
)


@dataclass(frozen=True)
class CisMatrixInput:
    """Per-chromosome selected units, aggregated matrices, and baseline."""

    chrom: str
    units: list[SelectedUnit]
    matrices: UnitMatrices
    n_bins_chrom: int
    cohesion: CohesionContext


@dataclass(frozen=True)
class CisMatrixGroup:
    """One chromosome group in `cis_matrices.h5`."""

    chrom: str
    unit_id: np.ndarray
    tad_id: np.ndarray
    unit_start: np.ndarray
    unit_end: np.ndarray
    unit_center: np.ndarray
    n_working_bins: np.ndarray
    alpha: float
    distance_bin_edges: np.ndarray
    observed: np.ndarray
    expected: np.ndarray
    log_oe: np.ndarray
    dc_log_freq: np.ndarray
    log_oe_excess: np.ndarray
    log_oe_baseline: np.ndarray


@dataclass(frozen=True)
class CisMatrices:
    """Dense unit-level cis matrices and the settings needed to interpret them."""

    storage: str
    clip_min: float
    clip_max: float
    working_binsize: int
    groups: list[CisMatrixGroup]


def build_cis_matrices(
    inputs: list[CisMatrixInput],
    *,
    working_binsize: int,
    distance_bins: int,
    clip_min: float,
    clip_max: float,
) -> CisMatrices:
    """Build write-ready dense float32 matrices in `cis_leaves` unit order."""
    if working_binsize <= 0:
        raise ClusteringError("working_binsize must be positive for matrix output")

    groups: list[CisMatrixGroup] = []
    unit_base = 0

    for item in inputs:
        n_units = len(item.units)
        matrices = item.matrices
        expected_shape = (n_units, n_units)
        for name in ("observed", "expected", "log_oe"):
            values = getattr(matrices, name)
            if values.shape != expected_shape:
                raise ClusteringError(
                    f"{item.chrom}: {name} matrix has shape {values.shape}, "
                    f"expected {expected_shape}",
                    chrom=item.chrom,
                    matrix=name,
                )

        context = item.cohesion
        for name in ("log_oe", "distance_bin"):
            values = getattr(context, name)
            if values.shape != expected_shape:
                raise ClusteringError(
                    f"{item.chrom}: cohesion {name} has shape {values.shape}, "
                    f"expected {expected_shape}",
                    chrom=item.chrom,
                    matrix=name,
                )

        edges = distance_bin_edges(item.n_bins_chrom, distance_bins)
        expected_baseline = len(edges) - 1
        if context.baseline.shape != (expected_baseline,):
            raise ClusteringError(
                f"{item.chrom}: cohesion baseline has shape "
                f"{context.baseline.shape}, expected {(expected_baseline,)}",
                chrom=item.chrom,
                matrix="log_oe_baseline",
            )

        unit_ids = np.array(
            [unit_base + index for index in range(n_units)],
            dtype=np.int64,
        )
        starts = np.array([unit.start for unit in item.units], dtype=np.int64)
        ends = np.array([unit.end for unit in item.units], dtype=np.int64)
        tad_ids = np.array(
            [-1 if unit.tad_id is None else unit.tad_id for unit in item.units],
            dtype=np.int64,
        )
        centers = (starts.astype(np.float64) + ends.astype(np.float64)) / 2.0
        n_bins = np.array(
            [unit.n_working_bins for unit in item.units],
            dtype=np.int64,
        )
        dc_log_freq = double_centered_log_frequency(
            matrices.observed,
            matrices.alpha,
            clip_min=clip_min,
            clip_max=clip_max,
            label=f"units:{item.chrom}",
        )
        log_oe_excess = np.asarray(
            context.log_oe_excess_matrix(),
            dtype=MATRIX_DTYPE,
        )

        groups.append(
            CisMatrixGroup(
                chrom=item.chrom,
                unit_id=unit_ids,
                tad_id=tad_ids,
                unit_start=starts,
                unit_end=ends,
                unit_center=centers,
                n_working_bins=n_bins,
                alpha=float(matrices.alpha),
                distance_bin_edges=edges,
                observed=np.asarray(matrices.observed, dtype=MATRIX_DTYPE),
                expected=np.asarray(matrices.expected, dtype=MATRIX_DTYPE),
                log_oe=np.asarray(matrices.log_oe, dtype=MATRIX_DTYPE),
                dc_log_freq=np.asarray(dc_log_freq, dtype=MATRIX_DTYPE),
                log_oe_excess=log_oe_excess,
                log_oe_baseline=np.asarray(context.baseline, dtype=np.float64),
            )
        )
        unit_base += n_units

    return CisMatrices(
        storage="dense",
        clip_min=clip_min,
        clip_max=clip_max,
        working_binsize=working_binsize,
        groups=groups,
    )


def write_cis_matrices(
    path: str | Path,
    matrices: CisMatrices,
    manifest: dict[str, Any],
) -> None:
    """Write the dense unit-level matrices and embedded manifest to HDF5."""
    with h5py.File(path, "w") as handle:
        handle.attrs["matrix_schema_version"] = MATRIX_SCHEMA_VERSION
        handle.attrs["storage"] = matrices.storage
        handle.attrs["clip_min"] = matrices.clip_min
        handle.attrs["clip_max"] = matrices.clip_max
        handle.attrs["working_binsize"] = matrices.working_binsize

        for group_data in matrices.groups:
            group = handle.create_group(group_data.chrom)
            group.attrs["alpha"] = group_data.alpha
            group.attrs["distance_bin_edges"] = group_data.distance_bin_edges

            group.create_dataset("unit_id", data=group_data.unit_id)
            tad_ids = group.create_dataset(
                "tad_id",
                data=group_data.tad_id,
            )
            tad_ids.attrs["null_value"] = -1
            group.create_dataset("unit_start", data=group_data.unit_start)
            group.create_dataset("unit_end", data=group_data.unit_end)
            group.create_dataset(
                "unit_center",
                data=group_data.unit_center,
            )
            group.create_dataset(
                "n_working_bins",
                data=group_data.n_working_bins,
            )
            for name in (
                "log_oe",
                "observed",
                "expected",
                "dc_log_freq",
                "log_oe_excess",
            ):
                group.create_dataset(
                    name,
                    data=getattr(group_data, name),
                    compression=MATRIX_COMPRESSION,
                    compression_opts=MATRIX_COMPRESSION_LEVEL,
                    shuffle=True,
                    chunks=True,
                )
            group.create_dataset(
                "log_oe_baseline",
                data=group_data.log_oe_baseline,
            )

        handle.create_dataset(
            "manifest",
            data=json.dumps(manifest, separators=(",", ":"), ensure_ascii=True),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
