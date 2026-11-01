"""Process-parallel cis chromosome selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cooler
import numpy as np
from threadpoolctl import threadpool_limits

from .config import ClusterConfig
from .domains import ChromosomeTree, UnitSelection, select_units
from .matrix import canonicalize_pc1_sign, compute_fine_unit_matrices
from .mcool import open_cooler
from .processes import process_context
from .signals import ignore_sigint
from .sources import CisSource, make_source

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChromosomeSelection:
    """Cis matrix, phase-1 unit selection, and canonicalized PCs."""

    chrom: str
    tree: ChromosomeTree
    unit_selection: UnitSelection
    pcs: np.ndarray
    unit_bins: np.ndarray


def run_cis_serial(
    config: ClusterConfig,
    source: CisSource,
    sizes: dict[str, int],
    chromosomes: list[str],
    cooler_object: cooler.Cooler,
) -> list[ChromosomeSelection]:
    """Select units for each chromosome in the current process."""
    results: list[ChromosomeSelection] = []
    for chrom in chromosomes:
        selection = _select_chromosome(
            config,
            source,
            sizes,
            chrom,
            cooler_object,
        )
        _log_selection(selection)
        results.append(selection)
    return results


def run_cis_parallel(
    config: ClusterConfig,
    sizes: dict[str, int],
    chromosomes: list[str],
    resolution: int,
) -> list[ChromosomeSelection]:
    """Select units in a process pool; results stay in chromosome order.

    Workers rebuild the source from `config`. Do not pass large data through
    `initargs`: Pool pickles them once per worker even with forkserver.
    """
    context = process_context()
    collected: dict[str, ChromosomeSelection] = {}
    pool = None
    try:
        pool = context.Pool(
            processes=config.workers,
            initializer=_init_worker,
            initargs=(config, sizes, resolution),
        )
        with pool:
            for selection in pool.imap_unordered(
                _process_chromosome,
                chromosomes,
                chunksize=1,
            ):
                _log_selection(selection)
                collected[selection.chrom] = selection
    finally:
        if pool is not None:
            pool.terminate()
    return [collected[chrom] for chrom in chromosomes]


def _log_selection(selection: ChromosomeSelection) -> None:
    units = selection.unit_selection.units
    n_tads = sum(unit.unit_kind == "tad" for unit in units)
    n_residuals = sum(unit.unit_kind == "residual" for unit in units)
    n_bins = sum(unit.unit_kind == "bin" for unit in units)
    if n_bins == len(units):
        LOG.info("[CIS:UNIT] %s: %d bin units selected", selection.chrom, n_bins)
    else:
        LOG.info(
            "[CIS:UNIT] %s: %d units selected (%d TAD, %d residual)",
            selection.chrom,
            len(units),
            n_tads,
            n_residuals,
        )


_WORKER_COOLER = None
_WORKER_CONFIG = None
_WORKER_SOURCE = None
_WORKER_SIZES = None
_WORKER_LIMITER = None


def _init_worker(
    config: ClusterConfig,
    sizes: dict[str, int],
    resolution: int,
) -> None:
    global _WORKER_COOLER, _WORKER_CONFIG, _WORKER_SOURCE, _WORKER_SIZES
    global _WORKER_LIMITER

    ignore_sigint()
    _WORKER_LIMITER = threadpool_limits(limits=1)
    _WORKER_CONFIG = config
    _WORKER_SOURCE = make_source(config.annotation_path)
    _WORKER_SIZES = sizes
    _WORKER_COOLER = open_cooler(config.mcool_path, resolution)


def _process_chromosome(chrom: str) -> ChromosomeSelection:
    assert _WORKER_CONFIG is not None
    assert _WORKER_SOURCE is not None
    assert _WORKER_SIZES is not None
    assert _WORKER_COOLER is not None
    return _select_chromosome(
        _WORKER_CONFIG,
        _WORKER_SOURCE,
        _WORKER_SIZES,
        chrom,
        _WORKER_COOLER,
    )


def _select_chromosome(
    config: ClusterConfig,
    source: CisSource,
    sizes: dict[str, int],
    chrom: str,
    cooler_object: cooler.Cooler,
) -> ChromosomeSelection:
    tree, bin_map = source.build_units(cooler_object, chrom, sizes[chrom])
    fine = compute_fine_unit_matrices(
        cooler_object,
        chrom,
        bin_map,
        tree.n_fine,
        n_distance_bins=config.oe_distance_bins,
        pseudocount=config.oe_pseudocount_scale,
        clip_min=config.oe_clip_min,
        clip_max=config.oe_clip_max,
        n_pcs=config.tree_pca_dim,
    )
    selection = select_units(
        tree,
        fine,
        n_pcs=config.tree_pca_dim,
        pseudocount=config.oe_pseudocount_scale,
        clip_min=config.oe_clip_min,
        clip_max=config.oe_clip_max,
        dp_lambda_scale=config.dp_lambda_scale,
        tree_branch_k=config.tree_branch_k,
        seed=config.seed,
        unit_kind=source.unit_kind,
        with_residuals=config.with_residuals,
    )
    pcs = canonicalize_pc1_sign(selection.matrices.pcs)
    unit_bins = np.asarray(
        [unit.n_working_bins for unit in selection.units],
        dtype=np.int64,
    )
    return ChromosomeSelection(
        chrom=chrom,
        tree=tree,
        unit_selection=selection,
        pcs=pcs,
        unit_bins=unit_bins,
    )
