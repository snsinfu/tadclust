"""Process-parallel cis chromosome selection."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from .config import ClusterConfig
from .domains import (
    ChromosomeTree,
    DomainSelection,
    assign_bin_counts,
    build_chromosome_tree,
    map_bins_to_fine,
    select_domains,
)
from .matrix import canonicalize_pc1_sign, compute_fine_unit_matrices
from .mcool import bin_centers, open_cooler
from .tads import TadTable


@dataclass(frozen=True)
class ChromosomeSelection:
    """Cis matrix, DP selection, and canonicalized PCs for one chromosome."""

    chrom: str
    tree: ChromosomeTree
    domain_selection: DomainSelection
    pcs: np.ndarray
    domain_bins: np.ndarray


def run_cis_serial(
    config: ClusterConfig,
    tads: TadTable,
    sizes: dict[str, int],
    chromosomes: list[str],
    cooler_object: Any,
) -> list[ChromosomeSelection]:
    """Select domains for each chromosome in the current process."""
    return [
        _select_chromosome(
            config,
            tads,
            sizes,
            chrom,
            cooler_object,
        )
        for chrom in chromosomes
    ]


def run_cis_parallel(
    config: ClusterConfig,
    tads: TadTable,
    sizes: dict[str, int],
    chromosomes: list[str],
    resolution: int,
) -> list[ChromosomeSelection]:
    """Select domains in a spawn pool; results stay in chromosome order."""
    context = mp.get_context("spawn")
    with context.Pool(
        processes=config.workers,
        initializer=_init_worker,
        initargs=(config, tads, sizes, resolution),
    ) as pool:
        return pool.map(
            _process_chromosome,
            chromosomes,
            chunksize=1,
        )


_WORKER_COOLER = None
_WORKER_CONFIG = None
_WORKER_TADS = None
_WORKER_SIZES = None
_WORKER_LIMITER = None


def _init_worker(
    config: ClusterConfig,
    tads: TadTable,
    sizes: dict[str, int],
    resolution: int,
) -> None:
    global _WORKER_COOLER, _WORKER_CONFIG, _WORKER_TADS, _WORKER_SIZES
    global _WORKER_LIMITER

    _WORKER_LIMITER = threadpool_limits(limits=1)
    _WORKER_CONFIG = config
    _WORKER_TADS = tads
    _WORKER_SIZES = sizes
    _WORKER_COOLER = open_cooler(config.mcool_path, resolution)


def _process_chromosome(chrom: str) -> ChromosomeSelection:
    assert _WORKER_CONFIG is not None
    assert _WORKER_TADS is not None
    assert _WORKER_SIZES is not None
    assert _WORKER_COOLER is not None
    return _select_chromosome(
        _WORKER_CONFIG,
        _WORKER_TADS,
        _WORKER_SIZES,
        chrom,
        _WORKER_COOLER,
    )


def _select_chromosome(
    config: ClusterConfig,
    tads: TadTable,
    sizes: dict[str, int],
    chrom: str,
    cooler_object: Any,
) -> ChromosomeSelection:
    tree = build_chromosome_tree(tads, chrom, sizes[chrom])
    centers = bin_centers(cooler_object, chrom)
    assign_bin_counts(tree, centers)
    bin_map = map_bins_to_fine(tree, centers)
    fine = compute_fine_unit_matrices(
        cooler_object,
        chrom,
        bin_map,
        tree.n_fine,
        n_distance_bins=config.oe_distance_bins,
        pseudocount=config.oe_pseudocount,
        clip_min=config.oe_clip_min,
        clip_max=config.oe_clip_max,
        n_pcs=config.tree_n_pcs,
    )
    selection = select_domains(
        tree,
        fine,
        n_pcs=config.tree_n_pcs,
        pseudocount=config.oe_pseudocount,
        clip_min=config.oe_clip_min,
        clip_max=config.oe_clip_max,
        dp_lambda_scale=config.dp_lambda_scale,
        tree_branch_k=config.tree_branch_k,
        seed=config.seed,
    )
    pcs = canonicalize_pc1_sign(selection.matrices.pcs)
    domain_bins = np.asarray(
        [domain.n_bins for domain in selection.domains],
        dtype=np.int64,
    )
    return ChromosomeSelection(
        chrom=chrom,
        tree=tree,
        domain_selection=selection,
        pcs=pcs,
        domain_bins=domain_bins,
    )
