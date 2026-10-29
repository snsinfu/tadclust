"""Phase-1 unit selection for uniform mcool bins."""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy import sparse

from .domains import (
    ChromosomeTree,
    DomainNode,
    DomainSelection,
    FineUnit,
    SelectedDomain,
    total_weighted_variance,
)
from .errors import ComputationError
from .matrix import DomainMatrices, FineUnitMatrices


def build_uniform_tree(
    starts: np.ndarray,
    ends: np.ndarray,
    chrom: str,
    chromsize: int,
) -> ChromosomeTree:
    """Build a tree whose leaves are the working-resolution bins."""
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    _check_bins(starts, ends, chrom, chromsize)

    root = DomainNode(
        node_id=0,
        chrom=chrom,
        start=0,
        end=chromsize,
        source_row=None,
        fine_lo=0,
        fine_hi=len(starts),
    )
    nodes = {root.node_id: root}
    fine_units: list[FineUnit] = []

    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        node_id = index + 1
        node = DomainNode(
            node_id=node_id,
            chrom=chrom,
            start=int(start),
            end=int(end),
            source_row=None,
            parent_id=root.node_id,
            fine_lo=index,
            fine_hi=index + 1,
        )
        nodes[node_id] = node
        root.children.append(node_id)
        fine_units.append(
            FineUnit(
                fine_id=index,
                chrom=chrom,
                start=int(start),
                end=int(end),
                owner_node_id=node_id,
                is_tad=False,
                n_bins=1,
            )
        )

    return ChromosomeTree(
        chrom=chrom,
        chromsize=chromsize,
        nodes=nodes,
        root_id=root.node_id,
        fine_units=fine_units,
        n_bins_chrom=len(fine_units),
    )


def select_uniform_domains(
    tree: ChromosomeTree,
    fine: FineUnitMatrices,
) -> DomainSelection:
    """Use every bin as one selected domain; no DP or re-aggregation."""
    n_fine = tree.n_fine
    if fine.bins_per_fine_unit.shape[0] != n_fine:
        raise ComputationError(
            f"{tree.chrom}: bin counts do not match the uniform tree",
            chrom=tree.chrom,
        )

    node_ids = sorted(
        (node_id for node_id in tree.nodes if node_id != tree.root_id),
        key=lambda node_id: tree.nodes[node_id].start,
    )
    domains: list[SelectedDomain] = []
    for node_id in node_ids:
        node = tree.nodes[node_id]
        indices = np.arange(node.fine_lo, node.fine_hi, dtype=np.int64)
        domains.append(
            SelectedDomain(
                domain_id=len(domains),
                chrom=tree.chrom,
                start=node.start,
                end=node.end,
                kind="bin",
                source_row=None,
                fine_indices=indices,
                n_bins=int(fine.bins_per_fine_unit[indices].sum()),
            )
        )

    if not domains:
        raise ComputationError(
            f"{tree.chrom}: no uniform-bin domains",
            chrom=tree.chrom,
        )

    matrices = DomainMatrices(
        observed=fine.observed,
        expected=fine.expected,
        log_oe=fine.log_oe,
        self_oe=fine.self_oe,
        alpha=fine.alpha,
        pcs=fine.pcs,
        eigenvalues=fine.eigenvalues,
        variance_fraction=fine.variance_fraction,
    )
    return DomainSelection(
        tree=tree,
        domains=domains,
        fine_to_domain=np.arange(n_fine, dtype=np.int64),
        domain_map=cast(sparse.csr_matrix, sparse.eye(n_fine, format="csr")),
        matrices=matrices,
        cut=list(node_ids),
        cut_size=len(domains),
        objective=0.0,
        total_variance=total_weighted_variance(
            fine.pcs,
            fine.bins_per_fine_unit,
        ),
        dp_lambda=0.0,
    )


def _check_bins(
    starts: np.ndarray,
    ends: np.ndarray,
    chrom: str,
    chromsize: int,
) -> None:
    if starts.ndim != 1 or ends.ndim != 1 or starts.size != ends.size:
        raise ComputationError(
            f"{chrom}: bin starts and ends must be one-dimensional and equal length",
            chrom=chrom,
        )
    if starts.size == 0:
        raise ComputationError(f"{chrom}: no working bins", chrom=chrom)
    if starts[0] != 0 or ends[-1] != chromsize:
        raise ComputationError(
            f"{chrom}: bins do not cover [0, {chromsize})",
            chrom=chrom,
        )
    if np.any(ends <= starts):
        raise ComputationError(
            f"{chrom}: bin end must be greater than start",
            chrom=chrom,
        )
    if np.any(starts[1:] != ends[:-1]):
        raise ComputationError(f"{chrom}: bins are not contiguous", chrom=chrom)
