"""Laminar TAD trees, fine units, and SSE-DP domain selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import polars as pl
from scipy import sparse

from .errors import ComputationError, InputError
from .matrix import DomainMatrices, FineUnitMatrices, compute_domain_matrices
from .tads import TadTable


@dataclass
class DomainNode:
    """One TAD annotation node or the synthetic chromosome root."""

    node_id: int
    chrom: str
    start: int
    end: int
    source_row: int | None
    parent_id: int | None = None
    children: list[int] = field(default_factory=list)
    fine_lo: int = 0
    fine_hi: int = 0


@dataclass
class FineUnit:
    """A contiguous observation unit: an annotation leaf or a gap."""

    fine_id: int
    chrom: str
    start: int
    end: int
    owner_node_id: int
    is_tad: bool
    n_bins: int = 0


@dataclass
class ChromosomeTree:
    """A laminar TAD tree with a contiguous fine-unit tiling."""

    chrom: str
    chromsize: int
    nodes: dict[int, DomainNode]
    root_id: int
    fine_units: list[FineUnit]
    n_bins_chrom: int = 0

    @property
    def n_fine(self) -> int:
        return len(self.fine_units)


@dataclass
class SelectedDomain:
    """One row of the selected partition: a TAD or a residual gap."""

    domain_id: int
    chrom: str
    start: int
    end: int
    kind: str
    source_row: int | None
    fine_indices: np.ndarray
    n_bins: int


@dataclass(frozen=True)
class DomainSelection:
    """Selected domains, their matrices, PCs, and DP diagnostics."""

    tree: ChromosomeTree
    domains: list[SelectedDomain]
    fine_to_domain: np.ndarray
    domain_map: sparse.csr_array | sparse.csr_matrix
    matrices: DomainMatrices
    cut: list[int]
    cut_size: int
    objective: float
    total_variance: float
    dp_lambda: float


def build_chromosome_tree(
    tads: TadTable,
    chrom: str,
    chromsize: int,
) -> ChromosomeTree:
    """Build a laminar tree and a contiguous fine-unit tiling."""
    # Build a TAD tree by scanning a flat, sorted list of interval annotations.
    # Stack is used to keep track of nesting.
    table = tads.rows.filter(pl.col("chrom") == chrom).sort(
        ["start", "end", "source_row"],
        descending=[False, True, False],
    )
    if table.height == 0:
        raise InputError(f"{chrom}: no TAD rows")

    # A whole-chromosome annotation row is the level-0 root, not a selectable
    # TAD. The synthetic root below already represents that interval.
    whole_chromosome = (pl.col("start") == 0) & (pl.col("end") == chromsize)
    table = table.filter(~whole_chromosome)

    root = DomainNode(
        node_id=0,
        chrom=chrom,
        start=0,
        end=chromsize,
        source_row=None,
    )
    nodes = {root.node_id: root}
    stack = [root.node_id]

    for row in table.iter_rows(named=True):
        start = row["start"]
        end = row["end"]
        source_row = row["source_row"]
        _check_interval(start, end, chromsize, chrom, source_row)

        node = DomainNode(
            node_id=len(nodes),
            chrom=chrom,
            start=start,
            end=end,
            source_row=source_row,
        )

        # Close finished nesting levels.
        while stack:
            candidate = nodes[stack[-1]]
            if candidate.start <= start and candidate.end >= end:
                break
            if candidate.end > start:
                raise InputError(
                    f"{chrom}: TAD intervals overlap but are not nested",
                    row=source_row,
                )
            stack.pop()

        if not stack:
            raise InputError(
                f"{chrom}: TAD interval is outside the chromosome root",
                row=source_row,
            )

        parent = nodes[stack[-1]]
        node.parent_id = parent.node_id
        parent.children.append(node.node_id)
        nodes[node.node_id] = node
        stack.append(node.node_id)

    for node in nodes.values():
        node.children.sort(key=lambda child_id: nodes[child_id].start)
    _check_siblings(nodes, chrom)

    fine_units: list[FineUnit] = []

    # Walk the completed containment tree to find unannotated gaps between
    # sibling TADs; gaps become fine units that later form residual domains.
    def visit(node_id: int) -> None:
        node = nodes[node_id]
        fine_lo = len(fine_units)

        if not node.children:
            fine_units.append(
                FineUnit(
                    fine_id=len(fine_units),
                    chrom=chrom,
                    start=node.start,
                    end=node.end,
                    owner_node_id=node_id,
                    is_tad=node_id != root.node_id,
                )
            )
        else:
            # Find gaps between siblings.
            cursor = node.start
            for child_id in node.children:
                child = nodes[child_id]
                if child.start > cursor:
                    fine_units.append(
                        FineUnit(
                            fine_id=len(fine_units),
                            chrom=chrom,
                            start=cursor,
                            end=child.start,
                            owner_node_id=node_id,
                            is_tad=False,
                        )
                    )
                visit(child_id)
                cursor = max(cursor, child.end)

            # Gap after the last child.
            if cursor < node.end:
                fine_units.append(
                    FineUnit(
                        fine_id=len(fine_units),
                        chrom=chrom,
                        start=cursor,
                        end=node.end,
                        owner_node_id=node_id,
                        is_tad=False,
                    )
                )

        node.fine_lo = fine_lo
        node.fine_hi = len(fine_units)

    visit(root.node_id)
    _check_fine_units(fine_units, chrom, chromsize)

    return ChromosomeTree(
        chrom=chrom,
        chromsize=chromsize,
        nodes=nodes,
        root_id=root.node_id,
        fine_units=fine_units,
    )


def build_flat_tree(
    starts: np.ndarray,
    ends: np.ndarray,
    chrom: str,
    chromsize: int,
) -> ChromosomeTree:
    """Build a tree whose leaves are contiguous flat units, such as mcool bins."""
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    _check_flat_bins(starts, ends, chrom, chromsize)

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


def is_disjoint(tree: ChromosomeTree) -> bool:
    """Return True when every TAD row is a direct child of the root."""
    return all(
        node.parent_id == tree.root_id
        for node in tree.nodes.values()
        if node.node_id != tree.root_id
    )


def assign_bin_counts(
    tree: ChromosomeTree,
    bin_centers: np.ndarray,
) -> None:
    """Assign the number of working bins whose centers fall in each fine unit."""
    centers = _sorted_centers(bin_centers)
    for fine in tree.fine_units:
        lo = int(np.searchsorted(centers, fine.start, side="left"))
        hi = int(np.searchsorted(centers, fine.end, side="left"))
        fine.n_bins = max(0, hi - lo)
    tree.n_bins_chrom = len(centers)


def map_bins_to_fine(
    tree: ChromosomeTree,
    bin_centers: np.ndarray,
) -> np.ndarray:
    """Map each working bin center to a fine-unit index."""
    centers = _sorted_centers(bin_centers)
    mapping = np.full(len(centers), -1, dtype=np.int64)

    for fine in tree.fine_units:
        lo = int(np.searchsorted(centers, fine.start, side="left"))
        hi = int(np.searchsorted(centers, fine.end, side="left"))
        mapping[lo:hi] = fine.fine_id

    if np.any(mapping < 0):
        starts = np.asarray(
            [fine.start for fine in tree.fine_units],
            dtype=np.float64,
        )
        missing = mapping < 0
        mapping[missing] = np.searchsorted(starts, centers[missing], side="right") - 1
        mapping = np.clip(mapping, 0, tree.n_fine - 1)
    return mapping


def total_weighted_variance(
    pcs: np.ndarray,
    bins_per_fine_unit: np.ndarray,
) -> float:
    """Return the total weighted variance of fine-unit PCs."""
    weights = np.asarray(bins_per_fine_unit, dtype=np.float64)
    mask = weights > 0
    values = np.asarray(pcs, dtype=np.float64)
    if not mask.any() or values.shape[1] == 0:
        return 1.0

    selected = values[mask]
    selected_weights = weights[mask]
    weight_sum = selected_weights.sum()
    mean = (selected * selected_weights[:, None]).sum(axis=0) / weight_sum
    variance = ((selected - mean) ** 2 * selected_weights[:, None]).sum(axis=0) / (
        weight_sum
    )
    total = np.nansum(variance)
    return total if total > 0 else 1.0


def node_costs(
    tree: ChromosomeTree,
    pcs: np.ndarray,
    bins_per_fine_unit: np.ndarray,
    total_variance: float,
    dp_lambda: float,
) -> dict[int, float]:
    """Return the SSE-based cost of keeping each node as one domain."""
    values = np.asarray(pcs, dtype=np.float64)
    weights = np.asarray(bins_per_fine_unit, dtype=np.float64)
    costs: dict[int, float] = {}

    for node_id, node in tree.nodes.items():
        lo, hi = node.fine_lo, node.fine_hi
        node_weights = weights[lo:hi]
        keep = node_weights > 0
        if not keep.any():
            costs[node_id] = math.inf
            continue

        active_weights = node_weights[keep]
        mass = active_weights.sum()
        if mass <= 0:
            costs[node_id] = math.inf
            continue

        active_values = values[lo:hi][keep]
        mean = (active_values * active_weights[:, None]).sum(axis=0) / mass
        sse = np.nansum((active_values - mean) ** 2 * active_weights[:, None])
        costs[node_id] = sse / total_variance + dp_lambda

    return costs


def antichain_dp(
    tree: ChromosomeTree,
    costs: dict[int, float],
    masses: dict[int, int],
) -> list[int]:
    """Choose a minimum-cost antichain of TAD nodes below the root."""
    nodes = tree.nodes
    order: list[int] = []
    stack = [tree.root_id]
    while stack:
        node_id = stack.pop()
        order.append(node_id)
        stack.extend(nodes[node_id].children)

    # keep: one domain for the whole node. expand: replace it by all children.
    # The synthetic root can only expand, so it is never a selected domain.
    best: dict[int, float] = {}
    expand: dict[int, float] = {}
    for node_id in reversed(order):
        node = nodes[node_id]
        if node.children:
            expand[node_id] = sum(best[child] for child in node.children)
        else:
            expand[node_id] = math.inf if masses[node_id] > 0 else 0.0

        if node_id == tree.root_id:
            best[node_id] = expand[node_id]
        elif masses[node_id] <= 0:
            best[node_id] = 0.0
        else:
            keep = costs[node_id] if np.isfinite(costs[node_id]) else math.inf
            best[node_id] = min(keep, expand[node_id])

    if best[tree.root_id] <= 0 or not np.isfinite(best[tree.root_id]):
        return [
            node_id
            for node_id, node in nodes.items()
            if node_id != tree.root_id and not node.children and masses[node_id] > 0
        ]

    selected: list[int] = []

    def choose(node_id: int) -> None:
        node = nodes[node_id]
        if masses[node_id] <= 0:
            return
        if node_id == tree.root_id:
            for child in node.children:
                choose(child)
            return
        keep = costs[node_id] if np.isfinite(costs[node_id]) else math.inf
        if keep <= expand[node_id]:
            selected.append(node_id)
        elif node.children:
            for child in node.children:
                choose(child)

    choose(tree.root_id)
    return selected


def selected_domains(
    tree: ChromosomeTree,
    cut: list[int],
    *,
    kind: str = "tad",
) -> tuple[list[SelectedDomain], np.ndarray, sparse.csr_array | sparse.csr_matrix]:
    """Turn a selected antichain plus uncovered fine units into a partition."""
    n_fine = tree.n_fine
    covered = np.zeros(n_fine, dtype=bool)
    domains: list[SelectedDomain] = []

    for node_id in cut:
        node = tree.nodes[node_id]
        if node_id == tree.root_id:
            raise ComputationError(
                f"{tree.chrom}: the synthetic root cannot be selected",
                chrom=tree.chrom,
            )
        if covered[node.fine_lo : node.fine_hi].any():
            raise ComputationError(
                f"{tree.chrom}: selected TAD domains overlap",
                chrom=tree.chrom,
            )
        covered[node.fine_lo : node.fine_hi] = True
        indices = np.arange(node.fine_lo, node.fine_hi, dtype=np.int64)
        domains.append(
            SelectedDomain(
                domain_id=-1,
                chrom=tree.chrom,
                start=node.start,
                end=node.end,
                kind=kind,
                source_row=node.source_row,
                fine_indices=indices,
                n_bins=int(sum(tree.fine_units[i].n_bins for i in indices)),
            )
        )

    uncovered = np.flatnonzero(~covered)
    if uncovered.size:
        breaks = np.flatnonzero(np.diff(uncovered) > 1)
        run_starts = np.r_[0, breaks + 1]
        run_ends = np.r_[breaks + 1, len(uncovered)]
        for start_index, end_index in zip(run_starts, run_ends):
            run = uncovered[start_index:end_index]
            domains.append(
                SelectedDomain(
                    domain_id=-1,
                    chrom=tree.chrom,
                    start=tree.fine_units[run[0]].start,
                    end=tree.fine_units[run[-1]].end,
                    kind="residual",
                    source_row=None,
                    fine_indices=run,
                    n_bins=int(sum(tree.fine_units[i].n_bins for i in run)),
                )
            )

    domains.sort(key=lambda domain: domain.start)
    for domain_id, domain in enumerate(domains):
        domain.domain_id = domain_id

    fine_to_domain = np.full(n_fine, -1, dtype=np.int64)
    domain_map = sparse.lil_matrix(
        (n_fine, len(domains)),
        dtype=np.float64,
    )
    for domain in domains:
        domain_map[domain.fine_indices, domain.domain_id] = 1.0
        fine_to_domain[domain.fine_indices] = domain.domain_id

    if np.any(fine_to_domain < 0):
        raise ComputationError(
            f"{tree.chrom}: fine units were not assigned to a domain",
            chrom=tree.chrom,
        )

    return domains, fine_to_domain, domain_map.tocsr()


def select_domains(
    tree: ChromosomeTree,
    fine: FineUnitMatrices,
    *,
    n_pcs: int,
    pseudocount: float,
    clip_min: float,
    clip_max: float,
    dp_lambda_scale: float,
    tree_branch_k: int,
    seed: int,
    selected_kind: str = "tad",
) -> DomainSelection:
    """Select domains, aggregate O/E, and compute domain PCs."""
    bins_per_fine_unit = np.asarray(
        fine.bins_per_fine_unit,
        dtype=np.float64,
    )
    n_data = int((bins_per_fine_unit > 0).sum())
    if n_data == 0:
        raise ComputationError(
            f"{tree.chrom}: no fine units contain working bins",
            chrom=tree.chrom,
        )

    total_variance = total_weighted_variance(fine.pcs, bins_per_fine_unit)
    if is_disjoint(tree):
        cut = [node_id for node_id in tree.nodes if node_id != tree.root_id]
        objective = 0.0
        lambda_value = 0.0
    else:
        lambda_value = (
            dp_lambda_scale * 0.5 * (tree_branch_k - 1) * math.log(max(2, n_data))
        )
        masses = {
            node_id: int(bins_per_fine_unit[node.fine_lo : node.fine_hi].sum())
            for node_id, node in tree.nodes.items()
        }
        costs = node_costs(
            tree,
            fine.pcs,
            bins_per_fine_unit,
            total_variance,
            lambda_value,
        )
        cut = antichain_dp(tree, costs, masses)
        objective = sum(costs[node_id] for node_id in cut)

    domains, fine_to_domain, domain_map = selected_domains(
        tree,
        cut,
        kind=selected_kind,
    )
    if np.array_equal(fine_to_domain, np.arange(tree.n_fine, dtype=np.int64)):
        matrices = DomainMatrices.from_fine_unit_matrices(fine)
    else:
        matrices = compute_domain_matrices(
            domain_map,
            fine.observed,
            fine.expected,
            pseudocount=pseudocount,
            clip_min=clip_min,
            clip_max=clip_max,
            n_pcs=n_pcs,
            label=f"domains:{tree.chrom}",
        )
    return DomainSelection(
        tree=tree,
        domains=domains,
        fine_to_domain=fine_to_domain,
        domain_map=domain_map,
        matrices=matrices,
        cut=cut,
        cut_size=len(cut),
        objective=objective,
        total_variance=total_variance,
        dp_lambda=lambda_value,
    )


def _check_flat_bins(
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


def _check_interval(
    start: int,
    end: int,
    chromsize: int,
    chrom: str,
    source_row: int,
) -> None:
    if start is None or end is None:
        raise InputError(
            f"{chrom}: TAD interval has a missing coordinate",
            row=source_row,
        )
    if start < 0 or end > chromsize or end <= start:
        raise InputError(
            f"{chrom}: TAD {start}-{end} is outside [0, {chromsize})",
            row=source_row,
        )


def _check_siblings(
    nodes: dict[int, DomainNode],
    chrom: str,
) -> None:
    for node in nodes.values():
        ordered = sorted(
            node.children,
            key=lambda child_id: nodes[child_id].start,
        )
        for left_id, right_id in pairwise(ordered):
            left = nodes[left_id]
            right = nodes[right_id]
            if left.end > right.start:
                raise InputError(
                    f"{chrom}: TAD intervals overlap but are not nested",
                    row=right.source_row,
                )


def _check_fine_units(
    fine_units: list[FineUnit],
    chrom: str,
    chromsize: int,
) -> None:
    if not fine_units:
        raise InputError(f"{chrom}: fine-unit tiling is empty")

    if fine_units[0].start != 0 or fine_units[-1].end != chromsize:
        raise InputError(f"{chrom}: fine-unit tiling does not cover the chromosome")

    for previous, current in pairwise(fine_units):
        if previous.end != current.start:
            raise InputError(f"{chrom}: fine-unit tiling is not contiguous")


def _sorted_centers(bin_centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(bin_centers, dtype=np.float64)
    if centers.ndim != 1:
        raise ComputationError("bin centers must be one-dimensional")
    if centers.size > 1 and np.any(np.diff(centers) < 0):
        raise ComputationError("bin centers must be sorted")
    return centers
