"""Laminar TAD trees, fine units, and SSE-DP unit selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import polars as pl
from scipy import sparse

from .errors import ClusteringError, InputError
from .matrix import FineUnitMatrices, UnitMatrices, compute_unit_matrices
from .tads import TadTable


@dataclass
class UnitNode:
    """One TAD annotation node or the synthetic chromosome root."""

    node_id: int
    chrom: str
    start: int
    end: int
    tad_table_row: int | None
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
    n_working_bins: int = 0


@dataclass
class ChromosomeTree:
    """A laminar TAD tree with a contiguous fine-unit tiling."""

    chrom: str
    chromsize: int
    nodes: dict[int, UnitNode]
    root_id: int
    fine_units: list[FineUnit]
    n_bins_chrom: int = 0

    @property
    def n_fine(self) -> int:
        return len(self.fine_units)


@dataclass
class SelectedUnit:
    """One selected unit: a TAD, an optional residual gap, or a bin."""

    unit_id: int
    chrom: str
    start: int
    end: int
    unit_kind: str
    tad_table_row: int | None
    fine_indices: np.ndarray
    n_working_bins: int


@dataclass(frozen=True)
class UnitSelection:
    """Selected units, their matrices, PCs, and DP diagnostics.

    `fine_to_unit` holds the selected unit index for each fine unit. It is
    `-1` for an uncovered fine unit when residual units are not included.
    """

    tree: ChromosomeTree
    units: list[SelectedUnit]
    fine_to_unit: np.ndarray
    unit_map: sparse.csr_array | sparse.csr_matrix
    matrices: UnitMatrices
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
        ["start", "end", "tad_table_row"],
        descending=[False, True, False],
    )
    if table.height == 0:
        raise InputError(f"{chrom}: no TAD rows")

    # A whole-chromosome annotation row is the level-0 root, not a selectable
    # TAD. The synthetic root below already represents that interval.
    whole_chromosome = (pl.col("start") == 0) & (pl.col("end") == chromsize)
    table = table.filter(~whole_chromosome)

    root = UnitNode(
        node_id=0,
        chrom=chrom,
        start=0,
        end=chromsize,
        tad_table_row=None,
    )
    nodes = {root.node_id: root}
    stack = [root.node_id]

    for row in table.iter_rows(named=True):
        start = row["start"]
        end = row["end"]
        tad_table_row = row["tad_table_row"]
        _check_interval(start, end, chromsize, chrom, tad_table_row)

        node = UnitNode(
            node_id=len(nodes),
            chrom=chrom,
            start=start,
            end=end,
            tad_table_row=tad_table_row,
        )

        # Close finished nesting levels.
        while stack:
            candidate = nodes[stack[-1]]
            if candidate.start <= start and candidate.end >= end:
                break
            if candidate.end > start:
                raise InputError(
                    f"{chrom}: TAD intervals overlap but are not nested",
                    row=tad_table_row,
                )
            stack.pop()

        if not stack:
            raise InputError(
                f"{chrom}: TAD interval is outside the chromosome root",
                row=tad_table_row,
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
    # sibling TADs; gaps become fine units that later form residual units.
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

    root = UnitNode(
        node_id=0,
        chrom=chrom,
        start=0,
        end=chromsize,
        tad_table_row=None,
        fine_lo=0,
        fine_hi=len(starts),
    )
    nodes = {root.node_id: root}
    fine_units: list[FineUnit] = []

    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        node_id = index + 1
        node = UnitNode(
            node_id=node_id,
            chrom=chrom,
            start=int(start),
            end=int(end),
            tad_table_row=None,
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
                n_working_bins=1,
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
        fine.n_working_bins = max(0, hi - lo)
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
    """Return the SSE-based cost of keeping each node as one unit."""
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

    # keep: one unit for the whole node. expand: replace it by all children.
    # The synthetic root can only expand, so it is never a selected unit.
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


def selected_units(
    tree: ChromosomeTree,
    cut: list[int],
    *,
    unit_kind: str = "tad",
    with_residuals: bool = False,
) -> tuple[list[SelectedUnit], np.ndarray, sparse.csr_array | sparse.csr_matrix]:
    """Turn a selected antichain into units.

    Uncovered fine units become residual units only when `with_residuals`
    is true. Otherwise they are left unassigned in `fine_to_unit`.
    """
    n_fine = tree.n_fine
    covered = np.zeros(n_fine, dtype=bool)
    units: list[SelectedUnit] = []

    for node_id in cut:
        node = tree.nodes[node_id]
        if node_id == tree.root_id:
            raise ClusteringError(
                f"{tree.chrom}: the synthetic root cannot be selected",
                chrom=tree.chrom,
            )
        if covered[node.fine_lo : node.fine_hi].any():
            raise ClusteringError(
                f"{tree.chrom}: selected TAD units overlap",
                chrom=tree.chrom,
            )
        covered[node.fine_lo : node.fine_hi] = True
        indices = np.arange(node.fine_lo, node.fine_hi, dtype=np.int64)
        units.append(
            SelectedUnit(
                unit_id=-1,
                chrom=tree.chrom,
                start=node.start,
                end=node.end,
                unit_kind=unit_kind,
                tad_table_row=node.tad_table_row,
                fine_indices=indices,
                n_working_bins=int(
                    sum(tree.fine_units[i].n_working_bins for i in indices)
                ),
            )
        )

    if with_residuals:
        uncovered = np.flatnonzero(~covered)
        if uncovered.size:
            breaks = np.flatnonzero(np.diff(uncovered) > 1)
            run_starts = np.r_[0, breaks + 1]
            run_ends = np.r_[breaks + 1, len(uncovered)]
            for start_index, end_index in zip(run_starts, run_ends):
                run = uncovered[start_index:end_index]
                units.append(
                    SelectedUnit(
                        unit_id=-1,
                        chrom=tree.chrom,
                        start=tree.fine_units[run[0]].start,
                        end=tree.fine_units[run[-1]].end,
                        unit_kind="residual",
                        tad_table_row=None,
                        fine_indices=run,
                        n_working_bins=int(
                            sum(tree.fine_units[i].n_working_bins for i in run)
                        ),
                    )
                )

    units.sort(key=lambda unit: unit.start)
    for unit_id, unit in enumerate(units):
        unit.unit_id = unit_id

    fine_to_unit = np.full(n_fine, -1, dtype=np.int64)
    unit_map = sparse.lil_matrix(
        (n_fine, len(units)),
        dtype=np.float64,
    )
    for unit in units:
        unit_map[unit.fine_indices, unit.unit_id] = 1.0
        fine_to_unit[unit.fine_indices] = unit.unit_id

    if with_residuals and np.any(fine_to_unit < 0):
        raise ClusteringError(
            f"{tree.chrom}: fine units were not assigned to a unit",
            chrom=tree.chrom,
        )

    return units, fine_to_unit, unit_map.tocsr()


def select_units(
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
    unit_kind: str = "tad",
    with_residuals: bool = False,
) -> UnitSelection:
    """Select units, aggregate O/E, and compute unit PCs.

    The DP always runs on the complete fine-unit set, including gaps. Residual
    units are added after the cut only when `with_residuals` is true.
    """
    bins_per_fine_unit = np.asarray(
        fine.bins_per_fine_unit,
        dtype=np.float64,
    )
    n_data = int((bins_per_fine_unit > 0).sum())
    if n_data == 0:
        raise ClusteringError(
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

    units, fine_to_unit, unit_map = selected_units(
        tree,
        cut,
        unit_kind=unit_kind,
        with_residuals=with_residuals,
    )
    if not units:
        raise ClusteringError(
            f"{tree.chrom}: no selectable TAD units; "
            "enable with_residuals or supply TAD intervals",
            chrom=tree.chrom,
        )
    if np.array_equal(fine_to_unit, np.arange(tree.n_fine, dtype=np.int64)):
        matrices = UnitMatrices.from_fine_unit_matrices(fine)
    else:
        matrices = compute_unit_matrices(
            unit_map,
            fine.observed,
            fine.expected,
            pseudocount=pseudocount,
            clip_min=clip_min,
            clip_max=clip_max,
            n_pcs=n_pcs,
            label=f"units:{tree.chrom}",
        )
    return UnitSelection(
        tree=tree,
        units=units,
        fine_to_unit=fine_to_unit,
        unit_map=unit_map,
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
        raise ClusteringError(
            f"{chrom}: bin starts and ends must be one-dimensional and equal length",
            chrom=chrom,
        )
    if starts.size == 0:
        raise ClusteringError(f"{chrom}: no working bins", chrom=chrom)
    if starts[0] != 0 or ends[-1] != chromsize:
        raise ClusteringError(
            f"{chrom}: bins do not cover [0, {chromsize})",
            chrom=chrom,
        )
    if np.any(ends <= starts):
        raise ClusteringError(
            f"{chrom}: bin end must be greater than start",
            chrom=chrom,
        )
    if np.any(starts[1:] != ends[:-1]):
        raise ClusteringError(f"{chrom}: bins are not contiguous", chrom=chrom)


def _check_interval(
    start: int,
    end: int,
    chromsize: int,
    chrom: str,
    tad_table_row: int,
) -> None:
    if start is None or end is None:
        raise InputError(
            f"{chrom}: TAD interval has a missing coordinate",
            row=tad_table_row,
        )
    if start < 0 or end > chromsize or end <= start:
        raise InputError(
            f"{chrom}: TAD {start}-{end} is outside [0, {chromsize})",
            row=tad_table_row,
        )


def _check_siblings(
    nodes: dict[int, UnitNode],
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
                    row=right.tad_table_row,
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
        raise ClusteringError("bin centers must be one-dimensional")
    if centers.size > 1 and np.any(np.diff(centers) < 0):
        raise ClusteringError("bin centers must be sorted")
    return centers
