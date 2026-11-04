"""Per-node maximum spanning trees for association excess."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ClusteringError


@dataclass(frozen=True)
class NodeMst:
    """A node's maximum spanning tree with edge weights sorted ascending.

    Edge endpoints are indices into the chromosome-local unit set used to build
    the tree. They are canonicalized so each edge has `u < v`.
    """

    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_weight: np.ndarray
    bottleneck: float | None
    components_at_zero: int | None
    lcc_frac_at_zero: float | None


def maximum_spanning_tree(
    excess: np.ndarray,
    indices: np.ndarray,
) -> NodeMst:
    """Return the maximum spanning tree over `indices` with excess weights."""
    values = np.asarray(excess, dtype=np.float64)
    selected = np.sort(np.asarray(indices, dtype=np.int64))
    n_units = selected.size

    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ClusteringError(
            f"association matrix has shape {values.shape}, expected a square matrix",
            matrix="mst",
        )
    if n_units and (int(selected[0]) < 0 or int(selected[-1]) >= values.shape[0]):
        raise ClusteringError(
            "node unit index is out of bounds for the association matrix",
            matrix="mst",
        )

    if n_units < 2:
        return NodeMst(
            edge_u=np.empty(0, dtype=np.int64),
            edge_v=np.empty(0, dtype=np.int64),
            edge_weight=np.empty(0, dtype=np.float64),
            bottleneck=None,
            components_at_zero=None,
            lcc_frac_at_zero=None,
        )

    weights = values[np.ix_(selected, selected)]
    if not np.isfinite(weights).all():
        raise ClusteringError(
            "association matrix contains non-finite weights",
            matrix="mst",
        )

    local_u, local_v, edge_weight = _prim_maximum(weights)
    low = np.minimum(local_u, local_v)
    high = np.maximum(local_u, local_v)
    order = np.lexsort((high, low, edge_weight))
    local_u = low[order]
    local_v = high[order]
    edge_weight = edge_weight[order]

    components, lcc_frac = _zero_threshold_components(
        n_units,
        local_u,
        local_v,
        edge_weight,
    )

    return NodeMst(
        edge_u=selected[local_u],
        edge_v=selected[local_v],
        edge_weight=edge_weight,
        bottleneck=float(edge_weight[0]),
        components_at_zero=components,
        lcc_frac_at_zero=lcc_frac,
    )


def _prim_maximum(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return max spanning tree edges from a dense symmetric weight matrix."""
    n_units = weights.shape[0]
    in_tree = np.zeros(n_units, dtype=bool)
    best = np.full(n_units, -np.inf, dtype=np.float64)
    parent = np.full(n_units, -1, dtype=np.int64)
    best[0] = np.inf

    edge_u = np.empty(n_units - 1, dtype=np.int64)
    edge_v = np.empty(n_units - 1, dtype=np.int64)
    edge_weight = np.empty(n_units - 1, dtype=np.float64)
    count = 0

    for _ in range(n_units):
        candidates = np.where(in_tree, -np.inf, best)
        unit = int(np.argmax(candidates))
        in_tree[unit] = True

        if parent[unit] >= 0:
            edge_u[count] = int(parent[unit])
            edge_v[count] = unit
            edge_weight[count] = float(weights[parent[unit], unit])
            count += 1

        improve = (~in_tree) & (weights[unit] > best)
        best[improve] = weights[unit, improve]
        parent[improve] = unit

    if count != n_units - 1:
        raise ClusteringError(
            "maximum spanning tree did not connect all node units",
            matrix="mst",
        )
    return edge_u, edge_v, edge_weight


def _zero_threshold_components(
    n_units: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_weight: np.ndarray,
) -> tuple[int, float]:
    """Return component count and largest-component fraction at weight >= 0."""
    parent = np.arange(n_units, dtype=np.int64)
    size = np.ones(n_units, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for edge_index in range(edge_weight.size):
        if edge_weight[edge_index] < 0.0:
            continue

        root_u = find(int(edge_u[edge_index]))
        root_v = find(int(edge_v[edge_index]))
        if root_u == root_v:
            continue

        if size[root_u] < size[root_v]:
            root_u, root_v = root_v, root_u
        parent[root_v] = root_u
        size[root_u] += size[root_v]

    roots = np.array([find(index) for index in range(n_units)], dtype=np.int64)
    components = int(np.unique(roots).size)
    largest = int(np.bincount(roots).max())
    return components, float(largest / n_units)
