"""Deterministic branching k-means trees over unit PCs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clustering import derive_seed, informative_mask, kmeans_labels
from .errors import ClusteringError

CHROM_ROOT_LABEL = "chrom"
GENOME_ROOT_LABEL = "genome"
_TREE_SEED_NAMESPACE = 0
_AB_SEED_NAMESPACE = 1


@dataclass
class TreeNode:
    """One node in a branching k-means tree."""

    node_id: int
    parent_id: int | None
    children: list[int] = field(default_factory=list)
    depth: int = 0
    path: list[int] = field(default_factory=list)
    unit_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    n_working_bins: int = 0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    sse: float = 0.0
    split_gain: float | None = None
    stop_reason: str | None = None
    ab_label: str | None = None
    ab_frac_a: float | None = None
    ab_frac_b: float | None = None
    ab_purity: float | None = None
    ab_signal: float | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class BranchingTree:
    """One complete tree with a tree-level root label."""

    nodes: list[TreeNode]
    root_id: int
    root_label: str


@dataclass
class CisForest:
    """One or more trees for a chromosome."""

    chrom: str
    trees: list[BranchingTree]
    node_offsets: list[int]
    groups: list[np.ndarray]


@dataclass
class AbAnnotation:
    """Per-unit A/B labels, feature ranks, and continuous signals.

    The A/B centroids are stored in the standardized PC subspace. The feature
    rank is used only to fit and orient the centroids, not to score units.
    Signals are raw signed projections on the A-B centroid axis. They are
    unbounded, positive on the A side, and expressed in standardized-PC units.
    """

    labels: np.ndarray
    feature_rank: np.ndarray
    fallback: bool
    fallback_reason: str | None
    signals: np.ndarray
    a_indices: np.ndarray
    b_indices: np.ndarray
    a_centroid_pc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b_centroid_pc: np.ndarray = field(default_factory=lambda: np.zeros(0))


def format_path_label(root_label: str, path: list[int]) -> str:
    """Format a root label and child-index path for display."""
    if not path:
        return root_label
    return f"{root_label}." + ".".join(str(index) for index in path)


def build_branching_tree(
    pcs: np.ndarray,
    unit_bins: np.ndarray,
    n_pcs: int,
    seed: int,
    branch_k: int,
    root_label: str = CHROM_ROOT_LABEL,
) -> BranchingTree:
    """Build a full tree by recursively splitting nodes with k-means.

    Node k-means seeds depend on each node's member set, so the resulting
    partitions are invariant to PC-axis sign flips and child reordering.
    Children are still ordered by ascending mean PC1: child 0 is the negative
    (B) side and the last child is the positive (A) side. Callers that need
    identical path labels across runs should canonicalize the PC1 sign before
    calling this function.
    """
    values = np.asarray(pcs, dtype=np.float64)
    bins = np.asarray(unit_bins, dtype=np.int64)
    if values.ndim != 2:
        raise ClusteringError("unit PCs must be a two-dimensional array")
    n_units = values.shape[0]
    if n_units == 0:
        raise ClusteringError("cannot build a tree from an empty unit set")
    if bins.shape[0] != n_units:
        raise ClusteringError("unit PCs and unit bin counts have different lengths")
    if branch_k < 2:
        raise ClusteringError("branch_k must be at least 2")
    if n_pcs < 1:
        raise ClusteringError("n_pcs must be at least 1")

    nodes: list[TreeNode] = []
    root_id = _new_node(
        nodes=nodes,
        parent_id=None,
        unit_indices=np.arange(n_units, dtype=np.int64),
        path=[],
        pcs=values,
        unit_bins=bins,
        n_pcs=n_pcs,
    )

    # Split order does not change the final full tree: every node is split
    # independently. Depth-first recursion keeps node ids deterministic without
    # a priority queue.
    if n_units > 1:
        _split_recursive(
            nodes,
            root_id,
            values,
            bins,
            n_pcs,
            branch_k,
            seed,
        )

    _leaf_lookup(nodes, n_units)
    return BranchingTree(
        nodes=nodes,
        root_id=root_id,
        root_label=root_label,
    )


def _node_seed(base_seed: int, unit_indices: np.ndarray) -> int:
    """Return a k-means seed determined by the node's member set."""
    members = np.sort(np.asarray(unit_indices, dtype=np.int64))
    return derive_seed(base_seed, _TREE_SEED_NAMESPACE, *map(int, members))


def _split_recursive(
    nodes: list[TreeNode],
    node_id: int,
    pcs: np.ndarray,
    unit_bins: np.ndarray,
    n_pcs: int,
    branch_k: int,
    seed: int,
) -> None:
    node = nodes[node_id]
    # Seed from the member set, not the path: feature-track orientation can
    # reverse child order while leaving the k-means objective unchanged.
    node_seed = _node_seed(seed, node.unit_indices)
    parts = _split_member_units(
        pcs,
        node.unit_indices,
        n_pcs=n_pcs,
        branch_k=branch_k,
        seed=node_seed,
    )
    if parts is None:
        node.stop_reason = "degenerate"
        return

    _append_children(
        nodes,
        node_id,
        parts,
        pcs,
        unit_bins,
        n_pcs,
    )
    for child_id in node.children:
        _split_recursive(
            nodes,
            child_id,
            pcs,
            unit_bins,
            n_pcs,
            branch_k,
            seed,
        )


def _split_member_units(
    pcs: np.ndarray,
    unit_indices: np.ndarray,
    *,
    n_pcs: int,
    branch_k: int,
    seed: int,
) -> list[np.ndarray] | None:
    # Small nodes become singletons directly. Larger nodes use k-means, then
    # sort children by ascending mean PC1 so paths do not depend on k-means
    # label order.
    if unit_indices.size <= 1:
        return None
    if unit_indices.size <= branch_k:
        parts = [unit_indices[index : index + 1] for index in range(unit_indices.size)]
        return _sort_by_mean_pc1(parts, pcs)

    labels, _ = kmeans_labels(
        pcs[unit_indices],
        branch_k,
        n_pcs,
        seed,
    )
    unique, inverse = np.unique(labels, return_inverse=True)
    if unique.size < 2:
        return None

    parts = [unit_indices[inverse == cluster] for cluster in range(unique.size)]
    return _sort_by_mean_pc1(parts, pcs)


def _sort_by_mean_pc1(
    parts: list[np.ndarray],
    pcs: np.ndarray,
) -> list[np.ndarray]:
    if pcs.shape[1] == 0:
        return parts
    return sorted(
        parts,
        key=lambda part: pcs[part, 0].mean(),
    )


def _append_children(
    nodes: list[TreeNode],
    parent_id: int,
    parts: list[np.ndarray],
    pcs: np.ndarray,
    unit_bins: np.ndarray,
    n_pcs: int,
) -> None:
    parent = nodes[parent_id]
    child_sse = 0.0

    for child_index, unit_indices in enumerate(parts):
        path = [*parent.path, child_index]
        child_id = _new_node(
            nodes=nodes,
            parent_id=parent_id,
            unit_indices=unit_indices,
            path=path,
            pcs=pcs,
            unit_bins=unit_bins,
            n_pcs=n_pcs,
        )
        parent.children.append(child_id)
        child_sse += nodes[child_id].sse

    # Relative SSE reduction; leaves and unsplit nodes keep split_gain None.
    parent.split_gain = (parent.sse - child_sse) / max(
        parent.sse,
        np.finfo(float).tiny,
    )


def _new_node(
    *,
    nodes: list[TreeNode],
    parent_id: int | None,
    unit_indices: np.ndarray,
    path: list[int],
    pcs: np.ndarray,
    unit_bins: np.ndarray,
    n_pcs: int,
) -> int:
    node_id = len(nodes)
    values = _node_values(pcs, unit_indices, n_pcs)
    centroid = values.mean(axis=0)
    sse = float(((values - centroid) ** 2).sum())

    nodes.append(
        TreeNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=len(path),
            path=list(path),
            unit_indices=unit_indices,
            n_working_bins=int(unit_bins[unit_indices].sum()),
            centroid=centroid,
            sse=sse,
        )
    )
    return node_id


def _node_values(
    pcs: np.ndarray,
    unit_indices: np.ndarray,
    n_pcs: int,
) -> np.ndarray:
    if pcs.shape[1] == 0:
        return np.zeros((unit_indices.size, 1), dtype=np.float64)
    n_features = max(1, min(n_pcs, pcs.shape[1]))
    return np.asarray(pcs[unit_indices, :n_features], dtype=np.float64)


def _leaf_lookup(
    nodes: list[TreeNode],
    n_units: int,
) -> np.ndarray:
    leaf_of_unit = np.full(n_units, -1, dtype=np.int64)
    for node in nodes:
        if node.is_leaf:
            leaf_of_unit[node.unit_indices] = node.node_id

    if np.any(leaf_of_unit < 0):
        raise ClusteringError("tree left some units unassigned")
    return leaf_of_unit


def build_cis_forest(
    chrom: str,
    pcs: np.ndarray,
    unit_bins: np.ndarray,
    *,
    n_pcs: int,
    branch_k: int,
    seed: int,
    node_base: int = 0,
    clusterable: np.ndarray | None = None,
) -> CisForest:
    """Build a single-root cis forest for one chromosome."""
    values = np.asarray(pcs)
    bins = np.asarray(unit_bins, dtype=np.int64)
    mask = _resolve_clusterable(values, n_pcs, clusterable)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        if values.shape[0] == 1:
            indices = np.arange(1, dtype=np.int64)
        else:
            return _make_cis_forest(
                chrom,
                trees=[],
                groups=[],
                node_base=node_base,
            )

    tree = build_branching_tree(
        values[indices],
        bins[indices],
        n_pcs=n_pcs,
        seed=seed,
        branch_k=branch_k,
        root_label=CHROM_ROOT_LABEL,
    )
    return _make_cis_forest(
        chrom,
        trees=[tree],
        groups=[indices],
        node_base=node_base,
    )


def _resolve_clusterable(
    values: np.ndarray,
    n_features: int,
    clusterable: np.ndarray | None,
) -> np.ndarray:
    """Return a validated unit mask of rows used for clustering."""
    if clusterable is None:
        return informative_mask(values, n_features)

    mask = np.asarray(clusterable, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
        raise ClusteringError("clusterable mask length does not match the unit count")
    return mask


def _make_cis_forest(
    chrom: str,
    *,
    trees: list[BranchingTree],
    groups: list[np.ndarray],
    node_base: int,
) -> CisForest:
    offsets: list[int] = []
    offset = node_base

    # Trees keep local node ids. Offsets make node ids unique across a
    # multi-tree forest such as the A/B-rooted pair.
    for tree in trees:
        offsets.append(offset)
        offset += len(tree.nodes)

    return CisForest(
        chrom=chrom,
        trees=trees,
        node_offsets=offsets,
        groups=groups,
    )


def annotate_ab(
    features: np.ndarray,
    feature_rank: np.ndarray,
    n_units: int,
    seed: int,
    *,
    n_pcs: int,
    clusterable: np.ndarray | None = None,
) -> AbAnnotation:
    """Assign per-unit A/B labels and continuous signals.

    The feature matrix must end with the standardized feature rank. K-means runs
    in that augmented space, but signals and labels are derived in the projected
    standardized PC subspace. Signals are raw signed projections on the A-B
    centroid axis; positive is the A side.
    """
    values = np.asarray(features, dtype=np.float64)
    feature_ranks = np.asarray(feature_rank, dtype=np.float64)
    if feature_ranks.shape[0] != n_units:
        raise ClusteringError("A/B feature-rank length does not match the unit count")
    if values.shape[0] != n_units:
        raise ClusteringError("A/B feature length does not match the unit count")
    if values.ndim != 2 or values.shape[1] == 0:
        raise ClusteringError("A/B features must have at least one column")

    labels = np.full(n_units, None, dtype=object)
    signals = np.full(n_units, np.nan, dtype=np.float64)
    empty = np.zeros(0, dtype=np.int64)
    mask = _resolve_clusterable(values, values.shape[1], clusterable)

    if n_units < 2:
        raise ClusteringError("A/B split requires at least two units")

    partition = _ab_partition(
        values,
        n_units,
        seed,
        mask,
        n_pcs,
    )
    if partition is None:
        return AbAnnotation(
            labels=labels,
            feature_rank=feature_ranks,
            fallback=True,
            fallback_reason="ab_degenerate",
            signals=signals,
            a_indices=empty,
            b_indices=empty,
        )

    (
        partition_labels,
        partition_signals,
        a_indices,
        b_indices,
        a_centroid_pc,
        b_centroid_pc,
    ) = partition
    return AbAnnotation(
        labels=partition_labels,
        feature_rank=feature_ranks,
        fallback=False,
        fallback_reason=None,
        signals=partition_signals,
        a_indices=a_indices,
        b_indices=b_indices,
        a_centroid_pc=a_centroid_pc,
        b_centroid_pc=b_centroid_pc,
    )


def _ab_signal(
    point: np.ndarray,
    a_centroid_pc: np.ndarray,
    b_centroid_pc: np.ndarray,
) -> float:
    """Return the raw signed projection on the A-B centroid axis.

    The signal is positive on the A side and expressed in standardized-PC
    units. It is not normalized by the centroid separation and is not clipped.
    """
    direction = a_centroid_pc - b_centroid_pc
    scale = float(np.linalg.norm(direction))
    if scale <= 0.0:
        return 0.0
    midpoint = (a_centroid_pc + b_centroid_pc) / 2.0
    return float(np.dot(point - midpoint, direction / scale))


def _ab_partition(
    features: np.ndarray,
    n_units: int,
    seed: int,
    clusterable: np.ndarray,
    n_pcs: int,
) -> (
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
    | None
):
    pc_dim = min(n_pcs, features.shape[1] - 1)
    if pc_dim < 1:
        return None

    indices = np.flatnonzero(clusterable)
    if indices.size < 2:
        return None

    subset = features[indices]
    cluster_labels, _ = kmeans_labels(
        subset,
        min(2, indices.size),
        features.shape[1],
        derive_seed(seed, _AB_SEED_NAMESPACE),
    )
    unique, inverse = np.unique(cluster_labels, return_inverse=True)
    if unique.size < 2:
        return None

    centroids = np.asarray(
        [subset[inverse == cluster].mean(axis=0) for cluster in range(unique.size)]
    )
    feature_rank_means = np.asarray(
        [subset[inverse == cluster, -1].mean() for cluster in range(unique.size)]
    )
    order = np.argsort(feature_rank_means, kind="stable")

    a_cluster = int(order[-1])
    b_cluster = int(order[0])
    a_centroid_pc = centroids[a_cluster, :pc_dim]
    b_centroid_pc = centroids[b_cluster, :pc_dim]
    signals = np.full(n_units, np.nan, dtype=np.float64)

    for unit in indices:
        point = features[unit, :pc_dim]
        # Positive signal is the A side, matching the track convention A > B.
        signals[unit] = _ab_signal(point, a_centroid_pc, b_centroid_pc)

    labels = np.full(n_units, None, dtype=object)
    a_indices = indices[signals[indices] > 0.0]
    b_indices = indices[signals[indices] < 0.0]
    if a_indices.size == 0 or b_indices.size == 0:
        return None
    labels[a_indices] = "A"
    labels[b_indices] = "B"

    return (
        labels,
        signals,
        a_indices,
        b_indices,
        a_centroid_pc,
        b_centroid_pc,
    )


def annotate_cis_forest(
    forest: CisForest,
    annotation: AbAnnotation,
    features: np.ndarray,
    *,
    n_pcs: int,
) -> None:
    """Attach per-node A/B composition, labels, and continuous signals.

    Signals use the node's mean standardized PC vector and the projected A/B
    centroids. They are raw signed projections on the A-B axis, positive on
    the A side.
    """
    values = np.asarray(features, dtype=np.float64)
    pc_dim = min(n_pcs, values.shape[1] - 1)
    pc_values = values[:, :pc_dim]
    has_centroids = (
        annotation.a_centroid_pc.size > 0 and annotation.b_centroid_pc.size > 0
    )

    for tree, group in zip(forest.trees, forest.groups):
        for node in tree.nodes:
            unit_ids = group[node.unit_indices]

            if has_centroids and unit_ids.size:
                node_feature = pc_values[unit_ids].mean(axis=0)
                node.ab_signal = _ab_signal(
                    node_feature,
                    annotation.a_centroid_pc,
                    annotation.b_centroid_pc,
                )
            else:
                node.ab_signal = None

            unit_labels = annotation.labels[unit_ids]
            n_total = int(unit_labels.size)
            n_a = int(np.sum(unit_labels == "A"))
            n_b = int(np.sum(unit_labels == "B"))
            if n_total and n_a + n_b:
                frac_a = n_a / n_total
                frac_b = n_b / n_total
                node.ab_frac_a = frac_a
                node.ab_frac_b = frac_b
                node.ab_purity = max(frac_a, frac_b)
            else:
                node.ab_frac_a = None
                node.ab_frac_b = None
                node.ab_purity = None

            # The node label follows the signed centroid signal. Nodes exactly
            # on the boundary stay unlabeled; fractions tell consumers whether
            # a node is mixed.
            if node.ab_signal is None or node.ab_signal == 0.0:
                node.ab_label = None
            elif node.ab_signal > 0.0:
                node.ab_label = "A"
            else:
                node.ab_label = "B"


def build_ab_forest(
    chrom: str,
    pcs: np.ndarray,
    unit_bins: np.ndarray,
    *,
    annotation: AbAnnotation,
    n_pcs: int,
    branch_k: int,
    seed: int,
    node_base: int = 0,
    clusterable: np.ndarray | None = None,
) -> CisForest:
    """Build an A/B-rooted forest, or a fallback single-root forest."""
    values = np.asarray(pcs)
    bins = np.asarray(unit_bins, dtype=np.int64)

    if annotation.fallback:
        mask = _resolve_clusterable(values, n_pcs, clusterable)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            if values.shape[0] == 1:
                indices = np.arange(1, dtype=np.int64)
            else:
                return _make_cis_forest(
                    chrom,
                    trees=[],
                    groups=[],
                    node_base=node_base,
                )
        tree = build_branching_tree(
            values[indices],
            bins[indices],
            n_pcs=n_pcs,
            seed=seed,
            branch_k=branch_k,
            root_label=CHROM_ROOT_LABEL,
        )
        if annotation.fallback_reason is not None:
            tree.nodes[tree.root_id].stop_reason = annotation.fallback_reason
        return _make_cis_forest(
            chrom,
            trees=[tree],
            groups=[indices],
            node_base=node_base,
        )

    a_indices = annotation.a_indices
    b_indices = annotation.b_indices
    a_tree = build_branching_tree(
        values[a_indices],
        bins[a_indices],
        n_pcs=n_pcs,
        seed=seed,
        branch_k=branch_k,
        root_label="A",
    )
    b_tree = build_branching_tree(
        values[b_indices],
        bins[b_indices],
        n_pcs=n_pcs,
        seed=seed,
        branch_k=branch_k,
        root_label="B",
    )
    return _make_cis_forest(
        chrom,
        trees=[a_tree, b_tree],
        groups=[a_indices, b_indices],
        node_base=node_base,
    )
