"""Deterministic branching k-means trees over unit PCs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clustering import derive_seed, informative_mask, kmeans_labels
from .config import AB_SPLIT_POLICY_CHOICES
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
    ab_purity: float | None = None
    ab_score: float | None = None

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
    """Per-unit A/B labels, feature ranks, and continuous scores.

    The A/B centroids are stored in the standardized PC subspace. The feature
    rank is used only to fit and orient the centroids, not to score units.
    """

    labels: np.ndarray
    feature_rank: np.ndarray
    split_policy: str
    fallback: bool
    fallback_reason: str | None
    scores: np.ndarray
    margins: np.ndarray
    a_indices: np.ndarray
    b_indices: np.ndarray
    a_centroids_pc: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
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

    Children are ordered by ascending mean PC1: child 0 is the negative (B)
    side and the last child is the positive (A) side. Callers should
    canonicalize the PC1 sign before calling this function when deterministic
    paths matter.
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
    node_seed = derive_seed(seed, _TREE_SEED_NAMESPACE, *node.path)
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
    split_policy: str,
    seed: int,
    *,
    n_pcs: int,
    clusterable: np.ndarray | None = None,
) -> AbAnnotation:
    """Assign per-unit A/B labels and boundary scores.

    The feature matrix must end with the standardized feature rank. K-means runs
    in that augmented space, but scores and labels are derived in the projected
    standardized PC subspace.
    """
    values = np.asarray(features, dtype=np.float64)
    feature_ranks = np.asarray(feature_rank, dtype=np.float64)
    if feature_ranks.shape[0] != n_units:
        raise ClusteringError("A/B feature-rank length does not match the unit count")
    if values.shape[0] != n_units:
        raise ClusteringError("A/B feature length does not match the unit count")
    if values.ndim != 2 or values.shape[1] == 0:
        raise ClusteringError("A/B features must have at least one column")
    if split_policy not in AB_SPLIT_POLICY_CHOICES:
        raise ClusteringError(f"unknown A/B split policy: {split_policy!r}")

    labels = np.full(n_units, None, dtype=object)
    scores = np.full(n_units, np.nan, dtype=np.float64)
    margins = np.full(n_units, np.nan, dtype=np.float64)
    empty = np.zeros(0, dtype=np.int64)
    mask = _resolve_clusterable(values, values.shape[1], clusterable)

    if n_units < 2:
        raise ClusteringError("A/B split requires at least two units")

    partition = _ab_partition(
        values,
        n_units,
        split_policy,
        seed,
        mask,
        n_pcs,
    )
    if partition is None:
        return AbAnnotation(
            labels=labels,
            feature_rank=feature_ranks,
            split_policy=split_policy,
            fallback=True,
            fallback_reason="ab_degenerate",
            scores=scores,
            margins=margins,
            a_indices=empty,
            b_indices=empty,
        )

    (
        partition_labels,
        partition_scores,
        partition_margins,
        a_indices,
        b_indices,
        a_centroids_pc,
        b_centroid_pc,
    ) = partition
    return AbAnnotation(
        labels=partition_labels,
        feature_rank=feature_ranks,
        split_policy=split_policy,
        fallback=False,
        fallback_reason=None,
        scores=partition_scores,
        margins=partition_margins,
        a_indices=a_indices,
        b_indices=b_indices,
        a_centroids_pc=a_centroids_pc,
        b_centroid_pc=b_centroid_pc,
    )


def _ab_partition(
    features: np.ndarray,
    n_units: int,
    split_policy: str,
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
    n_clusters = 2 if split_policy == "kmeans2" else 3
    cluster_labels, _ = kmeans_labels(
        subset,
        min(n_clusters, indices.size),
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

    if split_policy == "kmeans2" or unique.size == 2:
        a_clusters = [int(order[-1])]
        b_clusters = [int(order[0])]
    else:
        a_clusters = [int(order[-1]), int(order[-2])]
        b_clusters = [int(order[0])]

    a_centroids_pc = centroids[a_clusters][:, :pc_dim]
    b_centroid_pc = centroids[b_clusters[0]][:pc_dim]
    scores = np.full(n_units, np.nan, dtype=np.float64)
    margins = np.full(n_units, np.nan, dtype=np.float64)

    for unit in indices:
        point = features[unit, :pc_dim]
        a_distance = np.min(np.linalg.norm(a_centroids_pc - point, axis=1))
        b_distance = np.linalg.norm(b_centroid_pc - point)
        # Positive score is the A side, matching the track convention A > B.
        score = b_distance - a_distance
        scores[unit] = score
        margins[unit] = abs(score)

    labels = np.full(n_units, None, dtype=object)
    a_indices = indices[scores[indices] > 0.0]
    b_indices = indices[scores[indices] < 0.0]
    if a_indices.size == 0 or b_indices.size == 0:
        return None
    labels[a_indices] = "A"
    labels[b_indices] = "B"

    return (
        labels,
        scores,
        margins,
        a_indices,
        b_indices,
        a_centroids_pc,
        b_centroid_pc,
    )


def annotate_cis_forest(
    forest: CisForest,
    annotation: AbAnnotation,
    features: np.ndarray,
    *,
    n_pcs: int,
) -> None:
    """Attach per-node A/B composition, feature rank, labels, and boundary scores.

    Scores use the node's mean standardized PC vector and the projected A/B
    centroids.
    """
    values = np.asarray(features, dtype=np.float64)
    pc_dim = min(n_pcs, values.shape[1] - 1)
    pc_values = values[:, :pc_dim]
    has_centroids = (
        annotation.a_centroids_pc.size > 0 and annotation.b_centroid_pc.size > 0
    )

    for tree, group in zip(forest.trees, forest.groups):
        for node in tree.nodes:
            unit_ids = group[node.unit_indices]

            if has_centroids and unit_ids.size:
                node_feature = pc_values[unit_ids].mean(axis=0)
                a_distance = float(
                    np.min(
                        np.linalg.norm(
                            annotation.a_centroids_pc - node_feature,
                            axis=1,
                        )
                    )
                )
                b_distance = float(
                    np.linalg.norm(annotation.b_centroid_pc - node_feature)
                )
                node.ab_score = b_distance - a_distance
            else:
                node.ab_score = None

            unit_labels = annotation.labels[unit_ids]
            known = np.asarray(
                [label is not None for label in unit_labels],
                dtype=bool,
            )
            if known.any():
                n_a = int(np.sum(unit_labels[known] == "A"))
                frac_a = n_a / int(known.sum())
                node.ab_frac_a = frac_a
                node.ab_purity = max(frac_a, 1.0 - frac_a)
            else:
                node.ab_frac_a = None
                node.ab_purity = None

            # The node label follows the signed centroid score. Nodes exactly
            # on the boundary stay unlabeled; fractions tell consumers whether
            # a node is mixed.
            if node.ab_score is None or node.ab_score == 0.0:
                node.ab_label = None
            elif node.ab_score > 0.0:
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
