"""Deterministic branching k-means trees over domain PCs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clustering import derive_seed, informative_mask, kmeans_labels
from .config import AB_SPLIT_CHOICES
from .errors import ComputationError

AB_NODE_PURITY = 0.8
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
    domain_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    n_bins: int = 0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    sse: float = 0.0
    split_gain: float | None = None
    stop_reason: str | None = None
    ab_label: str | None = None
    ab_frac_a: float | None = None
    ab_purity: float | None = None
    ab_rank_mean: float | None = None

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
    """Per-domain A/B labels, ranks, and continuous scores."""

    labels: np.ndarray
    rank: np.ndarray
    split: str
    fallback: bool
    fallback_reason: str | None
    scores: np.ndarray
    margins: np.ndarray
    a_indices: np.ndarray
    b_indices: np.ndarray


def format_path_label(root_label: str, path: list[int]) -> str:
    """Format a root label and child-index path for display."""
    if not path:
        return root_label
    return f"{root_label}." + ".".join(str(index) for index in path)


def build_branching_tree(
    pcs: np.ndarray,
    domain_bins: np.ndarray,
    n_pcs: int,
    seed: int,
    branch_k: int,
    root_label: str = "root",
) -> BranchingTree:
    """Build a full tree by recursively splitting nodes with k-means.

    Children are ordered by descending mean PC1. Callers should canonicalize
    the PC1 sign before calling this function when deterministic paths matter.
    """
    values = np.asarray(pcs, dtype=np.float64)
    bins = np.asarray(domain_bins, dtype=np.int64)
    if values.ndim != 2:
        raise ComputationError("domain PCs must be a two-dimensional array")
    n_domains = values.shape[0]
    if n_domains == 0:
        raise ComputationError("cannot build a tree from an empty domain set")
    if bins.shape[0] != n_domains:
        raise ComputationError(
            "domain PCs and domain bin counts have different lengths"
        )
    if branch_k < 2:
        raise ComputationError("branch_k must be at least 2")
    if n_pcs < 1:
        raise ComputationError("n_pcs must be at least 1")

    nodes: list[TreeNode] = []
    root_id = _new_node(
        nodes=nodes,
        parent_id=None,
        domain_indices=np.arange(n_domains, dtype=np.int64),
        path=[],
        pcs=values,
        domain_bins=bins,
        n_pcs=n_pcs,
    )

    # Split order does not change the final full tree: every node is split
    # independently. Depth-first recursion keeps node ids deterministic without
    # a priority queue.
    if n_domains > 1:
        _split_recursive(
            nodes,
            root_id,
            values,
            bins,
            n_pcs,
            branch_k,
            seed,
        )

    _leaf_lookup(nodes, n_domains)
    return BranchingTree(
        nodes=nodes,
        root_id=root_id,
        root_label=root_label,
    )


def _split_recursive(
    nodes: list[TreeNode],
    node_id: int,
    pcs: np.ndarray,
    domain_bins: np.ndarray,
    n_pcs: int,
    branch_k: int,
    seed: int,
) -> None:
    node = nodes[node_id]
    node_seed = derive_seed(seed, _TREE_SEED_NAMESPACE, *node.path)
    parts = _split_member_domains(
        pcs,
        node.domain_indices,
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
        domain_bins,
        n_pcs,
    )
    for child_id in node.children:
        _split_recursive(
            nodes,
            child_id,
            pcs,
            domain_bins,
            n_pcs,
            branch_k,
            seed,
        )


def _split_member_domains(
    pcs: np.ndarray,
    domain_indices: np.ndarray,
    *,
    n_pcs: int,
    branch_k: int,
    seed: int,
) -> list[np.ndarray] | None:
    # Small nodes become singletons directly. Larger nodes use k-means, then
    # sort children by mean PC1 so paths do not depend on k-means label order.
    if domain_indices.size <= 1:
        return None
    if domain_indices.size <= branch_k:
        parts = [
            domain_indices[index : index + 1] for index in range(domain_indices.size)
        ]
        return _sort_by_mean_pc1(parts, pcs)

    labels, _ = kmeans_labels(
        pcs[domain_indices],
        branch_k,
        n_pcs,
        seed,
    )
    unique, inverse = np.unique(labels, return_inverse=True)
    if unique.size < 2:
        return None

    parts = [domain_indices[inverse == cluster] for cluster in range(unique.size)]
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
        reverse=True,
    )


def _append_children(
    nodes: list[TreeNode],
    parent_id: int,
    parts: list[np.ndarray],
    pcs: np.ndarray,
    domain_bins: np.ndarray,
    n_pcs: int,
) -> None:
    parent = nodes[parent_id]
    child_sse = 0.0

    for child_index, domain_indices in enumerate(parts):
        path = [*parent.path, child_index]
        child_id = _new_node(
            nodes=nodes,
            parent_id=parent_id,
            domain_indices=domain_indices,
            path=path,
            pcs=pcs,
            domain_bins=domain_bins,
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
    domain_indices: np.ndarray,
    path: list[int],
    pcs: np.ndarray,
    domain_bins: np.ndarray,
    n_pcs: int,
) -> int:
    node_id = len(nodes)
    values = _node_values(pcs, domain_indices, n_pcs)
    centroid = values.mean(axis=0)
    sse = float(((values - centroid) ** 2).sum())

    nodes.append(
        TreeNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=len(path),
            path=list(path),
            domain_indices=domain_indices,
            n_bins=int(domain_bins[domain_indices].sum()),
            centroid=centroid,
            sse=sse,
        )
    )
    return node_id


def _node_values(
    pcs: np.ndarray,
    domain_indices: np.ndarray,
    n_pcs: int,
) -> np.ndarray:
    if pcs.shape[1] == 0:
        return np.zeros((domain_indices.size, 1), dtype=np.float64)
    n_features = max(1, min(n_pcs, pcs.shape[1]))
    return np.asarray(pcs[domain_indices, :n_features], dtype=np.float64)


def _leaf_lookup(
    nodes: list[TreeNode],
    n_domains: int,
) -> np.ndarray:
    leaf_of_domain = np.full(n_domains, -1, dtype=np.int64)
    for node in nodes:
        if node.is_leaf:
            leaf_of_domain[node.domain_indices] = node.node_id

    if np.any(leaf_of_domain < 0):
        raise ComputationError("tree left some domains unassigned")
    return leaf_of_domain


def build_cis_forest(
    chrom: str,
    pcs: np.ndarray,
    domain_bins: np.ndarray,
    *,
    n_pcs: int,
    branch_k: int,
    seed: int,
    node_base: int = 0,
    clusterable: np.ndarray | None = None,
) -> CisForest:
    """Build a single-root cis forest for one chromosome."""
    values = np.asarray(pcs)
    bins = np.asarray(domain_bins, dtype=np.int64)
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
        root_label="root",
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
    """Return a validated domain mask of rows used for clustering."""
    if clusterable is None:
        return informative_mask(values, n_features)

    mask = np.asarray(clusterable, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
        raise ComputationError(
            "clusterable mask length does not match the domain count"
        )
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
    rank: np.ndarray,
    n_domains: int,
    ab_split: str,
    seed: int,
    clusterable: np.ndarray | None = None,
) -> AbAnnotation:
    """Assign per-domain A/B labels and boundary scores.

    The feature matrix must end with the standardized signal rank.
    """
    values = np.asarray(features, dtype=np.float64)
    ranks = np.asarray(rank, dtype=np.float64)
    if ranks.shape[0] != n_domains:
        raise ComputationError("A/B rank length does not match the domain count")
    if values.shape[0] != n_domains:
        raise ComputationError("A/B feature length does not match the domain count")
    if values.ndim != 2 or values.shape[1] == 0:
        raise ComputationError("A/B features must have at least one column")
    if ab_split not in AB_SPLIT_CHOICES:
        raise ComputationError(f"unknown A/B split policy: {ab_split!r}")

    labels = np.full(n_domains, None, dtype=object)
    scores = np.full(n_domains, np.nan, dtype=np.float64)
    margins = np.full(n_domains, np.nan, dtype=np.float64)
    empty = np.zeros(0, dtype=np.int64)
    mask = _resolve_clusterable(values, values.shape[1], clusterable)

    if n_domains < 2:
        raise ComputationError("A/B split requires at least two domains")

    partition = _ab_partition(values, n_domains, ab_split, seed, mask)
    if partition is None:
        return AbAnnotation(
            labels=labels,
            rank=ranks,
            split=ab_split,
            fallback=True,
            fallback_reason="ab_degenerate",
            scores=scores,
            margins=margins,
            a_indices=empty,
            b_indices=empty,
        )

    partition_labels, partition_scores, partition_margins, a_indices, b_indices = (
        partition
    )
    return AbAnnotation(
        labels=partition_labels,
        rank=ranks,
        split=ab_split,
        fallback=False,
        fallback_reason=None,
        scores=partition_scores,
        margins=partition_margins,
        a_indices=a_indices,
        b_indices=b_indices,
    )


def _ab_partition(
    features: np.ndarray,
    n_domains: int,
    ab_split: str,
    seed: int,
    clusterable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    indices = np.flatnonzero(clusterable)
    if indices.size < 2:
        return None

    subset = features[indices]
    n_clusters = 2 if ab_split == "kmeans2" else 3
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
    rank_means = np.asarray(
        [subset[inverse == cluster, -1].mean() for cluster in range(unique.size)]
    )
    order = np.argsort(rank_means, kind="stable")

    if ab_split == "kmeans2" or unique.size == 2:
        a_clusters = [int(order[-1])]
        b_clusters = [int(order[0])]
    else:
        a_clusters = [int(order[-1]), int(order[-2])]
        b_clusters = [int(order[0])]

    a_indices = indices[np.flatnonzero(np.isin(inverse, a_clusters))]
    b_indices = indices[np.flatnonzero(np.isin(inverse, b_clusters))]
    if a_indices.size == 0 or b_indices.size == 0:
        return None

    a_centroids = centroids[a_clusters]
    b_centroid = centroids[b_clusters[0]]
    labels = np.full(n_domains, None, dtype=object)
    labels[a_indices] = "A"
    labels[b_indices] = "B"
    scores = np.full(n_domains, np.nan, dtype=np.float64)
    margins = np.full(n_domains, np.nan, dtype=np.float64)

    for domain in indices:
        a_distance = np.min(np.linalg.norm(a_centroids - features[domain], axis=1))
        b_distance = np.linalg.norm(b_centroid - features[domain])
        # Positive score is the A side, matching the track convention A > B.
        score = b_distance - a_distance
        scores[domain] = score
        margins[domain] = abs(score)

    return labels, scores, margins, a_indices, b_indices


def annotate_cis_forest(
    forest: CisForest,
    labels: np.ndarray,
    rank: np.ndarray,
    purity: float = AB_NODE_PURITY,
) -> None:
    """Attach per-node A/B composition fields to every node."""
    for tree, group in zip(forest.trees, forest.groups):
        for node in tree.nodes:
            domain_ids = group[node.domain_indices]
            domain_ranks = rank[domain_ids]
            node.ab_rank_mean = (
                float(domain_ranks.mean()) if domain_ranks.size else None
            )

            domain_labels = labels[domain_ids]
            known = np.asarray(
                [label is not None for label in domain_labels],
                dtype=bool,
            )
            if not known.any():
                node.ab_label = None
                node.ab_frac_a = None
                node.ab_purity = None
                continue

            n_a = int(np.sum(domain_labels[known] == "A"))
            frac_a = n_a / int(known.sum())
            node.ab_frac_a = frac_a
            node.ab_purity = max(frac_a, 1.0 - frac_a)
            if frac_a >= purity:
                node.ab_label = "A"
            elif frac_a <= 1.0 - purity:
                node.ab_label = "B"
            else:
                node.ab_label = "mixed"


def build_ab_forest(
    chrom: str,
    pcs: np.ndarray,
    domain_bins: np.ndarray,
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
    bins = np.asarray(domain_bins, dtype=np.int64)

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
            root_label="root",
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
