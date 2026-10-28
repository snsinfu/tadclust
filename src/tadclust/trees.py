"""Deterministic branching k-means trees over domain PCs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clustering import kmeans_labels
from .errors import ComputationError


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

    Children are ordered by mean PC1. Callers should canonicalize the PC1 sign
    before calling this function when deterministic paths matter.
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
    parts = _split_member_domains(
        pcs,
        node.domain_indices,
        n_pcs=n_pcs,
        branch_k=branch_k,
        seed=seed,
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
    return sorted(parts, key=lambda part: pcs[part, 0].mean())


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


def single_node_tree(
    n_domains: int,
    domain_bins: np.ndarray,
    root_label: str = "root",
) -> BranchingTree:
    """Build a one-node tree, used when PCs have no positive mode."""
    if n_domains < 1:
        raise ComputationError("cannot build a tree from an empty domain set")

    bins = np.asarray(domain_bins, dtype=np.int64)
    if bins.shape[0] != n_domains:
        raise ComputationError(
            "domain count and domain bin counts have different lengths"
        )

    root = TreeNode(
        node_id=0,
        parent_id=None,
        domain_indices=np.arange(n_domains, dtype=np.int64),
        n_bins=int(bins.sum()),
        centroid=np.zeros(0, dtype=np.float64),
        path=[],
    )
    return BranchingTree(
        nodes=[root],
        root_id=root.node_id,
        root_label=root_label,
    )


def build_cis_forest(
    chrom: str,
    pcs: np.ndarray,
    domain_bins: np.ndarray,
    *,
    n_pcs: int,
    branch_k: int,
    seed: int,
    node_base: int = 0,
) -> CisForest:
    """Build a single-root cis forest for one chromosome."""
    tree = build_branching_tree(
        pcs,
        domain_bins,
        n_pcs=n_pcs,
        seed=seed,
        branch_k=branch_k,
        root_label="root",
    )
    n_domains = np.asarray(pcs).shape[0]
    return _make_cis_forest(
        chrom,
        trees=[tree],
        groups=[np.arange(n_domains, dtype=np.int64)],
        node_base=node_base,
    )


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
