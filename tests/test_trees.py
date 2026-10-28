"""Tests for deterministic branching k-means trees."""

from __future__ import annotations

import numpy as np
import pytest

from tadclust.clustering import kmeans_labels
from tadclust.errors import ComputationError
from tadclust.trees import (
    build_branching_tree,
    build_cis_forest,
    format_path_label,
    single_node_tree,
)


def leaf_by_domain(tree):
    leaves = {}
    for node in tree.nodes:
        if node.is_leaf:
            for domain in node.domain_indices.tolist():
                leaves[domain] = node
    return leaves


def test_kmeans_labels_clamps_and_handles_single_cluster() -> None:
    labels, centers = kmeans_labels(
        np.array([[0.0], [10.0]]),
        5,
        1,
        0,
    )
    assert set(labels.tolist()) == {0, 1}
    assert centers.shape == (2, 1)

    labels, centers = kmeans_labels(
        np.array([[1.0], [2.0]]),
        1,
        1,
        0,
    )
    assert labels.tolist() == [0, 0]
    assert centers.shape == (1, 1)

    with pytest.raises(ComputationError):
        kmeans_labels(np.empty((0, 1)), 2, 1, 0)


def test_single_domain_is_root_leaf() -> None:
    tree = build_branching_tree(
        np.array([[1.0]]),
        np.array([5]),
        n_pcs=1,
        seed=0,
        branch_k=3,
    )

    root = tree.nodes[tree.root_id]
    assert root.is_leaf
    assert root.path == []
    assert format_path_label(tree.root_label, root.path) == "root"
    assert root.depth == 0
    assert root.domain_indices.tolist() == [0]
    assert root.n_bins == 5
    assert root.split_gain is None
    assert len(tree.nodes) == 1


def test_binary_tree_paths_and_gain() -> None:
    pcs = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    bins = np.array([10, 20, 30, 40])
    tree = build_branching_tree(
        pcs,
        bins,
        n_pcs=1,
        seed=0,
        branch_k=2,
    )

    root = tree.nodes[tree.root_id]
    assert not root.is_leaf
    assert root.domain_indices.tolist() == [0, 1, 2, 3]
    assert root.n_bins == 100
    assert root.centroid.tolist() == [0.0]
    assert root.sse == 20.0
    assert root.split_gain == pytest.approx(0.8)
    assert len(root.children) == 2

    left = tree.nodes[root.children[0]]
    right = tree.nodes[root.children[1]]
    assert left.domain_indices.tolist() == [0, 1]
    assert right.domain_indices.tolist() == [2, 3]
    assert left.path == [0]
    assert right.path == [1]
    assert format_path_label(tree.root_label, left.path) == "root.0"
    assert format_path_label(tree.root_label, right.path) == "root.1"
    assert left.centroid.tolist() == [-2.0]
    assert right.centroid.tolist() == [2.0]
    assert left.n_bins == 30
    assert right.n_bins == 70

    leaves = leaf_by_domain(tree)
    assert [
        format_path_label(tree.root_label, leaves[domain].path) for domain in range(4)
    ] == [
        "root.0.0",
        "root.0.1",
        "root.1.0",
        "root.1.1",
    ]
    assert [leaves[domain].depth for domain in range(4)] == [2, 2, 2, 2]
    assert sorted(leaves) == [0, 1, 2, 3]


def test_child_order_is_mean_pc1() -> None:
    pcs = np.array([[-5.0], [-3.0], [-1.0], [1.0], [3.0], [5.0]])
    tree = build_branching_tree(
        pcs,
        np.ones(6, dtype=np.int64),
        n_pcs=1,
        seed=0,
        branch_k=3,
    )

    root = tree.nodes[tree.root_id]
    mean_values = [
        pcs[tree.nodes[child].domain_indices, 0].mean() for child in root.children
    ]
    assert len(root.children) == 3
    assert mean_values == sorted(mean_values)


@pytest.mark.parametrize("branch_k", [2, 3, 4])
def test_branch_arity(branch_k) -> None:
    pcs = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
    tree = build_branching_tree(
        pcs,
        np.ones(12, dtype=np.int64),
        n_pcs=1,
        seed=0,
        branch_k=branch_k,
    )

    assert len(tree.nodes[tree.root_id].children) == branch_k


def test_singleton_split_orders_by_pc1() -> None:
    pcs = np.array([[3.0], [1.0], [-1.0]])
    tree = build_branching_tree(
        pcs,
        np.ones(3, dtype=np.int64),
        n_pcs=1,
        seed=0,
        branch_k=3,
    )

    root = tree.nodes[tree.root_id]
    assert [tree.nodes[child].domain_indices.tolist() for child in root.children] == [
        [2],
        [1],
        [0],
    ]
    assert [tree.nodes[child].path for child in root.children] == [[0], [1], [2]]
    assert root.split_gain == pytest.approx(1.0)


def test_degenerate_split_stops(monkeypatch) -> None:
    def degenerate(pcs, k, n_pcs, seed, n_init=10):
        return np.zeros(len(pcs), dtype=np.int64), np.zeros((1, 1))

    monkeypatch.setattr("tadclust.trees.kmeans_labels", degenerate)
    tree = build_branching_tree(
        np.array([[0.0], [1.0], [2.0], [3.0]]),
        np.ones(4, dtype=np.int64),
        n_pcs=1,
        seed=0,
        branch_k=2,
    )

    root = tree.nodes[tree.root_id]
    assert root.is_leaf
    assert root.stop_reason == "degenerate"
    assert root.domain_indices.tolist() == [0, 1, 2, 3]


def test_single_node_tree() -> None:
    tree = single_node_tree(3, np.array([1, 2, 3]))

    root = tree.nodes[tree.root_id]
    assert root.is_leaf
    assert root.domain_indices.tolist() == [0, 1, 2]
    assert root.n_bins == 6
    assert root.sse == 0.0
    assert format_path_label(tree.root_label, root.path) == "root"
    assert len(tree.nodes) == 1


def test_build_cis_forest_groups_and_offsets() -> None:
    pcs = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    forest = build_cis_forest(
        "chr1",
        pcs,
        np.ones(4, dtype=np.int64),
        n_pcs=1,
        branch_k=2,
        seed=0,
        node_base=10,
    )

    assert forest.node_offsets == [10]
    assert forest.groups[0].tolist() == [0, 1, 2, 3]
    assert forest.trees[0].root_label == "root"

    leaves = leaf_by_domain(forest.trees[0])
    global_leaf_ids = [
        leaves[domain].node_id + forest.node_offsets[0] for domain in range(4)
    ]
    assert len(set(global_leaf_ids)) == 4
    assert all(node_id >= 10 for node_id in global_leaf_ids)
    assert [
        format_path_label(forest.trees[0].root_label, leaves[domain].path)
        for domain in range(4)
    ] == [
        "root.0.0",
        "root.0.1",
        "root.1.0",
        "root.1.1",
    ]


def test_invalid_tree_inputs() -> None:
    with pytest.raises(ComputationError):
        build_branching_tree(
            np.empty((0, 1)),
            np.empty(0, dtype=np.int64),
            n_pcs=1,
            seed=0,
            branch_k=2,
        )
    with pytest.raises(ComputationError):
        build_branching_tree(
            np.array([[0.0], [1.0]]),
            np.ones(2, dtype=np.int64),
            n_pcs=1,
            seed=0,
            branch_k=1,
        )
    with pytest.raises(ComputationError):
        build_branching_tree(
            np.array([[0.0], [1.0]]),
            np.ones(3, dtype=np.int64),
            n_pcs=1,
            seed=0,
            branch_k=2,
        )
    with pytest.raises(ComputationError):
        build_branching_tree(
            np.array([[0.0], [1.0]]),
            np.ones(2, dtype=np.int64),
            n_pcs=0,
            seed=0,
            branch_k=2,
        )
