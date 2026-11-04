"""Tests for maximum spanning tree node summaries."""

from __future__ import annotations

import numpy as np
import pytest

from tadclust.errors import ClusteringError
from tadclust.mst import maximum_spanning_tree


def test_maximum_spanning_tree_selects_strongest_links() -> None:
    excess = np.array(
        [
            [0.0, 3.0, 1.0, 0.5],
            [3.0, 0.0, 2.0, 4.0],
            [1.0, 2.0, 0.0, 5.0],
            [0.5, 4.0, 5.0, 0.0],
        ]
    )

    mst = maximum_spanning_tree(excess, np.arange(4))

    assert mst.edge_u.tolist() == [0, 1, 2]
    assert mst.edge_v.tolist() == [1, 3, 3]
    assert mst.edge_weight.tolist() == [3.0, 4.0, 5.0]
    assert mst.bottleneck == pytest.approx(3.0)
    assert mst.components_at_zero == 1
    assert mst.lcc_frac_at_zero == pytest.approx(1.0)


def test_maximum_spanning_tree_returns_selected_indices() -> None:
    excess = np.full((4, 4), 9.0)
    np.fill_diagonal(excess, 0.0)

    mst = maximum_spanning_tree(excess, np.array([3, 1]))

    assert mst.edge_u.tolist() == [1]
    assert mst.edge_v.tolist() == [3]
    assert mst.edge_weight.tolist() == [9.0]


def test_zero_threshold_components_separate_weak_units() -> None:
    excess = np.array(
        [
            [0.0, 2.0, -1.0, -3.0],
            [2.0, 0.0, -2.0, -4.0],
            [-1.0, -2.0, 0.0, -5.0],
            [-3.0, -4.0, -5.0, 0.0],
        ]
    )

    mst = maximum_spanning_tree(excess, np.arange(4))

    assert mst.edge_weight.tolist() == [-3.0, -1.0, 2.0]
    assert mst.bottleneck == pytest.approx(-3.0)
    assert mst.components_at_zero == 3
    assert mst.lcc_frac_at_zero == pytest.approx(0.5)


def test_zero_threshold_keeps_a_fully_negative_node_split() -> None:
    excess = np.array(
        [
            [0.0, -1.0, -2.0],
            [-1.0, 0.0, -3.0],
            [-2.0, -3.0, 0.0],
        ]
    )

    mst = maximum_spanning_tree(excess, np.arange(3))

    assert mst.edge_weight.tolist() == [-2.0, -1.0]
    assert mst.components_at_zero == 3
    assert mst.lcc_frac_at_zero == pytest.approx(1.0 / 3.0)


def test_zero_threshold_connects_an_enriched_node() -> None:
    excess = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )

    mst = maximum_spanning_tree(excess, np.arange(3))

    assert mst.edge_weight.tolist() == [2.0, 3.0]
    assert mst.components_at_zero == 1
    assert mst.lcc_frac_at_zero == pytest.approx(1.0)


def test_singleton_node_has_no_edges() -> None:
    mst = maximum_spanning_tree(np.zeros((1, 1)), np.array([0]))

    assert mst.edge_u.size == 0
    assert mst.edge_v.size == 0
    assert mst.edge_weight.size == 0
    assert mst.bottleneck is None
    assert mst.components_at_zero is None
    assert mst.lcc_frac_at_zero is None


def test_two_unit_node_reports_its_single_edge() -> None:
    excess = np.array([[0.0, -0.5], [-0.5, 0.0]])

    mst = maximum_spanning_tree(excess, np.array([1, 0]))

    assert mst.edge_u.tolist() == [0]
    assert mst.edge_v.tolist() == [1]
    assert mst.edge_weight.tolist() == [-0.5]
    assert mst.bottleneck == pytest.approx(-0.5)
    assert mst.components_at_zero == 2
    assert mst.lcc_frac_at_zero == pytest.approx(0.5)


def test_equal_weights_are_deterministic_under_index_reordering() -> None:
    excess = np.ones((4, 4))
    np.fill_diagonal(excess, 0.0)

    first = maximum_spanning_tree(excess, np.array([0, 1, 2, 3]))
    second = maximum_spanning_tree(excess, np.array([3, 1, 0, 2]))

    assert first.edge_u.tolist() == second.edge_u.tolist()
    assert first.edge_v.tolist() == second.edge_v.tolist()
    assert first.edge_weight.tolist() == second.edge_weight.tolist()
    assert first.edge_u.tolist() == [0, 0, 0]
    assert first.edge_v.tolist() == [1, 2, 3]


def test_out_of_bounds_index_raises() -> None:
    with pytest.raises(ClusteringError, match="out of bounds"):
        maximum_spanning_tree(np.zeros((2, 2)), np.array([0, 2]))


def test_non_square_matrix_raises() -> None:
    with pytest.raises(ClusteringError, match="square"):
        maximum_spanning_tree(np.zeros((2, 3)), np.array([0, 1]))
