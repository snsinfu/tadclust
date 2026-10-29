"""Tests for A/B splitting, scores, and rooted forests."""

from __future__ import annotations

import numpy as np
import pytest

from tadclust.errors import ComputationError
from tadclust.track import ab_feature_matrix
from tadclust.trees import (
    AbAnnotation,
    annotate_ab,
    annotate_cis_forest,
    build_ab_forest,
    build_cis_forest,
)


def test_annotate_ab_kmeans2() -> None:
    pcs = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    rank = np.array([0.0, 0.1, 0.9, 1.0])
    features = ab_feature_matrix(pcs, rank, 1)

    annotation = annotate_ab(features, rank, 4, "kmeans2", 0)

    assert not annotation.fallback
    assert annotation.labels.tolist() == ["B", "B", "A", "A"]
    assert annotation.a_indices.tolist() == [2, 3]
    assert annotation.b_indices.tolist() == [0, 1]
    assert np.all(annotation.scores[annotation.a_indices] > 0)
    assert np.all(annotation.scores[annotation.b_indices] < 0)
    assert np.corrcoef(annotation.scores, rank)[0, 1] > 0
    assert np.allclose(
        annotation.margins,
        np.abs(annotation.scores),
    )


def test_annotate_ab_kmeans3_merges_two_highest_signal_clusters() -> None:
    pcs = np.array([[-3.0], [-1.0], [0.0], [2.0], [3.0]])
    rank = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    features = ab_feature_matrix(pcs, rank, 1)

    annotation = annotate_ab(features, rank, 5, "kmeans3", 0)

    assert not annotation.fallback
    assert annotation.a_indices.size + annotation.b_indices.size == 5
    assert set(annotation.labels.tolist()) == {"A", "B"}
    assert 4 in annotation.a_indices.tolist()
    assert 0 in annotation.b_indices.tolist()
    assert (
        annotation.rank[annotation.a_indices].mean()
        > annotation.rank[annotation.b_indices].mean()
    )


def test_annotate_ab_rejects_too_few_domains() -> None:
    with pytest.raises(ComputationError):
        annotate_ab(
            np.zeros((1, 2)),
            np.array([0.5]),
            1,
            "kmeans2",
            0,
        )


def test_annotate_ab_degenerate(monkeypatch) -> None:
    def degenerate(pcs, k, n_pcs, seed, n_init=10):
        return np.zeros(len(pcs), dtype=np.int64), np.zeros((1, 2))

    monkeypatch.setattr("tadclust.trees.kmeans_labels", degenerate)
    annotation = annotate_ab(
        np.zeros((4, 2)),
        np.array([0.0, 0.25, 0.5, 1.0]),
        4,
        "kmeans2",
        0,
    )

    assert annotation.fallback
    assert annotation.fallback_reason == "ab_degenerate"


def test_annotate_ab_rejects_invalid_inputs() -> None:
    with pytest.raises(ComputationError):
        annotate_ab(np.zeros((2, 2)), np.zeros(3), 3, "kmeans2", 0)
    with pytest.raises(ComputationError):
        annotate_ab(np.zeros((2, 2)), np.zeros(2), 2, "bad", 0)


def test_annotate_cis_forest_composition() -> None:
    pcs = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    forest = build_cis_forest(
        "chr1",
        pcs,
        np.ones(4, dtype=np.int64),
        n_pcs=1,
        branch_k=2,
        seed=0,
    )
    labels = np.array(["A", "A", "B", "B"], dtype=object)
    rank = np.array([0.0, 0.0, 1.0, 1.0])

    annotate_cis_forest(forest, labels, rank)

    root = forest.trees[0].nodes[forest.trees[0].root_id]
    assert root.ab_label == "mixed"
    assert root.ab_frac_a == 0.5
    assert root.ab_purity == 0.5
    assert root.ab_rank_mean == 0.5

    for node in forest.trees[0].nodes:
        if node.is_leaf:
            assert node.ab_label in {"A", "B"}
            assert node.ab_frac_a in {0.0, 1.0}


def test_build_ab_forest_two_roots() -> None:
    pcs = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    bins = np.ones(4, dtype=np.int64)
    rank = np.array([0.0, 0.1, 0.9, 1.0])
    features = ab_feature_matrix(pcs, rank, 1)
    annotation = annotate_ab(features, rank, 4, "kmeans2", 0)

    forest = build_ab_forest(
        "chr1",
        pcs,
        bins,
        annotation=annotation,
        n_pcs=1,
        branch_k=2,
        seed=0,
        node_base=100,
    )

    assert [tree.root_label for tree in forest.trees] == ["A", "B"]
    assert forest.node_offsets[0] == 100
    assert forest.node_offsets[1] == 100 + len(forest.trees[0].nodes)
    assert forest.groups[0].tolist() == annotation.a_indices.tolist()
    assert forest.groups[1].tolist() == annotation.b_indices.tolist()

    annotate_cis_forest(forest, annotation.labels, annotation.rank)
    assert forest.trees[0].nodes[0].ab_label == "A"
    assert forest.trees[1].nodes[0].ab_label == "B"


def test_build_ab_forest_degenerate_fallback() -> None:
    annotation = AbAnnotation(
        labels=np.array([None, None], dtype=object),
        rank=np.array([0.5, 0.5]),
        split="kmeans2",
        fallback=True,
        fallback_reason="ab_degenerate",
        scores=np.full(2, np.nan),
        margins=np.full(2, np.nan),
        a_indices=np.zeros(0, dtype=np.int64),
        b_indices=np.zeros(0, dtype=np.int64),
    )
    forest = build_ab_forest(
        "chr1",
        np.array([[0.0], [1.0]]),
        np.array([1, 1], dtype=np.int64),
        annotation=annotation,
        n_pcs=1,
        branch_k=2,
        seed=0,
    )

    assert len(forest.trees) == 1
    assert forest.trees[0].root_label == "root"
    assert forest.trees[0].nodes[0].stop_reason == "ab_degenerate"
    assert forest.groups[0].tolist() == [0, 1]
