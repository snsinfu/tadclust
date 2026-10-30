"""Tests for A/B splitting, scores, and rooted forests."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from tadclust.errors import ClusteringError
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

    annotation = annotate_ab(features, rank, 4, "kmeans2", 0, n_pcs=1)

    assert not annotation.fallback
    assert annotation.labels.tolist() == ["B", "B", "A", "A"]
    assert annotation.a_indices.tolist() == [2, 3]
    assert annotation.b_indices.tolist() == [0, 1]
    assert np.all(annotation.scores[annotation.a_indices] > 0)
    assert np.all(annotation.scores[annotation.b_indices] < 0)
    a_distance = np.linalg.norm(
        features[:, :1] - annotation.a_centroids_pc[0],
        axis=1,
    )
    b_distance = np.linalg.norm(
        features[:, :1] - annotation.b_centroid_pc,
        axis=1,
    )
    np.testing.assert_allclose(annotation.scores, b_distance - a_distance)
    assert np.allclose(
        annotation.margins,
        np.abs(annotation.scores),
    )


def test_ab_score_uses_pc_space_not_rank() -> None:
    pcs = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    rank1 = np.array([0.0, 0.1, 0.9, 1.0])
    rank2 = np.array([0.0, 0.0, 100.0, 100.0])
    features1 = ab_feature_matrix(pcs, rank1, 1)
    features2 = ab_feature_matrix(pcs, rank2, 1)
    annotation = AbAnnotation(
        labels=np.array(["B", "B", "A", "A"], dtype=object),
        feature_rank=rank1,
        split_policy="kmeans2",
        fallback=False,
        fallback_reason=None,
        scores=np.zeros(4),
        margins=np.zeros(4),
        a_indices=np.array([2, 3]),
        b_indices=np.array([0, 1]),
        a_centroids_pc=features1[[2, 3], :1].mean(axis=0, keepdims=True),
        b_centroid_pc=features1[[0, 1], :1].mean(axis=0),
    )
    forest1 = build_cis_forest(
        "chr1",
        pcs,
        np.ones(4, dtype=np.int64),
        n_pcs=1,
        branch_k=2,
        seed=0,
    )
    forest2 = build_cis_forest(
        "chr1",
        pcs,
        np.ones(4, dtype=np.int64),
        n_pcs=1,
        branch_k=2,
        seed=0,
    )

    annotate_cis_forest(forest1, annotation, features1, n_pcs=1)
    annotate_cis_forest(forest2, annotation, features2, n_pcs=1)

    scores1 = [node.ab_score for node in forest1.trees[0].nodes]
    scores2 = [node.ab_score for node in forest2.trees[0].nodes]
    assert scores1 == pytest.approx(scores2)


def test_annotate_ab_kmeans3_merges_two_highest_feature_clusters() -> None:
    pcs = np.array([[-3.0], [-1.0], [0.0], [2.0], [3.0]])
    rank = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    features = ab_feature_matrix(pcs, rank, 1)

    annotation = annotate_ab(features, rank, 5, "kmeans3", 0, n_pcs=1)

    assert not annotation.fallback
    assert annotation.a_indices.size + annotation.b_indices.size == 5
    assert set(annotation.labels.tolist()) == {"A", "B"}
    assert 4 in annotation.a_indices.tolist()
    assert 0 in annotation.b_indices.tolist()
    assert annotation.a_centroids_pc.shape[0] == 2
    assert annotation.b_centroid_pc.shape == (1,)
    for index, label in enumerate(annotation.labels):
        score = annotation.scores[index]
        if label == "A":
            assert score > 0
        elif label == "B":
            assert score < 0


def test_ab_labels_follow_pc_score() -> None:
    pcs = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    rank = np.array([0.0, 0.1, 0.9, 1.0])
    features = ab_feature_matrix(pcs, rank, 1)

    annotation = annotate_ab(features, rank, 4, "kmeans2", 0, n_pcs=1)

    for index, label in enumerate(annotation.labels):
        score = annotation.scores[index]
        if label == "A":
            assert score > 0
        elif label == "B":
            assert score < 0
        else:
            assert label is None
            assert np.isnan(score) or score == 0.0


def test_ab_degenerate_when_pc_centroids_coincide() -> None:
    pcs = np.zeros((4, 1))
    rank = np.array([0.0, 0.1, 0.9, 1.0])
    features = ab_feature_matrix(pcs, rank, 1)

    annotation = annotate_ab(features, rank, 4, "kmeans2", 0, n_pcs=1)

    assert annotation.fallback
    assert annotation.fallback_reason == "ab_degenerate"
    assert annotation.labels.tolist() == [None, None, None, None]
    assert np.isnan(annotation.scores).all()
    assert annotation.a_indices.size == 0
    assert annotation.b_indices.size == 0


def test_annotate_ab_excludes_uninformative_units() -> None:
    pcs = np.array([[0.0], [1.0], [2.0], [3.0]])
    rank = np.array([0.0, 0.2, 0.8, 1.0])
    features = ab_feature_matrix(pcs, rank, 1)
    clusterable = np.array([False, True, True, True])

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        annotation = annotate_ab(
            features,
            rank,
            4,
            "kmeans2",
            0,
            n_pcs=1,
            clusterable=clusterable,
        )

    assert annotation.labels[0] is None
    assert np.isnan(annotation.scores[0])
    assert np.isnan(annotation.margins[0])
    assert 0 not in annotation.a_indices.tolist()
    assert 0 not in annotation.b_indices.tolist()
    assert set(annotation.a_indices.tolist() + annotation.b_indices.tolist()) == {
        1,
        2,
        3,
    }


def test_annotate_ab_rejects_too_few_units() -> None:
    with pytest.raises(ClusteringError):
        annotate_ab(
            np.zeros((1, 2)),
            np.array([0.5]),
            1,
            "kmeans2",
            0,
            n_pcs=1,
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
        n_pcs=1,
    )

    assert annotation.fallback
    assert annotation.fallback_reason == "ab_degenerate"


def test_annotate_ab_rejects_invalid_inputs() -> None:
    with pytest.raises(ClusteringError):
        annotate_ab(np.zeros((2, 2)), np.zeros(3), 3, "kmeans2", 0, n_pcs=1)
    with pytest.raises(ClusteringError):
        annotate_ab(np.zeros((2, 2)), np.zeros(2), 2, "bad", 0, n_pcs=1)


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
    features = ab_feature_matrix(pcs, rank, 1)
    annotation = AbAnnotation(
        labels=labels,
        feature_rank=rank,
        split_policy="kmeans2",
        fallback=False,
        fallback_reason=None,
        scores=np.zeros(4),
        margins=np.zeros(4),
        a_indices=np.array([0, 1]),
        b_indices=np.array([2, 3]),
    )

    annotate_cis_forest(forest, annotation, features, n_pcs=1)

    root = forest.trees[0].nodes[forest.trees[0].root_id]
    assert root.ab_label is None
    assert root.ab_frac_a == 0.5
    assert root.ab_purity == 0.5
    assert root.ab_feature_rank_mean == 0.5

    for node in forest.trees[0].nodes:
        assert node.ab_label is None
        if node.is_leaf:
            assert node.ab_frac_a in {0.0, 1.0}


def test_build_ab_forest_two_roots() -> None:
    pcs = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    bins = np.ones(4, dtype=np.int64)
    rank = np.array([0.0, 0.1, 0.9, 1.0])
    features = ab_feature_matrix(pcs, rank, 1)
    annotation = annotate_ab(features, rank, 4, "kmeans2", 0, n_pcs=1)

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

    annotate_cis_forest(forest, annotation, features, n_pcs=1)
    assert forest.trees[0].nodes[0].ab_label == "A"
    assert forest.trees[1].nodes[0].ab_label == "B"


def test_build_ab_forest_degenerate_fallback() -> None:
    annotation = AbAnnotation(
        labels=np.array([None, None], dtype=object),
        feature_rank=np.array([0.5, 0.5]),
        split_policy="kmeans2",
        fallback=True,
        fallback_reason="ab_degenerate",
        scores=np.full(2, np.nan),
        margins=np.full(2, np.nan),
        a_indices=np.zeros(0, dtype=np.int64),
        b_indices=np.zeros(0, dtype=np.int64),
    )
    forest = build_ab_forest(
        "chr1",
        np.array([[1.0], [2.0]]),
        np.array([1, 1], dtype=np.int64),
        annotation=annotation,
        n_pcs=1,
        branch_k=2,
        seed=0,
    )

    assert len(forest.trees) == 1
    assert forest.trees[0].root_label == "chrom"
    assert forest.trees[0].nodes[0].stop_reason == "ab_degenerate"
    assert forest.groups[0].tolist() == [0, 1]


def test_annotate_cis_forest_node_scores_match_leaf_units() -> None:
    pcs = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    rank = np.array([0.0, 0.1, 0.9, 1.0])
    features = ab_feature_matrix(pcs, rank, 1)
    annotation = annotate_ab(features, rank, 4, "kmeans2", 0, n_pcs=1)
    forest = build_cis_forest(
        "chr1",
        pcs,
        np.ones(4, dtype=np.int64),
        n_pcs=1,
        branch_k=2,
        seed=0,
    )

    annotate_cis_forest(forest, annotation, features, n_pcs=1)

    for tree, group in zip(forest.trees, forest.groups):
        for node in tree.nodes:
            if not node.is_leaf:
                continue
            unit = int(group[node.unit_indices[0]])
            assert node.ab_score == pytest.approx(annotation.scores[unit])
            assert node.ab_label == annotation.labels[unit]

    root = forest.trees[0].nodes[forest.trees[0].root_id]
    assert root.ab_score is not None
    assert root.ab_score == pytest.approx(0.0)
    if root.ab_score == 0.0:
        assert root.ab_label is None
    else:
        assert root.ab_label == ("A" if root.ab_score > 0 else "B")


def test_node_label_follows_score_without_mixed_threshold() -> None:
    pcs = np.array([[-3.0], [-2.0], [-1.0], [3.0]])
    rank = np.array([0.0, 0.1, 0.2, 1.0])
    features = ab_feature_matrix(pcs, rank, 1)
    labels = np.array(["A", "A", "A", "B"], dtype=object)
    annotation = AbAnnotation(
        labels=labels,
        feature_rank=rank,
        split_policy="kmeans2",
        fallback=False,
        fallback_reason=None,
        scores=np.zeros(4),
        margins=np.zeros(4),
        a_indices=np.array([0, 1, 2]),
        b_indices=np.array([3]),
        a_centroids_pc=features[[0, 1, 2], :1].mean(axis=0, keepdims=True),
        b_centroid_pc=features[3, :1],
    )
    forest = build_cis_forest(
        "chr1",
        pcs,
        np.ones(4, dtype=np.int64),
        n_pcs=1,
        branch_k=2,
        seed=0,
    )

    annotate_cis_forest(forest, annotation, features, n_pcs=1)

    root = forest.trees[0].nodes[forest.trees[0].root_id]
    assert root.ab_frac_a == pytest.approx(0.75)
    assert root.ab_purity == pytest.approx(0.75)
    assert root.ab_score is not None
    assert root.ab_score > 0
    assert root.ab_label == "A"
