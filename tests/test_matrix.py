"""Tests for cis O/E aggregation and principal components."""

from __future__ import annotations

import numpy as np
import pytest

from tadclust.errors import ClusteringError
from tadclust.matrix import (
    area_weighted_mean_oe,
    canonicalize_pc1_sign,
    compute_fine_unit_matrices,
    compute_top_pcs,
    compute_unit_matrices,
    cosine_normalize,
    distance_bin_edges,
    expected_by_distance,
    log2_ratio,
    orient_pcs_to_feature,
    pcs_from_log_oe,
    scaled_pseudocount,
    symmetrize_matrix,
)
from tadclust.mcool import open_cooler


def _matrix_with_zero_rows() -> np.ndarray:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(8, 8))
    matrix = matrix + matrix.T
    for index in (2, 4, 6):
        matrix[index, :] = 0.0
        matrix[:, index] = 0.0
    return matrix


def test_distance_bin_edges_are_increasing() -> None:
    edges = distance_bin_edges(100, 5)

    assert edges[0] == 1
    assert edges[-1] == 100
    assert np.all(np.diff(edges) > 0)
    assert distance_bin_edges(2, 5).tolist() == [1, 2]


def test_expected_by_distance_uses_diagonal_and_distance_means() -> None:
    distance = np.array([0, 0, 1, 1])
    observed = np.array([10.0, 20.0, 2.0, 4.0])

    expected = expected_by_distance(distance, observed, 2, 5)

    assert expected.tolist() == [15.0, 15.0, 3.0, 3.0]


def test_expected_by_distance_fills_empty_bins() -> None:
    distance = np.array([1, 1, 100])
    observed = np.array([10.0, 20.0, 100.0])

    expected = expected_by_distance(distance, observed, 100, 3)

    assert np.all(np.isfinite(expected))
    assert np.all(expected > 0)
    assert expected[0] == 15.0
    assert np.isclose(expected[2], 100.0)


def test_expected_by_distance_does_not_floor_zero_expected() -> None:
    expected = expected_by_distance(
        np.array([1, 1]),
        np.array([0.0, 0.0]),
        10,
        3,
    )

    assert expected.tolist() == [0.0, 0.0]
    assert scaled_pseudocount(expected, 0.5) == 0.5


def test_expected_by_distance_handles_diagonal_only() -> None:
    expected = expected_by_distance(
        np.array([0, 0]),
        np.array([5.0, 7.0]),
        10,
        3,
    )

    assert expected.tolist() == [6.0, 6.0]


def test_pseudocount_scales_with_smallest_positive_expected() -> None:
    expected = np.array([[4.0, 0.0], [np.nan, 2.0]])

    assert scaled_pseudocount(expected, 0.5) == 1.0
    assert scaled_pseudocount(np.zeros((2, 2)), 0.5) == 0.5


def test_log2_ratio_handles_infinities_and_asymmetric_clip() -> None:
    observed = np.array([0.0, 100.0, 1.0, 0.0])
    expected = np.array([1.0, 1.0, 0.0, 0.0])

    values = log2_ratio(
        observed,
        expected,
        0.0,
        clip_min=-2.0,
        clip_max=3.0,
    )

    assert values.tolist() == [-2.0, 3.0, 3.0, 0.0]


def test_symmetrize_fills_both_triangles() -> None:
    matrix = np.array([[1.0, 2.0], [0.0, 3.0]])

    assert symmetrize_matrix(matrix).tolist() == [[1.0, 2.0], [2.0, 3.0]]


def test_cosine_normalize_is_symmetric_and_keeps_zero_rows() -> None:
    matrix = np.array(
        [
            [3.0, 0.0, 4.0],
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 3.0],
        ]
    )

    normalized = cosine_normalize(matrix)

    assert np.allclose(normalized, normalized.T)
    assert np.isclose(normalized[0, 0], 3.0 / 25.0)
    assert np.isclose(normalized[0, 2], 4.0 / 25.0)
    assert np.all(normalized[1] == 0)
    assert np.all(normalized[:, 1] == 0)


def test_cosine_normalize_uses_row_and_column_norms() -> None:
    matrix = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    row_norm = np.linalg.norm(matrix, axis=1)
    col_norm = np.linalg.norm(matrix, axis=0)

    normalized = cosine_normalize(matrix)

    expected = matrix / (row_norm[:, None] * col_norm[None, :])
    assert np.allclose(normalized, expected)


def test_top_pcs_pads_and_sorts() -> None:
    embedding, eigenvalues, fractions = compute_top_pcs(
        np.diag([5.0, 3.0, 0.0, 0.0]),
        3,
    )

    assert embedding.shape == (4, 3)
    assert eigenvalues.shape == (3,)
    assert np.all(np.diff(eigenvalues) <= 0)
    assert np.isclose(fractions.sum(), 1.0)
    assert np.allclose(fractions, eigenvalues / eigenvalues.sum())


def test_top_pcs_single_row_is_zero() -> None:
    embedding, eigenvalues, fractions = compute_top_pcs(np.array([[7.0]]), 3)

    assert embedding.shape == (1, 3)
    assert np.all(embedding == 0)
    assert np.all(eigenvalues == 0)
    assert np.all(fractions == 0)


def test_top_pcs_sparse_path() -> None:
    embedding, eigenvalues, fractions = compute_top_pcs(
        np.diag([5.0, 3.0, 1.0, 0.0]),
        2,
        dense_max=1,
    )

    assert embedding.shape == (4, 2)
    assert np.allclose(eigenvalues, [5.0, 3.0])
    assert np.isclose(fractions.sum(), 1.0)


def test_top_pcs_keeps_zero_rows_exact() -> None:
    matrix = cosine_normalize(_matrix_with_zero_rows())

    embedding, _, _ = compute_top_pcs(matrix, 5)

    assert np.all(embedding[[2, 4, 6]] == 0.0)


def test_top_pcs_sparse_path_keeps_zero_rows_exact() -> None:
    matrix = cosine_normalize(_matrix_with_zero_rows())

    embedding, _, _ = compute_top_pcs(matrix, 2, dense_max=1)

    assert np.all(embedding[[2, 4, 6]] == 0.0)


def test_top_pcs_rejects_zero_matrix() -> None:
    with pytest.raises(ClusteringError) as info:
        compute_top_pcs(np.zeros((2, 2)), 2, label="test")

    assert info.value.matrix == "test"


def test_pcs_from_oe_rejects_zero_matrix() -> None:
    with pytest.raises(ClusteringError):
        pcs_from_log_oe(np.zeros((2, 2)), 2, label="cis")


def test_orient_pcs_to_feature_makes_positive_a() -> None:
    pcs = np.array([[2.0], [1.0], [-1.0], [-2.0]])
    rank = np.array([0.1, 0.2, 0.8, 0.9])

    result = orient_pcs_to_feature(pcs, rank)

    assert np.corrcoef(result[:, 0], rank)[0, 1] > 0
    assert np.all(result[2:, 0] > 0)


def test_orient_pcs_to_feature_flips_every_negative_column() -> None:
    pcs = np.array([[2.0, 1.0], [1.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]])
    rank = np.array([0.1, 0.2, 0.8, 0.9])

    result = orient_pcs_to_feature(pcs, rank)

    assert np.corrcoef(result[:, 0], rank)[0, 1] > 0
    assert np.corrcoef(result[:, 1], rank)[0, 1] > 0


def test_canonicalize_pc1_sign_flips_largest_magnitude() -> None:
    values = np.array([[-3.0, 1.0], [1.0, 2.0]])

    result = canonicalize_pc1_sign(values)

    assert result[0, 0] == 3.0
    assert result[1, 0] == -1.0
    assert values[0, 0] == -3.0


def test_canonicalize_pc1_sign_leaves_positive_max_unchanged() -> None:
    values = np.array([[3.0, 1.0], [-1.0, 2.0]])

    assert np.array_equal(canonicalize_pc1_sign(values), values)


def test_compute_fine_unit_matrices_on_synthetic_mcool(mcool) -> None:
    cooler_object = open_cooler(mcool, 500)
    bin_to_fine_unit = np.array([0, 0, 1, 1], dtype=np.int64)

    matrices = compute_fine_unit_matrices(
        cooler_object,
        "chr1",
        bin_to_fine_unit,
        2,
        n_distance_bins=5,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        n_pcs=2,
    )

    assert matrices.observed.shape == (2, 2)
    assert matrices.expected.shape == (2, 2)
    assert matrices.log_oe.shape == (2, 2)
    assert matrices.bins_per_fine_unit.tolist() == [2, 2]
    assert np.all(np.isfinite(np.diag(matrices.log_oe)))
    assert matrices.alpha > 0
    assert matrices.pcs.shape == (2, 2)
    assert matrices.eigenvalues.shape == (2,)


def test_compute_fine_unit_matrices_rejects_out_of_range_bin_map(
    mcool,
) -> None:
    cooler_object = open_cooler(mcool, 500)
    bin_to_fine_unit = np.array([0, 0, 1], dtype=np.int64)

    with pytest.raises(ClusteringError):
        compute_fine_unit_matrices(
            cooler_object,
            "chr1",
            bin_to_fine_unit,
            2,
            n_distance_bins=5,
            pseudocount=1.0,
            clip_min=-10.0,
            clip_max=10.0,
            n_pcs=2,
        )


def test_compute_unit_matrices_aggregate_fine_pairs() -> None:
    observed_fine = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 5.0, 6.0, 7.0],
            [3.0, 6.0, 8.0, 9.0],
            [4.0, 7.0, 9.0, 10.0],
        ]
    )
    expected_fine = np.ones((4, 4))
    unit_map = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    matrices = compute_unit_matrices(
        unit_map,
        observed_fine,
        expected_fine,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        n_pcs=1,
    )

    assert np.allclose(matrices.observed, unit_map.T @ observed_fine @ unit_map)
    assert np.allclose(matrices.expected, unit_map.T @ expected_fine @ unit_map)
    assert np.all(np.isfinite(np.diag(matrices.log_oe)))
    assert matrices.pcs.shape == (2, 1)


def test_area_weighted_mean_oe_uses_pair_and_diagonal_weights() -> None:
    log_oe = np.array([[2.0, 1.0], [1.0, 3.0]])
    unit_bins = np.array([1, 2])

    value = area_weighted_mean_oe(
        log_oe,
        unit_bins,
        np.array([0, 1]),
    )

    assert value == pytest.approx(2.0)


def test_area_weighted_mean_oe_singleton_returns_diagonal() -> None:
    log_oe = np.array([[2.0, 1.0], [1.0, 3.0]])
    unit_bins = np.array([1, 2])

    value = area_weighted_mean_oe(
        log_oe,
        unit_bins,
        np.array([1]),
    )

    assert value == pytest.approx(3.0)


def test_area_weighted_mean_oe_zero_weight_falls_back_to_mean() -> None:
    log_oe = np.array([[2.0, 1.0], [1.0, 3.0]])
    unit_bins = np.array([0, 0])

    value = area_weighted_mean_oe(
        log_oe,
        unit_bins,
        np.array([0, 1]),
    )

    assert value == pytest.approx(1.75)


def test_area_weighted_mean_oe_empty_indices_is_none() -> None:
    value = area_weighted_mean_oe(
        np.zeros((0, 0)),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
    )

    assert value is None
