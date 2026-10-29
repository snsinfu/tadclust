"""Tests for cis O/E aggregation and principal components."""

from __future__ import annotations

import numpy as np
import pytest

from tadclust.errors import ComputationError, NoPositiveModeError
from tadclust.matrix import (
    canonicalize_pc1_sign,
    compute_domain_matrices,
    compute_fine_unit_matrices,
    compute_top_pcs,
    cosine_normalize,
    distance_bin_edges,
    expected_by_distance,
    log2_ratio,
    orient_pc1_to_signal,
    pcs_from_log_oe,
    scaled_pseudocount,
    symmetrize_matrix,
)
from tadclust.mcool import open_cooler


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


def test_top_pcs_rejects_zero_matrix() -> None:
    with pytest.raises(NoPositiveModeError) as info:
        compute_top_pcs(np.zeros((2, 2)), 2, label="test")

    assert info.value.matrix == "test"


def test_pcs_from_oe_rejects_zero_matrix() -> None:
    with pytest.raises(NoPositiveModeError):
        pcs_from_log_oe(np.zeros((2, 2)), 2, label="cis")


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
    assert matrices.self_oe.shape == (2,)
    assert matrices.bins_per_fine_unit.tolist() == [2, 2]
    assert np.allclose(np.diag(matrices.log_oe), 0.0)
    assert matrices.alpha > 0
    assert matrices.pcs.shape == (2, 2)
    assert matrices.eigenvalues.shape == (2,)


def test_compute_fine_unit_matrices_rejects_out_of_range_bin_map(
    mcool,
) -> None:
    cooler_object = open_cooler(mcool, 500)
    bin_to_fine_unit = np.array([0, 0, 1], dtype=np.int64)

    with pytest.raises(ComputationError):
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


def test_compute_domain_matrices_aggregate_fine_pairs() -> None:
    observed_fine = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 5.0, 6.0, 7.0],
            [3.0, 6.0, 8.0, 9.0],
            [4.0, 7.0, 9.0, 10.0],
        ]
    )
    expected_fine = np.ones((4, 4))
    domain_map = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    matrices = compute_domain_matrices(
        domain_map,
        observed_fine,
        expected_fine,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        n_pcs=1,
    )

    assert np.allclose(matrices.observed, domain_map.T @ observed_fine @ domain_map)
    assert np.allclose(matrices.expected, domain_map.T @ expected_fine @ domain_map)
    assert np.allclose(np.diag(matrices.log_oe), 0.0)
    assert matrices.self_oe.shape == (2,)
    assert matrices.pcs.shape == (2, 1)
