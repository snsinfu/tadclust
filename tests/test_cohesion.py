"""Tests for node cohesion metrics."""

from __future__ import annotations

import numpy as np
import pytest

from tadclust.cohesion import build_cohesion_context
from tadclust.domains import SelectedUnit
from tadclust.errors import ClusteringError


def make_unit(
    start: int,
    end: int,
    *,
    unit_id: int = 0,
    n_working_bins: int = 1,
) -> SelectedUnit:
    return SelectedUnit(
        unit_id=unit_id,
        chrom="chr1",
        start=start,
        end=end,
        unit_kind="tad",
        tad_table_row=unit_id,
        fine_indices=np.zeros(0, dtype=np.int64),
        n_working_bins=n_working_bins,
    )


def make_context(
    log_oe: np.ndarray,
    *,
    working_binsize: int = 1000,
    n_working_bins: tuple[int, ...] = (1, 1, 2),
):
    units = [
        make_unit(index * 1000, (index + 1) * 1000, unit_id=index, n_working_bins=bins)
        for index, bins in enumerate(n_working_bins)
    ]
    return build_cohesion_context(
        units,
        log_oe,
        n_bins_chrom=4,
        working_binsize=working_binsize,
        distance_bins=60,
    )


def test_baseline_uses_area_weighted_distance_bins() -> None:
    log_oe = np.array(
        [
            [1.0, 2.0, 4.0],
            [2.0, 3.0, 6.0],
            [4.0, 6.0, 5.0],
        ]
    )
    context = make_context(log_oe)

    # Pair weights: (0,1)=1, (0,2)=2, (1,2)=2.
    assert context.baseline[0] == pytest.approx((2.0 + 2.0 * 6.0) / 3.0)
    assert context.baseline[1] == pytest.approx(4.0)
    assert context.baseline[2] == pytest.approx((2.0 + 2.0 * 4.0 + 2.0 * 6.0) / 5.0)


def test_node_metrics_match_direct_submatrix_formula() -> None:
    log_oe = np.array(
        [
            [1.0, 2.0, 4.0],
            [2.0, 3.0, 6.0],
            [4.0, 6.0, 5.0],
        ]
    )
    context = make_context(log_oe)

    cross, excess = context.node_metrics(np.array([0, 1], dtype=np.int64))
    assert cross == pytest.approx(2.0)
    assert excess == pytest.approx(2.0 - (2.0 + 2.0 * 6.0) / 3.0)

    cross, excess = context.node_metrics(np.array([0, 2], dtype=np.int64))
    assert cross == pytest.approx(4.0)
    assert excess == pytest.approx(0.0)


def test_node_metrics_use_actual_pair_distance_bins() -> None:
    log_oe = np.array(
        [
            [1.0, 2.0, 4.0],
            [2.0, 3.0, 6.0],
            [4.0, 6.0, 5.0],
        ]
    )
    context = make_context(log_oe)

    cross, excess = context.node_metrics(np.array([0, 1, 2], dtype=np.int64))
    weights = np.array([1.0, 1.0, 2.0])
    pair_weights = np.array([1.0, 2.0, 2.0])
    values = np.array([2.0, 4.0, 6.0])
    baseline = context.baseline[np.array([0, 1, 0])]
    expected_cross = float(np.dot(pair_weights, values) / pair_weights.sum())
    expected_excess = float(
        np.dot(pair_weights, values - baseline) / pair_weights.sum()
    )
    assert weights.tolist() == context.unit_bins.tolist()
    assert cross == pytest.approx(expected_cross)
    assert excess == pytest.approx(expected_excess)


def test_singleton_node_has_null_metrics() -> None:
    context = make_context(np.array([[1.0]]), n_working_bins=(1,))

    assert context.node_metrics(np.array([0], dtype=np.int64)) == (None, None)
    assert context.node_metrics(np.zeros(0, dtype=np.int64)) == (None, None)


def test_zero_weight_falls_back_to_unweighted_means() -> None:
    log_oe = np.array(
        [
            [1.0, 2.0, 4.0],
            [2.0, 3.0, 6.0],
            [4.0, 6.0, 5.0],
        ]
    )
    context = make_context(log_oe, n_working_bins=(0, 0, 0))

    cross, excess = context.node_metrics(np.array([0, 1, 2], dtype=np.int64))
    pair_values = np.array([2.0, 4.0, 6.0])
    pair_bins = np.array([0, 1, 0])
    assert cross == pytest.approx(pair_values.mean())
    assert excess == pytest.approx((pair_values - context.baseline[pair_bins]).mean())


def test_empty_baseline_bins_use_global_mean() -> None:
    context = make_context(
        np.array(
            [
                [1.0, 2.0, 4.0],
                [2.0, 3.0, 6.0],
                [4.0, 6.0, 5.0],
            ]
        )
    )

    # Bin 2 has no pairs; it falls back to the global weighted mean.
    assert context.baseline[2] == pytest.approx((2.0 + 2.0 * 4.0 + 2.0 * 6.0) / 5.0)


def test_rejects_mismatched_matrix_shape() -> None:
    units = [make_unit(0, 1000), make_unit(1000, 2000)]

    with pytest.raises(ClusteringError):
        build_cohesion_context(
            units,
            np.zeros((3, 3)),
            n_bins_chrom=4,
            working_binsize=1000,
            distance_bins=60,
        )


def test_rejects_non_positive_working_binsize() -> None:
    with pytest.raises(ClusteringError):
        build_cohesion_context(
            [make_unit(0, 1000)],
            np.ones((1, 1)),
            n_bins_chrom=4,
            working_binsize=0,
            distance_bins=60,
        )
