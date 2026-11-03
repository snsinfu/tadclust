"""Tests for node cohesion metrics."""

from __future__ import annotations

import h5py
import numpy as np
import polars as pl
import pytest
from conftest import write_mcool, write_tads

from tadclust.cohesion import build_cohesion_context
from tadclust.config import ClusterConfig
from tadclust.domains import SelectedUnit
from tadclust.errors import ClusteringError
from tadclust.matrix_io import MATRIX_FILE
from tadclust.pipeline import cluster
from tadclust.writer import write_outputs


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


def base_tads(tmp_path):
    return write_tads(
        tmp_path,
        [
            ("chr1", 0, 1000),
            ("chr1", 1000, 2000),
            ("chr2", 0, 1000),
            ("chr2", 1000, 2000),
        ],
    )


def four_tads(tmp_path):
    rows = []
    for chrom in ("chr1", "chr2"):
        for start in range(0, 2000, 500):
            rows.append((chrom, start, start + 500))
    return write_tads(tmp_path, rows)


def make_config(mcool, tads, tmp_path, **overrides):
    values = {
        "mcool_path": mcool,
        "annotation_path": tads,
        "output_dir": tmp_path / "out",
        "working_binsize": 500,
        "tree_pca_dim": 2,
        "tree_branch_k": 2,
    }
    values.update(overrides)
    return ClusterConfig(**values)


def dataset(group: h5py.Group, name: str) -> h5py.Dataset:
    value = group[name]
    assert isinstance(value, h5py.Dataset)
    return value


def direct_cohesion(
    group: h5py.Group,
    positions: np.ndarray,
    working_binsize: int,
) -> tuple[float, float]:
    log_oe = dataset(group, "log_oe")[()].astype(np.float64)
    centers = dataset(group, "unit_center")[()]
    unit_bins = dataset(group, "n_working_bins")[()].astype(np.float64)
    edges = np.asarray(group.attrs["distance_bin_edges"], dtype=np.float64)

    distance = np.floor(
        np.abs(centers[:, None] - centers[None, :]) / working_binsize
    ).astype(np.int64)
    np.maximum(distance, 1, out=distance)
    np.fill_diagonal(distance, 0)
    bins = np.clip(
        np.searchsorted(edges, distance, side="right") - 1,
        0,
        len(edges) - 2,
    )

    weights = unit_bins[:, None] * unit_bins[None, :]
    upper = np.triu_indices(centers.size, 1)
    pair_weights = weights[upper]
    pair_values = log_oe[upper]
    pair_bins = bins[upper]
    total = pair_weights.sum()
    global_mean = (
        float(np.dot(pair_weights, pair_values) / total)
        if total > 0
        else float(pair_values.mean())
    )
    sums = np.bincount(
        pair_bins,
        weights=pair_weights * pair_values,
        minlength=len(edges) - 1,
    )
    totals = np.bincount(
        pair_bins,
        weights=pair_weights,
        minlength=len(edges) - 1,
    )
    baseline = np.full(len(edges) - 1, global_mean)
    valid = totals > 0
    baseline[valid] = sums[valid] / totals[valid]

    sub = log_oe[np.ix_(positions, positions)]
    sub_bins = bins[np.ix_(positions, positions)]
    sub_bins_values = unit_bins[positions]
    sub_weights = sub_bins_values[:, None] * sub_bins_values[None, :]
    sub_upper = np.triu_indices(positions.size, 1)
    values = sub[sub_upper]
    bins = sub_bins[sub_upper]
    weights_selected = sub_weights[sub_upper]
    total = weights_selected.sum()
    if total > 0:
        cross = float(np.dot(weights_selected, values) / total)
        excess = float(np.dot(weights_selected, values - baseline[bins]) / total)
    else:
        cross = float(values.mean())
        excess = float((values - baseline[bins]).mean())
    return cross, excess


def test_cluster_nodes_have_cross_oe(mcool, tmp_path) -> None:
    result = cluster(make_config(mcool, base_tads(tmp_path), tmp_path))

    branch = result.cis_nodes.filter(~pl.col("is_leaf"))
    leaves = result.cis_nodes.filter(pl.col("is_leaf"))
    assert branch.height > 0
    assert branch["cross_oe"].null_count() == 0
    assert branch["cross_oe_excess"].null_count() == 0
    assert branch["cross_oe"].is_finite().all()
    assert branch["cross_oe_excess"].is_finite().all()
    assert leaves["cross_oe"].null_count() == leaves.height
    assert leaves["cross_oe_excess"].null_count() == leaves.height


def test_ab_nodes_have_cross_oe(mcool, tmp_path) -> None:
    track = tmp_path / "track.bedGraph"
    lines = []
    for chrom in ("chr1", "chr2"):
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0)):
            lines.append(f"{chrom}\t{index * 500}\t{(index + 1) * 500}\t{value}")
    track.write_text("\n".join(lines) + "\n")
    config = make_config(
        mcool,
        four_tads(tmp_path),
        tmp_path,
        ab_feature_track=track,
    )
    result = cluster(config)

    assert result.cis_nodes_ab is not None
    branch = result.cis_nodes_ab.filter(~pl.col("is_leaf"))
    assert branch.height > 0
    assert branch["cross_oe"].null_count() == 0
    assert branch["cross_oe_excess"].null_count() == 0
    assert branch["cross_oe"].is_finite().all()


def test_frequency_source_still_populates_cohesion(mcool, tmp_path) -> None:
    result = cluster(
        make_config(
            mcool,
            base_tads(tmp_path),
            tmp_path,
            cis_clustering_source="frequency",
        )
    )

    branch = result.cis_nodes.filter(~pl.col("is_leaf"))
    assert branch["cross_oe"].is_finite().all()
    assert branch["cross_oe_excess"].is_finite().all()


def test_cohesion_matches_direct_submatrix(mcool, tmp_path) -> None:
    config = make_config(mcool, four_tads(tmp_path), tmp_path)
    result = cluster(config)
    write_outputs(result, config)
    working_binsize = int(result.params["resolved"]["working_binsize"])

    out = tmp_path / "out"
    with h5py.File(out / MATRIX_FILE, "r") as handle:
        for chrom in ("chr1", "chr2"):
            group = handle[chrom]
            assert isinstance(group, h5py.Group)
            group_units = dataset(group, "unit_id")[()]
            leaves = result.cis_leaves.filter(pl.col("chrom") == chrom).sort("unit_id")
            branch_rows = result.cis_nodes.filter(
                (pl.col("chrom") == chrom) & (~pl.col("is_leaf"))
            )
            assert branch_rows.height > 0

            for row in branch_rows.iter_rows(named=True):
                path_label = row["path_label"]
                if row["path"] == []:
                    descendants = leaves
                else:
                    prefix = path_label + "."
                    descendants = leaves.filter(
                        pl.col("path_label").str.starts_with(prefix)
                    )
                unit_ids = descendants["unit_id"].to_numpy()
                positions = np.searchsorted(group_units, unit_ids)
                assert np.array_equal(group_units[positions], unit_ids)
                expected_cross, expected_excess = direct_cohesion(
                    group,
                    positions,
                    working_binsize,
                )
                assert row["cross_oe"] == pytest.approx(expected_cross, abs=1e-3)
                assert row["cross_oe_excess"] == pytest.approx(
                    expected_excess,
                    abs=1e-3,
                )


def test_unclustered_rows_have_null_cohesion(tmp_path) -> None:
    mcool = write_mcool(
        tmp_path,
        resolutions=(500,),
        chroms=("chr1",),
        chrom_size=3000,
        empty_bins={0, 1},
    )
    config = ClusterConfig(
        mcool_path=mcool,
        annotation_path=None,
        output_dir=tmp_path / "out",
        working_binsize=500,
        tree_pca_dim=2,
        tree_branch_k=2,
    )
    result = cluster(config)

    rows = result.cis_nodes.filter(pl.col("node_id").is_null())
    assert rows.height > 0
    assert rows["cross_oe"].null_count() == rows.height
    assert rows["cross_oe_excess"].null_count() == rows.height
