"""End-to-end tests for the cis matrix HDF5 product."""

from __future__ import annotations

import json
from typing import cast

import h5py
import numpy as np
import polars as pl
import pytest
from conftest import write_tads

import tadclust
from tadclust.config import ClusterConfig
from tadclust.matrix import double_centered_log_frequency, log2_ratio
from tadclust.matrix_io import MATRIX_DATASETS, MATRIX_FILE, MATRIX_SCHEMA_VERSION
from tadclust.pipeline import cluster
from tadclust.writer import write_outputs


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


def make_uniform_config(mcool, tmp_path, **overrides):
    values = {
        "mcool_path": mcool,
        "annotation_path": None,
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


def chrom_groups(handle: h5py.File) -> dict[str, h5py.Group]:
    groups: dict[str, h5py.Group] = {}
    for name in handle:
        if not isinstance(name, str) or name == "manifest":
            continue
        value = handle[name]
        assert isinstance(value, h5py.Group)
        groups[name] = value
    return groups


def read_leaves(out_dir):
    return pl.read_parquet(out_dir / "cis_leaves.parquet")


def leaf_rows(leaves, chrom):
    return leaves.filter(pl.col("chrom") == chrom).sort("unit_id")


def test_cis_matrices_output_and_manifest(mcool, tmp_path) -> None:
    config = make_config(mcool, base_tads(tmp_path), tmp_path)
    result = cluster(config)
    written = write_outputs(result, config)

    out = tmp_path / "out"
    matrix_path = out / MATRIX_FILE
    assert matrix_path.is_file()
    assert matrix_path.name in written

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["matrices"] == {
        "file": MATRIX_FILE,
        "schema_version": MATRIX_SCHEMA_VERSION,
        "storage": "dense",
        "datasets": list(MATRIX_DATASETS),
    }

    with h5py.File(matrix_path, "r") as handle:
        assert handle.attrs["matrix_schema_version"] == MATRIX_SCHEMA_VERSION
        assert handle.attrs["storage"] == "dense"
        assert handle.attrs["clip_min"] == -10.0
        assert handle.attrs["clip_max"] == 10.0
        assert handle.attrs["working_binsize"] == 500
        assert "source" not in handle.attrs

        groups = chrom_groups(handle)
        assert list(groups) == ["chr1", "chr2"]
        for group in groups.values():
            assert set(group) == set(MATRIX_DATASETS)
            assert "n_units" not in group.attrs
            assert dataset(group, "tad_table_row").attrs["null_value"] == -1

        raw = dataset(handle, "manifest")[()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        assert json.loads(raw) == manifest
        assert tadclust.read_manifest(matrix_path) == manifest


@pytest.mark.parametrize("source", ["oe", "frequency"])
def test_cis_matrix_contents(mcool, tmp_path, source) -> None:
    config = make_config(
        mcool,
        base_tads(tmp_path),
        tmp_path,
        cis_clustering_source=source,
    )
    result = cluster(config)
    write_outputs(result, config)

    out = tmp_path / "out"
    leaves = read_leaves(out)
    with h5py.File(out / MATRIX_FILE, "r") as handle:
        clip_min = cast(float, handle.attrs["clip_min"])
        clip_max = cast(float, handle.attrs["clip_max"])
        working_binsize = cast(int, handle.attrs["working_binsize"])
        groups = chrom_groups(handle)

        for chrom, group in groups.items():
            units = dataset(group, "unit_id")[()]
            expected_units = leaf_rows(leaves, chrom)["unit_id"].to_numpy()
            n = units.size
            assert np.array_equal(units, expected_units)
            assert n == leaf_rows(leaves, chrom).height

            observed = dataset(group, "observed")[()]
            expected = dataset(group, "expected")[()]
            log_oe = dataset(group, "log_oe")[()]
            dc_log_freq = dataset(group, "dc_log_freq")[()]
            log_oe_excess = dataset(group, "log_oe_excess")[()]
            log_oe_baseline = dataset(group, "log_oe_baseline")[()]
            alpha = cast(float, group.attrs["alpha"])
            edges = np.asarray(group.attrs["distance_bin_edges"], dtype=np.float64)

            for values in (
                observed,
                expected,
                log_oe,
                dc_log_freq,
                log_oe_excess,
            ):
                assert values.shape == (n, n)
                assert values.dtype == np.float32
                assert np.all(np.isfinite(values))
                assert np.allclose(values, values.T)

            assert log_oe_baseline.ndim == 1
            assert log_oe_baseline.dtype == np.float64
            assert log_oe_baseline.size == edges.size - 1
            assert np.all(np.isfinite(log_oe_baseline))
            assert np.all(np.diag(log_oe_excess) == 0.0)

            recomputed = log2_ratio(
                observed.astype(np.float64),
                expected.astype(np.float64),
                alpha,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            assert np.allclose(log_oe, recomputed, rtol=1e-3, atol=1e-3)

            recomputed_dc = double_centered_log_frequency(
                observed.astype(np.float64),
                alpha,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            assert np.allclose(
                dc_log_freq,
                recomputed_dc,
                rtol=1e-4,
                atol=1e-4,
            )

            centers = dataset(group, "unit_center")[()]
            distance = np.floor(
                np.abs(centers[:, None] - centers[None, :]) / working_binsize
            ).astype(np.int64)
            np.maximum(distance, 1, out=distance)
            np.fill_diagonal(distance, 0)
            distance_bin = np.clip(
                np.searchsorted(edges, distance, side="right") - 1,
                0,
                edges.size - 2,
            )
            unit_bins = dataset(group, "n_working_bins")[()].astype(np.float64)
            upper = np.triu_indices(n, 1)
            pair_weights = (unit_bins[:, None] * unit_bins[None, :])[upper]
            pair_values = log_oe.astype(np.float64)[upper]
            pair_bins = distance_bin[upper]
            total_weight = float(pair_weights.sum())
            global_mean = (
                float(np.dot(pair_weights, pair_values) / total_weight)
                if total_weight > 0.0
                else float(pair_values.mean())
            )
            sums = np.bincount(
                pair_bins,
                weights=pair_weights * pair_values,
                minlength=edges.size - 1,
            )
            totals = np.bincount(
                pair_bins,
                weights=pair_weights,
                minlength=edges.size - 1,
            )
            recomputed_baseline = np.full(edges.size - 1, global_mean)
            valid = totals > 0.0
            recomputed_baseline[valid] = sums[valid] / totals[valid]
            assert np.allclose(
                log_oe_baseline,
                recomputed_baseline,
                rtol=1e-4,
                atol=1e-4,
            )

            expected_excess = log_oe.astype(np.float64) - log_oe_baseline[distance_bin]
            np.fill_diagonal(expected_excess, 0.0)
            assert np.allclose(
                log_oe_excess,
                expected_excess,
                rtol=1e-3,
                atol=1e-3,
            )


def test_cis_matrix_units_match_cis_leaves(mcool, tmp_path) -> None:
    config = make_config(mcool, base_tads(tmp_path), tmp_path)
    result = cluster(config)
    write_outputs(result, config)

    out = tmp_path / "out"
    leaves = read_leaves(out)
    with h5py.File(out / MATRIX_FILE, "r") as handle:
        for chrom, group in chrom_groups(handle).items():
            rows = leaf_rows(leaves, chrom)
            assert np.array_equal(
                dataset(group, "unit_id")[()],
                rows["unit_id"].to_numpy(),
            )
            assert np.array_equal(
                dataset(group, "unit_start")[()],
                rows["unit_start"].to_numpy(),
            )
            assert np.array_equal(
                dataset(group, "unit_end")[()],
                rows["unit_end"].to_numpy(),
            )
            assert np.array_equal(
                dataset(group, "n_working_bins")[()],
                rows["n_working_bins"].to_numpy(),
            )
            expected_rows = rows["tad_table_row"].fill_null(-1).to_numpy()
            assert np.array_equal(
                dataset(group, "tad_table_row")[()],
                expected_rows,
            )
            starts = rows["unit_start"].to_numpy()
            ends = rows["unit_end"].to_numpy()
            assert np.allclose(
                dataset(group, "unit_center")[()],
                (starts + ends) / 2.0,
            )


def test_cis_matrices_match_serial_and_parallel(mcool, tmp_path) -> None:
    serial = cluster(make_config(mcool, base_tads(tmp_path), tmp_path))
    parallel = cluster(
        make_config(
            mcool,
            base_tads(tmp_path),
            tmp_path,
            output_dir=tmp_path / "parallel",
            workers=2,
        )
    )

    assert [group.chrom for group in serial.cis_matrices.groups] == [
        group.chrom for group in parallel.cis_matrices.groups
    ]
    for left, right in zip(
        serial.cis_matrices.groups,
        parallel.cis_matrices.groups,
        strict=True,
    ):
        for name in (
            "observed",
            "expected",
            "log_oe",
            "dc_log_freq",
            "log_oe_excess",
        ):
            assert np.array_equal(getattr(left, name), getattr(right, name))
        assert np.array_equal(left.log_oe_baseline, right.log_oe_baseline)
        assert np.array_equal(left.unit_id, right.unit_id)
        assert left.alpha == right.alpha


def test_cis_matrices_include_residual_units(
    gapped_mcool,
    gapped_tads,
    tmp_path,
) -> None:
    config = make_config(
        gapped_mcool,
        gapped_tads,
        tmp_path,
        with_residuals=True,
    )
    result = cluster(config)
    write_outputs(result, config)

    out = tmp_path / "out"
    leaves = read_leaves(out)
    assert "residual" in set(leaves["unit_kind"].to_list())

    with h5py.File(out / MATRIX_FILE, "r") as handle:
        for chrom, group in chrom_groups(handle).items():
            rows = leaf_rows(leaves, chrom)
            assert np.array_equal(
                dataset(group, "unit_id")[()],
                rows["unit_id"].to_numpy(),
            )
            tad_rows = dataset(group, "tad_table_row")[()]
            assert np.array_equal(
                tad_rows,
                rows["tad_table_row"].fill_null(-1).to_numpy(),
            )
            assert np.any(tad_rows == -1)


def test_cis_matrices_uniform_bins(mcool, tmp_path) -> None:
    config = make_uniform_config(mcool, tmp_path)
    result = cluster(config)
    write_outputs(result, config)

    out = tmp_path / "out"
    leaves = read_leaves(out)
    assert set(leaves["unit_kind"].to_list()) == {"bin"}

    with h5py.File(out / MATRIX_FILE, "r") as handle:
        for chrom, group in chrom_groups(handle).items():
            rows = leaf_rows(leaves, chrom)
            assert np.array_equal(
                dataset(group, "unit_id")[()],
                rows["unit_id"].to_numpy(),
            )
            assert np.all(dataset(group, "tad_table_row")[()] == -1)


def test_existing_outputs_work_without_matrix_file(mcool, tmp_path) -> None:
    config = make_config(mcool, base_tads(tmp_path), tmp_path)
    result = cluster(config)
    write_outputs(result, config)

    out = tmp_path / "out"
    (out / MATRIX_FILE).unlink()

    assert pl.read_parquet(out / "cis_nodes.parquet").height > 0
    assert pl.read_parquet(out / "cis_leaves.parquet").height == 4
    assert json.loads((out / "manifest.json").read_text())["matrices"]["file"] == (
        MATRIX_FILE
    )
    assert tadclust.read_manifest(out / "cis_nodes.parquet")["matrices"]["file"] == (
        MATRIX_FILE
    )
