"""Tests for the dense cis matrix HDF5 product."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from tadclust.cohesion import build_cohesion_context
from tadclust.domains import SelectedUnit
from tadclust.errors import ClusteringError
from tadclust.matrix import UnitMatrices, double_centered_log_frequency
from tadclust.matrix_io import (
    MATRIX_SCHEMA_VERSION,
    CisMatrixInput,
    build_cis_matrices,
    write_cis_matrices,
)

WORKING_BINSIZE = 1000


def make_unit(
    chrom: str,
    start: int,
    end: int,
    *,
    unit_id: int,
    n_working_bins: int = 2,
    tad_id: int | None = None,
) -> SelectedUnit:
    return SelectedUnit(
        unit_id=unit_id,
        chrom=chrom,
        start=start,
        end=end,
        unit_kind="tad",
        tad_id=tad_id,
        fine_indices=np.zeros(0, dtype=np.int64),
        n_working_bins=n_working_bins,
    )


def make_matrices(n: int, seed: int = 0) -> UnitMatrices:
    rng = np.random.default_rng(seed)
    observed = rng.uniform(1.0, 5.0, size=(n, n))
    observed = (observed + observed.T) / 2.0
    expected = np.full((n, n), 2.0)
    alpha = 0.5
    log_oe = np.log2((observed + alpha) / (expected + alpha))
    return UnitMatrices(
        observed=observed,
        expected=expected,
        log_oe=log_oe,
        alpha=alpha,
        pcs=np.zeros((n, 2)),
        eigenvalues=np.zeros(2),
        variance_fraction=np.zeros(2),
    )


def dataset(group: h5py.Group, name: str) -> h5py.Dataset:
    value = group[name]
    assert isinstance(value, h5py.Dataset)
    return value


def make_context(units, matrices, *, n_bins_chrom: int = 4):
    return build_cohesion_context(
        units,
        matrices.log_oe,
        n_bins_chrom,
        working_binsize=WORKING_BINSIZE,
        distance_bins=60,
    )


def test_build_cis_matrices_uses_global_unit_order_and_metadata() -> None:
    units_a = [
        make_unit("chr1", 0, 1000, unit_id=0, tad_id=4),
        make_unit("chr1", 1000, 2000, unit_id=1, n_working_bins=3),
    ]
    units_b = [make_unit("chr2", 0, 500, unit_id=0, tad_id=7)]
    matrices_a = make_matrices(2)
    matrices_b = make_matrices(1, seed=1)
    matrices = build_cis_matrices(
        [
            CisMatrixInput(
                "chr1",
                units_a,
                matrices_a,
                n_bins_chrom=4,
                cohesion=make_context(units_a, matrices_a),
            ),
            CisMatrixInput(
                "chr2",
                units_b,
                matrices_b,
                n_bins_chrom=4,
                cohesion=make_context(units_b, matrices_b),
            ),
        ],
        working_binsize=WORKING_BINSIZE,
        distance_bins=60,
        clip_min=-10.0,
        clip_max=10.0,
    )

    assert matrices.storage == "dense"
    assert matrices.working_binsize == WORKING_BINSIZE
    assert [group.chrom for group in matrices.groups] == ["chr1", "chr2"]

    first, second = matrices.groups
    assert first.unit_id.tolist() == [0, 1]
    assert second.unit_id.tolist() == [2]
    assert first.tad_id.tolist() == [4, -1]
    assert second.tad_id.tolist() == [7]
    assert first.unit_center.tolist() == [500.0, 1500.0]
    assert first.n_working_bins.tolist() == [2, 3]
    assert first.observed.dtype == np.float32
    assert first.expected.dtype == np.float32
    assert first.log_oe.dtype == np.float32
    assert first.dc_log_freq.dtype == np.float32
    assert first.log_oe_excess.dtype == np.float32
    assert first.log_oe_baseline.dtype == np.float64
    assert first.log_oe_baseline.shape == (len(first.distance_bin_edges) - 1,)
    assert np.all(np.diag(first.log_oe_excess) == 0.0)
    assert first.distance_bin_edges[0] == 1
    assert first.distance_bin_edges[-1] == 4


def test_build_cis_matrices_rejects_wrong_matrix_shape() -> None:
    unit = make_unit("chr1", 0, 1000, unit_id=0)
    matrices = make_matrices(2)
    context = build_cohesion_context(
        [unit],
        np.zeros((1, 1)),
        n_bins_chrom=4,
        working_binsize=WORKING_BINSIZE,
        distance_bins=60,
    )

    with pytest.raises(ClusteringError):
        build_cis_matrices(
            [
                CisMatrixInput(
                    "chr1",
                    [unit],
                    matrices,
                    n_bins_chrom=4,
                    cohesion=context,
                )
            ],
            working_binsize=WORKING_BINSIZE,
            distance_bins=60,
            clip_min=-10.0,
            clip_max=10.0,
        )


def test_build_cis_matrices_rejects_wrong_ab_pc_shape() -> None:
    unit = make_unit("chr1", 0, 1000, unit_id=0)
    matrices = make_matrices(1)
    with pytest.raises(ClusteringError):
        build_cis_matrices(
            [
                CisMatrixInput(
                    "chr1",
                    [unit],
                    matrices,
                    n_bins_chrom=4,
                    cohesion=make_context([unit], matrices),
                    ab_pc=np.zeros((2, 2)),
                )
            ],
            working_binsize=WORKING_BINSIZE,
            distance_bins=60,
            clip_min=-10.0,
            clip_max=10.0,
        )


def test_write_cis_matrices_round_trip(tmp_path) -> None:
    units_a = [
        make_unit("chr1", 0, 1000, unit_id=0, tad_id=0),
        make_unit("chr1", 1000, 2000, unit_id=1),
    ]
    matrices_a = make_matrices(2)
    context_a = make_context(units_a, matrices_a)
    units_b = [make_unit("chr2", 0, 1000, unit_id=0, tad_id=1)]
    matrices_b = make_matrices(1, seed=2)
    context_b = make_context(units_b, matrices_b)
    matrices = build_cis_matrices(
        [
            CisMatrixInput(
                "chr1",
                units_a,
                matrices_a,
                n_bins_chrom=4,
                cohesion=context_a,
            ),
            CisMatrixInput(
                "chr2",
                units_b,
                matrices_b,
                n_bins_chrom=4,
                cohesion=context_b,
            ),
        ],
        working_binsize=WORKING_BINSIZE,
        distance_bins=60,
        clip_min=-10.0,
        clip_max=10.0,
    )
    manifest = {
        "manifest_version": "1.3",
        "tadclust_version": "0.1.0a2",
        "created_utc": "2026-01-01T00:00:00Z",
    }
    path = tmp_path / "cis_matrices.h5"

    write_cis_matrices(path, matrices, manifest)

    with h5py.File(path, "r") as handle:
        assert handle.attrs["matrix_schema_version"] == MATRIX_SCHEMA_VERSION
        assert handle.attrs["storage"] == "dense"
        assert handle.attrs["clip_min"] == -10.0
        assert handle.attrs["clip_max"] == 10.0
        assert handle.attrs["working_binsize"] == WORKING_BINSIZE
        assert list(handle) == ["chr1", "chr2", "manifest"]
        manifest_data = handle["manifest"]
        assert isinstance(manifest_data, h5py.Dataset)
        raw_manifest = manifest_data[()]
        if isinstance(raw_manifest, bytes):
            raw_manifest = raw_manifest.decode("utf-8")
        assert raw_manifest == json.dumps(
            manifest,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        assert json.loads(raw_manifest) == manifest

        first = handle["chr1"]
        assert isinstance(first, h5py.Group)
        assert first.attrs["alpha"] == 0.5
        assert "n_units" not in first.attrs
        assert dataset(first, "unit_id")[()].tolist() == [0, 1]
        assert dataset(first, "tad_id")[()].tolist() == [0, -1]
        assert dataset(first, "unit_center")[()].tolist() == [500.0, 1500.0]
        for name in (
            "log_oe",
            "observed",
            "expected",
            "dc_log_freq",
            "log_oe_excess",
        ):
            values = dataset(first, name)[()]
            assert values.shape == (2, 2)
            assert values.dtype == np.float32
            assert np.allclose(values, values.T)

        excess = dataset(first, "log_oe_excess")[()]
        baseline = dataset(first, "log_oe_baseline")[()]
        assert baseline.dtype == np.float64
        assert baseline.shape == (len(matrices.groups[0].distance_bin_edges) - 1,)
        assert np.all(np.diag(excess) == 0.0)
        assert np.allclose(
            excess,
            context_a.log_oe_excess_matrix().astype(np.float32),
            atol=1e-5,
        )

        expected_dc = double_centered_log_frequency(
            matrices.groups[0].observed.astype(np.float64),
            matrices.groups[0].alpha,
            clip_min=-10.0,
            clip_max=10.0,
        )
        assert np.allclose(dataset(first, "dc_log_freq")[()], expected_dc, atol=1e-5)
