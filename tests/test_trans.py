"""Tests for trans O/E from the mcool pixel table and the trans tree."""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from tadclust.domains import SelectedUnit
from tadclust.errors import ClusteringError, McoolError
from tadclust.mcool import open_cooler
from tadclust.trans import (
    PairInfo,
    TransInfo,
    TransInput,
    TransMatrix,
    build_pixel_map,
    build_trans_tree,
    trans_matrix,
)


def make_unit(chrom, start, end, unit_id=0, n_working_bins=1):
    return SelectedUnit(
        unit_id=unit_id,
        chrom=chrom,
        start=start,
        end=end,
        unit_kind="tad",
        tad_table_row=None,
        fine_indices=np.zeros(0, dtype=np.int64),
        n_working_bins=n_working_bins,
    )


def two_chrom_inputs():
    return [
        TransInput(
            chrom="chr1",
            units=[
                make_unit("chr1", 0, 1000, 0, 2),
                make_unit("chr1", 1000, 2000, 1, 2),
            ],
        ),
        TransInput(
            chrom="chr2",
            units=[make_unit("chr2", 0, 2000, 2, 4)],
        ),
    ]


def one_unit_inputs(chroms):
    return [
        TransInput(
            chrom=chrom,
            units=[make_unit(chrom, 0, 2000, index, 4)],
        )
        for index, chrom in enumerate(chroms)
    ]


def test_build_pixel_map(mcool) -> None:
    cooler_object = open_cooler(mcool, 500)
    inputs = two_chrom_inputs()

    with h5py.File(mcool, "r") as handle:
        group = handle["resolutions/500"]
        assert isinstance(group, h5py.Group)
        pixel_map = build_pixel_map(cooler_object, group, inputs)

    assert pixel_map.bin_to_unit.tolist() == [0, 0, 1, 1, 2, 2, 2, 2]
    assert pixel_map.col_sums.tolist() == [2, 2, 4]
    assert pixel_map.unit_chrom.tolist() == [0, 0, 1]
    assert pixel_map.n_bins_chrom == {"chr1": 4, "chr2": 4}
    assert pixel_map.pair_info == {(0, 1): PairInfo(0, 2, 2, 1)}
    assert pixel_map.pair_counts == {(0, 1): 16}
    assert pixel_map.pair_count == 16


def test_trans_pair_null(three_chrom_mcool) -> None:
    cooler_object = open_cooler(three_chrom_mcool, 500)
    inputs = one_unit_inputs(("chr1", "chr2", "chr3"))

    trans = trans_matrix(
        three_chrom_mcool,
        500,
        cooler_object,
        inputs,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        null="pair",
    )

    assert trans.log_oe.shape == (3, 3)
    assert np.allclose(trans.log_oe, trans.log_oe.T)
    assert np.allclose(np.diag(trans.log_oe), 0.0)
    assert trans.info.pair_means == {
        ("chr1", "chr2"): 100.0,
        ("chr1", "chr3"): 1.0,
        ("chr2", "chr3"): 20.0,
    }
    assert trans.info.pair_count == 48
    assert trans.info.n_chrom == 3
    assert trans.info.alpha_mode == "per_pair"
    assert trans.info.grand_mean == pytest.approx(1936.0 / 48.0)
    assert trans.info.pair_alphas[("chr1", "chr2")] == pytest.approx(1600.0)
    assert trans.info.pair_alphas[("chr1", "chr3")] == pytest.approx(16.0)
    assert trans.info.pair_alphas[("chr2", "chr3")] == pytest.approx(320.0)
    assert np.allclose(trans.log_oe, 0.0)


def test_trans_grand_null(three_chrom_mcool) -> None:
    cooler_object = open_cooler(three_chrom_mcool, 500)
    inputs = one_unit_inputs(("chr1", "chr2", "chr3"))

    trans = trans_matrix(
        three_chrom_mcool,
        500,
        cooler_object,
        inputs,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        null="grand",
    )

    assert trans.info.alpha_mode == "global"
    assert trans.info.alpha == pytest.approx(trans.info.grand_mean * 16.0)
    assert all(
        alpha == pytest.approx(trans.info.alpha)
        for alpha in trans.info.pair_alphas.values()
    )
    assert np.allclose(trans.log_oe, trans.log_oe.T)
    assert np.allclose(np.diag(trans.log_oe), 0.0)
    assert np.any(np.abs(trans.log_oe) > 0)


def test_trans_single_chromosome_has_no_pairs(mcool) -> None:
    cooler_object = open_cooler(mcool, 500)
    inputs = one_unit_inputs(("chr1",))

    trans = trans_matrix(
        mcool,
        500,
        cooler_object,
        inputs,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
    )

    assert trans.log_oe.shape == (1, 1)
    assert np.all(trans.log_oe == 0)
    assert trans.info.pair_count == 0
    assert trans.info.pair_means == {}
    assert trans.info.pair_alphas == {}

    result = build_trans_tree(
        trans,
        np.array([4], dtype=np.int64),
        n_pcs=2,
        branch_k=3,
        seed=0,
    )
    assert result.pcs.shape == (1, 2)
    assert len(result.tree.nodes) == 1


def test_build_trans_tree_positive(three_chrom_mcool) -> None:
    cooler_object = open_cooler(three_chrom_mcool, 500)
    inputs = one_unit_inputs(("chr1", "chr2", "chr3"))
    trans = trans_matrix(
        three_chrom_mcool,
        500,
        cooler_object,
        inputs,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        null="grand",
    )

    result = build_trans_tree(
        trans,
        np.array([4, 4, 4], dtype=np.int64),
        n_pcs=2,
        branch_k=2,
        seed=0,
    )

    assert result.pcs.shape == (3, 2)
    assert result.eigenvalues.shape == (2,)
    assert result.tree.nodes[result.tree.root_id].unit_indices.tolist() == [
        0,
        1,
        2,
    ]
    assert result.unit_indices.tolist() == [0, 1, 2]


def test_build_trans_tree_excludes_uninformative_units() -> None:
    info = TransInfo(
        null="pair",
        alpha_mode="pair",
        grand_mean=1.0,
        alpha=1.0,
        min_bins=1.0,
        total=2.0,
        pair_count=2,
        n_chrom=2,
        pair_means={},
        pair_alphas={},
        pair_counts={},
    )
    trans = TransMatrix(
        log_oe=np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        info=info,
        unit_chrom=np.array([0, 0, 1], dtype=np.int64),
        col_sums=np.array([1, 1, 0], dtype=np.int64),
        chrom_names=["chr1", "chr2"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        result = build_trans_tree(
            trans,
            np.array([1, 1, 1], dtype=np.int64),
            n_pcs=1,
            branch_k=3,
            seed=0,
        )

    assert result.unit_indices.tolist() == [0, 1]
    root = result.tree.nodes[result.tree.root_id]
    assert root.unit_indices.tolist() == [0, 1]


def test_trans_no_positive_mode_raises(three_chrom_mcool) -> None:
    cooler_object = open_cooler(three_chrom_mcool, 500)
    inputs = one_unit_inputs(("chr1", "chr2", "chr3"))
    trans = trans_matrix(
        three_chrom_mcool,
        500,
        cooler_object,
        inputs,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        null="pair",
    )

    with pytest.raises(ClusteringError):
        build_trans_tree(
            trans,
            np.array([4, 4, 4], dtype=np.int64),
            n_pcs=2,
            branch_k=2,
            seed=0,
        )


def test_trans_chunking_matches(monkeypatch, three_chrom_mcool) -> None:
    cooler_object = open_cooler(three_chrom_mcool, 500)
    inputs = one_unit_inputs(("chr1", "chr2", "chr3"))

    monkeypatch.setattr("tadclust.trans.CHUNK_ROWS", 1)
    chunked = trans_matrix(
        three_chrom_mcool,
        500,
        cooler_object,
        inputs,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        null="grand",
    )

    monkeypatch.setattr("tadclust.trans.CHUNK_ROWS", 10_000_000)
    whole = trans_matrix(
        three_chrom_mcool,
        500,
        cooler_object,
        inputs,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        null="grand",
    )

    assert np.allclose(chunked.log_oe, whole.log_oe)


def test_trans_matrix_has_no_dense_unit_limit(tmp_path) -> None:
    inputs = [
        TransInput(
            chrom="chr1",
            units=[make_unit("chr1", i * 1000, (i + 1) * 1000, i) for i in range(3001)],
        ),
        TransInput(
            chrom="chr2",
            units=[
                make_unit("chr2", i * 1000, (i + 1) * 1000, 3001 + i)
                for i in range(3000)
            ],
        ),
    ]

    with pytest.raises(McoolError):
        trans_matrix(
            tmp_path / "missing.mcool",
            500,
            None,  # pyright: ignore[reportArgumentType]
            inputs,
            pseudocount=1.0,
            clip_min=-10.0,
            clip_max=10.0,
        )


def test_trans_parallel_matches_serial(three_chrom_mcool) -> None:
    cooler_object = open_cooler(three_chrom_mcool, 500)
    inputs = one_unit_inputs(("chr1", "chr2", "chr3"))

    serial = trans_matrix(
        three_chrom_mcool,
        500,
        cooler_object,
        inputs,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        null="grand",
        workers=1,
    )
    parallel = trans_matrix(
        three_chrom_mcool,
        500,
        cooler_object,
        inputs,
        pseudocount=1.0,
        clip_min=-10.0,
        clip_max=10.0,
        null="grand",
        workers=2,
    )

    assert np.allclose(serial.log_oe, parallel.log_oe)
    assert serial.info.pair_means == parallel.info.pair_means


def test_unknown_trans_null_is_rejected(three_chrom_mcool) -> None:
    cooler_object = open_cooler(three_chrom_mcool, 500)
    inputs = one_unit_inputs(("chr1", "chr2", "chr3"))

    with pytest.raises(ClusteringError):
        trans_matrix(
            three_chrom_mcool,
            500,
            cooler_object,
            inputs,
            pseudocount=1.0,
            clip_min=-10.0,
            clip_max=10.0,
            null="uniform",
        )
