"""Tests for the compute-only cluster pipeline."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from conftest import write_mcool, write_tads, write_track

import tadclust.domains as domains_module
from tadclust.config import ClusterConfig
from tadclust.errors import ClusteringError, InputError
from tadclust.pipeline import cluster


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


def ab_tads(tmp_path):
    rows = []
    for chrom in ("chr1", "chr2"):
        for start in range(0, 2000, 500):
            rows.append((chrom, start, start + 500))
    return write_tads(tmp_path, rows)


def ab_track(tmp_path):
    rows = []
    for chrom in ("chr1", "chr2"):
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0)):
            rows.append((chrom, index * 500, (index + 1) * 500, value))
    return write_track(tmp_path, rows)


def make_config(mcool, tads, tmp_path, **overrides):
    values = {
        "mcool_path": mcool,
        "annotation_path": tads,
        "output_dir": tmp_path / "out",
        "working_binsize": 500,
        "tree_n_pcs": 2,
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
        "tree_n_pcs": 2,
        "tree_branch_k": 2,
    }
    values.update(overrides)
    return ClusterConfig(**values)


def assert_node_references(nodes) -> None:
    ids = {node_id for node_id in nodes["node_id"].to_list() if node_id is not None}
    for row in nodes.iter_rows(named=True):
        if row["parent_id"] is not None:
            assert row["parent_id"] in ids
        assert set(row["children"]).issubset(ids)


def test_cluster_cis_only(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    result = cluster(make_config(mcool, tads, tmp_path))

    assert result.cis_leaves.height == 4
    assert result.cis_leaves["domain_id"].to_list() == [0, 1, 2, 3]
    assert result.cis_leaves["pcs"].dtype == pl.Array(pl.Float64, 2)
    assert result.cis_leaves["root_label"].unique().to_list() == ["chrom"]
    assert result.cis_leaves["source_row"].to_list() == [0, 1, 2, 3]
    assert "ab_label" not in result.cis_leaves.columns
    assert "trans_path" not in result.cis_leaves.columns

    assert "ab_label" not in result.cis_nodes.columns
    assert set(result.cis_nodes["root_label"].unique().to_list()) == {"chrom"}
    assert result.cis_nodes["parent_id"].null_count() == 2
    assert_node_references(result.cis_nodes)

    assert result.cis_leaves_ab is None
    assert result.cis_nodes_ab is None
    assert result.trans_nodes is None

    assert result.qc["n_domains"] == 4
    assert result.qc["n_tads"] == 4
    assert result.qc["n_residuals"] == 0
    assert result.qc["trans"] == {"enabled": False}
    assert result.qc["ab"] == {"enabled": False}
    assert set(result.qc["cis"]["per_chromosome"]) == {"chr1", "chr2"}

    assert result.params["working_binsize"] == 500
    assert result.params["working_binsize_mode"] == "explicit"
    assert result.params["input_mode"] == "annotation"
    assert result.params["chromosomes"] == ["chr1", "chr2"]
    assert result.inputs["mcool"] == str(mcool)
    assert result.inputs["feature_track"] is None
    assert result.output_files == [
        "cis_leaves.parquet",
        "cis_leaves.tsv",
        "cis_nodes.parquet",
    ]
    assert not (tmp_path / "out").exists()


def test_cluster_uniform_bins(mcool, tmp_path) -> None:
    result = cluster(make_uniform_config(mcool, tmp_path))

    assert result.cis_leaves.height == 8
    assert result.cis_leaves["kind"].unique().to_list() == ["bin"]
    assert result.cis_leaves["source_row"].null_count() == 8
    assert result.qc["n_domains"] == 8
    assert result.qc["n_tads"] == 0
    assert result.qc["n_bin_domains"] == 8
    assert result.qc["n_residuals"] == 0
    assert result.params["input_mode"] == "uniform_bins"
    assert result.params["annotation_path"] is None
    assert result.params["working_binsize"] == 500
    assert result.params["working_binsize_mode"] == "explicit"
    assert result.inputs["annotation"] is None
    for chromosome_qc in result.qc["cis"]["per_chromosome"].values():
        assert chromosome_qc["n_bin_domains"] == 4
        assert chromosome_qc["dp_cut_size"] == 4
        assert chromosome_qc["dp_objective"] == 0.0
        assert chromosome_qc["dp_lambda"] == 0.0


def test_uniform_bins_and_disjoint_tads_skip_dp(
    monkeypatch,
    mcool,
    tmp_path,
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("DP must not run")

    monkeypatch.setattr(domains_module, "antichain_dp", fail)

    uniform = cluster(make_uniform_config(mcool, tmp_path))
    assert uniform.cis_leaves.height == 8

    disjoint = cluster(
        make_config(
            mcool,
            base_tads(tmp_path),
            tmp_path,
            output_dir=tmp_path / "disjoint",
        )
    )
    assert disjoint.cis_leaves.height == 4


def test_hierarchical_tads_use_dp(monkeypatch, mcool, tmp_path) -> None:
    tads = write_tads(
        tmp_path,
        [
            ("chr1", 0, 1500),
            ("chr1", 0, 500),
            ("chr1", 500, 1500),
            ("chr1", 1500, 2000),
            ("chr2", 0, 1500),
            ("chr2", 0, 500),
            ("chr2", 500, 1500),
            ("chr2", 1500, 2000),
        ],
    )

    def fail(*args, **kwargs):
        raise AssertionError("DP must run for hierarchical annotations")

    monkeypatch.setattr(domains_module, "antichain_dp", fail)

    with pytest.raises(AssertionError, match="DP must run"):
        cluster(make_config(mcool, tads, tmp_path))


def test_cluster_uniform_bins_parallel_matches_serial(mcool, tmp_path) -> None:
    serial = cluster(
        make_uniform_config(
            mcool,
            tmp_path,
            output_dir=tmp_path / "serial",
        )
    )
    parallel = cluster(
        make_uniform_config(
            mcool,
            tmp_path,
            output_dir=tmp_path / "parallel",
            workers=2,
        )
    )

    columns = [
        "domain_id",
        "chrom",
        "start",
        "end",
        "kind",
        "source_row",
        "n_bins",
    ]
    assert serial.cis_leaves.select(columns).equals(parallel.cis_leaves.select(columns))
    assert np.allclose(
        np.vstack(serial.cis_leaves["pcs"].to_list()),
        np.vstack(parallel.cis_leaves["pcs"].to_list()),
    )


def test_cluster_with_trans(three_chrom_mcool, trans_tads, tmp_path) -> None:
    result = cluster(
        make_config(
            three_chrom_mcool,
            trans_tads,
            tmp_path,
            trans_enabled=True,
            trans_null="grand",
        )
    )

    assert result.trans_nodes is not None
    assert "trans_root_label" in result.cis_leaves.columns
    assert "trans_path" in result.cis_leaves.columns
    assert "trans_path_label" in result.cis_leaves.columns
    assert "trans_depth" in result.cis_leaves.columns
    assert result.trans_nodes["root_label"].unique().to_list() == ["genome"]
    assert set(
        result.cis_leaves["trans_root_label"].drop_nulls().unique().to_list()
    ) == {"genome"}
    assert {"chroms", "chrom_domain_counts"}.issubset(result.trans_nodes.columns)
    assert result.qc["trans"]["enabled"] is True
    assert result.qc["trans"]["skipped"] is None
    assert result.qc["trans"]["n_domains"] == 6
    assert result.qc["trans"]["alpha"] is not None
    pair_alphas = result.qc["trans"]["pair_alphas"]
    assert len(pair_alphas) == 3
    assert {tuple(entry["pair"]) for entry in pair_alphas} == {
        ("chr1", "chr2"),
        ("chr1", "chr3"),
        ("chr2", "chr3"),
    }
    assert all(
        entry["alpha"] == pytest.approx(result.qc["trans"]["alpha"])
        for entry in pair_alphas
    )

    root = result.trans_nodes.filter(result.trans_nodes["node_id"] == 0).row(
        0, named=True
    )
    assert root["chroms"] == ["chr1", "chr2", "chr3"]
    assert sum(root["chrom_domain_counts"]) == 6
    assert "cis_leaves.tsv" in result.output_files
    assert result.output_files[-1] == "trans_nodes.parquet"


def test_cluster_trans_no_positive_mode_raises(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)

    with pytest.raises(ClusteringError):
        cluster(
            make_config(
                mcool,
                tads,
                tmp_path,
                trans_enabled=True,
            )
        )


def test_cluster_with_ab(mcool, tmp_path) -> None:
    tads = ab_tads(tmp_path)
    track = ab_track(tmp_path)
    result = cluster(
        make_config(
            mcool,
            tads,
            tmp_path,
            ab_feature_track=track,
        )
    )

    assert result.cis_leaves_ab is not None
    assert result.cis_nodes_ab is not None
    assert "ab_label" in result.cis_nodes.columns
    assert "ab_label" in result.cis_leaves.columns
    assert result.cis_leaves["ab_label"].to_list() == (
        result.cis_leaves_ab["ab_label"].to_list()
    )
    np.testing.assert_allclose(
        result.cis_leaves["ab_feature_rank"].to_numpy(),
        result.cis_leaves_ab["ab_feature_rank"].to_numpy(),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.cis_leaves["ab_score"].to_numpy(),
        result.cis_leaves_ab["ab_score"].to_numpy(),
        equal_nan=True,
    )
    assert "ab_label" in result.cis_leaves_ab.columns
    assert result.cis_leaves_ab.height == 8
    assert set(result.cis_leaves_ab["ab_label"].drop_nulls().to_list()) <= {
        "A",
        "B",
    }
    feature_ranks = np.asarray(
        result.cis_leaves_ab["ab_feature_rank"].to_numpy(),
        dtype=np.float64,
    )
    assert feature_ranks.min() >= 0.0
    assert feature_ranks.max() <= 1.0
    assert set(result.cis_nodes_ab["root_label"].to_list()) >= {"A", "B"}

    ab_qc = result.qc["ab"]
    assert ab_qc["enabled"] is True
    assert ab_qc["split_policy"] == "kmeans2"
    assert ab_qc["n_domains_a"] + ab_qc["n_domains_b"] == 8
    assert set(ab_qc["per_chromosome"]) == {"chr1", "chr2"}
    assert "n_nodes_a" not in ab_qc
    assert set(ab_qc["unconstrained_forest"]) == {
        "n_nodes_a",
        "n_nodes_b",
    }
    for chromosome_qc in ab_qc["per_chromosome"].values():
        assert "n_nodes_a" not in chromosome_qc
        assert set(chromosome_qc["unconstrained_forest"]) == {
            "n_nodes_a",
            "n_nodes_b",
        }
    assert result.output_files == [
        "cis_leaves.parquet",
        "cis_leaves.tsv",
        "cis_nodes.parquet",
        "cis_leaves_ab.parquet",
        "cis_leaves_ab.tsv",
        "cis_nodes_ab.parquet",
    ]


def test_cluster_chromosome_selection(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    result = cluster(
        make_config(
            mcool,
            tads,
            tmp_path,
            chromosomes=["chr2"],
        )
    )

    assert result.cis_leaves["chrom"].unique().to_list() == ["chr2"]
    assert result.params["chromosomes"] == ["chr2"]
    assert set(result.qc["cis"]["per_chromosome"]) == {"chr2"}


def test_cluster_trans_skipped_for_single_chromosome(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    result = cluster(
        make_config(
            mcool,
            tads,
            tmp_path,
            chromosomes=["chr1"],
            trans_enabled=True,
        )
    )

    assert result.trans_nodes is None
    assert "trans_path" not in result.cis_leaves.columns
    assert result.qc["trans"]["enabled"] is True
    assert result.qc["trans"]["skipped"] == "no_chromosome_pairs"


def test_cluster_adaptive_resolution(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    result = cluster(
        make_config(
            mcool,
            tads,
            tmp_path,
            working_binsize=None,
        )
    )

    assert result.params["working_binsize"] == 500
    assert result.params["working_binsize_mode"] == "adaptive"


def test_cluster_rejects_unknown_chromosome(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    with pytest.raises(InputError):
        cluster(
            make_config(
                mcool,
                tads,
                tmp_path,
                chromosomes=["chr3"],
            )
        )


def test_cluster_rejects_chromosome_without_tad_rows(mcool, tmp_path) -> None:
    tads = write_tads(tmp_path, [("chr2", 0, 1000)])
    with pytest.raises(InputError):
        cluster(
            make_config(
                mcool,
                tads,
                tmp_path,
                chromosomes=["chr1"],
            )
        )


def test_cluster_parallel_matches_serial(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    serial = cluster(
        make_config(
            mcool,
            tads,
            tmp_path,
            output_dir=tmp_path / "serial",
            workers=1,
        )
    )
    parallel = cluster(
        make_config(
            mcool,
            tads,
            tmp_path,
            output_dir=tmp_path / "parallel",
            workers=2,
        )
    )

    columns = [
        "domain_id",
        "chrom",
        "start",
        "end",
        "kind",
        "source_row",
        "n_bins",
    ]
    assert serial.cis_leaves.select(columns).equals(parallel.cis_leaves.select(columns))
    assert np.allclose(
        np.vstack(serial.cis_leaves["pcs"].to_list()),
        np.vstack(parallel.cis_leaves["pcs"].to_list()),
    )
    node_columns = [
        "chrom",
        "node_id",
        "parent_id",
        "depth",
        "path",
        "is_leaf",
    ]
    assert serial.cis_nodes.select(node_columns).equals(
        parallel.cis_nodes.select(node_columns)
    )


def test_zero_contact_cis_domain_is_unclustered(tmp_path) -> None:
    mcool = write_mcool(
        tmp_path,
        resolutions=(500,),
        chroms=("chr1",),
        chrom_size=3000,
        empty_bins={0, 1},
    )
    result = cluster(make_uniform_config(mcool, tmp_path))

    leaves = result.cis_leaves.sort("start")
    unclustered = leaves.filter(pl.col("start") < 1000)
    clustered = leaves.filter(pl.col("start") >= 1000)
    assert unclustered["path"].to_list() == [None, None]
    assert unclustered["path_label"].to_list() == [None, None]
    assert unclustered["root_label"].to_list() == [None, None]
    assert unclustered["depth"].to_list() == [None, None]
    assert clustered["path"].null_count() == 0

    node_unclustered = result.cis_nodes.filter(pl.col("node_id").is_null())
    assert node_unclustered.height == 2
    assert node_unclustered["domain_id"].to_list() == [0, 1]
    assert node_unclustered["is_leaf"].to_list() == [True, True]
    assert node_unclustered["path"].to_list() == [None, None]
    assert result.qc["cis"]["n_unclustered"] == 2
    assert result.qc["cis"]["per_chromosome"]["chr1"]["n_unclustered"] == 2


def test_zero_contact_ab_domain_is_unclustered(tmp_path) -> None:
    mcool = write_mcool(
        tmp_path,
        resolutions=(500,),
        chroms=("chr1",),
        chrom_size=3000,
        empty_bins={0, 1},
    )
    track = write_track(
        tmp_path,
        [
            ("chr1", index * 500, (index + 1) * 500, float(index + 1))
            for index in range(6)
        ],
    )

    result = cluster(make_uniform_config(mcool, tmp_path, ab_feature_track=track))

    assert result.cis_leaves_ab is not None
    leaves = result.cis_leaves_ab.sort("start")
    unclustered = leaves.filter(pl.col("start") < 1000)
    clustered = leaves.filter(pl.col("start") >= 1000)
    assert unclustered["path"].to_list() == [None, None]
    assert unclustered["path_label"].to_list() == [None, None]
    assert unclustered["ab_label"].to_list() == [None, None]
    assert unclustered["ab_score"].to_list() == [None, None]
    assert clustered["path"].null_count() == 0

    assert result.cis_nodes_ab is not None
    node_unclustered = result.cis_nodes_ab.filter(pl.col("node_id").is_null())
    assert node_unclustered.height == 2
    assert node_unclustered["ab_label"].to_list() == [None, None]
    assert node_unclustered["ab_score"].to_list() == [None, None]
    assert result.qc["ab"]["n_unclustered"] == 2


def test_zero_contact_trans_domains_are_unclustered(tmp_path) -> None:
    mcool = write_mcool(
        tmp_path,
        resolutions=(500,),
        chroms=("chr1", "chr2", "chr3"),
        cross_count_fn=lambda i, j: 5 + i + j,
        omit_cross_pairs={("chr1", "chr3"), ("chr2", "chr3")},
    )
    tads = write_tads(
        tmp_path,
        [
            (chrom, start, start + 1000)
            for chrom in ("chr1", "chr2", "chr3")
            for start in (0, 1000)
        ],
    )

    result = cluster(
        make_config(
            mcool,
            tads,
            tmp_path,
            trans_enabled=True,
        )
    )

    chr3 = result.cis_leaves.filter(pl.col("chrom") == "chr3")
    assert chr3["trans_path"].to_list() == [None, None]
    assert chr3["trans_path_label"].to_list() == [None, None]
    other = result.cis_leaves.filter(pl.col("chrom") != "chr3")
    assert other["trans_path"].null_count() == 0
    assert result.trans_nodes is not None
    trans_leaf = result.trans_nodes.filter(pl.col("is_leaf"))
    assert trans_leaf["domain_id"].null_count() == 0
    trans_chroms = {
        chrom for chroms in result.trans_nodes["chroms"].to_list() for chrom in chroms
    }
    assert trans_chroms == {"chr1", "chr2"}
    assert result.qc["trans"]["n_clustered"] == 4
    assert result.qc["trans"]["n_unclustered"] == 2
