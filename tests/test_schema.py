"""Golden parquet-schema and manifest tests for synthetic end-to-end runs."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from conftest import write_tads, write_track

import tadclust
from tadclust.config import ClusterConfig
from tadclust.pipeline import cluster
from tadclust.writer import SCHEMA_VERSION, write_outputs


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
        "tree_n_pcs": 3,
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
        "tree_n_pcs": 3,
        "tree_branch_k": 2,
    }
    values.update(overrides)
    return ClusterConfig(**values)


def schema_items(path: Path):
    return list(pl.read_parquet(path).schema.items())


def tsv_columns(path: Path):
    return pl.read_csv(path, separator="\t", infer_schema_length=0).columns


def cis_leaves_schema(include_trans: bool, include_ab: bool):
    schema = [
        ("unit_id", pl.Int64),
        ("chrom", pl.String),
        ("start", pl.Int64),
        ("end", pl.Int64),
        ("unit_kind", pl.String),
        ("tad_table_row", pl.Int64),
        ("root_label", pl.String),
        ("path", pl.List(pl.Int64)),
        ("path_label", pl.String),
        ("depth", pl.Int64),
        ("pc", pl.Array(pl.Float64, 3)),
        ("n_working_bins", pl.Int64),
        ("node_oe", pl.Float64),
    ]
    if include_ab:
        schema.extend(
            [
                ("ab_label", pl.String),
                ("ab_feature_rank", pl.Float64),
                ("ab_score", pl.Float64),
            ]
        )
    if include_trans:
        schema.extend(
            [
                ("trans_root_label", pl.String),
                ("trans_path", pl.List(pl.Int64)),
                ("trans_path_label", pl.String),
                ("trans_depth", pl.Int64),
            ]
        )
    return schema


def cis_nodes_schema(include_ab: bool, n_pcs: int = 3):
    schema = [
        ("chrom", pl.String),
        ("node_id", pl.Int64),
        ("parent_id", pl.Int64),
        ("depth", pl.Int64),
        ("root_label", pl.String),
        ("path", pl.List(pl.Int64)),
        ("path_label", pl.String),
        ("is_leaf", pl.Boolean),
        ("n_units", pl.Int64),
        ("node_internal_sse", pl.Float64),
        ("node_split_gain", pl.Float64),
        ("child_ids", pl.List(pl.Int64)),
        ("stop_reason", pl.String),
        ("n_working_bins", pl.Int64),
        ("unit_id", pl.Int64),
        ("start", pl.Int64),
        ("end", pl.Int64),
        ("unit_kind", pl.String),
        ("tad_table_row", pl.Int64),
        ("pc", pl.Array(pl.Float64, n_pcs)),
        ("node_oe", pl.Float64),
    ]
    if include_ab:
        schema.extend(
            [
                ("ab_label", pl.String),
                ("ab_feature_rank", pl.Float64),
                ("ab_score", pl.Float64),
                ("ab_feature_rank_mean", pl.Float64),
                ("ab_frac_a", pl.Float64),
                ("ab_purity", pl.Float64),
            ]
        )
    return schema


CIS_LEAVES_TSV_COLUMNS = [
    "unit_id",
    "chrom",
    "start",
    "end",
    "unit_kind",
    "tad_table_row",
    "root_label",
    "path_label",
    "depth",
    "n_working_bins",
    "node_oe",
]

TRANS_LEAVES_TSV_COLUMNS = [
    "trans_root_label",
    "trans_path_label",
    "trans_depth",
]

CIS_LEAVES_AB_COLUMNS = [
    "ab_label",
    "ab_feature_rank",
    "ab_score",
]

CIS_LEAVES_AB_TSV_COLUMNS = [
    "unit_id",
    "chrom",
    "start",
    "end",
    "unit_kind",
    "tad_table_row",
    "root_label",
    "path_label",
    "depth",
    "ab_label",
    "ab_feature_rank",
    "ab_score",
]

CIS_LEAVES_AB_SCHEMA = [
    ("unit_id", pl.Int64),
    ("chrom", pl.String),
    ("start", pl.Int64),
    ("end", pl.Int64),
    ("unit_kind", pl.String),
    ("tad_table_row", pl.Int64),
    ("root_label", pl.String),
    ("path", pl.List(pl.Int64)),
    ("path_label", pl.String),
    ("depth", pl.Int64),
    ("ab_label", pl.String),
    ("ab_feature_rank", pl.Float64),
    ("ab_score", pl.Float64),
]

TRANS_NODES_SCHEMA = [
    ("node_id", pl.Int64),
    ("parent_id", pl.Int64),
    ("unit_id", pl.Int64),
    ("depth", pl.Int64),
    ("root_label", pl.String),
    ("path", pl.List(pl.Int64)),
    ("path_label", pl.String),
    ("is_leaf", pl.Boolean),
    ("n_units", pl.Int64),
    ("node_internal_sse", pl.Float64),
    ("node_split_gain", pl.Float64),
    ("child_ids", pl.List(pl.Int64)),
    ("stop_reason", pl.String),
    ("n_working_bins", pl.Int64),
    ("chroms", pl.List(pl.String)),
    ("chrom_unit_counts", pl.List(pl.Int64)),
]


def test_cis_only_golden_schema(mcool, tmp_path) -> None:
    config = make_config(mcool, base_tads(tmp_path), tmp_path)
    result = cluster(config)
    write_outputs(result, config)

    out = tmp_path / "out"
    assert schema_items(out / "cis_leaves.parquet") == cis_leaves_schema(False, False)
    assert schema_items(out / "cis_nodes.parquet") == cis_nodes_schema(False)
    assert tsv_columns(out / "cis_leaves.tsv") == CIS_LEAVES_TSV_COLUMNS
    assert not (out / "cis_leaves_ab.parquet").exists()
    assert not (out / "cis_leaves_ab.tsv").exists()
    assert not (out / "cis_nodes_ab.parquet").exists()
    assert not (out / "trans_nodes.parquet").exists()


def test_trans_golden_schema(three_chrom_mcool, trans_tads, tmp_path) -> None:
    config = make_config(
        three_chrom_mcool,
        trans_tads,
        tmp_path,
        trans_enabled=True,
        trans_null="grand",
    )
    result = cluster(config)
    write_outputs(result, config)

    out = tmp_path / "out"
    assert schema_items(out / "cis_leaves.parquet") == cis_leaves_schema(True, False)
    assert schema_items(out / "cis_nodes.parquet") == cis_nodes_schema(False)
    assert schema_items(out / "trans_nodes.parquet") == TRANS_NODES_SCHEMA
    assert tsv_columns(out / "cis_leaves.tsv") == (
        CIS_LEAVES_TSV_COLUMNS + TRANS_LEAVES_TSV_COLUMNS
    )


def test_ab_golden_schema(mcool, tmp_path) -> None:
    config = make_config(
        mcool,
        ab_tads(tmp_path),
        tmp_path,
        ab_feature_track=ab_track(tmp_path),
        ab_split_policy="kmeans2",
    )
    result = cluster(config)
    write_outputs(result, config)

    out = tmp_path / "out"
    assert schema_items(out / "cis_leaves.parquet") == cis_leaves_schema(False, True)
    assert schema_items(out / "cis_nodes.parquet") == cis_nodes_schema(True)
    assert schema_items(out / "cis_leaves_ab.parquet") == CIS_LEAVES_AB_SCHEMA
    assert schema_items(out / "cis_nodes_ab.parquet") == cis_nodes_schema(True)
    assert tsv_columns(out / "cis_leaves.tsv") == (
        CIS_LEAVES_TSV_COLUMNS + CIS_LEAVES_AB_COLUMNS
    )
    assert tsv_columns(out / "cis_leaves_ab.tsv") == CIS_LEAVES_AB_TSV_COLUMNS
    assert not (out / "trans_nodes.parquet").exists()


def test_uniform_bins_golden_schema(mcool, tmp_path) -> None:
    config = make_uniform_config(mcool, tmp_path)
    result = cluster(config)
    write_outputs(result, config)

    out = tmp_path / "out"
    assert schema_items(out / "cis_leaves.parquet") == cis_leaves_schema(False, False)
    assert schema_items(out / "cis_nodes.parquet") == cis_nodes_schema(False)
    assert tsv_columns(out / "cis_leaves.tsv") == CIS_LEAVES_TSV_COLUMNS

    leaves = pl.read_parquet(out / "cis_leaves.parquet")
    assert set(leaves["unit_kind"].to_list()) == {"bin"}
    assert leaves["tad_table_row"].null_count() == leaves.height

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["inputs"]["annotation"] is None
    assert manifest["params"]["input_mode"] == "uniform_bins"
    assert manifest["params"]["annotation_path"] is None
    assert manifest["qc"]["n_bin_units"] == 8
    assert manifest["qc"]["n_tads"] == 0
    assert manifest["qc"]["n_residuals"] == 0


def test_manifest_contract(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    config = make_config(mcool, tads, tmp_path)
    result = cluster(config)
    write_outputs(result, config)

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert list(manifest) == [
        "schema_version",
        "tadclust_version",
        "created_utc",
        "inputs",
        "cli",
        "params",
        "qc",
        "output_files",
    ]
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["tadclust_version"] == tadclust.__version__
    assert manifest["created_utc"].endswith("Z")
    assert manifest["inputs"] == {
        "mcool": str(mcool),
        "annotation": str(tads),
        "feature_track": None,
    }
    assert manifest["cli"] is None
    assert manifest["params"]["working_binsize"] == 500
    assert manifest["params"]["working_binsize_mode"] == "explicit"
    assert manifest["params"]["ab_split_policy"] is None
    assert manifest["qc"]["n_units"] == 4
    assert manifest["output_files"] == [
        "cis_leaves.parquet",
        "cis_leaves.tsv",
        "cis_nodes.parquet",
        "manifest.json",
    ]
    assert "checksums" not in manifest
    assert "contact_contrast" not in json.dumps(manifest)


def test_cis_leaves_derived_from_cis_nodes(mcool, tmp_path) -> None:
    config = make_config(mcool, base_tads(tmp_path), tmp_path)
    result = cluster(config)

    leaf_columns = [column for column in result.cis_leaves.columns]
    assert set(leaf_columns) <= set(result.cis_nodes.columns)

    derived = (
        result.cis_nodes.filter(pl.col("is_leaf")).select(leaf_columns).sort("unit_id")
    )
    expected = result.cis_leaves.select(leaf_columns).sort("unit_id")
    assert derived.equals(expected)


def test_trans_leaves_derived_from_cis_and_trans_nodes(
    three_chrom_mcool,
    trans_tads,
    tmp_path,
) -> None:
    config = make_config(
        three_chrom_mcool,
        trans_tads,
        tmp_path,
        trans_enabled=True,
        trans_null="grand",
    )
    result = cluster(config)
    assert result.trans_nodes is not None

    trans_leaf = result.trans_nodes.filter(pl.col("is_leaf"))
    assert trans_leaf["unit_id"].null_count() == 0
    trans_columns = trans_leaf.select(
        "unit_id",
        pl.col("root_label").alias("trans_root_label"),
        pl.col("path").alias("trans_path"),
        pl.col("path_label").alias("trans_path_label"),
        pl.col("depth").alias("trans_depth"),
    )

    base_columns = [
        column
        for column in result.cis_leaves.columns
        if not column.startswith("trans_")
    ]
    derived = (
        result.cis_nodes.filter(pl.col("is_leaf"))
        .select(base_columns)
        .join(trans_columns, on="unit_id", how="left")
        .select(result.cis_leaves.columns)
        .sort("unit_id")
    )
    assert derived.equals(result.cis_leaves.sort("unit_id"))


def test_ab_leaves_derived_from_ab_nodes(mcool, tmp_path) -> None:
    config = make_config(
        mcool,
        ab_tads(tmp_path),
        tmp_path,
        ab_feature_track=ab_track(tmp_path),
        ab_split_policy="kmeans2",
    )
    result = cluster(config)
    assert result.cis_nodes_ab is not None
    assert result.cis_leaves_ab is not None
    assert set(result.cis_leaves.columns) <= set(result.cis_nodes.columns)

    leaf_columns = list(result.cis_leaves_ab.columns)
    assert set(leaf_columns) <= set(result.cis_nodes_ab.columns)

    derived = (
        result.cis_nodes_ab.filter(pl.col("is_leaf"))
        .select(leaf_columns)
        .sort("unit_id")
    )
    expected = result.cis_leaves_ab.select(leaf_columns).sort("unit_id")
    assert derived.equals(expected)
