"""Golden parquet-schema and manifest tests for synthetic end-to-end runs."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

import tadclust
from tadclust.config import ClusterConfig
from tadclust.pipeline import cluster
from tadclust.writer import SCHEMA_VERSION, write_outputs

from conftest import write_tads, write_track


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


def schema_items(path: Path):
    return list(pl.read_parquet(path).schema.items())


def cis_leaves_schema(include_trans: bool):
    schema = [
        ("domain_id", pl.Int64),
        ("chrom", pl.String),
        ("start", pl.Int64),
        ("end", pl.Int64),
        ("kind", pl.String),
        ("source_row", pl.Int64),
        ("n_bins", pl.Int64),
        ("self_oe", pl.Float64),
        ("pcs", pl.Array(pl.Float64, 3)),
        ("root_label", pl.String),
        ("path", pl.List(pl.Int64)),
        ("path_label", pl.String),
        ("depth", pl.Int64),
    ]
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


def cis_nodes_schema(include_ab: bool):
    schema = [
        ("chrom", pl.String),
        ("node_id", pl.Int64),
        ("parent_id", pl.Int64),
        ("depth", pl.Int64),
        ("root_label", pl.String),
        ("path", pl.List(pl.Int64)),
        ("path_label", pl.String),
        ("is_leaf", pl.Boolean),
        ("n_domains", pl.Int64),
        ("n_bins", pl.Int64),
        ("sse", pl.Float64),
        ("split_gain", pl.Float64),
        ("children", pl.List(pl.Int64)),
        ("stop_reason", pl.String),
    ]
    if include_ab:
        schema.extend(
            [
                ("ab_label", pl.String),
                ("ab_frac_a", pl.Float64),
                ("ab_purity", pl.Float64),
                ("ab_rank_mean", pl.Float64),
            ]
        )
    return schema


CIS_LEAVES_AB_SCHEMA = [
    ("domain_id", pl.Int64),
    ("chrom", pl.String),
    ("start", pl.Int64),
    ("end", pl.Int64),
    ("kind", pl.String),
    ("source_row", pl.Int64),
    ("root_label", pl.String),
    ("path", pl.List(pl.Int64)),
    ("path_label", pl.String),
    ("depth", pl.Int64),
    ("ab_label", pl.String),
    ("ab_rank", pl.Float64),
    ("ab_score", pl.Float64),
    ("ab_margin", pl.Float64),
]

TRANS_NODES_SCHEMA = [
    ("node_id", pl.Int64),
    ("parent_id", pl.Int64),
    ("depth", pl.Int64),
    ("root_label", pl.String),
    ("path", pl.List(pl.Int64)),
    ("path_label", pl.String),
    ("is_leaf", pl.Boolean),
    ("n_domains", pl.Int64),
    ("n_bins", pl.Int64),
    ("sse", pl.Float64),
    ("split_gain", pl.Float64),
    ("children", pl.List(pl.Int64)),
    ("stop_reason", pl.String),
    ("chroms", pl.List(pl.String)),
    ("chrom_domain_counts", pl.List(pl.Int64)),
]


def test_cis_only_golden_schema(mcool, tmp_path) -> None:
    config = make_config(mcool, base_tads(tmp_path), tmp_path)
    result = cluster(config)
    write_outputs(result, config)

    out = tmp_path / "out"
    assert schema_items(out / "cis_leaves.parquet") == cis_leaves_schema(False)
    assert schema_items(out / "cis_nodes.parquet") == cis_nodes_schema(False)
    assert not (out / "cis_leaves_ab.parquet").exists()
    assert not (out / "cis_nodes_ab.parquet").exists()
    assert not (out / "trans_nodes.parquet").exists()


def test_trans_golden_schema(mcool, tmp_path) -> None:
    config = make_config(
        mcool,
        base_tads(tmp_path),
        tmp_path,
        trans_enabled=True,
    )
    result = cluster(config)
    write_outputs(result, config)

    out = tmp_path / "out"
    assert schema_items(out / "cis_leaves.parquet") == cis_leaves_schema(True)
    assert schema_items(out / "cis_nodes.parquet") == cis_nodes_schema(False)
    assert schema_items(out / "trans_nodes.parquet") == TRANS_NODES_SCHEMA


def test_ab_golden_schema(mcool, tmp_path) -> None:
    config = make_config(
        mcool,
        ab_tads(tmp_path),
        tmp_path,
        ab_signal_track=ab_track(tmp_path),
        ab_split="kmeans2",
    )
    result = cluster(config)
    write_outputs(result, config)

    out = tmp_path / "out"
    assert schema_items(out / "cis_leaves.parquet") == cis_leaves_schema(False)
    assert schema_items(out / "cis_nodes.parquet") == cis_nodes_schema(True)
    assert schema_items(out / "cis_leaves_ab.parquet") == CIS_LEAVES_AB_SCHEMA
    assert schema_items(out / "cis_nodes_ab.parquet") == cis_nodes_schema(True)
    assert not (out / "trans_nodes.parquet").exists()


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
        "signal_track": None,
    }
    assert manifest["cli"] is None
    assert manifest["params"]["working_binsize"] == 500
    assert manifest["params"]["working_binsize_mode"] == "explicit"
    assert manifest["qc"]["n_domains"] == 4
    assert manifest["output_files"] == [
        "cis_leaves.parquet",
        "cis_nodes.parquet",
        "manifest.json",
    ]
    assert "checksums" not in manifest
    assert "contact_contrast" not in json.dumps(manifest)
