"""Tests for parquet schemas, manifest writing, and overwrite handling."""

from __future__ import annotations

import json

import polars as pl
import pytest

import tadclust
from tadclust.config import ClusterConfig
from tadclust.errors import OutputError
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
        "tree_n_pcs": 2,
        "tree_branch_k": 2,
    }
    values.update(overrides)
    return ClusterConfig(**values)


def test_write_cis_only_outputs(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    config = make_config(mcool, tads, tmp_path)
    result = cluster(config)

    written = write_outputs(
        result,
        config,
        cli_args=["--mcool", str(mcool)],
    )

    assert set(written) == {
        "cis_leaves.parquet",
        "cis_nodes.parquet",
        "manifest.json",
    }
    for path in written.values():
        assert path.is_file()
    assert not list((tmp_path / "out").glob("*.tmp"))

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
    assert manifest["inputs"] == result.inputs
    assert manifest["cli"] == {"argv": ["--mcool", str(mcool)]}
    assert manifest["params"]["working_binsize"] == 500
    assert manifest["qc"]["n_domains"] == 4
    assert manifest["output_files"] == [
        "cis_leaves.parquet",
        "cis_nodes.parquet",
        "manifest.json",
    ]
    assert "checksums" not in manifest

    leaves = pl.read_parquet(tmp_path / "out" / "cis_leaves.parquet")
    nodes = pl.read_parquet(tmp_path / "out" / "cis_nodes.parquet")
    assert leaves.columns == [
        "domain_id",
        "chrom",
        "start",
        "end",
        "kind",
        "source_row",
        "n_bins",
        "self_oe",
        "pc1",
        "pc2",
        "root_label",
        "path",
        "path_label",
        "depth",
    ]
    assert leaves.schema["path"] == pl.List(pl.Int64)
    assert leaves.schema["depth"] == pl.Int64
    assert leaves.schema["self_oe"] == pl.Float64
    assert nodes.schema["parent_id"] == pl.Int64
    assert nodes.schema["children"] == pl.List(pl.Int64)
    assert nodes.schema["is_leaf"] == pl.Boolean
    assert nodes.schema["split_gain"] == pl.Float64
    assert "ab_label" not in nodes.columns


def test_write_ab_and_trans_outputs(mcool, tmp_path) -> None:
    tads = ab_tads(tmp_path)
    track = ab_track(tmp_path)
    config = make_config(
        mcool,
        tads,
        tmp_path,
        ab_signal_track=track,
        ab_split="kmeans2",
        trans_enabled=True,
    )
    result = cluster(config)

    write_outputs(result, config)

    out = tmp_path / "out"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["output_files"] == [
        "cis_leaves.parquet",
        "cis_nodes.parquet",
        "cis_leaves_ab.parquet",
        "cis_nodes_ab.parquet",
        "trans_nodes.parquet",
        "manifest.json",
    ]

    leaves = pl.read_parquet(out / "cis_leaves.parquet")
    nodes = pl.read_parquet(out / "cis_nodes.parquet")
    leaves_ab = pl.read_parquet(out / "cis_leaves_ab.parquet")
    nodes_ab = pl.read_parquet(out / "cis_nodes_ab.parquet")
    trans_nodes = pl.read_parquet(out / "trans_nodes.parquet")

    assert "trans_path" in leaves.columns
    assert leaves.schema["trans_path"] == pl.List(pl.Int64)
    assert "ab_label" in nodes.columns
    assert "ab_score" in leaves_ab.columns
    assert "ab_label" in nodes_ab.columns
    assert trans_nodes.schema["chroms"] == pl.List(pl.String)
    assert trans_nodes.schema["chrom_domain_counts"] == pl.List(pl.Int64)


def test_nonempty_output_dir_requires_overwrite(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    config = make_config(mcool, tads, tmp_path)
    result = cluster(config)
    write_outputs(result, config)
    unrelated = tmp_path / "out" / "keep.txt"
    unrelated.write_text("keep")

    with pytest.raises(OutputError):
        write_outputs(result, config)

    write_outputs(result, config, overwrite=True)
    assert unrelated.read_text() == "keep"


def test_overwrite_removes_stale_optional_tables(mcool, tmp_path) -> None:
    ab_config = make_config(
        mcool,
        ab_tads(tmp_path),
        tmp_path,
        ab_signal_track=ab_track(tmp_path),
        ab_split="kmeans2",
        trans_enabled=True,
    )
    write_outputs(cluster(ab_config), ab_config)
    out = tmp_path / "out"
    assert (out / "cis_leaves_ab.parquet").exists()
    assert (out / "cis_nodes_ab.parquet").exists()
    assert (out / "trans_nodes.parquet").exists()

    config = make_config(mcool, base_tads(tmp_path), tmp_path)
    result = cluster(config)
    write_outputs(result, config, overwrite=True)

    assert not (out / "cis_leaves_ab.parquet").exists()
    assert not (out / "cis_nodes_ab.parquet").exists()
    assert not (out / "trans_nodes.parquet").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["output_files"] == [
        "cis_leaves.parquet",
        "cis_nodes.parquet",
        "manifest.json",
    ]


def test_output_path_must_be_a_directory(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    target = tmp_path / "out"
    target.write_text("not a directory")
    config = make_config(mcool, tads, tmp_path, output_dir=target)
    result = cluster(config)

    with pytest.raises(OutputError):
        write_outputs(result, config)
