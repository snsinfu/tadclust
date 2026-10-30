"""Tests for parquet schemas, manifest writing, and overwrite handling."""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
import pytest
from conftest import write_mcool, write_tads, write_track

import tadclust
from tadclust.config import ClusterConfig
from tadclust.errors import OutputError
from tadclust.pipeline import cluster
from tadclust.writer import (
    SCHEMA_VERSION,
    _stage_tsv,
    _temporary_path,
    write_outputs,
)

CIS_LEAVES_TSV_COLUMNS = [
    "unit_id",
    "chrom",
    "unit_start",
    "unit_end",
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
    "ab_score",
]
CIS_LEAVES_AB_TSV_COLUMNS = [
    "unit_id",
    "chrom",
    "unit_start",
    "unit_end",
    "unit_kind",
    "tad_table_row",
    "root_label",
    "path_label",
    "depth",
    "ab_label",
    "ab_score",
]


def read_tsv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", infer_schema_length=0)


def test_stage_tsv_replaces_nulls_with_type_sentinels(tmp_path) -> None:
    table = pl.DataFrame(
        {
            "i": [1, None],
            "f": [2.5, None],
            "s": ["x", None],
            "b": [True, None],
        }
    )
    schema = {
        "i": pl.Int64,
        "f": pl.Float64,
        "s": pl.String,
        "b": pl.Boolean,
    }

    temporary, _ = _stage_tsv(table, tmp_path / "out.tsv", schema)

    assert temporary.read_text().splitlines() == [
        "i\tf\ts\tb",
        "1\t2.5\tx\ttrue",
        "-1\tNaN\tNA\tNA",
    ]


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


def test_temporary_paths_are_unique(tmp_path) -> None:
    target = tmp_path / "cis_leaves.parquet"

    first = _temporary_path(target)
    second = _temporary_path(target)

    assert first != second
    assert first.parent == target.parent
    assert second.parent == target.parent
    assert first.name.endswith(".tmp")
    assert second.name.endswith(".tmp")


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
        "cis_leaves.tsv",
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
    assert manifest["qc"]["n_units"] == 4
    assert manifest["output_files"] == [
        "cis_leaves.parquet",
        "cis_leaves.tsv",
        "cis_nodes.parquet",
        "manifest.json",
    ]
    assert "checksums" not in manifest

    leaves = pl.read_parquet(tmp_path / "out" / "cis_leaves.parquet")
    leaves_tsv = read_tsv(tmp_path / "out" / "cis_leaves.tsv")
    nodes = pl.read_parquet(tmp_path / "out" / "cis_nodes.parquet")
    assert leaves.columns == [
        "unit_id",
        "chrom",
        "unit_start",
        "unit_end",
        "unit_kind",
        "tad_table_row",
        "root_label",
        "path",
        "path_label",
        "depth",
        "pc",
        "n_working_bins",
        "node_oe",
    ]
    assert leaves.schema["path"] == pl.List(pl.Int64)
    assert leaves.schema["depth"] == pl.Int64
    assert leaves.schema["node_oe"] == pl.Float64
    assert leaves.schema["pc"] == pl.Array(pl.Float64, 2)
    assert nodes.schema["parent_id"] == pl.Int64
    assert nodes.schema["child_ids"] == pl.List(pl.Int64)
    assert nodes.schema["is_leaf"] == pl.Boolean
    assert nodes.schema["node_split_gain"] == pl.Float64
    assert nodes.schema["unit_id"] == pl.Int64
    assert nodes.schema["pc"] == pl.Array(pl.Float64, 2)
    assert nodes.schema["node_oe"] == pl.Float64
    assert "ab_label" not in nodes.columns

    assert leaves_tsv.columns == CIS_LEAVES_TSV_COLUMNS
    assert leaves_tsv.height == leaves.height
    assert set(leaves_tsv["path_label"].to_list()) == set(
        leaves["path_label"].to_list()
    )
    assert "pc" not in leaves_tsv.columns
    assert "path" not in leaves_tsv.columns


def test_uniform_bin_tsv_nulls_use_sentinels(tmp_path) -> None:
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
    config = make_config(
        mcool,
        None,
        tmp_path,
        annotation_path=None,
        ab_feature_track=track,
    )
    result = cluster(config)
    write_outputs(result, config)
    out = tmp_path / "out"

    parquet = pl.read_parquet(out / "cis_leaves.parquet")
    assert parquet["tad_table_row"].null_count() == parquet.height

    for name in ("cis_leaves.tsv", "cis_leaves_ab.tsv"):
        text = (out / name).read_text()
        for line in text.splitlines():
            assert all(cell != "" for cell in line.split("\t"))

    leaves = read_tsv(out / "cis_leaves.tsv")
    assert leaves["tad_table_row"].to_list() == ["-1"] * leaves.height
    unclustered = leaves.filter(pl.col("unit_start").cast(pl.Int64) < 1000)
    assert unclustered["root_label"].to_list() == ["NA", "NA"]
    assert unclustered["path_label"].to_list() == ["NA", "NA"]
    assert unclustered["depth"].to_list() == ["-1", "-1"]
    assert unclustered["ab_label"].to_list() == ["NA", "NA"]
    assert unclustered["ab_score"].to_list() == ["NaN", "NaN"]

    leaves_ab = read_tsv(out / "cis_leaves_ab.tsv")
    unclustered_ab = leaves_ab.filter(pl.col("unit_start").cast(pl.Int64) < 1000)
    assert unclustered_ab["ab_label"].to_list() == ["NA", "NA"]
    assert unclustered_ab["ab_score"].to_list() == ["NaN", "NaN"]


def test_write_ab_and_trans_outputs(
    three_chrom_mcool,
    trans_tads,
    trans_track,
    tmp_path,
) -> None:
    config = make_config(
        three_chrom_mcool,
        trans_tads,
        tmp_path,
        ab_feature_track=trans_track,
        ab_split_policy="kmeans2",
        trans_enabled=True,
        trans_null="grand",
    )
    result = cluster(config)

    write_outputs(result, config)

    out = tmp_path / "out"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["output_files"] == [
        "cis_leaves.parquet",
        "cis_leaves.tsv",
        "cis_nodes.parquet",
        "cis_leaves_ab.parquet",
        "cis_leaves_ab.tsv",
        "cis_nodes_ab.parquet",
        "trans_nodes.parquet",
        "manifest.json",
    ]

    leaves = pl.read_parquet(out / "cis_leaves.parquet")
    leaves_tsv = read_tsv(out / "cis_leaves.tsv")
    nodes = pl.read_parquet(out / "cis_nodes.parquet")
    leaves_ab = pl.read_parquet(out / "cis_leaves_ab.parquet")
    leaves_ab_tsv = read_tsv(out / "cis_leaves_ab.tsv")
    nodes_ab = pl.read_parquet(out / "cis_nodes_ab.parquet")
    trans_nodes = pl.read_parquet(out / "trans_nodes.parquet")

    assert "trans_path" in leaves.columns
    assert leaves.schema["trans_path"] == pl.List(pl.Int64)
    assert "ab_label" in leaves.columns
    assert "ab_margin" not in leaves.columns
    assert {"ab_label", "ab_score", "ab_frac_a", "ab_frac_b", "ab_purity"} <= set(
        nodes.columns
    )
    assert "ab_margin" not in nodes.columns
    assert "ab_score" in leaves_ab.columns
    assert "ab_margin" not in leaves_ab.columns
    assert "ab_label" in nodes_ab.columns
    assert "ab_margin" not in nodes_ab.columns
    assert trans_nodes.schema["chroms"] == pl.List(pl.String)
    assert trans_nodes.schema["chrom_unit_counts"] == pl.List(pl.Int64)
    assert trans_nodes.schema["unit_id"] == pl.Int64

    assert leaves_tsv.columns == (
        CIS_LEAVES_TSV_COLUMNS + CIS_LEAVES_AB_COLUMNS + TRANS_LEAVES_TSV_COLUMNS
    )
    assert "trans_path" not in leaves_tsv.columns
    assert "trans_path_label" in leaves_tsv.columns
    assert leaves_ab_tsv.columns == CIS_LEAVES_AB_TSV_COLUMNS
    assert "path" not in leaves_ab_tsv.columns


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


def test_write_outputs_restores_old_outputs_on_failure(
    mcool,
    tmp_path,
    monkeypatch,
) -> None:
    tads = base_tads(tmp_path)
    config = make_config(mcool, tads, tmp_path)
    result = cluster(config)
    write_outputs(result, config)

    out = tmp_path / "out"
    leaves = out / "cis_leaves.parquet"
    sentinel = b"old leaves"
    leaves.write_bytes(sentinel)
    old_nodes = (out / "cis_nodes.parquet").read_bytes()
    old_manifest = (out / "manifest.json").read_bytes()

    real_replace = os.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        if not failed and destination == out / "cis_nodes.parquet":
            failed = True
            raise OSError("simulated write failure")
        return real_replace(source, destination)

    monkeypatch.setattr("tadclust.writer.os.replace", fail_once)

    with pytest.raises(OSError):
        write_outputs(result, config, overwrite=True)

    assert leaves.read_bytes() == sentinel
    assert (out / "cis_nodes.parquet").read_bytes() == old_nodes
    assert (out / "manifest.json").read_bytes() == old_manifest
    assert not list(out.glob("*.tmp"))


def test_overwrite_removes_stale_optional_tables(
    three_chrom_mcool,
    trans_tads,
    trans_track,
    tmp_path,
) -> None:
    ab_config = make_config(
        three_chrom_mcool,
        trans_tads,
        tmp_path,
        ab_feature_track=trans_track,
        ab_split_policy="kmeans2",
        trans_enabled=True,
        trans_null="grand",
    )
    write_outputs(cluster(ab_config), ab_config)
    out = tmp_path / "out"
    assert (out / "cis_leaves.tsv").exists()
    assert (out / "cis_leaves_ab.parquet").exists()
    assert (out / "cis_leaves_ab.tsv").exists()
    assert (out / "cis_nodes_ab.parquet").exists()
    assert (out / "trans_nodes.parquet").exists()

    config = make_config(three_chrom_mcool, base_tads(tmp_path), tmp_path)
    result = cluster(config)
    write_outputs(result, config, overwrite=True)

    assert (out / "cis_leaves.tsv").exists()
    assert not (out / "cis_leaves_ab.parquet").exists()
    assert not (out / "cis_leaves_ab.tsv").exists()
    assert not (out / "cis_nodes_ab.parquet").exists()
    assert not (out / "trans_nodes.parquet").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["output_files"] == [
        "cis_leaves.parquet",
        "cis_leaves.tsv",
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
