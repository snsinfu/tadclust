"""Tests for the command-line interface."""

from __future__ import annotations

import json

import polars as pl
import pytest
from conftest import write_tads, write_track

from tadclust import __version__
from tadclust.cli import main


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


def cli_args(mcool, tads, tmp_path, *extra):
    return [
        "--mcool",
        str(mcool),
        "--tad-table",
        str(tads),
        "--out-dir",
        str(tmp_path / "out"),
        "--working-binsize",
        "500",
        "--tree-n-pcs",
        "2",
        "--tree-branch-k",
        "2",
        *extra,
    ]


def uniform_cli_args(mcool, tmp_path, *extra):
    return [
        "--mcool",
        str(mcool),
        "--uniform-bins",
        "--out-dir",
        str(tmp_path / "out"),
        "--working-binsize",
        "500",
        "--tree-n-pcs",
        "2",
        "--tree-branch-k",
        "2",
        *extra,
    ]


def test_cli_cis_run(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    args = cli_args(mcool, tads, tmp_path)

    main(args)

    out = tmp_path / "out"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["cli"]["argv"] == args
    assert manifest["params"]["working_binsize"] == 500
    assert manifest["qc"]["n_domains"] == 4
    assert (out / "cis_leaves.parquet").exists()
    assert not (out / "trans_nodes.parquet").exists()


def test_cli_uniform_bins_run(mcool, tmp_path) -> None:
    args = uniform_cli_args(mcool, tmp_path)

    main(args)

    out = tmp_path / "out"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["params"]["input_mode"] == "uniform_bins"
    assert manifest["params"]["annotation_path"] is None
    assert manifest["inputs"]["annotation"] is None
    assert manifest["qc"]["n_bin_domains"] == 8
    assert manifest["qc"]["n_tads"] == 0

    leaves = pl.read_parquet(out / "cis_leaves.parquet")
    assert set(leaves["kind"].to_list()) == {"bin"}
    assert leaves["source_row"].null_count() == leaves.height


def test_cli_uniform_bins_requires_working_binsize(mcool, tmp_path) -> None:
    args = [
        "--mcool",
        str(mcool),
        "--uniform-bins",
        "--out-dir",
        str(tmp_path / "out"),
    ]

    with pytest.raises(SystemExit) as info:
        main(args)
    assert "working_binsize is required when annotation_path is omitted" in str(
        info.value
    )


def test_cli_rejects_tad_table_and_uniform_bins(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    args = [*cli_args(mcool, tads, tmp_path), "--uniform-bins"]

    with pytest.raises(SystemExit) as info:
        main(args)
    assert info.value.code == 2


def test_help_lists_advanced(capsys) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--help"])

    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "advanced" in out
    assert "--oe-distance-bins" in out
    assert "--uniform-bins" in out


def test_cli_trans_run(three_chrom_mcool, trans_tads, tmp_path) -> None:
    args = cli_args(
        three_chrom_mcool,
        trans_tads,
        tmp_path,
        "--enable-trans",
        "--trans-null",
        "grand",
    )

    main(args)

    out = tmp_path / "out"
    assert (out / "trans_nodes.parquet").exists()
    leaves = pl.read_parquet(out / "cis_leaves.parquet")
    assert "trans_path" in leaves.columns


def test_cli_ab_run(mcool, tmp_path) -> None:
    tads = ab_tads(tmp_path)
    track = ab_track(tmp_path)
    args = cli_args(
        mcool,
        tads,
        tmp_path,
        "--ab-signal-track",
        str(track),
    )

    main(args)

    out = tmp_path / "out"
    assert (out / "cis_leaves_ab.parquet").exists()
    assert (out / "cis_nodes_ab.parquet").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["params"]["ab_split"] == "kmeans2"


def test_cli_ab_split_requires_track(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    args = cli_args(mcool, tads, tmp_path, "--ab-split", "kmeans3")

    with pytest.raises(SystemExit) as info:
        main(args)
    assert "ab_split requires ab_signal_track" in str(info.value)


def test_cli_overwrite(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    args = cli_args(mcool, tads, tmp_path)

    main(args)
    with pytest.raises(SystemExit) as info:
        main(args)
    assert "not empty" in str(info.value)
    main([*args, "--overwrite"])


def test_cli_invalid_config_returns_error(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    args = cli_args(mcool, tads, tmp_path, "--oe-pseudocount", "0")

    with pytest.raises(SystemExit) as info:
        main(args)
    assert "pseudocount" in str(info.value)


def test_cli_parallel_workers(three_chrom_mcool, trans_tads, tmp_path) -> None:
    args = cli_args(
        three_chrom_mcool,
        trans_tads,
        tmp_path,
        "--workers",
        "2",
        "--enable-trans",
        "--trans-null",
        "grand",
    )

    main(args)
    assert (tmp_path / "out" / "cis_leaves.parquet").exists()
    assert (tmp_path / "out" / "trans_nodes.parquet").exists()


def test_cli_version(capsys) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])

    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out
