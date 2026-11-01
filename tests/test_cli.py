"""Tests for the command-line interface."""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import polars as pl
import pytest
from conftest import write_tads, write_track

from tadclust import __version__
from tadclust.cli import main, parse_args


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
        "--output-dir",
        str(tmp_path / "out"),
        "--working-binsize",
        "500",
        "--tree-pca-dim",
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
        "--output-dir",
        str(tmp_path / "out"),
        "--working-binsize",
        "500",
        "--tree-pca-dim",
        "2",
        "--tree-branch-k",
        "2",
        *extra,
    ]


def test_cli_default_tree_branch_k() -> None:
    args = parse_args(
        [
            "--mcool",
            "test.mcool",
            "--tad-table",
            "domains.tsv",
            "--output-dir",
            "out",
        ]
    )

    assert args.tree_branch_k == 2
    assert args.with_residuals is False
    assert args.quiet is False


def test_cli_with_residuals_flag() -> None:
    args = parse_args(
        [
            "--mcool",
            "test.mcool",
            "--tad-table",
            "domains.tsv",
            "--output-dir",
            "out",
            "--with-residuals",
        ]
    )

    assert args.with_residuals is True


def test_cli_short_aliases(tmp_path) -> None:
    args = parse_args(
        [
            "--mcool",
            "test.mcool",
            "--tad-table",
            "domains.tsv",
            "-o",
            str(tmp_path / "out"),
            "-p",
            "2",
            "-s",
            "7",
            "-f",
            "-q",
        ]
    )

    assert args.output_dir == str(tmp_path / "out")
    assert args.workers == 2
    assert args.seed == 7
    assert args.overwrite is True
    assert args.quiet is True


@pytest.mark.parametrize(
    "flag",
    [
        "--out-dir",
        "--enable-trans",
        "--include-residuals",
        "--tree-n-pcs",
        "--oe-pseudocount",
    ],
)
def test_cli_old_flags_rejected(flag) -> None:
    argv = [
        "--mcool",
        "test.mcool",
        "--tad-table",
        "domains.tsv",
        "--output-dir",
        "out",
        flag,
    ]
    if flag not in ("--enable-trans", "--include-residuals"):
        argv.append("1")

    with pytest.raises(SystemExit) as info:
        parse_args(argv)
    assert info.value.code == 2


def test_cli_cis_run(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    args = cli_args(mcool, tads, tmp_path)

    main(args)

    out = tmp_path / "out"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["cli"]["argv"] == args
    assert manifest["params"]["resolved"]["working_binsize"] == 500
    assert manifest["qc"]["summary"]["n_units"] == 4
    assert (out / "cis_leaves.parquet").exists()
    assert (out / "cis_leaves.tsv").exists()
    assert not (out / "cis_leaves_ab.tsv").exists()
    assert not (out / "trans_nodes.parquet").exists()


def test_cli_log_format(capsys, mcool, tmp_path) -> None:
    args = cli_args(mcool, base_tads(tmp_path), tmp_path)

    main(args)

    stderr = capsys.readouterr().err
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - INFO - "
    assert re.search(pattern + r"\[CIS\] running", stderr, re.MULTILINE)
    assert re.search(pattern + r"\[WRITE\] wrote", stderr, re.MULTILINE)


def test_cli_quiet_suppresses_info(capsys, mcool, tmp_path) -> None:
    args = cli_args(mcool, base_tads(tmp_path), tmp_path, "-q")

    main(args)

    stderr = capsys.readouterr().err
    assert "INFO" not in stderr
    assert "[CIS]" not in stderr
    assert "[WRITE]" not in stderr


def test_cli_residuals_opt_in(gapped_mcool, gapped_tads, tmp_path) -> None:
    args = cli_args(gapped_mcool, gapped_tads, tmp_path)

    main(args)

    out = tmp_path / "out"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["params"]["requested"]["with_residuals"] is False
    assert manifest["qc"]["summary"]["n_residuals"] == 0
    leaves = pl.read_parquet(out / "cis_leaves.parquet")
    assert set(leaves["unit_kind"].to_list()) == {"tad"}

    main([*args, "--with-residuals", "--overwrite"])

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["params"]["requested"]["with_residuals"] is True
    assert manifest["qc"]["summary"]["n_residuals"] > 0
    leaves = pl.read_parquet(out / "cis_leaves.parquet")
    assert "residual" in set(leaves["unit_kind"].to_list())


def test_cli_uniform_bins_run(mcool, tmp_path) -> None:
    args = uniform_cli_args(mcool, tmp_path)

    main(args)

    out = tmp_path / "out"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["params"]["resolved"]["input_mode"] == "uniform_bins"
    assert manifest["inputs"]["annotation"] is None
    assert manifest["qc"]["summary"]["n_bin_units"] == 8
    assert manifest["qc"]["summary"]["n_tads"] == 0

    leaves = pl.read_parquet(out / "cis_leaves.parquet")
    assert set(leaves["unit_kind"].to_list()) == {"bin"}
    assert leaves["tad_table_row"].null_count() == leaves.height


def test_cli_uniform_bins_requires_working_binsize(mcool, tmp_path) -> None:
    args = [
        "--mcool",
        str(mcool),
        "--uniform-bins",
        "--output-dir",
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


def help_option_order(output: str) -> list[str]:
    main = output.split("\nadvanced:")[0]
    order = []
    for line in main.splitlines():
        if line.startswith("  -") and not line.startswith("   "):
            order.append(line.split()[0].rstrip(","))
    return order


def test_help_lists_advanced(capsys) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--help"])

    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "advanced" in out
    assert "--oe-distance-bins" in out
    assert "--oe-pseudocount-scale" in out
    assert "--tree-pca-dim" in out
    assert "--tree-branch-k" in out
    assert "--uniform-bins" in out
    assert "--with-residuals" in out
    assert "--with-trans" in out
    assert "--quiet" in out
    assert "--output-dir" in out


def test_help_option_order(capsys) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--help"])

    assert info.value.code == 0
    out = capsys.readouterr().out
    assert help_option_order(out) == [
        "--mcool",
        "--tad-table",
        "--uniform-bins",
        "--ab-feature-track",
        "-o",
        "--working-binsize",
        "--with-trans",
        "--with-residuals",
        "-p",
        "-s",
        "-f",
        "-q",
        "-V",
        "-h",
    ]


def test_cli_trans_run(three_chrom_mcool, trans_tads, tmp_path) -> None:
    args = cli_args(
        three_chrom_mcool,
        trans_tads,
        tmp_path,
        "--with-trans",
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
        "--ab-feature-track",
        str(track),
    )

    main(args)

    out = tmp_path / "out"
    assert (out / "cis_leaves.tsv").exists()
    assert (out / "cis_leaves_ab.parquet").exists()
    assert (out / "cis_leaves_ab.tsv").exists()
    assert (out / "cis_nodes_ab.parquet").exists()
    leaves = pl.read_parquet(out / "cis_leaves.parquet")
    assert "ab_label" in leaves.columns
    manifest = json.loads((out / "manifest.json").read_text())
    assert "ab_split_policy" not in manifest["params"]["requested"]


def test_cli_rejects_ab_split_policy(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    args = cli_args(mcool, tads, tmp_path, "--ab-split-policy", "kmeans3")

    with pytest.raises(SystemExit) as info:
        main(args)
    assert info.value.code == 2


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
    args = cli_args(mcool, tads, tmp_path, "--oe-pseudocount-scale", "0")

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
        "--with-trans",
        "--trans-null",
        "grand",
    )

    main(args)
    assert (tmp_path / "out" / "cis_leaves.parquet").exists()
    assert (tmp_path / "out" / "trans_nodes.parquet").exists()


def test_cli_interrupt_stops_parallel_workers(mcool, tmp_path) -> None:
    tads = base_tads(tmp_path)
    out = tmp_path / "out"
    helper = Path(__file__).with_name("interrupt_cli_helper.py")
    proc = subprocess.Popen(
        [sys.executable, str(helper), str(mcool), str(tads), str(out)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    stdout = proc.stdout
    assert stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        for line in stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    combined: list[str] = []
    try:
        deadline = time.monotonic() + 20.0
        busy_workers = 0
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if line is None:
                break
            combined.append(line)
            if "[TEST] worker busy" in line:
                busy_workers += 1
                if busy_workers == 2:
                    break

        assert busy_workers == 2, "".join(combined)
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=20)
        reader.join(timeout=5)
        while True:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                break
            if line is not None:
                combined.append(line)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    output = "".join(combined)
    assert proc.returncode == -signal.SIGINT
    assert "Traceback" not in output
    assert "KeyboardInterrupt" not in output
    assert not out.exists()


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_cli_version(flag, capsys) -> None:
    with pytest.raises(SystemExit) as info:
        main([flag])

    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out
