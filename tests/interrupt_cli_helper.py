"""Test helper: run the CLI with a slow parallel cis workload."""

from __future__ import annotations

import sys
import time
from typing import Any, NoReturn

from tadclust import cli, parallel
from tadclust.config import ClusterConfig

_original_select = parallel._select_chromosome


def _slow_select(*args: Any, **kwargs: Any) -> parallel.ChromosomeSelection:
    print("[TEST] worker busy", flush=True)
    time.sleep(60)
    return _original_select(*args, **kwargs)


parallel._select_chromosome = _slow_select


def slow_cluster(config: ClusterConfig) -> NoReturn:
    parallel.run_cis_parallel(
        config,
        {"chr1": 2000, "chr2": 2000},
        ["chr1", "chr2"],
        500,
    )
    raise AssertionError("SIGINT did not arrive")


def main() -> None:
    mcool, tads, out = sys.argv[1:4]
    cli.cluster = slow_cluster
    cli.main(
        [
            "--mcool",
            mcool,
            "--tad-table",
            tads,
            "--output-dir",
            out,
            "--working-binsize",
            "500",
            "--workers",
            "2",
        ]
    )


if __name__ == "__main__":
    main()
