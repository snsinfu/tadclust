"""Test helper: run the CLI with a slow parallel cis workload."""

from __future__ import annotations

import sys
import time
from typing import ClassVar, NoReturn, cast

import cooler
import numpy as np

from tadclust import cli
from tadclust.config import ClusterConfig
from tadclust.domains import ChromosomeTree
from tadclust.parallel import run_cis_parallel
from tadclust.sources import CisSource


class SlowSource:
    """Cis source that blocks long enough for the test to send SIGINT."""

    unit_kind: ClassVar[str] = "tad"

    def build_units(
        self,
        cooler_object: cooler.Cooler,
        chrom: str,
        chromsize: int,
    ) -> tuple[ChromosomeTree, np.ndarray]:
        print(f"[TEST] worker busy {chrom}", flush=True)
        time.sleep(60)
        raise AssertionError("SIGINT did not arrive")


def slow_cluster(config: ClusterConfig) -> NoReturn:
    run_cis_parallel(
        config,
        cast(CisSource, SlowSource()),
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
