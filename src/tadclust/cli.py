"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from . import __version__
from .config import (
    AB_SPLIT_CHOICES,
    TRANS_NULL_CHOICES,
    ClusterConfig,
)
from .errors import TadclustError
from .pipeline import cluster
from .writer import write_outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="tadclust",
        description="Hierarchical clustering of TADs",
    )
    parser.add_argument("--mcool", required=True, help="multi-resolution mcool")
    parser.add_argument(
        "--tad-table",
        required=True,
        help="tab-separated chrom/start/end TAD table",
    )
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument(
        "--working-binsize",
        type=int,
        default=None,
        help="working bin size in bp; omit for adaptive selection",
    )
    parser.add_argument(
        "--enable-trans",
        action="store_true",
        help="enable trans clustering",
    )
    parser.add_argument(
        "--ab-signal-track",
        default=None,
        help="bedGraph or BED signal track; enables A/B",
    )
    parser.add_argument(
        "--ab-split",
        choices=AB_SPLIT_CHOICES,
        default=None,
        help="A/B split policy; default kmeans2 with a signal track",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker processes for cis chromosomes",
    )
    parser.add_argument("--seed", type=int, default=0, help="k-means seed")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing known outputs",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    advanced = parser.add_argument_group("advanced")
    advanced.add_argument(
        "--oe-distance-bins",
        type=int,
        default=60,
        help="number of log-spaced distance-decay bins",
    )
    advanced.add_argument(
        "--oe-clip-min",
        type=float,
        default=-10.0,
        help="lower log2 O/E clip",
    )
    advanced.add_argument(
        "--oe-clip-max",
        type=float,
        default=10.0,
        help="upper log2 O/E clip",
    )
    advanced.add_argument(
        "--oe-pseudocount",
        type=float,
        default=1.0,
        help="pseudocount multiplier",
    )
    advanced.add_argument(
        "--dp-lambda",
        type=float,
        default=None,
        help="DP penalty; omit for the automatic value",
    )
    advanced.add_argument(
        "--tree-n-pcs",
        type=int,
        default=5,
        help="number of PCs used by tree splits",
    )
    advanced.add_argument(
        "--tree-branch-k",
        type=int,
        default=3,
        help="clusters per tree split",
    )
    advanced.add_argument(
        "--trans-null",
        choices=TRANS_NULL_CHOICES,
        default="pair",
        help="trans null model",
    )
    advanced.add_argument(
        "--max-dense-domains",
        type=int,
        default=6_000,
        help="dense trans-matrix domain limit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI; raise SystemExit on failure."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logger = logging.getLogger("tadclust")

    if args.ab_signal_track is None and args.ab_split is not None:
        raise SystemExit("tadclust: error: --ab-split requires --ab-signal-track")
    ab_split = args.ab_split or "kmeans2" if args.ab_signal_track is not None else None

    try:
        config = ClusterConfig(
            mcool_path=args.mcool,
            annotation_path=args.tad_table,
            output_dir=args.out_dir,
            resolution=args.working_binsize,
            oe_distance_bins=args.oe_distance_bins,
            oe_clip_min=args.oe_clip_min,
            oe_clip_max=args.oe_clip_max,
            oe_pseudocount=args.oe_pseudocount,
            dp_lambda=args.dp_lambda,
            tree_n_pcs=args.tree_n_pcs,
            tree_branch_k=args.tree_branch_k,
            trans_enabled=args.enable_trans,
            trans_null=args.trans_null,
            ab_signal_track=args.ab_signal_track,
            ab_split=ab_split,
            workers=args.workers,
            max_dense_domains=args.max_dense_domains,
            seed=args.seed,
        )
        result = cluster(config)
        written = write_outputs(
            result,
            config,
            cli_args=raw_args,
            overwrite=args.overwrite,
        )
    except TadclustError as exc:
        raise SystemExit(f"tadclust: error: {exc}") from exc

    logger.info(
        "wrote %d domains and %d file(s) to %s",
        result.qc["n_domains"],
        len(written),
        config.output_dir,
    )
