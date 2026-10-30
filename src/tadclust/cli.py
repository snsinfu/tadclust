"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from . import __version__
from .config import (
    AB_SPLIT_POLICY_CHOICES,
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--tad-table",
        default=None,
        help="tab-separated chrom/start/end TAD table",
    )
    source.add_argument(
        "--uniform-bins",
        action="store_true",
        help="cluster uniform mcool bins as units; requires --working-binsize",
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
        "--ab-feature-track",
        default=None,
        help="bedGraph or BED feature track; enables A/B",
    )
    parser.add_argument(
        "--ab-split-policy",
        choices=AB_SPLIT_POLICY_CHOICES,
        default=None,
        help="A/B split policy; default kmeans2 with a feature track",
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
        "--dp-lambda-scale",
        type=float,
        default=1.0,
        help="DP penalty multiplier on the automatic value",
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

    try:
        config = ClusterConfig(
            mcool_path=args.mcool,
            annotation_path=args.tad_table,
            output_dir=args.out_dir,
            working_binsize=args.working_binsize,
            oe_distance_bins=args.oe_distance_bins,
            oe_clip_min=args.oe_clip_min,
            oe_clip_max=args.oe_clip_max,
            oe_pseudocount=args.oe_pseudocount,
            dp_lambda_scale=args.dp_lambda_scale,
            tree_n_pcs=args.tree_n_pcs,
            tree_branch_k=args.tree_branch_k,
            trans_enabled=args.enable_trans,
            trans_null=args.trans_null,
            ab_feature_track=args.ab_feature_track,
            ab_split_policy=args.ab_split_policy,
            workers=args.workers,
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
        "wrote %d units and %d file(s) to %s",
        result.qc["n_units"],
        len(written),
        config.output_dir,
    )
