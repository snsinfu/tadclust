"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from . import __version__
from .config import (
    CIS_CONTACT_REPRESENTATION_CHOICES,
    TRANS_NULL_CHOICES,
    UNIT_SELECTION_CHOICES,
    ClusterConfig,
)
from .errors import TadclustError
from .pipeline import cluster
from .processes import prepare_process_context
from .signals import exit_by_sigint, install_interrupt_handler
from .writer import write_outputs

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="tadclust",
        description="Hierarchical clustering of TADs",
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("--mcool", required=True, help="multi-resolution mcool")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--tad-table",
        default=None,
        help="tab-separated or parquet chrom/start/end TAD table",
    )
    source.add_argument(
        "--uniform-bins",
        action="store_true",
        help="cluster uniform mcool bins as units; requires --working-binsize",
    )
    parser.add_argument(
        "--ab-feature-track",
        default=None,
        help="bedGraph or BED feature track; enables A/B",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="output directory",
    )
    parser.add_argument(
        "--working-binsize",
        type=int,
        default=None,
        help="working bin size in bp; omit for adaptive selection",
    )
    parser.add_argument(
        "--with-trans",
        action="store_true",
        help="enable trans clustering",
    )
    parser.add_argument(
        "--with-residuals",
        action="store_true",
        help="include uncovered-gap residual units in clustering and output",
    )
    parser.add_argument(
        "-p",
        "--workers",
        type=int,
        default=1,
        help="worker processes for cis chromosomes",
    )
    parser.add_argument("-s", "--seed", type=int, default=0, help="k-means seed")
    parser.add_argument(
        "-f",
        "--overwrite",
        action="store_true",
        help="replace existing known outputs",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress INFO logs",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit",
    )

    advanced = parser.add_argument_group("advanced")
    advanced.add_argument(
        "--distance-bins",
        type=int,
        default=60,
        help="number of log-spaced distance-decay bins",
    )
    advanced.add_argument(
        "--clip-min",
        type=float,
        default=-10.0,
        help=(
            "lower bound for clipping transformed contact values: log O/E, "
            "and double-centered log frequency after centering"
        ),
    )
    advanced.add_argument(
        "--clip-max",
        type=float,
        default=10.0,
        help=(
            "upper bound for clipping transformed contact values: log O/E, "
            "and double-centered log frequency after centering"
        ),
    )
    advanced.add_argument(
        "--pseudocount-scale",
        type=float,
        default=1.0,
        help=(
            "pseudocount scale on the smallest non-zero expected value, "
            "used by both cis contact representations"
        ),
    )
    advanced.add_argument(
        "--cis-contact-representation",
        choices=CIS_CONTACT_REPRESENTATION_CHOICES,
        default="log_oe",
        help=(
            "cis contact representation used for clustering: log_oe (default) uses "
            "distance-corrected log O/E and favors global/compartment-correlated "
            "structure; dc_log_freq uses double-centered log contact frequency and "
            "keeps deep branches local. A/B annotation always uses log_oe; trans "
            "always uses log_oe."
        ),
    )
    advanced.add_argument(
        "--unit-selection",
        choices=UNIT_SELECTION_CHOICES,
        default="dp",
        help=(
            "hierarchical annotation unit selection: dp (default) chooses an "
            "SSE-DP antichain; roots takes the top-level TAD of each hierarchy "
            "and skips the DP"
        ),
    )
    advanced.add_argument(
        "--dp-lambda-scale",
        type=float,
        default=1.0,
        help="DP penalty multiplier on the automatic value",
    )
    advanced.add_argument(
        "--tree-pca-dim",
        type=int,
        default=5,
        help="number of PCs used by tree splits",
    )
    advanced.add_argument(
        "--tree-branch-k",
        type=int,
        default=2,
        help="clusters per tree split",
    )
    advanced.add_argument(
        "--kmeans-n-init",
        type=int,
        default=100,
        help="k-means restarts per split (sklearn n_init; default 100)",
    )
    advanced.add_argument(
        "--trans-null",
        choices=TRANS_NULL_CHOICES,
        default="pair",
        help="trans null model",
    )
    advanced.add_argument(
        "--tad-table-metadata-key",
        action="append",
        default=None,
        metavar="KEY",
        help=(
            "copy this parquet TAD-table file-level metadata key into the "
            "manifest; repeatable"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI; raise SystemExit on error, or die by SIGINT on interrupt."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    install_interrupt_handler()
    try:
        _run(raw_args)
    except KeyboardInterrupt:
        exit_by_sigint()
    except TadclustError as exc:
        raise SystemExit(f"tadclust: error: {exc}") from exc


def _run(raw_args: list[str]) -> None:
    args = parse_args(raw_args)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        force=True,
    )

    config = ClusterConfig(
        mcool_path=args.mcool,
        tad_table_path=args.tad_table,
        tad_table_metadata_keys=tuple(args.tad_table_metadata_key or ()),
        output_dir=args.output_dir,
        working_binsize=args.working_binsize,
        distance_bins=args.distance_bins,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        pseudocount_scale=args.pseudocount_scale,
        cis_contact_representation=args.cis_contact_representation,
        dp_lambda_scale=args.dp_lambda_scale,
        unit_selection=args.unit_selection,
        with_residuals=args.with_residuals,
        tree_pca_dim=args.tree_pca_dim,
        tree_branch_k=args.tree_branch_k,
        kmeans_n_init=args.kmeans_n_init,
        with_trans=args.with_trans,
        trans_null=args.trans_null,
        ab_feature_track=args.ab_feature_track,
        workers=args.workers,
        seed=args.seed,
    )
    if config.workers > 1:
        prepare_process_context()
    result = cluster(config)
    written = write_outputs(
        result,
        config,
        cli_args=raw_args,
        overwrite=args.overwrite,
    )
    LOG.info(
        "[WRITE] wrote %d units and %d file(s) to %s",
        result.qc["summary"]["n_units"],
        len(written),
        config.output_dir,
    )
