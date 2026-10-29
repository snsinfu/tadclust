"""Programmatic configuration for a tadclust run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

AB_SPLIT_CHOICES = ("kmeans2", "kmeans3")
TRANS_NULL_CHOICES = ("pair", "grand")


@dataclass(frozen=True)
class ClusterConfig:
    """All inputs and technical parameters for one clustering run.

    Paths are stored exactly as supplied. The mcool and annotation paths must
    exist; the output directory is created by the writer.
    """

    mcool_path: str | Path
    annotation_path: str | Path
    output_dir: str | Path
    chromosomes: list[str] | None = None
    working_binsize: int | None = None
    tad_size_percentile: float = 10.0
    min_bins_per_tad: int = 3
    oe_distance_bins: int = 60
    oe_clip_min: float = -10.0
    oe_clip_max: float = 10.0
    oe_pseudocount: float = 1.0
    dp_lambda_scale: float = 1.0
    tree_n_pcs: int = 5
    tree_branch_k: int = 3
    trans_enabled: bool = False
    trans_null: str = "pair"
    ab_signal_track: str | Path | None = None
    ab_split: str | None = None
    workers: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if not Path(self.mcool_path).is_file():
            raise ConfigError(
                f"mcool not found: {self.mcool_path}",
                notes=("check the mcool_path supplied to ClusterConfig",),
                field="mcool_path",
            )
        if not Path(self.annotation_path).is_file():
            raise ConfigError(
                f"annotation not found: {self.annotation_path}",
                notes=("check the annotation_path supplied to ClusterConfig",),
                field="annotation_path",
            )

        if self.chromosomes is not None:
            if not self.chromosomes:
                raise ConfigError(
                    "chromosomes must be non-empty when given",
                    field="chromosomes",
                )
            if any(not str(chrom) for chrom in self.chromosomes):
                raise ConfigError(
                    "chromosome names must be non-empty",
                    field="chromosomes",
                )

        if self.working_binsize is not None and self.working_binsize <= 0:
            raise ConfigError(
                "working_binsize must be positive",
                field="working_binsize",
            )
        if not 0 < self.tad_size_percentile <= 100:
            raise ConfigError(
                "tad_size_percentile must be in (0, 100]",
                field="tad_size_percentile",
            )
        if self.min_bins_per_tad < 1:
            raise ConfigError(
                "min_bins_per_tad must be at least 1",
                field="min_bins_per_tad",
            )
        if self.oe_distance_bins < 2:
            raise ConfigError(
                "oe_distance_bins must be at least 2",
                field="oe_distance_bins",
            )
        if not self.oe_clip_min < self.oe_clip_max:
            raise ConfigError(
                "oe_clip_min must be less than oe_clip_max",
                field="oe_clip_min",
            )
        if self.oe_pseudocount <= 0:
            raise ConfigError(
                "oe_pseudocount must be positive",
                field="oe_pseudocount",
            )
        if self.dp_lambda_scale < 0:
            raise ConfigError(
                "dp_lambda_scale must be non-negative",
                field="dp_lambda_scale",
            )
        if self.tree_n_pcs < 1:
            raise ConfigError(
                "tree_n_pcs must be at least 1",
                field="tree_n_pcs",
            )
        if self.tree_branch_k < 2:
            raise ConfigError(
                "tree_branch_k must be at least 2",
                field="tree_branch_k",
            )
        if self.trans_null not in TRANS_NULL_CHOICES:
            raise ConfigError(
                f"trans_null must be one of {TRANS_NULL_CHOICES}, got {self.trans_null!r}",
                field="trans_null",
            )
        if self.workers < 1:
            raise ConfigError(
                "workers must be at least 1",
                field="workers",
            )
        if self.seed < 0:
            raise ConfigError(
                "seed must be non-negative",
                field="seed",
            )

        if self.ab_split is not None and self.ab_split not in AB_SPLIT_CHOICES:
            raise ConfigError(
                f"ab_split must be one of {AB_SPLIT_CHOICES}, got {self.ab_split!r}",
                field="ab_split",
            )
        if self.ab_signal_track is None and self.ab_split is not None:
            raise ConfigError(
                "ab_split requires ab_signal_track",
                notes=("supply a signal track or omit ab_split",),
                field="ab_split",
            )
        if (
            self.ab_signal_track is not None
            and not Path(self.ab_signal_track).is_file()
        ):
            raise ConfigError(
                f"signal track not found: {self.ab_signal_track}",
                notes=("check the ab_signal_track supplied to ClusterConfig",),
                field="ab_signal_track",
            )
