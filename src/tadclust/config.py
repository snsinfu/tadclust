"""Programmatic configuration for a tadclust run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

TRANS_NULL_CHOICES = ("pair", "grand")
CIS_CONTACT_REPRESENTATION_CHOICES = ("log_oe", "dc_log_freq")
UNIT_SELECTION_CHOICES = ("dp", "roots")


@dataclass(frozen=True)
class ClusterConfig:
    """All inputs and technical parameters for one clustering run.

    Paths are stored exactly as supplied. The mcool path must exist. When the
    TAD table path is supplied it must exist. When it is omitted, uniform
    working-resolution mcool bins are used and working_binsize is required.
    `tad_table_metadata_keys` copies named parquet file-level metadata keys into
    the manifest. The output directory is created by the writer.
    """

    mcool_path: str | Path
    tad_table_path: str | Path | None
    output_dir: str | Path
    tad_table_metadata_keys: tuple[str, ...] = ()
    chromosomes: list[str] | None = None
    working_binsize: int | None = None
    tad_size_percentile: float = 10.0
    min_bins_per_tad: int = 3
    distance_bins: int = 60
    clip_min: float = -10.0
    clip_max: float = 10.0
    pseudocount_scale: float = 1.0
    cis_contact_representation: str = "log_oe"
    dp_lambda_scale: float = 1.0
    unit_selection: str = "dp"
    with_residuals: bool = False
    tree_pca_dim: int = 5
    tree_branch_k: int = 2
    kmeans_n_init: int = 100
    with_trans: bool = False
    trans_null: str = "pair"
    ab_feature_track: str | Path | None = None
    workers: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if not Path(self.mcool_path).is_file():
            raise ConfigError(
                f"mcool not found: {self.mcool_path}",
                notes=("check the mcool_path supplied to ClusterConfig",),
                field="mcool_path",
            )
        if self.tad_table_path is not None and not Path(self.tad_table_path).is_file():
            raise ConfigError(
                f"TAD table not found: {self.tad_table_path}",
                notes=("check the tad_table_path supplied to ClusterConfig",),
                field="tad_table_path",
            )

        if self.tad_table_path is None and self.working_binsize is None:
            raise ConfigError(
                "working_binsize is required when tad_table_path is omitted",
                field="working_binsize",
            )

        if isinstance(self.tad_table_metadata_keys, str):
            raise ConfigError(
                "tad_table_metadata_keys must be a sequence of keys, not a string",
                field="tad_table_metadata_keys",
            )
        if any(not key for key in self.tad_table_metadata_keys):
            raise ConfigError(
                "tad_table_metadata_keys must contain non-empty strings",
                field="tad_table_metadata_keys",
            )
        if self.tad_table_metadata_keys and self.tad_table_path is None:
            raise ConfigError(
                "tad_table_metadata_keys requires a TAD table",
                notes=("metadata copying applies to parquet TAD-table input",),
                field="tad_table_metadata_keys",
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
        if self.distance_bins < 2:
            raise ConfigError(
                "distance_bins must be at least 2",
                field="distance_bins",
            )
        if not self.clip_min < self.clip_max:
            raise ConfigError(
                "clip_min must be less than clip_max",
                field="clip_min",
            )
        if self.pseudocount_scale <= 0:
            raise ConfigError(
                "pseudocount_scale must be positive",
                field="pseudocount_scale",
            )
        if self.cis_contact_representation not in CIS_CONTACT_REPRESENTATION_CHOICES:
            raise ConfigError(
                f"cis_contact_representation must be one of "
                f"{CIS_CONTACT_REPRESENTATION_CHOICES}, "
                f"got {self.cis_contact_representation!r}",
                field="cis_contact_representation",
            )
        if self.dp_lambda_scale < 0:
            raise ConfigError(
                "dp_lambda_scale must be non-negative",
                field="dp_lambda_scale",
            )
        if self.unit_selection not in UNIT_SELECTION_CHOICES:
            raise ConfigError(
                f"unit_selection must be one of {UNIT_SELECTION_CHOICES}, "
                f"got {self.unit_selection!r}",
                field="unit_selection",
            )
        if self.tad_table_path is None and self.unit_selection == "roots":
            raise ConfigError(
                "unit_selection='roots' requires a TAD table",
                notes=("uniform-bin mode has no TAD hierarchy to take roots from",),
                field="unit_selection",
            )
        if self.tree_pca_dim < 1:
            raise ConfigError(
                "tree_pca_dim must be at least 1",
                field="tree_pca_dim",
            )
        if self.tree_branch_k < 2:
            raise ConfigError(
                "tree_branch_k must be at least 2",
                field="tree_branch_k",
            )
        if self.kmeans_n_init < 1:
            raise ConfigError(
                "kmeans_n_init must be at least 1",
                field="kmeans_n_init",
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

        if (
            self.ab_feature_track is not None
            and not Path(self.ab_feature_track).is_file()
        ):
            raise ConfigError(
                f"feature track not found: {self.ab_feature_track}",
                notes=("check the ab_feature_track supplied to ClusterConfig",),
                field="ab_feature_track",
            )
