"""Feature-track parsing, overlap weighting, ranks, and A/B features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from .bedlike import read_bedlike
from .errors import ClusteringError, TrackError


@dataclass(frozen=True)
class FeatureTrack:
    """A chrom/start/end/value feature table."""

    table: pl.DataFrame

    def for_chrom(
        self,
        chrom: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return starts, ends, and values for one chromosome."""
        subset = self.table.filter(pl.col("chrom") == chrom)
        if subset.height == 0:
            return None
        return (
            subset["start"].to_numpy(),
            subset["end"].to_numpy(),
            subset["value"].to_numpy(),
        )


def read_feature_track(path: str | Path) -> FeatureTrack:
    """Read a tab-separated bedGraph or BED score track.

    Four-column input uses column 4 as the value. Five-or-more-column input
    uses column 5. Blank lines, lines starting with #, and track/browser
    headers are ignored.
    """
    source = Path(path)
    if not source.is_file():
        raise TrackError(
            f"feature track not found: {source}",
            path=str(source),
        )

    try:
        bedlike = read_bedlike(
            source,
            header_names=None,
            schema_overrides=_make_schema,
            skip_track_headers=True,
        )
    except Exception as exc:
        raise TrackError(
            f"cannot read feature track: {source}",
            path=str(source),
        ) from exc

    if bedlike.width < 4:
        raise TrackError(
            f"feature track has {bedlike.width} column(s); at least 4 are required",
            path=str(source),
        )

    # 5-column BED uses column 5; 4-column bedGraph uses column 4.
    value_column = 5 if bedlike.width >= 5 else 4
    table = bedlike.table.select(
        pl.col("column_1").alias("chrom"),
        pl.col("column_2").alias("start"),
        pl.col("column_3").alias("end"),
        pl.col(f"column_{value_column}").alias("value"),
    )

    bad_chrom = table.filter(
        pl.col("chrom").is_null() | (pl.col("chrom").str.strip_chars() == "")
    )
    if bad_chrom.height:
        raise TrackError(
            "feature track has an empty chromosome name",
            path=str(source),
        )

    missing = table.filter(pl.col("start").is_null() | pl.col("end").is_null())
    if missing.height:
        raise TrackError(
            "feature track has a missing start or end",
            path=str(source),
        )

    bad_interval = table.filter(
        (pl.col("start") < 0) | (pl.col("end") <= pl.col("start"))
    )
    if bad_interval.height:
        raise TrackError(
            "feature track has an interval with end <= start or start < 0",
            path=str(source),
        )

    bad_value = table.filter(pl.col("value").is_null() | ~pl.col("value").is_finite())
    if bad_value.height:
        raise TrackError(
            "feature track has a non-finite value",
            path=str(source),
        )

    return FeatureTrack(table=table.sort(["chrom", "start"]))


def _make_schema(has_header: bool, width: int) -> dict[str, type]:
    schema: dict[str, type] = {"column_1": pl.String}
    if width >= 2:
        schema["column_2"] = pl.Int64
    if width >= 3:
        schema["column_3"] = pl.Int64
    if width >= 4:
        value_column = 5 if width >= 5 else 4
        schema[f"column_{value_column}"] = pl.Float64
    return schema


def overlap_weighted_values(
    track: FeatureTrack,
    domains,
    chrom: str,
) -> np.ndarray:
    """Return overlap-length-weighted track values for selected domains."""
    values = np.full(len(domains), np.nan, dtype=np.float64)

    match track.for_chrom(chrom):
        case None:
            return values
        case starts, ends, track_values:
            pass

    if not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise TrackError(
            f"{chrom}: feature track has a missing start or end",
            chrom=chrom,
        )

    for index, domain in enumerate(domains):
        overlap = np.minimum(ends, domain.end) - np.maximum(
            starts,
            domain.start,
        )
        overlap = np.where(overlap > 0, overlap, 0.0)
        coverage = overlap.sum()
        if coverage > 0:
            values[index] = (overlap * track_values).sum() / coverage

    return values


def fill_and_rank(
    values: np.ndarray,
    chrom: str,
) -> tuple[np.ndarray, int]:
    """Fill uncovered domains with the chromosome median, then rank."""
    values = np.asarray(values, dtype=np.float64)
    covered = np.isfinite(values)
    n_uncovered = int((~covered).sum())
    if not covered.any():
        raise TrackError(
            f"{chrom}: no feature track intervals overlap selected domains",
            chrom=chrom,
        )

    filled = np.where(covered, values, np.nanmedian(values))
    if np.allclose(filled, filled[0]):
        raise TrackError(
            f"{chrom}: feature track is constant over selected domains",
            chrom=chrom,
        )

    ranks = rankdata(filled, method="average") / len(filled)
    return ranks, n_uncovered


def ab_feature_matrix(
    pcs: np.ndarray,
    feature_rank: np.ndarray,
    n_pcs: int,
) -> np.ndarray:
    """Build standardized [PC1..PCn, feature rank] features."""
    values = np.asarray(pcs, dtype=np.float64)
    ranks = np.asarray(feature_rank, dtype=np.float64)
    if values.ndim != 2:
        raise ClusteringError("domain PCs must be a two-dimensional array")
    if values.shape[0] != ranks.shape[0]:
        raise ClusteringError("PC and feature-rank lengths differ")

    n_features = max(1, min(n_pcs, values.shape[1])) if values.shape[1] else 0
    columns = [values[:, :n_features]] if n_features else []
    columns.append(ranks[:, None])
    features = np.column_stack(columns)

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    standardized = np.zeros_like(features)
    positive = std > 0
    standardized[:, positive] = (features[:, positive] - mean[positive]) / std[positive]
    return standardized
