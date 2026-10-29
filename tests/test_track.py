"""Tests for signal-track parsing and A/B feature construction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from tadclust.errors import TrackError
from tadclust.track import (
    SignalTrack,
    ab_feature_matrix,
    fill_and_rank,
    overlap_weighted_values,
    read_signal_track,
)


def write_track(tmp_path, text, name="track.bedGraph"):
    path = tmp_path / name
    path.write_text(text)
    return path


def track_values(track, chrom):
    values = track.for_chrom(chrom)
    assert values is not None
    return values


def test_read_bedgraph(tmp_path) -> None:
    path = write_track(
        tmp_path,
        "chr1\t0\t50\t1.0\nchr1\t50\t100\t3.0\nchr2\t0\t100\t2.0\n",
    )

    track = read_signal_track(path)

    starts, ends, values = track_values(track, "chr1")
    assert starts.tolist() == [0, 50]
    assert ends.tolist() == [50, 100]
    assert values.tolist() == [1.0, 3.0]
    assert track_values(track, "chr2")[2].tolist() == [2.0]
    assert track.for_chrom("chr3") is None


def test_read_five_column_bed_uses_column_five(tmp_path) -> None:
    path = write_track(
        tmp_path,
        "chr1\t0\t100\tpeak\t2.5\n",
        name="track.bed",
    )

    track = read_signal_track(path)

    assert track_values(track, "chr1")[2].tolist() == [2.5]


def test_comments_and_track_headers_are_ignored(tmp_path) -> None:
    path = write_track(
        tmp_path,
        "# comment\n"
        "\n"
        "track type=bedGraph name=test\n"
        "browser position chr1\n"
        "chr1\t0\t100\t1.0\n",
    )

    track = read_signal_track(path)

    assert track_values(track, "chr1")[2].tolist() == [1.0]


def test_comments_between_rows_are_skipped(tmp_path) -> None:
    path = write_track(
        tmp_path,
        "chr1\t0\t100\t1.0\n# between rows\nchr1\t100\t200\t2.0\n",
    )

    track = read_signal_track(path)

    assert track_values(track, "chr1")[2].tolist() == [1.0, 2.0]


def test_missing_track_is_rejected(tmp_path) -> None:
    with pytest.raises(TrackError):
        read_signal_track(tmp_path / "missing.bedGraph")


def test_empty_track_is_rejected(tmp_path) -> None:
    path = write_track(tmp_path, "# only comments\n")

    with pytest.raises(TrackError):
        read_signal_track(path)


def test_too_few_columns_is_rejected(tmp_path) -> None:
    path = write_track(tmp_path, "chr1\t0\t100\n")

    with pytest.raises(TrackError):
        read_signal_track(path)


@pytest.mark.parametrize(
    "text",
    [
        "chr1\t0\t100\tnan\n",
        "chr1\t0\t100\tinf\n",
        "chr1\t100\t100\t1.0\n",
        "chr1\t-1\t100\t1.0\n",
        "\t0\t100\t1.0\n",
        "chr1\t\t100\t1.0\n",
        "chr1\t0\t\t1.0\n",
    ],
)
def test_invalid_track_rows_are_rejected(tmp_path, text) -> None:
    path = write_track(tmp_path, text)

    with pytest.raises(TrackError):
        read_signal_track(path)


def test_overlap_weighted_values() -> None:
    track = SignalTrack(
        table=pl.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [0, 50],
                "end": [50, 100],
                "value": [1.0, 3.0],
            }
        )
    )
    domains = [
        SimpleNamespace(start=25, end=75),
        SimpleNamespace(start=200, end=300),
    ]

    values = overlap_weighted_values(track, domains, "chr1")

    assert values[0] == 2.0
    assert np.isnan(values[1])
    missing = overlap_weighted_values(track, domains, "chr2")
    assert np.all(np.isnan(missing))


def test_overlap_weighted_values_rejects_missing_coordinates() -> None:
    track = SignalTrack(
        table=pl.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [None],
                "end": [100],
                "value": [1.0],
            },
            schema={
                "chrom": pl.String,
                "start": pl.Int64,
                "end": pl.Int64,
                "value": pl.Float64,
            },
        )
    )

    with pytest.raises(TrackError):
        overlap_weighted_values(
            track,
            [SimpleNamespace(start=0, end=100)],
            "chr1",
        )


def test_fill_and_rank() -> None:
    ranks, n_uncovered = fill_and_rank(
        np.array([1.0, np.nan, 3.0]),
        "chr1",
    )

    assert n_uncovered == 1
    assert np.allclose(ranks, [1.0 / 3.0, 2.0 / 3.0, 1.0])


def test_fill_and_rank_rejects_no_coverage() -> None:
    with pytest.raises(TrackError):
        fill_and_rank(np.array([np.nan, np.nan]), "chr1")


def test_fill_and_rank_rejects_constant_track() -> None:
    with pytest.raises(TrackError):
        fill_and_rank(np.array([2.0, 2.0, np.nan]), "chr1")


def test_ab_feature_matrix_standardizes_columns() -> None:
    pcs = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    rank = np.array([0.0, 0.5, 1.0])

    features = ab_feature_matrix(pcs, rank, 2)

    assert features.shape == (3, 3)
    assert np.allclose(features.mean(axis=0), 0.0)
    assert np.isclose(features[:, 0].std(), 1.0)
    assert np.isclose(features[:, 2].std(), 1.0)
    assert np.all(features[:, 1] == 0.0)
