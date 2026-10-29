"""Tests for mcool access and resolution selection."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from tadclust.errors import McoolError
from tadclust.mcool import (
    ResolutionChoice,
    bin_centers,
    bin_intervals,
    choose_resolution,
    chrom_sizes,
    list_resolutions,
    open_cooler,
)


def test_list_resolutions(mcool) -> None:
    assert list_resolutions(mcool) == [500, 1000, 2000]


def test_missing_mcool_is_rejected(tmp_path) -> None:
    with pytest.raises(McoolError):
        list_resolutions(tmp_path / "missing.mcool")


def test_missing_resolutions_group_is_rejected(tmp_path) -> None:
    path = tmp_path / "plain.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("something")

    with pytest.raises(McoolError):
        list_resolutions(path)


def test_open_cooler_and_metadata(mcool) -> None:
    clr = open_cooler(mcool, 500)

    assert chrom_sizes(clr) == {"chr1": 2000, "chr2": 2000}
    centers = bin_centers(clr, "chr1")
    assert centers.tolist() == [250.0, 750.0, 1250.0, 1750.0]


def test_bin_intervals(mcool) -> None:
    clr = open_cooler(mcool, 500)

    starts, ends = bin_intervals(clr, "chr1")
    assert starts.tolist() == [0, 500, 1000, 1500]
    assert ends.tolist() == [500, 1000, 1500, 2000]


def test_unbalanced_cooler_is_rejected(unbalanced_mcool) -> None:
    with pytest.raises(McoolError) as info:
        open_cooler(unbalanced_mcool, 500)

    assert info.value.resolution == 500


def test_explicit_resolution_must_exist() -> None:
    choice = choose_resolution([500, 1000], [3000], resolution=1000)

    assert choice == ResolutionChoice(value=1000, mode="explicit")

    with pytest.raises(McoolError):
        choose_resolution([500, 1000], [3000], resolution=750)


def test_explicit_resolution_does_not_need_lengths() -> None:
    choice = choose_resolution([500, 1000], resolution=1000)

    assert choice == ResolutionChoice(value=1000, mode="explicit")


def test_adaptive_resolution_requires_lengths() -> None:
    with pytest.raises(McoolError):
        choose_resolution([500, 1000])


def test_adaptive_rule_uses_p10_over_min_bins() -> None:
    choice = choose_resolution([500, 1000, 2000, 5000], [3000])

    assert choice == ResolutionChoice(value=1000, mode="adaptive")


def test_adaptive_rule_uses_percentile() -> None:
    lengths = [1000] * 9 + [9000]

    choice = choose_resolution([500, 1000, 2000], lengths, percentile=10)

    assert choice.value == 500


def test_adaptive_rule_falls_back_to_smallest() -> None:
    choice = choose_resolution([500, 1000], [300])

    assert choice.value == 500


def test_adaptive_rule_uses_largest_when_target_is_large() -> None:
    choice = choose_resolution([500, 1000], [30000])

    assert choice.value == 1000


def test_empty_inputs_are_rejected() -> None:
    with pytest.raises(McoolError):
        choose_resolution([], [1000])

    with pytest.raises(McoolError):
        choose_resolution([500], np.array([], dtype=np.int64))
