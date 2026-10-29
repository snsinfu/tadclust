"""mcool access and working-resolution selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cooler
import h5py
import numpy as np

from .errors import McoolError


@dataclass(frozen=True)
class ResolutionChoice:
    """The selected working resolution and how it was chosen."""

    value: int
    mode: str


def list_resolutions(path: str | Path) -> list[int]:
    """Return the sorted resolutions present in an mcool file."""
    source = Path(path)
    if not source.is_file():
        raise McoolError(
            f"mcool not found: {source}",
            path=str(source),
        )

    try:
        with h5py.File(source, "r") as handle:
            group = handle.get("resolutions")
            if not isinstance(group, h5py.Group):
                raise McoolError(
                    f"{source}: no /resolutions group",
                    path=str(source),
                )
            names = list(group.keys())
    except McoolError:
        raise
    except OSError as exc:
        raise McoolError(
            f"cannot open mcool: {source}",
            path=str(source),
        ) from exc

    try:
        resolutions = sorted(int(name) for name in names)
    except ValueError as exc:
        raise McoolError(
            f"{source}: resolution group names must be integers",
            path=str(source),
        ) from exc

    if not resolutions:
        raise McoolError(
            f"{source}: no resolutions found",
            path=str(source),
        )
    return resolutions


def open_cooler(path: str | Path, resolution: int) -> cooler.Cooler:
    """Open one resolution of an mcool file and require balancing weights."""
    source = Path(path)
    uri = f"{source}::/resolutions/{resolution}"
    try:
        clr = cooler.Cooler(uri)
    except Exception as exc:
        raise McoolError(
            f"cannot open {source} at resolution {resolution}",
            path=str(source),
            resolution=resolution,
        ) from exc

    if "weight" not in clr.bins().columns:
        raise McoolError(
            f"{source} at resolution {resolution} has no bins/weight; "
            "run cooler balance first",
            path=str(source),
            resolution=resolution,
        )
    return clr


def chrom_sizes(clr: cooler.Cooler) -> dict[str, int]:
    """Return chromosome sizes in mcool order."""
    return {str(chrom): int(size) for chrom, size in clr.chromsizes.items()}


def bin_centers(clr: cooler.Cooler, chrom: str) -> np.ndarray:
    """Return working-resolution bin centers for one chromosome."""
    bins = clr.bins().fetch(chrom)
    if bins.empty:
        raise McoolError(
            f"{chrom}: no bins at the working resolution",
            path=str(clr.filename),
        )

    starts = np.asarray(bins["start"], dtype=np.int64)
    ends = np.asarray(bins["end"], dtype=np.int64)
    centers = (starts + ends) / 2.0
    return np.sort(centers.astype(np.float64))


def choose_resolution(
    available: Sequence[int],
    lengths: Sequence[int] | np.ndarray,
    *,
    resolution: int | None = None,
    percentile: float = 10.0,
    min_bins_per_tad: int = 3,
) -> ResolutionChoice:
    """Choose an explicit or annotation-adaptive working resolution.

    Adaptive selection uses the largest available resolution at or below
    percentile(TAD length) / min_bins_per_tad, or the smallest available
    resolution when none qualifies.
    """
    resolutions = sorted(int(value) for value in available)
    if not resolutions:
        raise McoolError("mcool has no resolutions")

    if resolution is not None:
        if resolution not in resolutions:
            raise McoolError(
                f"resolution {resolution} is not available; choices: {resolutions}",
                resolution=resolution,
            )
        return ResolutionChoice(value=resolution, mode="explicit")

    size_array = np.asarray(lengths, dtype=np.float64)
    if size_array.size == 0:
        raise McoolError("cannot choose a resolution from an empty TAD table")
    if min_bins_per_tad < 1:
        raise McoolError("min_bins_per_tad must be at least 1")

    target = float(np.percentile(size_array, percentile)) / min_bins_per_tad
    candidates = [value for value in resolutions if value <= target]
    chosen = max(candidates) if candidates else min(resolutions)
    return ResolutionChoice(value=chosen, mode="adaptive")
