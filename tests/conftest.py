"""Shared synthetic mcool fixtures."""

from __future__ import annotations

from collections.abc import Callable

import h5py
import numpy as np
import pandas as pd
import pytest


def write_mcool(
    tmp_path,
    *,
    resolutions: tuple[int, ...] = (500, 1000, 2000),
    chroms: tuple[str, ...] = ("chr1", "chr2"),
    chrom_size: int = 2000,
    with_weight: bool = True,
    cross_counts: dict[tuple[str, str], int] | None = None,
    cross_count_fn: Callable[[int, int], float] | None = None,
    empty_bins: set[int] | None = None,
    omit_cross_pairs: set[tuple[str, str]] | None = None,
):
    import cooler

    base_resolution = min(resolutions)
    bins = []
    for chrom in chroms:
        for start in range(0, chrom_size, base_resolution):
            row = {
                "chrom": chrom,
                "start": start,
                "end": min(start + base_resolution, chrom_size),
            }
            if with_weight:
                row["weight"] = 1.0
            bins.append(row)
    bins_table = pd.DataFrame(bins)

    pixels = []
    for i in range(len(bins_table)):
        for j in range(i, len(bins_table)):
            if empty_bins and (i in empty_bins or j in empty_bins):
                continue
            if i == j:
                count = 100
            elif bins_table.chrom[i] != bins_table.chrom[j]:
                key = tuple(sorted((bins_table.chrom[i], bins_table.chrom[j])))
                if omit_cross_pairs and key in omit_cross_pairs:
                    continue
                if cross_count_fn is not None:
                    count = cross_count_fn(i, j)
                else:
                    count = (cross_counts or {}).get(key, 5)
            elif i // 2 == j // 2:
                count = 30
            else:
                count = 2
            pixels.append((i, j, count))
    pixel_table = pd.DataFrame(pixels, columns=["bin1_id", "bin2_id", "count"])

    base = tmp_path / "base.cool"
    cooler.create_cooler(str(base), bins_table, pixel_table)

    path = tmp_path / "test.mcool"
    cooler.zoomify_cooler(
        str(base),
        str(path),
        sorted(resolutions),
        chunksize=100_000,
    )

    if with_weight:
        # zoomify_cooler drops the weight column at coarser resolutions.
        with h5py.File(path, "a") as handle:
            for resolution in resolutions:
                group = handle[f"resolutions/{resolution}"]
                assert isinstance(group, h5py.Group)
                bins_group = group["bins"]
                assert isinstance(bins_group, h5py.Group)
                if "weight" not in bins_group:
                    starts = bins_group["start"]
                    assert isinstance(starts, h5py.Dataset)
                    size = starts.shape[0]
                    bins_group.create_dataset(
                        "weight",
                        data=np.ones(size, dtype=np.float64),
                    )
    return path


@pytest.fixture
def mcool(tmp_path):
    return write_mcool(tmp_path)


@pytest.fixture
def unbalanced_mcool(tmp_path):
    return write_mcool(tmp_path, with_weight=False)


@pytest.fixture
def three_chrom_mcool(tmp_path):
    return write_mcool(
        tmp_path,
        resolutions=(500,),
        chroms=("chr1", "chr2", "chr3"),
        cross_counts={
            ("chr1", "chr2"): 100,
            ("chr1", "chr3"): 1,
            ("chr2", "chr3"): 20,
        },
    )


@pytest.fixture
def trans_tads(tmp_path):
    rows = []
    for chrom in ("chr1", "chr2", "chr3"):
        rows.extend(
            [
                (chrom, 0, 1000),
                (chrom, 1000, 2000),
            ]
        )
    return write_tads(tmp_path, rows, name="trans_tads.tsv")


@pytest.fixture
def trans_track(tmp_path):
    rows = []
    for chrom in ("chr1", "chr2", "chr3"):
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0)):
            rows.append((chrom, index * 500, (index + 1) * 500, value))
    return write_track(tmp_path, rows, name="trans_track.bedGraph")


def write_tads(tmp_path, rows, name="tads.tsv"):
    path = tmp_path / name
    lines = ["chrom\tstart\tend"]
    lines.extend(f"{chrom}\t{start}\t{end}" for chrom, start, end in rows)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_track(tmp_path, rows, name="track.bedGraph"):
    path = tmp_path / name
    lines = [f"{chrom}\t{start}\t{end}\t{value}" for chrom, start, end, value in rows]
    path.write_text("\n".join(lines) + "\n")
    return path
