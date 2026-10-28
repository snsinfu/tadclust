"""Shared synthetic mcool fixtures."""

from __future__ import annotations

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
            if i == j:
                count = 100
            elif bins_table.chrom[i] != bins_table.chrom[j]:
                count = 5
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
