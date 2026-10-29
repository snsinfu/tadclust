"""Phase-1 chromosome input sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TypeAlias

import cooler
import numpy as np

from .domains import (
    ChromosomeTree,
    assign_bin_counts,
    build_chromosome_tree,
    build_flat_tree,
    map_bins_to_fine,
)
from .errors import InputError
from .mcool import bin_centers, bin_intervals
from .tads import TadTable, check_against_chromsizes


@dataclass(frozen=True)
class AnnotationSource:
    """Chromosome units derived from a laminar annotation table."""

    table: TadTable
    path: str
    mode: ClassVar[str] = "annotation"
    domain_kind: ClassVar[str] = "tad"

    def resolution_lengths(self) -> np.ndarray:
        return self.table.lengths

    def validate(self, sizes: dict[str, int]) -> None:
        check_against_chromsizes(self.table, sizes, path=self.path)

    def select_chromosomes(
        self,
        available: list[str],
        requested: list[str] | None,
    ) -> list[str]:
        return _select_chromosomes(available, requested, set(self.table.chroms))

    def build_units(
        self,
        cooler_object: cooler.Cooler,
        chrom: str,
        chromsize: int,
    ) -> tuple[ChromosomeTree, np.ndarray]:
        tree = build_chromosome_tree(self.table, chrom, chromsize)
        centers = bin_centers(cooler_object, chrom)
        assign_bin_counts(tree, centers)
        return tree, map_bins_to_fine(tree, centers)


@dataclass(frozen=True)
class UniformBinSource:
    """Chromosome units derived directly from working-resolution mcool bins."""

    mode: ClassVar[str] = "uniform_bins"
    domain_kind: ClassVar[str] = "bin"

    def resolution_lengths(self) -> None:
        return None

    def validate(self, sizes: dict[str, int]) -> None:
        return None

    def select_chromosomes(
        self,
        available: list[str],
        requested: list[str] | None,
    ) -> list[str]:
        return _select_chromosomes(available, requested, None)

    def build_units(
        self,
        cooler_object: cooler.Cooler,
        chrom: str,
        chromsize: int,
    ) -> tuple[ChromosomeTree, np.ndarray]:
        starts, ends = bin_intervals(cooler_object, chrom)
        tree = build_flat_tree(starts, ends, chrom, chromsize)
        return tree, np.arange(tree.n_fine, dtype=np.int64)


CisSource: TypeAlias = AnnotationSource | UniformBinSource


def _select_chromosomes(
    available: list[str],
    requested: list[str] | None,
    required: set[str] | None,
) -> list[str]:
    if requested is None:
        chromosomes = [
            chrom for chrom in available if required is None or chrom in required
        ]
        if not chromosomes:
            if required is None:
                raise InputError("mcool has no chromosomes")
            raise InputError("no chromosomes with TAD rows")
        return chromosomes

    if len(set(requested)) != len(requested):
        raise InputError("duplicate chromosome in the requested list")

    unknown = [chrom for chrom in requested if chrom not in available]
    if unknown:
        raise InputError(f"unknown chromosome(s): {', '.join(unknown)}")
    if required is not None:
        missing = [chrom for chrom in requested if chrom not in required]
        if missing:
            raise InputError(f"no TAD rows for chromosome(s): {', '.join(missing)}")
    return list(requested)
