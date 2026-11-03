"""Node cohesion metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domains import SelectedUnit
from .errors import ClusteringError
from .matrix import distance_bin_edges


@dataclass(frozen=True)
class CohesionContext:
    """Per-chromosome unit matrices, pair-distance bins, and O/E baseline."""

    log_oe: np.ndarray
    unit_bins: np.ndarray
    distance_bin: np.ndarray
    baseline: np.ndarray

    def node_metrics(
        self,
        indices: np.ndarray,
    ) -> tuple[float | None, float | None]:
        """Return cross-unit O/E and its distance-matched excess."""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size < 2:
            return None, None

        sub = self.log_oe[np.ix_(indices, indices)]
        sub_bins = self.distance_bin[np.ix_(indices, indices)]
        weights = self.unit_bins[indices]
        pair_weights = weights[:, None] * weights[None, :]
        upper = np.triu_indices(indices.size, 1)
        values = sub[upper]
        pair_bins = sub_bins[upper]
        selected_weights = pair_weights[upper]
        total = float(selected_weights.sum())

        if total > 0.0:
            cross_oe = float(np.dot(selected_weights, values) / total)
            excess = float(
                np.dot(
                    selected_weights,
                    values - self.baseline[pair_bins],
                )
                / total
            )
        else:
            cross_oe = float(values.mean())
            excess = float((values - self.baseline[pair_bins]).mean())
        return cross_oe, excess


def build_cohesion_context(
    units: list[SelectedUnit],
    log_oe: np.ndarray,
    n_bins_chrom: int,
    *,
    working_binsize: int,
    distance_bins: int,
) -> CohesionContext:
    """Build the per-chromosome context used for node cohesion metrics."""
    values = np.asarray(log_oe, dtype=np.float64)
    n_units = len(units)
    if values.shape != (n_units, n_units):
        raise ClusteringError(
            f"log O/E matrix has shape {values.shape}, expected {(n_units, n_units)}",
            matrix="cohesion",
        )
    if working_binsize <= 0:
        raise ClusteringError(
            "working_binsize must be positive for cohesion metrics",
            matrix="cohesion",
        )

    centers = np.array(
        [(unit.start + unit.end) / 2.0 for unit in units],
        dtype=np.float64,
    )
    distance = np.floor(
        np.abs(centers[:, None] - centers[None, :]) / working_binsize
    ).astype(np.int32)
    np.maximum(distance, 1, out=distance)
    np.fill_diagonal(distance, 0)

    edges = distance_bin_edges(n_bins_chrom, distance_bins)
    n_distance_bins = len(edges) - 1
    distance_bin = np.clip(
        np.searchsorted(edges, distance, side="right") - 1,
        0,
        n_distance_bins - 1,
    )
    unit_bins = np.array(
        [unit.n_working_bins for unit in units],
        dtype=np.int64,
    )
    weights = unit_bins[:, None] * unit_bins[None, :]

    upper = np.triu_indices(n_units, 1)
    pair_weights = weights[upper]
    pair_values = values[upper]
    pair_bins = distance_bin[upper]
    baseline = np.zeros(n_distance_bins, dtype=np.float64)

    if pair_values.size:
        total = float(pair_weights.sum())
        if total > 0.0:
            global_mean = float(np.dot(pair_weights, pair_values) / total)
        else:
            global_mean = float(pair_values.mean())

        sums = np.bincount(
            pair_bins,
            weights=pair_weights * pair_values,
            minlength=n_distance_bins,
        )
        totals = np.bincount(
            pair_bins,
            weights=pair_weights,
            minlength=n_distance_bins,
        )
        baseline.fill(global_mean)
        valid = totals > 0.0
        baseline[valid] = sums[valid] / totals[valid]

    return CohesionContext(
        log_oe=values,
        unit_bins=unit_bins,
        distance_bin=distance_bin,
        baseline=baseline,
    )
