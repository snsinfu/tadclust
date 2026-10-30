"""Shared k-means helper."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .errors import ClusteringError


def derive_seed(base_seed: int, *key: int) -> int:
    """Return a stable 32-bit seed for one k-means call."""
    sequence = [int(base_seed), len(key), *key]
    return int(np.random.SeedSequence(sequence).generate_state(1)[0])


def informative_mask(features: np.ndarray, n_features: int) -> np.ndarray:
    """Return True for rows with at least one non-zero clustering feature."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ClusteringError("features must be a two-dimensional array")
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=bool)
    n_features = max(1, min(int(n_features), values.shape[1]))
    return np.any(values[:, :n_features] != 0.0, axis=1)


def kmeans_labels(
    pcs: np.ndarray,
    k: int,
    n_pcs: int,
    seed: int,
    n_init: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster PC rows and return labels plus cluster centers."""
    values = np.asarray(pcs, dtype=np.float64)
    n_rows = values.shape[0]
    if n_rows == 0:
        raise ClusteringError("cannot cluster an empty set")

    n_clusters = max(1, min(k, n_rows))
    n_features = max(1, min(n_pcs, values.shape[1])) if values.shape[1] else 0
    data = (
        values[:, :n_features]
        if n_features
        else np.zeros((n_rows, 1), dtype=np.float64)
    )

    # Identical profiles cannot form distinct clusters. Uninformative rows are
    # filtered before this point; this is a safeguard for duplicate features.
    if n_clusters > 1:
        n_clusters = min(n_clusters, int(np.unique(data, axis=0).shape[0]))

    if n_clusters == 1:
        return (
            np.zeros(n_rows, dtype=np.int64),
            data.mean(axis=0, keepdims=True),
        )

    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,  # pyright: ignore[reportArgumentType]
        random_state=seed,
    )
    labels = model.fit_predict(data).astype(np.int64)
    return labels, np.asarray(model.cluster_centers_, dtype=np.float64)
