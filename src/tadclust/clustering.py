"""Shared k-means helper."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .errors import ComputationError


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
        raise ComputationError("cannot cluster an empty set")

    n_clusters = max(1, min(k, n_rows))
    n_features = max(1, min(n_pcs, values.shape[1])) if values.shape[1] else 0
    data = (
        values[:, :n_features]
        if n_features
        else np.zeros((n_rows, 1), dtype=np.float64)
    )

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
