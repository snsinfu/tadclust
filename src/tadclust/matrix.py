"""Cis observed/expected matrices and principal components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

from .errors import ComputationError, McoolError, NoPositiveModeError

DENSE_PCA_MAX = 6000


@dataclass(frozen=True)
class ObservedExpected:
    """Observed, expected, log-ratio, self O/E, and pseudocount."""

    observed: np.ndarray
    expected: np.ndarray
    log_oe: np.ndarray
    self_oe: np.ndarray
    alpha: float


@dataclass(frozen=True)
class FineUnitMatrices(ObservedExpected):
    """Fine-unit O/E matrices, bin counts, and principal components."""

    bins_per_fine_unit: np.ndarray
    pcs: np.ndarray
    eigenvalues: np.ndarray
    variance_fraction: np.ndarray


@dataclass(frozen=True)
class DomainMatrices(ObservedExpected):
    """Selected-domain O/E matrices and principal components."""

    pcs: np.ndarray
    eigenvalues: np.ndarray
    variance_fraction: np.ndarray


def distance_bin_edges(
    n_chromosome_bins: int,
    n_distance_bins: int,
) -> np.ndarray:
    """Build log-spaced off-diagonal distance-bin edges."""
    if n_distance_bins < 1:
        raise ComputationError("n_distance_bins must be at least 1")
    if n_chromosome_bins <= 2:
        return np.array([1.0, 2.0])

    upper = float(max(2, n_chromosome_bins))
    raw = np.geomspace(1.0, upper, num=max(3, n_distance_bins + 1))
    edges = np.unique(np.round(raw).astype(np.int64))
    edges = edges[edges >= 1]
    if edges.size < 2:
        return np.array([1.0, 2.0])
    return edges.astype(np.float64)


def expected_by_distance(
    distance: np.ndarray,
    observed: np.ndarray,
    n_chromosome_bins: int,
    n_distance_bins: int,
) -> np.ndarray:
    """Estimate expected counts from the observed distance decay."""
    distance = np.asarray(distance, dtype=np.int64)
    observed = np.asarray(observed, dtype=np.float64)
    if distance.size == 0:
        raise McoolError("cis matrix has no pixels")

    edges = distance_bin_edges(n_chromosome_bins, n_distance_bins)
    n_distance_bins_used = len(edges) - 1
    bin_index = np.clip(
        np.searchsorted(edges, distance, side="right") - 1,
        0,
        n_distance_bins_used - 1,
    )

    expected = np.zeros_like(observed, dtype=np.float64)
    diagonal = distance == 0
    off_diagonal = ~diagonal

    if diagonal.any():
        expected[diagonal] = float(np.mean(observed[diagonal]))

    if off_diagonal.any():
        sums = np.bincount(
            bin_index[off_diagonal],
            weights=observed[off_diagonal],
            minlength=n_distance_bins_used,
        )
        counts = np.bincount(
            bin_index[off_diagonal],
            minlength=n_distance_bins_used,
        )
        means = np.zeros(n_distance_bins_used, dtype=np.float64)
        valid = counts > 0
        if not valid.any():
            raise McoolError("distance-expected O/E: no off-diagonal pixels")
        means[valid] = sums[valid] / counts[valid]

        if not valid.all():
            bin_centers = np.sqrt(edges[:-1] * edges[1:])
            log_bin_centers = np.log(bin_centers)
            log_means = np.log(np.maximum(means[valid], np.finfo(float).tiny))
            means = np.exp(
                np.interp(log_bin_centers, log_bin_centers[valid], log_means)
            )

        expected[off_diagonal] = means[bin_index[off_diagonal]]

    expected[expected <= 0] = np.finfo(float).tiny
    return expected


def scaled_pseudocount(expected: np.ndarray, multiplier: float) -> float:
    """Scale the pseudocount by the smallest positive expected value."""
    values = np.asarray(expected, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return float(max(multiplier, np.finfo(float).tiny))
    return float(max(multiplier * float(positive.min()), np.finfo(float).tiny))


def log2_ratio(
    observed: np.ndarray,
    expected: np.ndarray,
    alpha: float,
    *,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """Compute log2 O/E with a pseudocount and asymmetric clipping."""
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log2((observed + alpha) / (expected + alpha))
    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=clip_max,
        neginf=clip_min,
    )
    return np.clip(values, clip_min, clip_max)


def symmetrize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return a full symmetric matrix from one triangle."""
    values = np.asarray(matrix, dtype=np.float64)
    return values + values.T - np.diag(np.diag(values))


def cosine_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows and columns by their L2 norms."""
    values = np.nan_to_num(
        np.asarray(matrix, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    norm = np.linalg.norm(values, axis=1)
    inverse = np.zeros_like(norm)
    valid = norm > 0
    inverse[valid] = 1.0 / norm[valid]
    normalized = values * inverse[:, None] * inverse[None, :]
    return np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def compute_top_pcs(
    matrix: np.ndarray,
    n_pcs: int,
    *,
    dense_max: int = DENSE_PCA_MAX,
    label: str = "matrix",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the top principal components and their variance fractions.

    The O/E matrix is symmetric, so eigh is sufficient and cheaper than SVD;
    negative eigenvalues are clipped to zero.
    """
    if n_pcs < 1:
        raise ComputationError(
            "n_pcs must be at least 1",
            matrix=label,
        )

    matrix_values = np.nan_to_num(
        np.asarray(matrix, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    n_rows = int(matrix_values.shape[0])
    if n_rows == 0:
        raise ComputationError(
            "cannot compute PCs of an empty matrix",
            matrix=label,
        )
    if n_rows == 1:
        zeros = np.zeros(n_pcs, dtype=np.float64)
        return np.zeros((1, n_pcs)), zeros.copy(), zeros.copy()

    matrix_values = (matrix_values + matrix_values.T) / 2.0
    n_components = max(1, min(n_pcs, n_rows - 1))
    try:
        if n_rows <= dense_max:
            eigenvalues, eigenvectors = linalg.eigh(
                matrix_values,
                subset_by_index=[n_rows - n_components, n_rows - 1],
            )
        else:
            eigenvalues, eigenvectors = sparse_linalg.eigsh(
                sparse.csr_matrix(matrix_values),
                k=n_components,
                which="LA",
                tol=1e-4,  # pyright: ignore[reportArgumentType]
                maxiter=max(1000, n_rows * 10),
            )
    except Exception as exc:
        raise ComputationError(
            f"eigendecomposition failed: {exc}",
            matrix=label,
        ) from exc

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)

    if eigenvalues.size == 0 or eigenvalues[0] <= 1e-12:
        raise NoPositiveModeError(
            "eigendecomposition found no positive mode",
            matrix=label,
        )

    eigenvalues = np.maximum(eigenvalues, 0.0)
    embedding = eigenvectors * np.sqrt(eigenvalues)[None, :]

    if embedding.shape[1] < n_pcs:
        padding = n_pcs - embedding.shape[1]
        embedding = np.pad(embedding, ((0, 0), (0, padding)))
        eigenvalues = np.pad(eigenvalues, (0, padding))

    total = float(eigenvalues.sum())
    fractions = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    return embedding, eigenvalues, fractions


def pcs_from_log_oe(
    matrix: np.ndarray,
    n_pcs: int,
    *,
    label: str = "cis",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCs from the cosine-normalized O/E matrix."""
    return compute_top_pcs(cosine_normalize(matrix), n_pcs, label=label)


def canonicalize_pc1_sign(embedding: np.ndarray) -> np.ndarray:
    """Flip PC1 so its largest-magnitude entry is positive."""
    values = np.array(embedding, dtype=np.float64, copy=True)
    if values.shape[1] == 0:
        return values

    column = values[:, 0]
    index = int(np.argmax(np.abs(column)))
    if column[index] < 0:
        values[:, 0] *= -1.0
    return values


def orient_pc1_to_signal(
    pcs: np.ndarray,
    rank: np.ndarray,
) -> np.ndarray:
    """Flip PC1 so positive PC1 is the high-signal (A) side."""
    values = np.array(pcs, dtype=np.float64, copy=True)
    ranks = np.asarray(rank, dtype=np.float64)
    if values.ndim != 2:
        raise ComputationError("domain PCs must be a two-dimensional array")
    if values.shape[0] != ranks.shape[0]:
        raise ComputationError("PC and signal-rank lengths differ")
    if values.shape[1] == 0 or values.shape[0] < 2:
        return values
    if np.std(values[:, 0]) == 0 or np.std(ranks) == 0:
        return values
    if np.corrcoef(values[:, 0], ranks)[0, 1] < 0:
        values[:, 0] *= -1.0
    return values


def compute_fine_unit_matrices(
    cooler_object: Any,
    chrom: str,
    bin_to_fine_unit: np.ndarray,
    n_fine: int,
    *,
    n_distance_bins: int,
    pseudocount: float,
    clip_min: float,
    clip_max: float,
    n_pcs: int,
    balance: bool = True,
) -> FineUnitMatrices:
    """Aggregate cis O/E to fine units and compute fine-unit PCs."""
    if n_fine < 1:
        raise ComputationError(
            f"{chrom}: no fine units",
            chrom=chrom,
        )

    bin_map = np.asarray(bin_to_fine_unit, dtype=np.int64)
    if bin_map.ndim != 1:
        raise ComputationError(
            f"{chrom}: bin map must be one-dimensional",
            chrom=chrom,
        )
    if np.any(bin_map < -1) or np.any(bin_map >= n_fine):
        raise ComputationError(
            f"{chrom}: bin map values must be in [-1, {n_fine})",
            chrom=chrom,
        )

    cis_matrix = cooler_object.matrix(
        balance=bool(balance),
        sparse=True,
    ).fetch(chrom)
    cis_matrix = cis_matrix.tocoo()
    cis_matrix.sum_duplicates()

    upper_triangle = sparse.triu(cis_matrix, k=0).tocoo()
    if upper_triangle.nnz == 0:
        upper_triangle = sparse.tril(cis_matrix, k=0).T.tocoo()
    if upper_triangle.nnz == 0:
        raise McoolError(
            f"{chrom}: balanced cis matrix has no non-zero pixels",
        )

    pixel_row = upper_triangle.row.astype(np.int64, copy=False)
    pixel_col = upper_triangle.col.astype(np.int64, copy=False)
    observed_pixels = upper_triangle.data.astype(np.float64, copy=False)
    pixel_distance = pixel_col - pixel_row

    usable_pixels = np.isfinite(observed_pixels) & (observed_pixels > 0)
    if not usable_pixels.any():
        raise McoolError(f"{chrom}: no usable cis pixels after filtering")
    pixel_row = pixel_row[usable_pixels]
    pixel_col = pixel_col[usable_pixels]
    observed_pixels = observed_pixels[usable_pixels]
    pixel_distance = pixel_distance[usable_pixels]

    if pixel_row.max() >= bin_map.size or pixel_col.max() >= bin_map.size:
        raise ComputationError(
            f"{chrom}: pixel bin index is outside the bin map",
            chrom=chrom,
        )

    n_chrom_bins = int(cooler_object.extent(chrom)[1] - cooler_object.extent(chrom)[0])
    expected_pixels = expected_by_distance(
        pixel_distance,
        observed_pixels,
        n_chrom_bins,
        n_distance_bins,
    )

    fine_unit_row = bin_map[pixel_row]
    fine_unit_col = bin_map[pixel_col]
    valid = (fine_unit_row >= 0) & (fine_unit_col >= 0)
    fine_unit_row = fine_unit_row[valid]
    fine_unit_col = fine_unit_col[valid]
    observed_pixels = observed_pixels[valid]
    expected_pixels = expected_pixels[valid]
    if not fine_unit_row.size:
        raise McoolError(f"{chrom}: no pixels map to fine units")

    flat_index = fine_unit_row * n_fine + fine_unit_col
    n_flat_cells = n_fine * n_fine
    observed_fine = np.bincount(
        flat_index,
        weights=observed_pixels,
        minlength=n_flat_cells,
    ).reshape(n_fine, n_fine)
    expected_fine = np.bincount(
        flat_index,
        weights=expected_pixels,
        minlength=n_flat_cells,
    ).reshape(n_fine, n_fine)

    observed_fine = symmetrize_matrix(observed_fine)
    expected_fine = symmetrize_matrix(expected_fine)
    mapped_bins = bin_map[bin_map >= 0]
    bins_per_fine_unit = np.bincount(
        mapped_bins,
        minlength=n_fine,
    ).astype(np.int64)

    alpha = scaled_pseudocount(expected_fine, pseudocount)
    log_oe = log2_ratio(
        observed_fine,
        expected_fine,
        alpha,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    self_oe = np.diag(log_oe).copy()
    np.fill_diagonal(log_oe, 0.0)

    pcs, eigenvalues, fractions = pcs_from_log_oe(
        log_oe,
        n_pcs,
        label=f"cis:{chrom}",
    )
    return FineUnitMatrices(
        observed=observed_fine,
        expected=expected_fine,
        log_oe=log_oe,
        self_oe=self_oe,
        alpha=alpha,
        bins_per_fine_unit=bins_per_fine_unit,
        pcs=pcs,
        eigenvalues=eigenvalues,
        variance_fraction=fractions,
    )


def compute_domain_matrices(
    domain_map: Any,
    observed_fine: np.ndarray,
    expected_fine: np.ndarray,
    *,
    pseudocount: float,
    clip_min: float,
    clip_max: float,
    n_pcs: int,
    label: str = "domains",
) -> DomainMatrices:
    """Aggregate fine O/E to selected domains and compute domain PCs."""
    observed = np.asarray(
        domain_map.T @ observed_fine @ domain_map,
        dtype=np.float64,
    )
    expected = np.asarray(
        domain_map.T @ expected_fine @ domain_map,
        dtype=np.float64,
    )

    alpha = scaled_pseudocount(expected, pseudocount)
    log_oe = log2_ratio(
        observed,
        expected,
        alpha,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    self_oe = np.diag(log_oe).copy()
    np.fill_diagonal(log_oe, 0.0)

    pcs, eigenvalues, fractions = pcs_from_log_oe(
        log_oe,
        n_pcs,
        label=label,
    )
    return DomainMatrices(
        observed=observed,
        expected=expected,
        log_oe=log_oe,
        self_oe=self_oe,
        alpha=alpha,
        pcs=pcs,
        eigenvalues=eigenvalues,
        variance_fraction=fractions,
    )
