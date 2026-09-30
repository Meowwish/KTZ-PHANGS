"""Forward transfer for annular mean subtraction in PHANGS correlations.

The abundance maps are detrended by subtracting the sample mean in each
radial annulus.  This module applies the same linear projector to a
stationary-equivalent covariance curve on the actual finite pixel mask.  The
expensive part is linear in the input covariance and is therefore cached as
two small response matrices; likelihood evaluations then require only matrix
products.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve


TRANSFER_VERSION = 1


def _radial_bin_sum(
    grid: np.ndarray, lag_bin: np.ndarray, selected: np.ndarray, n_bins: int,
) -> np.ndarray:
    return np.bincount(
        lag_bin[selected], weights=grid.ravel()[selected], minlength=n_bins,
    )


def build_annular_detrending_transfer(
    valid: np.ndarray,
    radius_grid_kpc: np.ndarray,
    pixel_size_kpc: float,
    separation_edges_kpc: np.ndarray,
    separation_centres_kpc: np.ndarray,
    annulus_width_kpc: float,
) -> dict[str, np.ndarray]:
    """Build the exact annular projector for a separation-only covariance.

    The input covariance is represented by its zero-lag value followed by its
    values at ``separation_centres_kpc``.  Linear interpolation supplies the
    covariance at every Cartesian lag; it is set to zero beyond the final
    fitted separation.  For each basis vector we evaluate ``P C P.T`` on the
    observed mask, where ``P`` subtracts the arithmetic mean of the valid
    cells in the same annulus.  The returned numerator and denominator
    matrices reproduce the exhaustive pair-product estimator.
    """
    valid = np.asarray(valid, dtype=bool)
    radius_grid_kpc = np.asarray(radius_grid_kpc, dtype=float)
    separation_edges_kpc = np.asarray(separation_edges_kpc, dtype=float)
    separation_centres_kpc = np.asarray(separation_centres_kpc, dtype=float)
    if valid.shape != radius_grid_kpc.shape:
        raise ValueError("valid and radius_grid_kpc must have equal shapes")
    if annulus_width_kpc <= 0.0 or pixel_size_kpc <= 0.0:
        raise ValueError("pixel and annulus widths must be positive")

    height, width = valid.shape
    annulus_index = np.floor(radius_grid_kpc / annulus_width_kpc).astype(int)
    annulus_labels = np.unique(annulus_index[valid])
    annulus_masks = [valid & (annulus_index == label) for label in annulus_labels]
    annulus_counts = np.asarray([np.sum(mask) for mask in annulus_masks])

    lag_y = np.arange(-(height - 1), height) * pixel_size_kpc
    lag_x = np.arange(-(width - 1), width) * pixel_size_kpc
    lag_x_grid, lag_y_grid = np.meshgrid(lag_x, lag_y)
    lag_radius = np.hypot(lag_x_grid, lag_y_grid)
    lag_bin = np.digitize(lag_radius.ravel(), separation_edges_kpc) - 1
    n_bins = len(separation_centres_kpc)
    selected_lag = (
        (lag_bin >= 0) & (lag_bin < n_bins) & (lag_radius.ravel() > 0.0)
    )
    valid_float = valid.astype(float)
    pair_grid = fftconvolve(valid_float, valid_float[::-1, ::-1], mode="full")

    n_coordinates = n_bins + 1
    numerator_matrix = np.empty((n_bins, n_coordinates), dtype=float)
    denominator_matrix = np.empty_like(numerator_matrix)
    interpolation_radius = np.concatenate(([0.0], separation_centres_kpc))

    for coordinate_index in range(n_coordinates):
        coordinate = np.zeros(n_coordinates)
        coordinate[coordinate_index] = 1.0
        covariance_lag = np.interp(
            lag_radius, interpolation_radius, coordinate, left=coordinate[0],
            right=0.0,
        )

        annular_means_at_pixel = np.asarray([
            fftconvolve(mask.astype(float), covariance_lag, mode="same") / count
            for mask, count in zip(annulus_masks, annulus_counts)
        ])
        annulus_pair_means = np.asarray([
            [np.mean(mean_map[target_mask]) for target_mask in annulus_masks]
            for mean_map in annular_means_at_pixel
        ])

        first_term = pair_grid * covariance_lag
        row_mean_term = np.zeros_like(first_term)
        annulus_mean_term = np.zeros_like(first_term)
        for source_index, source_mask in enumerate(annulus_masks):
            row_mean_term += fftconvolve(
                source_mask.astype(float),
                (annular_means_at_pixel[source_index] * valid)[::-1, ::-1],
                mode="full",
            )
            pair_mean_map = np.zeros(valid.shape, dtype=float)
            for target_index, target_mask in enumerate(annulus_masks):
                pair_mean_map[target_mask] = annulus_pair_means[
                    source_index, target_index
                ]
            annulus_mean_term += fftconvolve(
                source_mask.astype(float), pair_mean_map[::-1, ::-1],
                mode="full",
            )
        projected_pair_sum = (
            first_term - row_mean_term - row_mean_term[::-1, ::-1]
            + annulus_mean_term
        )
        numerator_matrix[:, coordinate_index] = _radial_bin_sum(
            projected_pair_sum, lag_bin, selected_lag, n_bins,
        )

        projected_variance = np.zeros(valid.shape, dtype=float)
        for annulus_position, annulus_mask in enumerate(annulus_masks):
            projected_variance[annulus_mask] = (
                coordinate[0]
                - 2.0 * annular_means_at_pixel[annulus_position][annulus_mask]
                + annulus_pair_means[annulus_position, annulus_position]
            )
        endpoint_sum = 0.5 * (
            fftconvolve(
                projected_variance, valid_float[::-1, ::-1], mode="full"
            )
            + fftconvolve(
                valid_float, projected_variance[::-1, ::-1], mode="full"
            )
        )
        denominator_matrix[:, coordinate_index] = _radial_bin_sum(
            endpoint_sum, lag_bin, selected_lag, n_bins,
        )

    return {
        "numerator_matrix": numerator_matrix,
        "denominator_matrix": denominator_matrix,
        "annulus_labels": annulus_labels,
        "annulus_counts": annulus_counts,
    }


def transfer_signature(
    valid: np.ndarray,
    radius_grid_kpc: np.ndarray,
    pixel_size_kpc: float,
    separation_edges_kpc: np.ndarray,
    separation_centres_kpc: np.ndarray,
    annulus_width_kpc: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(valid, dtype=np.uint8).tobytes())
    digest.update(np.ascontiguousarray(radius_grid_kpc, dtype=float).tobytes())
    digest.update(json.dumps({
        "version": TRANSFER_VERSION,
        "pixel_size_kpc": float(pixel_size_kpc),
        "annulus_width_kpc": float(annulus_width_kpc),
        "separation_edges_kpc": np.asarray(separation_edges_kpc).tolist(),
        "separation_centres_kpc": np.asarray(separation_centres_kpc).tolist(),
    }, sort_keys=True).encode())
    return digest.hexdigest()


def load_or_build_annular_detrending_transfer(
    path: str | Path,
    valid: np.ndarray,
    radius_grid_kpc: np.ndarray,
    pixel_size_kpc: float,
    separation_edges_kpc: np.ndarray,
    separation_centres_kpc: np.ndarray,
    annulus_width_kpc: float,
) -> dict[str, np.ndarray]:
    """Load a matching response matrix, or build and cache it."""
    path = Path(path)
    signature = transfer_signature(
        valid, radius_grid_kpc, pixel_size_kpc, separation_edges_kpc,
        separation_centres_kpc, annulus_width_kpc,
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["signature"].item()) == signature:
                return {
                    name: cached[name] for name in cached.files
                    if name != "signature"
                }
    result = build_annular_detrending_transfer(
        valid, radius_grid_kpc, pixel_size_kpc, separation_edges_kpc,
        separation_centres_kpc, annulus_width_kpc,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, signature=np.asarray(signature), **result)
    return result


def projected_numerator_and_denominator(
    correlation: np.ndarray,
    transfer: dict[str, np.ndarray],
    zero_lag: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return annularly projected pair numerator and endpoint denominator."""
    coordinates = np.concatenate(([zero_lag], np.asarray(correlation, dtype=float)))
    return (
        transfer["numerator_matrix"] @ coordinates,
        transfer["denominator_matrix"] @ coordinates,
    )


def apply_annular_detrending_transfer(
    correlation: np.ndarray,
    transfer: dict[str, np.ndarray],
    zero_lag: float = 1.0,
) -> np.ndarray:
    """Apply annular mean subtraction to a stationary-equivalent curve."""
    numerator, denominator = projected_numerator_and_denominator(
        correlation, transfer, zero_lag=zero_lag,
    )
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )
