"""PHANGS morphology-mask utilities for the four-galaxy KTZ application."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
import numpy as np


ARM_LABELS = (5, 6)
INTERARM_LABELS = (4, 7)
TEMPLATE_LABELS = ARM_LABELS + INTERARM_LABELS


def sample_simple_environment_mask(
    mask_path: str | Path,
    x_grid_kpc: np.ndarray,
    y_grid_kpc: np.ndarray,
    project_face_on_to_sky: Callable,
) -> np.ndarray:
    """Nearest-neighbour sample the released simple mask on a face-on grid."""
    mask_path = Path(mask_path)
    with fits.open(mask_path, memmap=True) as hdul:
        mask = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header.copy()
    # These released masks inherit SIP coefficients from an already resampled
    # Spitzer mosaic although CTYPE is not marked ``-SIP``.  Removing those
    # stale coefficients reproduces the catalogue ENVIRONMENT assignments
    # slightly more accurately (better than 99.2 per cent for all four maps).
    for key in list(header):
        if key in {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"} or key.startswith(
            ("A_", "B_", "AP_", "BP_")
        ):
            del header[key]
    wcs = WCS(header).celestial
    ra_deg, dec_deg = project_face_on_to_sky(x_grid_kpc, y_grid_kpc)
    pixel_x, pixel_y = wcs.world_to_pixel_values(ra_deg, dec_deg)
    sampled = map_coordinates(
        mask, [pixel_y, pixel_x], order=0, mode="constant", cval=np.nan,
        prefilter=False,
    )
    return sampled.reshape(np.shape(x_grid_kpc))


def arm_fractions_by_radius(
    labels: np.ndarray,
    radius_kpc: np.ndarray,
    radial_edges_kpc: np.ndarray,
) -> np.ndarray:
    """Return arm covering fraction among arm/interarm mask pixels per annulus."""
    labels = np.asarray(labels)
    radius_kpc = np.asarray(radius_kpc, dtype=float)
    radial_edges_kpc = np.asarray(radial_edges_kpc, dtype=float)
    arm = np.isin(labels, ARM_LABELS)
    template_domain = np.isin(labels, TEMPLATE_LABELS)
    radial_bin = np.digitize(radius_kpc, radial_edges_kpc) - 1
    fractions = np.full(len(radial_edges_kpc) - 1, np.nan)
    for index in range(len(fractions)):
        selected = template_domain & (radial_bin == index)
        if np.any(selected):
            fractions[index] = np.mean(arm[selected])
    finite = np.isfinite(fractions)
    if not np.any(finite):
        raise ValueError("The PHANGS mask has no arm/interarm pixels on this grid.")
    fractions = np.interp(
        np.arange(len(fractions)), np.flatnonzero(finite), fractions[finite]
    )
    return np.clip(fractions, 1.0e-3, 1.0 - 1.0e-3)


def centred_arm_indicator(
    labels: np.ndarray,
    radius_kpc: np.ndarray,
    radial_edges_kpc: np.ndarray,
    arm_fractions: np.ndarray,
) -> np.ndarray:
    """Construct B_arm-<B_arm>_R, setting non-arm-domain pixels to zero."""
    labels = np.asarray(labels)
    radius_kpc = np.asarray(radius_kpc, dtype=float)
    radial_bin = np.clip(
        np.digitize(radius_kpc, radial_edges_kpc) - 1,
        0, len(arm_fractions) - 1,
    )
    arm = np.isin(labels, ARM_LABELS)
    template_domain = np.isin(labels, TEMPLATE_LABELS)
    result = np.zeros(labels.shape, dtype=float)
    result[template_domain] = (
        arm[template_domain].astype(float)
        - arm_fractions[radial_bin[template_domain]]
    )
    return result


def calibrate_mask_contrast(
    base_indicator: np.ndarray,
    source_flux_grid: np.ndarray,
    source_valid: np.ndarray,
    radius_kpc: np.ndarray,
    azimuth_rad: np.ndarray,
    radial_edges_kpc: np.ndarray,
    seed: int,
    n_bootstrap: int = 400,
) -> tuple[float, float, np.ndarray]:
    """Fit the binary-mask contrast to annulus-normalised H-alpha emission.

    The fitted model is F/<F>_R = 1 + A_mask (B_arm-f_arm).  A 12-by-4
    spatial block bootstrap supplies the quoted uncertainty.
    """
    source_valid = np.asarray(source_valid, dtype=bool)
    flux_grid = np.asarray(source_flux_grid, dtype=float)
    radial_bin = np.digitize(radius_kpc, radial_edges_kpc) - 1
    relative_flux = np.full(flux_grid.shape, np.nan)
    for index in range(len(radial_edges_kpc) - 1):
        selected = (
            source_valid & (radial_bin == index)
            & np.isfinite(flux_grid) & (flux_grid > 0)
        )
        if np.any(selected):
            relative_flux[selected] = flux_grid[selected] / np.mean(
                flux_grid[selected]
            )
    selected = (
        source_valid & np.isfinite(relative_flux)
        & np.isfinite(base_indicator)
        & (base_indicator != 0.0)
    )
    x = base_indicator[selected]
    y = relative_flux[selected] - 1.0
    denominator = np.sum(x**2)
    if denominator <= 0.0:
        raise ValueError("Cannot calibrate an empty arm template.")
    amplitude = float(max(np.sum(x * y) / denominator, 1.0e-3))

    source_radius = radius_kpc[selected]
    source_azimuth = np.mod(azimuth_rad[selected], 2.0 * np.pi)
    radial_quartiles = np.sqrt(np.linspace(
        np.nanmin(source_radius)**2, np.nanmax(source_radius)**2, 5
    ))
    radial_block = np.clip(
        np.digitize(source_radius, radial_quartiles) - 1, 0, 3
    )
    azimuth_block = np.floor(source_azimuth / (2.0 * np.pi) * 12).astype(int)
    block_id = 12 * radial_block + azimuth_block
    unique_blocks = np.unique(block_id)
    block_numerator = np.asarray([
        np.sum(x[block_id == block] * y[block_id == block])
        for block in unique_blocks
    ])
    block_denominator = np.asarray([
        np.sum(x[block_id == block]**2) for block in unique_blocks
    ])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for draw_index in range(n_bootstrap):
        draw = rng.integers(0, len(unique_blocks), len(unique_blocks))
        draws[draw_index] = max(
            np.sum(block_numerator[draw]) / np.sum(block_denominator[draw]),
            1.0e-3,
        )
    return amplitude, float(np.std(draws, ddof=1)), draws


def build_environment_template(
    root: str | Path,
    galaxy: str,
    x_grid_kpc: np.ndarray,
    y_grid_kpc: np.ndarray,
    radius_kpc: np.ndarray,
    azimuth_rad: np.ndarray,
    project_face_on_to_sky: Callable,
    source_valid: np.ndarray,
    source_flux: np.ndarray,
    radial_width_kpc: float,
    radial_maximum_kpc: float,
    seed: int,
) -> dict[str, np.ndarray | float | str]:
    """Build the H-alpha-calibrated PHANGS arm-mask source template."""
    root = Path(root)
    mask_galaxy = "NGC0628" if galaxy.upper() == "NGC628" else galaxy.upper()
    mask_path = (
        root / "data" / "phangs" / "environment_masks" / "PHANGS_env_masks"
        / "env_masks_simple" / f"{mask_galaxy}_env_mask_simple.fits"
    )
    labels = sample_simple_environment_mask(
        mask_path, x_grid_kpc, y_grid_kpc, project_face_on_to_sky
    )
    midpoint_axis = np.linspace(
        float(x_grid_kpc[0, 0]), float(x_grid_kpc[0, -1]),
        2 * x_grid_kpc.shape[1] - 1,
    )
    midpoint_x, midpoint_y = np.meshgrid(midpoint_axis, midpoint_axis)
    midpoint_radius = np.hypot(midpoint_x, midpoint_y)
    midpoint_labels = sample_simple_environment_mask(
        mask_path, midpoint_x, midpoint_y, project_face_on_to_sky
    )
    radial_edges = np.arange(
        0.0, radial_maximum_kpc + radial_width_kpc, radial_width_kpc
    )
    arm_fractions = arm_fractions_by_radius(
        midpoint_labels, midpoint_radius, radial_edges
    )
    base_indicator = centred_arm_indicator(
        labels, radius_kpc, radial_edges, arm_fractions
    )
    midpoint_base_indicator = centred_arm_indicator(
        midpoint_labels, midpoint_radius, radial_edges, arm_fractions
    )
    source_flux_grid = np.full(x_grid_kpc.shape, np.nan)
    source_flux_grid[source_valid] = source_flux
    amplitude, amplitude_error, bootstrap = calibrate_mask_contrast(
        base_indicator, source_flux_grid, source_valid, radius_kpc,
        azimuth_rad, radial_edges, seed,
    )
    modulation = amplitude * base_indicator
    midpoint_modulation = amplitude * midpoint_base_indicator
    minimum = float(min(np.min(modulation), np.min(midpoint_modulation)))
    positive_limit = None if minimum >= 0.0 else 0.98 / abs(minimum)
    digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    return {
        "mask_path": str(mask_path),
        "mask_sha256": digest,
        "labels": labels,
        "midpoint_labels": midpoint_labels,
        "arm_fractions": arm_fractions,
        "radial_edges_kpc": radial_edges,
        "base_indicator": base_indicator,
        "midpoint_base_indicator": midpoint_base_indicator,
        "modulation": modulation,
        "midpoint_modulation": midpoint_modulation,
        "contrast_amplitude": float(amplitude),
        "contrast_amplitude_error": float(amplitude_error),
        "contrast_bootstrap": bootstrap,
        "positive_scale_limit": positive_limit,
        "unit_rate_minimum": 1.0 + minimum,
        "arm_area_fraction": float(np.mean(np.isin(labels, ARM_LABELS))),
        "template_domain_fraction": float(
            np.mean(np.isin(labels, TEMPLATE_LABELS))
        ),
    }
