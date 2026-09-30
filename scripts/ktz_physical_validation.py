"""Exploratory external diagnostics for the four PHANGS--MUSE KTZ fits.

This script deliberately uses observables that were not constructed from the
metallicity residual values.  It (i) estimates the excess pair correlation of
qualified PHANGS H II-region centroids relative to the same radial-plus-arm
intensity used by the KTZ likelihood, and (ii) removes the NGC 4303 large-scale
tail from the likelihood and asks whether the small-scale posterior predicts
that held-out tail.  Both checks are exploratory: the source and abundance
selections share observational structure, and the harmonic H-alpha geometry was
examined after the tail was identified.  Machine-readable tables and the manuscript
figure are written below ``outputs/implications`` and ``figs``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phangs_annular_transfer import apply_annular_detrending_transfer
from scripts.phangs_template_ktz import (
    correlation_from_template_basis,
    interpolate_regular_basis,
)


OUTPUT_ROOT = ROOT / "outputs" / "implications"
FIGURE_ROOT = ROOT / "figs"
PIXEL_SIZE_KPC = 0.06
TARGET_FWHM_KPC = 0.12
ANNULUS_WIDTH_KPC = 0.20
SOURCE_SEPARATION_EDGES = np.arange(0.0, 2.52 + 1.0e-9, 0.12)
SOURCE_SEPARATION_CENTRES = 0.5 * (
    SOURCE_SEPARATION_EDGES[:-1] + SOURCE_SEPARATION_EDGES[1:]
)


@dataclass(frozen=True)
class GalaxyConfig:
    name: str
    catalogue: Path
    output_dir: Path
    r25_kpc: float

    @property
    def stem(self) -> str:
        return self.name.lower()

    @property
    def map_path(self) -> Path:
        return self.output_dir / (
            f"{self.stem}_muse_scal_metallicity_residual_pixel_map.npz"
        )

    @property
    def morphology_path(self) -> Path:
        return self.output_dir / f"{self.stem}_morphology_summary.json"

    @property
    def chain_path(self) -> Path:
        return self.output_dir / f"{self.stem}_transport_chain.npz"


GALAXIES = (
    GalaxyConfig(
        "NGC628", ROOT / "data/phangs/NGC0628_catalog.fits",
        OUTPUT_ROOT, 13.96846123970602,
    ),
    GalaxyConfig(
        "NGC1566", ROOT / "data/phangs/NGC1566/NGC1566_catalog.fits",
        OUTPUT_ROOT / "NGC1566", 18.5354034316284,
    ),
    GalaxyConfig(
        "NGC4254", ROOT / "data/phangs/NGC4254/NGC4254_catalog.fits",
        OUTPUT_ROOT / "NGC4254", 9.526590513180578,
    ),
    GalaxyConfig(
        "NGC4303", ROOT / "data/phangs/NGC4303/NGC4303_catalog.fits",
        OUTPUT_ROOT / "NGC4303", 16.81334394292804,
    ),
)


def load_npz(path: Path, omit: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Materialise a compressed archive so repeated interpolation is fast."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files if key not in omit}


def numeric_column(table: Table, name: str) -> np.ndarray:
    return np.asarray(table[name], dtype=float)


def self_pair_histogram(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Unique unordered-pair histogram from cumulative KD-tree counts."""
    tree = cKDTree(points)
    cumulative = tree.count_neighbors(tree, edges, cumulative=True)
    return np.diff((cumulative - len(points)) / 2.0)


def cross_pair_histogram(
    first: np.ndarray, second: np.ndarray, edges: np.ndarray,
) -> np.ndarray:
    cumulative = cKDTree(first).count_neighbors(
        cKDTree(second), edges, cumulative=True,
    )
    return np.diff(cumulative)


def landy_szalay(
    data_points: np.ndarray,
    random_points: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Landy--Szalay excess correlation for an inhomogeneous finite window."""
    n_data = len(data_points)
    n_random = len(random_points)
    dd = self_pair_histogram(data_points, edges) / (
        n_data * (n_data - 1.0) / 2.0
    )
    rr = self_pair_histogram(random_points, edges) / (
        n_random * (n_random - 1.0) / 2.0
    )
    dr = cross_pair_histogram(data_points, random_points, edges) / (
        n_data * n_random
    )
    return np.divide(
        dd - 2.0 * dr + rr,
        rr,
        out=np.full_like(rr, np.nan, dtype=float),
        where=rr > 0.0,
    )


def fit_gaussian_excess_curve(
    curve: np.ndarray, covariance: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit A exp[-r^2/(2 ell^2)] above the catalogue deblending scale."""
    selected = (
        (SOURCE_SEPARATION_CENTRES >= 0.18)
        & (SOURCE_SEPARATION_CENTRES <= 2.46)
        & np.isfinite(curve)
        & np.isfinite(np.diag(covariance))
    )
    fit_covariance = covariance[np.ix_(selected, selected)]
    fit_covariance = (
        0.50 * fit_covariance
        + 0.50 * np.diag(np.diag(fit_covariance))
        + 1.0e-5 * np.eye(np.sum(selected))
    )
    cholesky = np.linalg.cholesky(fit_covariance)
    radii = SOURCE_SEPARATION_CENTRES[selected]

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        amplitude, length = np.exp(log_parameters)
        model = amplitude * np.exp(-0.5 * (radii / length) ** 2)
        return np.linalg.solve(cholesky, curve[selected] - model)

    result = least_squares(
        residual,
        np.log([0.15, 0.60]),
        bounds=(np.log([1.0e-3, 0.06]), np.log([2.0, 3.0])),
    )
    dof = max(1, int(np.sum(selected)) - 2)
    return np.exp(result.x), float(np.sum(result.fun**2) / dof)


def source_posterior_summary(
    config: GalaxyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the source-pair excess specified by the fitted KTZ parameters."""
    chain_data = load_npz(config.chain_path)
    chain = chain_data["internal_chain"]
    burn = int(chain_data["burn_steps"])
    samples = chain[burn::5].reshape(-1, chain.shape[-1])
    rng = np.random.default_rng(91_000 + int(config.name[3:]))
    indices = rng.choice(len(samples), min(4000, len(samples)), replace=False)
    samples = samples[indices]
    amplitude = np.exp(samples[:, 5] - samples[:, 2])
    length = np.exp(samples[:, 6])
    curves = amplitude[:, None] * np.exp(
        -0.5 * (SOURCE_SEPARATION_CENTRES[None, :] / length[:, None]) ** 2
    )
    return (
        curves,
        np.percentile(amplitude, [16, 50, 84]),
        np.percentile(length, [16, 50, 84]),
    )


def source_pair_diagnostic(config: GalaxyConfig) -> dict[str, object]:
    """Measure H II-region excess pairs after conditioning on KTZ geometry."""
    catalogue = Table.read(config.catalogue)
    quality = (
        (numeric_column(catalogue, "FLAG_EDGE") == 0)
        & (numeric_column(catalogue, "FLAG_STAR") == 0)
        & (numeric_column(catalogue, "BPT_NII") == 0)
        & np.isfinite(numeric_column(catalogue, "HA6562_LUM_CORR"))
        & (numeric_column(catalogue, "HA6562_LUM_CORR") > 0)
        & (numeric_column(catalogue, "DEPROJ_R_R25") >= 0.10)
        & (numeric_column(catalogue, "DEPROJ_R_R25") <= 0.40)
    )
    radius = numeric_column(catalogue, "DEPROJ_R_R25")[quality] * config.r25_kpc
    azimuth = np.mod(numeric_column(catalogue, "DEPROJ_PHI")[quality], 2.0 * np.pi)
    observed_points = np.column_stack((
        radius * np.cos(azimuth), radius * np.sin(azimuth),
    ))

    map_data = load_npz(config.map_path)
    morphology = json.loads(config.morphology_path.read_text(encoding="utf-8"))
    x_grid = map_data["x_grid_kpc"]
    y_grid = map_data["y_grid_kpc"]
    radius_grid = np.hypot(x_grid, y_grid)
    source_window = np.isfinite(map_data["halpha_flux"])
    source_modulation = map_data["arm_source_modulation"]
    rate_log_slope = float(morphology["event_rate_log_slope_kpc_inv"])
    reference_radius = float(morphology["r_ref_kpc"])
    intensity = np.where(
        source_window,
        np.exp(rate_log_slope * (radius_grid - reference_radius))
        * (1.0 + source_modulation),
        0.0,
    )
    intensity = np.clip(intensity, 0.0, None)
    valid_indices = np.flatnonzero(source_window)
    probabilities = intensity.ravel()[valid_indices]
    probabilities = probabilities / np.sum(probabilities)

    rng = np.random.default_rng(75_210 + int(config.name[3:]))
    n_random = 8 * len(observed_points)
    random_indices = rng.choice(
        valid_indices, size=n_random, replace=True, p=probabilities,
    )
    random_points = np.column_stack((
        x_grid.ravel()[random_indices]
        + (rng.random(n_random) - 0.5) * PIXEL_SIZE_KPC,
        y_grid.ravel()[random_indices]
        + (rng.random(n_random) - 0.5) * PIXEL_SIZE_KPC,
    ))
    random_azimuth = np.mod(
        np.arctan2(random_points[:, 1], random_points[:, 0]), 2.0 * np.pi,
    )

    observed_curve = landy_szalay(
        observed_points, random_points, SOURCE_SEPARATION_EDGES,
    )
    observed_sector = np.floor(12.0 * azimuth / (2.0 * np.pi)).astype(int)
    random_sector = np.floor(
        12.0 * random_azimuth / (2.0 * np.pi)
    ).astype(int)
    jackknife_curves = []
    for sector in range(12):
        jackknife_curves.append(landy_szalay(
            observed_points[observed_sector != sector],
            random_points[random_sector != sector],
            SOURCE_SEPARATION_EDGES,
        ))
    jackknife_curves = np.asarray(jackknife_curves)
    difference = jackknife_curves - np.nanmean(jackknife_curves, axis=0)
    difference = np.nan_to_num(difference, nan=0.0)
    covariance = 11.0 / 12.0 * difference.T @ difference
    uncertainty = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    fitted_parameters, reduced_quadratic = fit_gaussian_excess_curve(
        observed_curve, covariance,
    )

    jackknife_parameters = np.asarray([
        fit_gaussian_excess_curve(curve, covariance)[0]
        for curve in jackknife_curves
    ])
    log_jackknife = np.log(jackknife_parameters)
    log_mean = np.mean(log_jackknife, axis=0)
    log_standard_error = np.sqrt(
        11.0 / 12.0
        * np.sum((log_jackknife - log_mean[None, :]) ** 2, axis=0)
    )
    fitted_interval = np.exp(
        np.log(fitted_parameters)[None, :]
        + np.asarray([-1.0, 0.0, 1.0])[:, None]
        * log_standard_error[None, :]
    )

    (posterior_curves, posterior_amplitude_interval,
     posterior_length_interval) = source_posterior_summary(config)
    posterior_interval = np.percentile(posterior_curves, [16, 50, 84], axis=0)
    output_table = config.output_dir / f"{config.stem}_hii_source_pair_test.csv"
    Table({
        "r_kpc": SOURCE_SEPARATION_CENTRES,
        "hii_landy_szalay": observed_curve,
        "hii_jackknife_error": uncertainty,
        "ktz_source_excess_p16": posterior_interval[0],
        "ktz_source_excess_median": posterior_interval[1],
        "ktz_source_excess_p84": posterior_interval[2],
        "included_in_gaussian_fit": SOURCE_SEPARATION_CENTRES >= 0.18,
    }).write(output_table, format="ascii.csv", overwrite=True)

    return {
        "config": config,
        "n_hii": len(observed_points),
        "curve": observed_curve,
        "uncertainty": uncertainty,
        "posterior_interval": posterior_interval,
        "posterior_amplitude_interval": posterior_amplitude_interval,
        "posterior_length_interval": posterior_length_interval,
        "fit": fitted_parameters,
        "fit_interval": fitted_interval,
        "fit_reduced_quadratic": reduced_quadratic,
        "output_table": output_table,
    }


def annular_residual_map(
    values: np.ndarray, valid: np.ndarray, radius: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    annulus = np.floor(radius / ANNULUS_WIDTH_KPC).astype(int)
    residual = np.full(values.shape, np.nan, dtype=float)
    retained = valid.copy()
    for label in np.unique(annulus[valid]):
        selected = valid & (annulus == label)
        if np.sum(selected) < 20:
            retained[selected] = False
            continue
        residual[selected] = values[selected] - np.mean(values[selected])
    return residual, retained & np.isfinite(residual)


def metallicity_jackknife_covariance(config: GalaxyConfig) -> np.ndarray:
    """Rebuild the spatial covariance used by the notebook from saved maps."""
    data = load_npz(config.map_path)
    abundance = data["metallicity_12_log_oh"]
    valid_base = data["valid_mask"].astype(bool) & np.isfinite(abundance)
    x_grid = data["x_grid_kpc"]
    y_grid = data["y_grid_kpc"]
    radius = np.hypot(x_grid, y_grid)
    azimuth = np.mod(np.arctan2(y_grid, x_grid), 2.0 * np.pi)
    separation_edges = np.arange(
        0.5 * PIXEL_SIZE_KPC,
        5.0 + 0.5 * PIXEL_SIZE_KPC + 1.0e-9,
        PIXEL_SIZE_KPC,
    )
    n_bins = len(separation_edges) - 1
    lag_y = np.arange(-(abundance.shape[0] - 1), abundance.shape[0]) * PIXEL_SIZE_KPC
    lag_x = np.arange(-(abundance.shape[1] - 1), abundance.shape[1]) * PIXEL_SIZE_KPC
    lag_x_grid, lag_y_grid = np.meshgrid(lag_x, lag_y)
    lag_radius = np.hypot(lag_x_grid, lag_y_grid)
    lag_bin = np.digitize(lag_radius.ravel(), separation_edges) - 1
    selected_lag = (
        (lag_bin >= 0) & (lag_bin < n_bins) & (lag_radius.ravel() > 0.0)
    )

    def curve(retained_region: np.ndarray) -> np.ndarray:
        valid = valid_base & retained_region
        residual, valid = annular_residual_map(abundance, valid, radius)
        residual = residual - np.mean(residual[valid])
        mask = valid.astype(float)
        field = np.where(valid, residual, 0.0)
        power = field**2
        pair_grid = fftconvolve(mask, mask[::-1, ::-1], mode="full")
        numerator_grid = fftconvolve(
            field, field[::-1, ::-1], mode="full",
        )
        endpoint_grid = 0.5 * (
            fftconvolve(power, mask[::-1, ::-1], mode="full")
            + fftconvolve(mask, power[::-1, ::-1], mode="full")
        )
        bins = lag_bin[selected_lag]
        numerator = np.bincount(
            bins, weights=numerator_grid.ravel()[selected_lag], minlength=n_bins,
        )
        endpoint = np.bincount(
            bins, weights=endpoint_grid.ravel()[selected_lag], minlength=n_bins,
        )
        return np.divide(
            numerator, endpoint, out=np.full(n_bins, np.nan), where=endpoint > 0,
        )

    azimuth_block = np.floor(12.0 * azimuth / (2.0 * np.pi)).astype(int)
    radial_edges = np.sqrt(np.linspace(
        (0.10 * config.r25_kpc) ** 2,
        (0.40 * config.r25_kpc) ** 2,
        5,
    ))
    radial_block = np.clip(np.digitize(radius, radial_edges) - 1, 0, 3)
    spatial_block = 12 * radial_block + azimuth_block
    labels = np.unique(spatial_block[valid_base])
    curves = np.asarray([curve(spatial_block != label) for label in labels])
    difference = np.nan_to_num(curves - np.nanmean(curves, axis=0), nan=0.0)
    return (len(labels) - 1.0) / len(labels) * difference.T @ difference


class SavedKTZModel:
    """Fast KTZ plus isotropic-C_NP evaluator backed by saved emulators."""

    def __init__(self, output_dir: Path, stem: str):
        self.output_dir = output_dir
        self.stem = stem
        self.poisson = load_npz(
            output_dir / f"{stem}_muse_scal_7p_ktz_full_pair_emulator.npz"
        )
        self.cnp = load_npz(
            output_dir / f"{stem}_muse_scal_7p_ktz_cnp_full_pair_emulator.npz"
        )
        self.transfer = load_npz(
            output_dir / f"{stem}_annular_detrending_transfer.npz",
            omit=("signature",),
        )
        self.diagnostics = np.genfromtxt(
            output_dir / f"{stem}_residual_covariance_diagnostics.csv",
            delimiter=",", names=True,
        )
        self.poisson_names = tuple(self.poisson["basis_names"].astype(str))
        self.cnp_names = tuple(self.cnp["basis_names"].astype(str))

    def __call__(self, internal: np.ndarray) -> np.ndarray | None:
        ell = float(np.exp(internal[0]))
        epsilon = float(1.0 / (1.0 + np.exp(-internal[1])))
        beta = float(np.exp(internal[2]))
        eta = float(np.exp(internal[3]))
        rate_log_slope = float(internal[4])
        source_amplitude = float(np.exp(internal[5] - internal[2]))
        source_length = float(np.exp(internal[6]))
        intrinsic_sigma2 = (epsilon * ell) ** 2
        diffusion_extent2 = ell**2 - intrinsic_sigma2
        observed_sigma2 = intrinsic_sigma2 + 0.5 * (
            TARGET_FWHM_KPC / 2.354820045
        ) ** 2
        observed_ell = np.sqrt(observed_sigma2 + diffusion_extent2)
        observed_epsilon = np.sqrt(observed_sigma2) / observed_ell
        basis = interpolate_regular_basis(
            (
                self.poisson["ell_grid"], self.poisson["epsilon_grid"],
                self.poisson["rate_log_slope_grid"],
            ),
            (observed_ell, observed_epsilon, rate_log_slope),
            self.poisson["basis_grid"], self.poisson_names,
            (True, True, False),
        )
        cnp_basis = interpolate_regular_basis(
            (
                self.cnp["ell_grid"], self.cnp["epsilon_grid"],
                self.cnp["rate_log_slope_grid"], self.cnp["source_length_grid"],
            ),
            (observed_ell, observed_epsilon, rate_log_slope, source_length),
            self.cnp["basis_grid"], self.cnp_names,
            (True, True, False, True),
        )
        if basis is None or cnp_basis is None:
            return None
        intrinsic = correlation_from_template_basis(
            basis, beta, eta, source_amplitude, cnp_basis,
        )
        projected = apply_annular_detrending_transfer(
            intrinsic, self.transfer, zero_lag=1.0,
        )
        return (
            self.diagnostics["noise_signal_fraction"] * projected
            + self.diagnostics["noise_additive_bias"]
        )


def stable_importance_weights(log_weight: np.ndarray) -> tuple[np.ndarray, float]:
    log_weight = log_weight - np.max(log_weight)
    weight = np.exp(log_weight)
    weight = weight / np.sum(weight)
    return weight, float(1.0 / np.sum(weight**2))


def weighted_percentile(
    values: np.ndarray, percentiles: tuple[float, ...], weights: np.ndarray,
) -> np.ndarray:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return np.interp(np.asarray(percentiles) / 100.0, cumulative, values[order])


def heldout_tail_diagnostic() -> dict[str, object]:
    """Condition NGC 4303 only on r <= 3 kpc and predict its negative tail."""
    config = next(item for item in GALAXIES if item.name == "NGC4303")
    jackknife_covariance = metallicity_jackknife_covariance(config)
    primary_diagnostics = np.genfromtxt(
        config.output_dir / "ngc4303_residual_covariance_diagnostics.csv",
        delimiter=",", names=True,
    )
    measurement_variance = primary_diagnostics["line_flux_mc_error"] ** 2
    selection_response = np.genfromtxt(
        config.output_dir / "ngc4303_selection_response.csv",
        delimiter=",", names=True,
    )
    selection_bias = selection_response["dynamic_minus_fixed_mean"]
    raw_covariance = (
        jackknife_covariance + np.diag(measurement_variance)
        + np.outer(selection_bias, selection_bias)
    )
    fit_covariance = (
        0.65 * raw_covariance
        + 0.35 * np.diag(np.diag(raw_covariance))
        + 0.010**2 * np.eye(len(raw_covariance))
    )
    radius = primary_diagnostics["r_kpc"]
    observed = primary_diagnostics["xi_centered"]
    training = radius < 3.0
    tail = (radius >= 3.3) & (radius <= 4.5)
    inverse_full = np.linalg.inv(fit_covariance)
    inverse_training = np.linalg.inv(fit_covariance[np.ix_(training, training)])

    geometry_specs = (
        ("PHANGS mask", "mask", config.output_dir, 0),
        (
            "direct H-alpha", "direct_halpha",
            OUTPUT_ROOT / "NGC4303_halpha_map_systematic", 1,
        ),
        (
            "H-alpha harmonic", "halpha_harmonic",
            OUTPUT_ROOT / "NGC4303_logspiral_systematic", 2,
        ),
    )
    results: dict[str, object] = {
        "r_kpc": radius,
        "observed": observed,
        "error": np.sqrt(np.diag(raw_covariance)),
        "training": training,
        "tail": tail,
    }
    output_columns: dict[str, np.ndarray] = {
        "r_kpc": radius,
        "xi_observed": observed,
        "xi_error": np.sqrt(np.diag(raw_covariance)),
        "used_for_training": training,
        "held_out_tail": tail,
    }
    summary_rows = []
    for geometry_label, safe_name, output_dir, seed_offset in geometry_specs:
        model = SavedKTZModel(output_dir, "ngc4303")
        chain_data = load_npz(output_dir / "ngc4303_transport_chain.npz")
        chain = chain_data["internal_chain"]
        burn = int(chain_data["burn_steps"])
        samples = chain[burn::5].reshape(-1, chain.shape[-1])
        rng = np.random.default_rng(43_030 + seed_offset)
        indices = rng.choice(len(samples), min(20_000, len(samples)), replace=False)
        samples = samples[indices]
        predictions = np.empty((len(samples), len(radius)))
        log_weight = np.empty(len(samples))
        for index, sample in enumerate(samples):
            prediction = model(sample)
            if prediction is None:
                raise RuntimeError("Saved posterior sample lies outside its emulator.")
            predictions[index] = prediction
            full_residual = observed - prediction
            training_residual = full_residual[training]
            full_quadratic = full_residual @ inverse_full @ full_residual
            training_quadratic = (
                training_residual @ inverse_training @ training_residual
            )
            log_weight[index] = 0.5 * (full_quadratic - training_quadratic)
        weights, effective_sample_size = stable_importance_weights(log_weight)

        prediction_interval = np.asarray([
            weighted_percentile(predictions[:, column], (16, 50, 84), weights)
            for column in range(predictions.shape[1])
        ]).T
        median_prediction = prediction_interval[1]
        tail_mean_draws = np.mean(predictions[:, tail], axis=1)
        tail_mean_interval = weighted_percentile(
            tail_mean_draws, (16, 50, 84), weights,
        )
        tail_covariance = fit_covariance[np.ix_(tail, tail)]
        tail_inverse = np.linalg.inv(tail_covariance)
        tail_residual = observed[tail] - median_prediction[tail]
        tail_quadratic = float(tail_residual @ tail_inverse @ tail_residual)
        for percentile_index, percentile in enumerate((16, 50, 84)):
            output_columns[f"{safe_name}_train_only_p{percentile}"] = (
                prediction_interval[percentile_index]
            )
        results[safe_name] = {
            "interval": prediction_interval,
            "effective_sample_size": effective_sample_size,
            "tail_mean_interval": tail_mean_interval,
            "tail_quadratic": tail_quadratic,
        }
        summary_rows.append((
            geometry_label, effective_sample_size, *tail_mean_interval,
            tail_quadratic,
        ))

    kt18_chain_data = load_npz(config.output_dir / "ngc4303_kt18_transport_chain.npz")
    kt18_chain = kt18_chain_data["internal_chain"]
    kt18_burn = int(kt18_chain_data["burn_steps"])
    kt18_samples = kt18_chain[kt18_burn::2].reshape(-1, 2)

    def kt18_curve(internal: np.ndarray) -> np.ndarray:
        ell = float(np.exp(internal[0]))
        epsilon = float(1.0 / (1.0 + np.exp(-internal[1])))
        sigma2 = (epsilon * ell) ** 2 + 0.5 * (
            TARGET_FWHM_KPC / 2.354820045
        ) ** 2
        observed_ell = np.sqrt(sigma2 + ell**2 - (epsilon * ell) ** 2)
        observed_epsilon = np.sqrt(sigma2) / observed_ell
        rho = radius / observed_ell
        from scipy.special import exp1
        intrinsic = (
            exp1(rho**2 / 4.0)
            - exp1(rho**2 / (4.0 * observed_epsilon**2))
        ) / (-2.0 * np.log(observed_epsilon))
        transfer = load_npz(
            config.output_dir / "ngc4303_annular_detrending_transfer.npz",
            omit=("signature",),
        )
        projected = apply_annular_detrending_transfer(
            intrinsic, transfer, zero_lag=1.0,
        )
        return (
            primary_diagnostics["noise_signal_fraction"] * projected
            + primary_diagnostics["noise_additive_bias"]
        )

    kt18_predictions = np.asarray([kt18_curve(sample) for sample in kt18_samples])
    kt18_log_weight = np.empty(len(kt18_samples))
    for index, prediction in enumerate(kt18_predictions):
        full_residual = observed - prediction
        training_residual = full_residual[training]
        kt18_log_weight[index] = 0.5 * (
            full_residual @ inverse_full @ full_residual
            - training_residual @ inverse_training @ training_residual
        )
    kt18_weights, kt18_ess = stable_importance_weights(kt18_log_weight)
    kt18_interval = np.asarray([
        weighted_percentile(kt18_predictions[:, column], (16, 50, 84), kt18_weights)
        for column in range(kt18_predictions.shape[1])
    ]).T
    for percentile_index, percentile in enumerate((16, 50, 84)):
        output_columns[f"kt18_train_only_p{percentile}"] = (
            kt18_interval[percentile_index]
        )
    kt18_tail_mean = weighted_percentile(
        np.mean(kt18_predictions[:, tail], axis=1), (16, 50, 84), kt18_weights,
    )
    kt18_tail_residual = observed[tail] - kt18_interval[1, tail]
    kt18_tail_inverse = np.linalg.inv(fit_covariance[np.ix_(tail, tail)])
    kt18_tail_quadratic = float(
        kt18_tail_residual @ kt18_tail_inverse @ kt18_tail_residual
    )
    results["kt18"] = {
        "interval": kt18_interval,
        "effective_sample_size": kt18_ess,
        "tail_mean_interval": kt18_tail_mean,
        "tail_quadratic": kt18_tail_quadratic,
    }
    summary_rows.append((
        "KT18", kt18_ess, *kt18_tail_mean, kt18_tail_quadratic,
    ))
    results["observed_tail_mean"] = float(np.mean(observed[tail]))

    Table(output_columns).write(
        OUTPUT_ROOT / "ngc4303_heldout_tail_prediction.csv",
        format="ascii.csv", overwrite=True,
    )
    Table(
        rows=summary_rows,
        names=(
            "model_or_geometry", "importance_effective_sample_size",
            "tail_mean_p16", "tail_mean_median", "tail_mean_p84",
            "tail_quadratic_at_median",
        ),
    ).write(
        OUTPUT_ROOT / "ngc4303_heldout_tail_summary.csv",
        format="ascii.csv", overwrite=True,
    )
    return results


def make_figure(
    source_results: list[dict[str, object]], heldout: dict[str, object],
) -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
    })
    figure, axes = plt.subplots(
        2, 3, figsize=(10.8, 6.5), constrained_layout=True,
    )
    source_axes = axes.ravel()[:4]
    for axis, result in zip(source_axes, source_results):
        config = result["config"]
        interval = result["posterior_interval"]
        axis.axhline(0.0, color="0.55", linewidth=0.7)
        axis.fill_between(
            SOURCE_SEPARATION_CENTRES, interval[0], interval[2],
            color="#beaed4", alpha=0.32, linewidth=0,
            label=r"metal-inferred $C_{\rm NP}$ 16--84%",
        )
        axis.plot(
            SOURCE_SEPARATION_CENTRES, interval[1], color="#7561a8",
            linewidth=1.7, label=r"metal-inferred $C_{\rm NP}$ median",
        )
        axis.errorbar(
            SOURCE_SEPARATION_CENTRES[1:], result["curve"][1:],
            yerr=result["uncertainty"][1:], fmt="o", ms=2.8,
            color="black", ecolor="0.35", elinewidth=0.8, capsize=1.5,
            label="H II-region excess pairs",
        )
        amplitude, length = result["fit"]
        fit_curve = amplitude * np.exp(
            -0.5 * (SOURCE_SEPARATION_CENTRES / length) ** 2
        )
        axis.plot(
            SOURCE_SEPARATION_CENTRES, fit_curve, color="#1b9e77",
            linestyle="--", linewidth=1.3, label="H II Gaussian fit",
        )
        axis.set_title(
            f"{config.name.replace('NGC', 'NGC ')}: "
            rf"$\ell_{{\rm HII}}={length:.2f}$ kpc"
        )
        axis.set_xlim(0.10, 2.50)
        axis.set_ylim(-0.10, 0.55)
        axis.set_xlabel(r"separation $r$ [kpc]")
        axis.set_ylabel(r"source excess $g_{\rm src}(r)-1$")
    handles, labels = source_axes[0].get_legend_handles_labels()
    source_axes[0].legend(handles, labels, loc="upper right", frameon=False)

    axis = axes[1, 1]
    radius = heldout["r_kpc"]
    axis.axhline(0.0, color="0.55", linewidth=0.7)
    axis.axvspan(0.0, 3.0, color="0.92", label="fit range")
    axis.axvspan(3.3, 4.5, color="#fdb863", alpha=0.16, label="held-out tail")
    for key, color, linestyle, label in (
        ("mask", "#4daf4a", "-", "KTZ: PHANGS mask"),
        ("direct_halpha", "#7b3294", "--", r"KTZ: direct H$\alpha$"),
        (
            "halpha_harmonic", "#d95f02", "-.",
            r"KTZ: H$\alpha$ harmonic",
        ),
        ("kt18", "#377eb8", ":", "KT18"),
    ):
        interval = heldout[key]["interval"]
        axis.fill_between(
            radius, interval[0], interval[2], color=color, alpha=0.12,
            linewidth=0,
        )
        axis.plot(radius, interval[1], color=color, linestyle=linestyle,
                  linewidth=1.6, label=label)
    axis.errorbar(
        radius, heldout["observed"], yerr=heldout["error"], fmt="o",
        ms=2.2, color="black", ecolor="0.4", elinewidth=0.6, capsize=1,
        label="NGC 4303 data",
    )
    axis.set_xlim(0.0, 5.0)
    axis.set_ylim(-0.14, 1.02)
    axis.set_xlabel(r"separation $r$ [kpc]")
    axis.set_ylabel(r"$\xi_{3,{\rm pp}}(r)$")
    axis.set_title("NGC 4303: train below 3 kpc, predict the tail")
    axis.legend(loc="upper right", frameon=False, ncol=1)

    axis = axes[1, 2]
    x = np.arange(4)
    hii_lengths = np.asarray([result["fit"][1] for result in source_results])
    hii_intervals = np.asarray([
        result["fit_interval"][:, 1] for result in source_results
    ])
    ktz_intervals = np.asarray([
        result["posterior_length_interval"] for result in source_results
    ])
    ktz_lengths = ktz_intervals[:, 1]
    axis.errorbar(
        x - 0.08, hii_lengths,
        yerr=np.vstack((
            hii_lengths - hii_intervals[:, 0],
            hii_intervals[:, 2] - hii_lengths,
        )),
        fmt="o", color="#1b9e77", capsize=2.5,
        label=r"H II pair scale $\ell_{\rm HII}$",
    )
    axis.errorbar(
        x + 0.08, ktz_lengths,
        yerr=np.vstack((
            ktz_lengths - ktz_intervals[:, 0],
            ktz_intervals[:, 2] - ktz_lengths,
        )),
        fmt="s", color="#7561a8", capsize=2.5,
        label=r"KTZ $\ell_{\rm src}$",
    )
    axis.set_xticks(x, ["628", "1566", "4254", "4303"])
    axis.set_yscale("log")
    axis.set_ylim(0.025, 3.2)
    axis.set_xlabel("NGC")
    axis.set_ylabel("source-clustering scale [kpc]")
    axis.set_title("Descriptive ordering of source-pair scales")
    axis.legend(loc="upper right", frameon=False)

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_ROOT / "phangs_ktz_physical_validation.pdf")
    figure.savefig(
        OUTPUT_ROOT / "phangs_ktz_physical_validation.png", dpi=220,
    )
    plt.close(figure)

    for result in source_results:
        config = result["config"]
        is_ngc4303 = config.name == "NGC4303"
        notebook_figure, notebook_axes = plt.subplots(
            1, 2 if is_ngc4303 else 1,
            figsize=(10.5 if is_ngc4303 else 5.4, 4.1),
            constrained_layout=True,
        )
        source_axis = (
            notebook_axes[0] if is_ngc4303 else notebook_axes
        )
        interval = result["posterior_interval"]
        source_axis.axhline(0.0, color="0.55", linewidth=0.7)
        source_axis.fill_between(
            SOURCE_SEPARATION_CENTRES, interval[0], interval[2],
            color="#beaed4", alpha=0.32, linewidth=0,
            label=r"metal-inferred $C_{\rm NP}$ 16--84%",
        )
        source_axis.plot(
            SOURCE_SEPARATION_CENTRES, interval[1], color="#7561a8",
            linewidth=1.7, label=r"metal-inferred $C_{\rm NP}$ median",
        )
        source_axis.errorbar(
            SOURCE_SEPARATION_CENTRES[1:], result["curve"][1:],
            yerr=result["uncertainty"][1:], fmt="o", ms=3.0,
            color="black", ecolor="0.35", elinewidth=0.8, capsize=1.5,
            label="H II-region excess pairs",
        )
        amplitude, length = result["fit"]
        source_axis.plot(
            SOURCE_SEPARATION_CENTRES,
            amplitude * np.exp(
                -0.5 * (SOURCE_SEPARATION_CENTRES / length) ** 2
            ),
            color="#1b9e77", linestyle="--", linewidth=1.3,
            label="H II Gaussian fit",
        )
        source_axis.set(
            xlim=(0.10, 2.50), ylim=(-0.10, 0.55),
            xlabel=r"separation $r$ [kpc]",
            ylabel=r"source excess $g_{\rm src}(r)-1$",
            title=(
                f"{config.name.replace('NGC', 'NGC ')}: exploratory source-pair comparison"
            ),
        )
        source_axis.legend(loc="upper right", frameon=False)

        if is_ngc4303:
            tail_axis = notebook_axes[1]
            radius = heldout["r_kpc"]
            tail_axis.axhline(0.0, color="0.55", linewidth=0.7)
            tail_axis.axvspan(0.0, 3.0, color="0.92", label="fit range")
            tail_axis.axvspan(
                3.3, 4.5, color="#fdb863", alpha=0.16,
                label="held-out tail",
            )
            for key, color, linestyle, label in (
                ("mask", "#4daf4a", "-", "KTZ: PHANGS mask"),
                (
                    "direct_halpha", "#7b3294", "--",
                    r"KTZ: direct H$\alpha$",
                ),
                (
                    "halpha_harmonic", "#d95f02", "-.",
                    r"KTZ: H$\alpha$ harmonic",
                ),
                ("kt18", "#377eb8", ":", "KT18"),
            ):
                model_interval = heldout[key]["interval"]
                tail_axis.fill_between(
                    radius, model_interval[0], model_interval[2],
                    color=color, alpha=0.12, linewidth=0,
                )
                tail_axis.plot(
                    radius, model_interval[1], color=color,
                    linestyle=linestyle, linewidth=1.6, label=label,
                )
            tail_axis.errorbar(
                radius, heldout["observed"], yerr=heldout["error"],
                fmt="o", ms=2.2, color="black", ecolor="0.4",
                elinewidth=0.6, capsize=1, label="NGC 4303 data",
            )
            tail_axis.set(
                xlim=(0.0, 5.0), ylim=(-0.14, 1.02),
                xlabel=r"separation $r$ [kpc]",
                ylabel=r"$\xi_{3,{\rm pp}}(r)$",
                title="Fit below 3 kpc; predict the negative tail",
            )
            tail_axis.legend(loc="upper right", frameon=False)
        notebook_figure.savefig(
            config.output_dir / f"{config.stem}_physical_validation.png",
            dpi=180,
        )
        plt.close(notebook_figure)


def write_summary(
    source_results: list[dict[str, object]], heldout: dict[str, object],
) -> None:
    rows = []
    for result in source_results:
        config = result["config"]
        length_interval = result["posterior_length_interval"]
        amplitude_interval = result["posterior_amplitude_interval"]
        rows.append((
            config.name, result["n_hii"],
            result["fit"][0], result["fit_interval"][0, 0],
            result["fit_interval"][2, 0],
            result["fit"][1], result["fit_interval"][0, 1],
            result["fit_interval"][2, 1],
            amplitude_interval[1], amplitude_interval[0], amplitude_interval[2],
            length_interval[1], length_interval[0], length_interval[2],
            result["fit_reduced_quadratic"],
        ))
    Table(
        rows=rows,
        names=(
            "galaxy", "n_hii", "hii_amplitude", "hii_amplitude_lo",
            "hii_amplitude_hi", "hii_length_kpc", "hii_length_lo_kpc",
            "hii_length_hi_kpc", "ktz_A_src_median", "ktz_A_src_p16",
            "ktz_A_src_p84", "ktz_ell_src_median_kpc",
            "ktz_ell_src_p16_kpc", "ktz_ell_src_p84_kpc",
            "hii_gaussian_reduced_quadratic",
        ),
    ).write(
        OUTPUT_ROOT / "phangs_hii_source_pair_summary.csv",
        format="ascii.csv", overwrite=True,
    )

    observed_tail = float(heldout["observed_tail_mean"])
    print(f"NGC 4303 observed held-out tail mean: {observed_tail:+.4f}")
    for key in ("mask", "direct_halpha", "halpha_harmonic", "kt18"):
        result = heldout[key]
        lo, median, hi = result["tail_mean_interval"]
        print(
            f"{key}: held-out mean {median:+.4f} [{lo:+.4f}, {hi:+.4f}], "
            f"importance ESS={result['effective_sample_size']:.1f}, "
            f"tail Q={result['tail_quadratic']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-heldout", action="store_true",
        help="only regenerate the independent H II-region source test",
    )
    arguments = parser.parse_args()
    source_results = [source_pair_diagnostic(config) for config in GALAXIES]
    if arguments.skip_heldout:
        return
    heldout = heldout_tail_diagnostic()
    write_summary(source_results, heldout)
    make_figure(source_results, heldout)


if __name__ == "__main__":
    main()
