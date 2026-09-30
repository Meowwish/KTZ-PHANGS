"""Diagnose whether posterior volume, rather than KTZ support, limits a fit.

For NGC 1566 and NGC 4303 this script separates the prior from the saved
posterior log density, locates the highest-likelihood retained chain state,
and compares it with the pointwise posterior-median curve.  The calculation
is deliberately diagnostic: it does not replace posterior summaries with a
single best-fitting state.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from ktz_physical_validation import (
    GALAXIES,
    SavedKTZModel,
    metallicity_jackknife_covariance,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "implications"
FIGURE_PATH = OUTPUT_ROOT / (
    "ngc1566_ngc4303_maximum_likelihood_diagnostic.png"
)
SUMMARY_PATH = OUTPUT_ROOT / (
    "ngc1566_ngc4303_maximum_likelihood_summary.csv"
)
TAIL_LOWER_KPC = 3.3
TAIL_UPPER_KPC = 4.5


def ktz_log_prior(
    internal: np.ndarray, morphology: dict[str, object],
) -> np.ndarray:
    """Evaluate the non-constant part of the saved seven-parameter prior."""
    epsilon = 1.0 / (1.0 + np.exp(-internal[..., 1]))
    rate_log_slope_mean = float(morphology["event_rate_log_slope_kpc_inv"])
    rate_log_slope_error = float(morphology["event_rate_log_slope_error_kpc_inv"])
    eta_prior_sigma = max(
        float(morphology["mask_contrast_amplitude_error"])
        / float(morphology["mask_contrast_amplitude"]),
        0.35,
    )
    return (
        np.log(2.0) - np.log1p(epsilon)
        - 0.5 * (internal[..., 3] / eta_prior_sigma) ** 2
        - 0.5 * ((internal[..., 4] - rate_log_slope_mean) / rate_log_slope_error) ** 2
    )


def kt18_log_prior(internal: np.ndarray) -> np.ndarray:
    """Evaluate the marginal transport prior used by the KT18 comparison."""
    epsilon = 1.0 / (1.0 + np.exp(-internal[..., 1]))
    ell_star = np.exp(internal[..., 0])
    kappa_t_star = 0.5 * ell_star**2 * (1.0 - epsilon**2)
    log_t_lower = np.maximum(
        np.log(0.05), np.log(kappa_t_star) - np.log(1.50),
    )
    log_t_upper = np.minimum(
        np.log(3.00), np.log(kappa_t_star) - np.log(0.05),
    )
    return (
        np.log(2.0) - np.log1p(epsilon)
        + np.log(log_t_upper - log_t_lower)
    )


def highest_likelihood_state(
    chain_path: Path, prior_function,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return the retained chain state with the smallest pure-likelihood chi2."""
    with np.load(chain_path, allow_pickle=False) as archive:
        burn = int(archive["burn_steps"])
        chain = archive["internal_chain"][burn:]
        log_posterior = archive["log_probability"][burn:]
    log_prior = prior_function(chain)
    chi_square = -2.0 * (log_posterior - log_prior)
    index = np.unravel_index(np.nanargmin(chi_square), chi_square.shape)
    best = np.asarray(chain[index], dtype=float)
    # Percentile ranks expose boundary or small-posterior-volume optima.
    thinned = chain[::20].reshape(-1, chain.shape[-1])
    ranks = np.asarray([
        np.mean(thinned[:, column] <= best[column])
        for column in range(chain.shape[-1])
    ])
    return best, float(chi_square[index]), ranks


def kt18_best_likelihood(galaxy: str) -> tuple[float, np.ndarray]:
    directory = OUTPUT_ROOT / galaxy
    stem = galaxy.lower()
    best, chi_square, _ = highest_likelihood_state(
        directory / f"{stem}_kt18_transport_chain.npz",
        kt18_log_prior,
    )
    return chi_square, best


def covariance_tail_significance(galaxy: str) -> tuple[float, float]:
    """Return the tail mean and its standard error under the fitted covariance."""
    config = next(item for item in GALAXIES if item.name == galaxy)
    diagnostics = np.genfromtxt(
        config.output_dir / f"{config.stem}_residual_covariance_diagnostics.csv",
        delimiter=",", names=True,
    )
    jackknife = metallicity_jackknife_covariance(config)
    measurement = np.diag(diagnostics["line_flux_mc_error"] ** 2)
    raw = jackknife + measurement
    covariance = (
        0.65 * raw + 0.35 * np.diag(np.diag(raw))
        + 0.010**2 * np.eye(len(raw))
    )
    selected = (
        (diagnostics["r_kpc"] >= TAIL_LOWER_KPC)
        & (diagnostics["r_kpc"] <= TAIL_UPPER_KPC)
    )
    tail_covariance = covariance[np.ix_(selected, selected)]
    weights = np.ones(np.count_nonzero(selected))
    standard_error = float(
        np.sqrt(weights @ tail_covariance @ weights) / len(weights)
    )
    return float(np.mean(diagnostics["xi_centered"][selected])), standard_error


def main() -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(12.2, 4.8), sharey=True, constrained_layout=True,
    )
    colours = {
        "PHANGS mask": "#159947",
        "H-alpha spiral": "#d95f02",
    }
    rows: list[tuple[object, ...]] = []

    for axis, galaxy in zip(axes, ("NGC1566", "NGC4303")):
        stem = galaxy.lower()
        primary = OUTPUT_ROOT / galaxy
        observed_table = np.genfromtxt(
            primary / f"{stem}_xi_observed_and_models.csv",
            delimiter=",", names=True,
        )
        radius = observed_table["r_kpc"]
        observed = observed_table["xi_observed"]
        error = observed_table["xi_jackknife_error"]
        tail = (
            (radius >= TAIL_LOWER_KPC) & (radius <= TAIL_UPPER_KPC)
        )
        tail_mean, tail_error = covariance_tail_significance(galaxy)
        kt18_chi_square, _ = kt18_best_likelihood(galaxy)

        axis.errorbar(
            radius, observed, yerr=error, fmt="o", ms=3.2, lw=0.8,
            color="black", ecolor="0.45", label="PHANGS--MUSE", zorder=6,
        )
        axis.plot(
            radius, observed_table["xi_KT18_median"], color="#2878b5",
            lw=1.6, label="KT18 posterior median",
        )

        geometries = (
            ("PHANGS mask", primary),
            ("H-alpha spiral", OUTPUT_ROOT / f"{galaxy}_logspiral_systematic"),
        )
        for geometry_label, directory in geometries:
            model_table = np.genfromtxt(
                directory / f"{stem}_xi_observed_and_models.csv",
                delimiter=",", names=True,
            )
            morphology = json.loads(
                (directory / f"{stem}_morphology_summary.json").read_text()
            )
            best, chi_square, ranks = highest_likelihood_state(
                directory / f"{stem}_transport_chain.npz",
                lambda values: ktz_log_prior(values, morphology),
            )
            model = SavedKTZModel(directory, stem)
            best_curve = np.concatenate(([1.0], model(best)))
            colour = colours[geometry_label]
            axis.plot(
                radius, model_table["xi_KTZ_median"], color=colour,
                lw=1.7, ls="--", label=f"{geometry_label}: posterior median",
            )
            axis.plot(
                radius, best_curve, color=colour, lw=2.2,
                label=f"{geometry_label}: best likelihood",
            )

            reference = float(morphology["regularised_chi2_reference_mean"])
            fit_summary = Table.read(
                directory / f"{stem}_ktz_fit_summary.csv",
                format="ascii.csv",
            )
            median_q = float(
                fit_summary["covariance_calibrated_discrepancy"][0]
            )
            rows.append((
                galaxy, geometry_label, chi_square, chi_square / reference,
                median_q, tail_mean, tail_error,
                float(np.mean(best_curve[tail])),
                kt18_chi_square, chi_square + 2.0 * 7,
                kt18_chi_square + 2.0 * 2,
                float(np.exp(best[0])),
                float(1.0 / (1.0 + np.exp(-best[1]))),
                float(np.exp(best[3])), float(best[4]),
                float(np.exp(best[5] - best[2])), float(np.exp(best[6])),
                float(ranks[3]), float(ranks[4]),
            ))
            del model
            gc.collect()

        axis.axhline(0.0, color="0.55", lw=0.8)
        axis.axvspan(
            TAIL_LOWER_KPC, TAIL_UPPER_KPC, color="0.8", alpha=0.16,
            zorder=0,
        )
        axis.set(
            xlim=(0.0, 5.0), ylim=(-0.13, 1.02),
            xlabel=r"separation $r$ [kpc]", title=galaxy,
        )

    axes[0].set_ylabel(r"$\xi_{3,\mathrm{pp}}(r)$")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 1.11), fontsize=8.5,
    )
    figure.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)

    Table(
        rows=rows,
        names=(
            "galaxy", "geometry", "best_likelihood_chi2",
            "best_likelihood_covariance_calibrated_Q",
            "posterior_median_covariance_calibrated_Q",
            "observed_mean_3p3_4p5", "observed_mean_standard_error",
            "best_likelihood_mean_3p3_4p5", "kt18_best_likelihood_chi2",
            "ktz_aic", "kt18_aic", "ell_star_kpc", "epsilon",
            "eta", "rate_log_slope_kpc_inv", "source_excess_amplitude",
            "ell_src_kpc", "eta_posterior_rank",
            "rate_log_slope_posterior_rank",
        ),
    ).write(SUMMARY_PATH, format="ascii.csv", overwrite=True)
    print(FIGURE_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
