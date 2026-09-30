#!/usr/bin/env python3
"""Add data-dependent selection propagation and null tests to four notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "NGC628_PHANGS_implications.ipynb",
    ROOT / "NGC1566_PHANGS_implications.ipynb",
    ROOT / "NGC4254_PHANGS_implications.ipynb",
    ROOT / "NGC4303_PHANGS_implications.ipynb",
]


SELECTION_HELPER = r'''

def li21_selection_mask(
    flux: dict[str, np.ndarray],
    metallicity: np.ndarray | None = None,
    corrected: dict[str, np.ndarray] | None = None,
    line_sn_min: float = LINE_SN_MIN,
    halpha_ew_min: float = HALPHA_EW_MIN_ANGSTROM,
) -> np.ndarray:
    """Re-evaluate every data-dependent Li21 cut for one flux realisation."""
    if metallicity is None or corrected is None:
        metallicity, _, corrected = scal_metallicity(flux)

    line_quality_draw = np.ones(pixel_X.shape, dtype=bool)
    for name in LINE_EXTENSIONS:
        signal_to_noise = np.divide(
            flux[name], line_error[name],
            out=np.full_like(flux[name], np.nan),
            where=np.isfinite(line_error[name]) & (line_error[name] > 0),
        )
        line_quality_draw &= (
            np.isfinite(flux[name]) & np.isfinite(line_error[name])
            & (signal_to_noise >= line_sn_min)
            & (line_coverage[name] >= LINE_COVERAGE_MIN)
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        bpt_x_draw = np.log10(corrected["nii"] / corrected["ha"])
        bpt_y_draw = np.log10(corrected["oiii"] / corrected["hb"])
        halpha_ew_draw = halpha_ew_scale * flux["ha"] / r_band_flux
    bpt_draw = (
        np.isfinite(bpt_x_draw) & np.isfinite(bpt_y_draw)
        & (bpt_x_draw < 0.40)
        & (bpt_y_draw < 0.61 / (bpt_x_draw - 0.47) + 1.19)
    )
    ew_draw = (
        np.isfinite(halpha_ew_draw) & (halpha_ew_draw >= halpha_ew_min)
        & np.isfinite(r_band_flux) & (r_band_coverage >= LINE_COVERAGE_MIN)
    )
    return (
        geometry_mask & line_quality_draw & bpt_draw & ew_draw
        & np.isfinite(metallicity)
    )


# This identity guards against the Monte Carlo selection drifting away from
# the nominal analysis definition through a duplicated implementation.
nominal_selection_check = li21_selection_mask(
    line_flux, metallicity_grid, corrected_line_flux
)
if not np.array_equal(nominal_selection_check, preliminary_valid_grid):
    raise RuntimeError("Nominal and re-evaluated Li21 selections disagree.")
'''


OLD_MC = r'''metallicity_mc_grid = np.empty(
    (N_FLUX_MC,) + pixel_X.shape, dtype=np.float32
)
for draw_index in range(N_FLUX_MC):
    drawn_flux = {
        name: line_flux[name] + line_error[name] * correlated_unit_noise(
            pixel_X.shape, noise_sigma_grid_pixel, flux_mc_rng
        )
        for name in LINE_EXTENSIONS
    }
    metallicity_mc_grid[draw_index] = scal_metallicity(drawn_flux)[0]
'''


NEW_MC = r'''metallicity_mc_grid = np.empty(
    (N_FLUX_MC,) + pixel_X.shape, dtype=np.float32
)
selection_mc_grid = np.empty(
    (N_FLUX_MC,) + pixel_X.shape, dtype=bool
)
for draw_index in range(N_FLUX_MC):
    drawn_flux = {
        name: line_flux[name] + line_error[name] * correlated_unit_noise(
            pixel_X.shape, noise_sigma_grid_pixel, flux_mc_rng
        )
        for name in LINE_EXTENSIONS
    }
    drawn_metallicity, _, drawn_corrected = scal_metallicity(drawn_flux)
    metallicity_mc_grid[draw_index] = drawn_metallicity
    selection_mc_grid[draw_index] = li21_selection_mask(
        drawn_flux, drawn_metallicity, drawn_corrected
    )
'''


OLD_STAT_SIGNATURE = r'''def fft_pair_product_statistic(
    region_mask: np.ndarray | None = None,
    centre_residual: bool = True,
    metallicity_values: np.ndarray | None = None,
):'''


NEW_STAT_SIGNATURE = r'''def fft_pair_product_statistic(
    region_mask: np.ndarray | None = None,
    centre_residual: bool = True,
    metallicity_values: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
):'''


OLD_STAT_VALID = r'''    if metallicity_values is None:
        metallicity_values = metallicity_grid
    valid = pixel_valid_grid & region_mask & np.isfinite(metallicity_values)
'''


NEW_STAT_VALID = r'''    if metallicity_values is None:
        metallicity_values = metallicity_grid
    if valid_mask is None:
        valid_mask = pixel_valid_grid
    valid = valid_mask & region_mask & np.isfinite(metallicity_values)
'''


OLD_MEASUREMENT_BLOCK = r'''# Re-evaluate the complete nonlinear abundance and detrending pipeline for
# every line-flux Monte Carlo map.  Keep the conservative nominal selection
# fixed; rare non-finite draws are replaced by the nominal value before the
# statistic is formed.
measurement_curves = np.empty((N_FLUX_MC, n_bins))
noise_numerator_means = np.empty((N_FLUX_MC, n_bins))
noise_endpoint_means = np.empty((N_FLUX_MC, n_bins))
for draw_index in range(N_FLUX_MC):
    draw_map = np.asarray(metallicity_mc_grid[draw_index], dtype=float)
    draw_map = np.where(np.isfinite(draw_map), draw_map, metallicity_grid)
    measurement_curves[draw_index] = fft_pair_product_statistic(
        metallicity_values=draw_map
    )[0]
    noise_map = draw_map - metallicity_grid
    noise_residual, _ = annular_residual_map(noise_map, pixel_valid_grid)
    noise_numerator, noise_endpoint, noise_counts = fft_pair_sums(
        noise_residual, pixel_valid_grid & np.isfinite(noise_residual)
    )
    noise_numerator_means[draw_index] = np.divide(
        noise_numerator, noise_counts, out=np.zeros(n_bins),
        where=noise_counts > 0,
    )
    noise_endpoint_means[draw_index] = np.divide(
        noise_endpoint, noise_counts, out=np.zeros(n_bins),
        where=noise_counts > 0,
    )

measurement_covariance = np.cov(measurement_curves, rowvar=False, ddof=1)
'''


NEW_MEASUREMENT_BLOCK = r'''# Re-evaluate the nonlinear abundance, selection, and detrending pipeline for
# every line-flux Monte Carlo realisation.  The fixed-mask curves are retained
# as a paired control; the likelihood covariance uses the dynamic-selection
# curves, so threshold crossings are propagated rather than conditioned away.
fixed_selection_measurement_curves = np.empty((N_FLUX_MC, n_bins))
dynamic_selection_measurement_curves = np.empty((N_FLUX_MC, n_bins))
noise_numerator_means = np.empty((N_FLUX_MC, n_bins))
noise_endpoint_means = np.empty((N_FLUX_MC, n_bins))
selection_pixel_counts = np.empty(N_FLUX_MC, dtype=int)
for draw_index in range(N_FLUX_MC):
    draw_map = np.asarray(metallicity_mc_grid[draw_index], dtype=float)
    fixed_draw_map = np.where(np.isfinite(draw_map), draw_map, metallicity_grid)
    fixed_selection_measurement_curves[draw_index] = (
        fft_pair_product_statistic(metallicity_values=fixed_draw_map)[0]
    )

    draw_valid = selection_mc_grid[draw_index] & np.isfinite(draw_map)
    selection_pixel_counts[draw_index] = int(np.sum(draw_valid))
    dynamic_selection_measurement_curves[draw_index] = (
        fft_pair_product_statistic(
            metallicity_values=draw_map, valid_mask=draw_valid
        )[0]
    )
    noise_map = draw_map - metallicity_grid
    noise_residual, _ = annular_residual_map(noise_map, draw_valid)
    noise_valid = draw_valid & np.isfinite(noise_residual)
    noise_numerator, noise_endpoint, noise_counts = fft_pair_sums(
        noise_residual, noise_valid
    )
    noise_numerator_means[draw_index] = np.divide(
        noise_numerator, noise_counts, out=np.zeros(n_bins),
        where=noise_counts > 0,
    )
    noise_endpoint_means[draw_index] = np.divide(
        noise_endpoint, noise_counts, out=np.zeros(n_bins),
        where=noise_counts > 0,
    )

measurement_curves = dynamic_selection_measurement_curves
selection_curve_difference = (
    dynamic_selection_measurement_curves
    - fixed_selection_measurement_curves
)
selection_response_bias = np.nanmean(selection_curve_difference, axis=0)
selection_bias_covariance = np.outer(
    selection_response_bias, selection_response_bias
)
measurement_covariance = np.cov(measurement_curves, rowvar=False, ddof=1)
'''


OLD_TOTAL_COVARIANCE = (
    "total_covariance_raw = jackknife_covariance + measurement_covariance\n"
)
NEW_TOTAL_COVARIANCE = r'''# The paired dynamic-minus-fixed mean is not subtracted from the data: the
# observed map has one realised selection.  Its magnitude is instead carried
# as a conservative, fully correlated one-sigma selection systematic.
total_covariance_raw = (
    jackknife_covariance + measurement_covariance
    + selection_bias_covariance
)
'''


DIAGNOSTIC_MARKDOWN = r'''## 4.1 Data-dependent selection response

The geometric coverage window is fixed, but the line-$S/N$, BPT, and
$W_{\rm H\alpha}$ decisions are functions of noisy measurements.  The flux
Monte Carlo therefore recomputes those three decisions in every realisation.
The dynamic-selection curves enter the measurement covariance; fixed-mask
curves are retained only as a paired diagnostic.  We do not subtract the
paired mean shift from the observed curve, because a plug-in bootstrap centred
on the noisy observed flux does not identify an unbiased correction.  Instead,
its outer product is included as a conservative correlated systematic.

Two robustness checks distinguish ordinary holes from endogenous selection.
First, the nominal curve is recomputed over a grid of line-$S/N$ and
$W_{\rm H\alpha}$ thresholds.  Second, line-ratio vectors are permuted within
0.2-kpc annuli while the observed H$\alpha$ brightness, errors, coverage, PSF
noise correlation, and complete selection logic are retained.  This destroys
intrinsic spatial abundance correlation while preserving radial one-point
line-ratio distributions and their coupling to detectability.  A recovered
non-zero curve is therefore a selection/null diagnostic, not a physical KTZ
prediction.  It is deliberately more limited than a native-cube injection:
the released, PSF-matched line maps are the starting point.
'''


DIAGNOSTIC_CODE = r'''selection_sn_grid = np.asarray([2.0, 3.0, 5.0])
selection_ew_grid = np.asarray([3.0, 6.0, 12.0])
threshold_curves = []
threshold_rows = []
for sn_cut in selection_sn_grid:
    for ew_cut in selection_ew_grid:
        threshold_valid = li21_selection_mask(
            line_flux, metallicity_grid, corrected_line_flux,
            line_sn_min=float(sn_cut), halpha_ew_min=float(ew_cut),
        ) & np.isfinite(metallicity_error_grid)
        threshold_curve, threshold_pairs, _, _, _, _ = (
            fft_pair_product_statistic(valid_mask=threshold_valid)
        )
        threshold_curves.append(threshold_curve)
        threshold_rows.append((
            float(sn_cut), float(ew_cut), int(np.sum(threshold_valid)),
            int(np.sum(threshold_pairs)),
            float(np.nanmax(np.abs(threshold_curve - xi_observed))),
        ))
threshold_curves = np.asarray(threshold_curves)


def annular_line_ratio_shuffle(
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Destroy spatial ratio correlation while preserving annular marginals."""
    ratio_pool = geometry_mask & np.isfinite(r_band_flux)
    for name in LINE_EXTENSIONS:
        ratio_pool &= (
            np.isfinite(line_flux[name]) & (line_flux[name] > 0)
            & np.isfinite(line_error[name]) & (line_error[name] > 0)
            & (line_coverage[name] >= LINE_COVERAGE_MIN)
        )
    ratio_pool &= np.isfinite(line_flux["ha"]) & (line_flux["ha"] > 0)

    shuffled = {
        name: np.full(pixel_X.shape, np.nan, dtype=float)
        for name in LINE_EXTENSIONS
    }
    shuffled["ha"][ratio_pool] = line_flux["ha"][ratio_pool]
    radial_edges = np.arange(
        0.0, R_MAX_KPC + ANNULUS_WIDTH_KPC, ANNULUS_WIDTH_KPC
    )
    radial_bin = np.digitize(pixel_R, radial_edges) - 1
    for bin_index in range(len(radial_edges) - 1):
        receiver = np.flatnonzero(ratio_pool & (radial_bin == bin_index))
        if len(receiver) < MIN_PIXELS_PER_ANNULUS:
            continue
        donor = rng.permutation(receiver)
        receiver_ha = line_flux["ha"].ravel()[receiver]
        donor_ha = line_flux["ha"].ravel()[donor]
        for name in LINE_EXTENSIONS:
            if name == "ha":
                continue
            donor_ratio = line_flux[name].ravel()[donor] / donor_ha
            shuffled[name].ravel()[receiver] = receiver_ha * donor_ratio
    return shuffled


N_SELECTION_NULL = 40
selection_null_rng = np.random.default_rng(
    91000 + int(''.join(character for character in GALAXY_NAME if character.isdigit()))
)
selection_null_noiseless = np.empty((N_SELECTION_NULL, n_bins))
selection_null_with_noise = np.empty((N_SELECTION_NULL, n_bins))
for null_index in range(N_SELECTION_NULL):
    null_flux = annular_line_ratio_shuffle(selection_null_rng)
    null_metallicity, _, null_corrected = scal_metallicity(null_flux)
    null_valid = li21_selection_mask(
        null_flux, null_metallicity, null_corrected
    )
    selection_null_noiseless[null_index] = fft_pair_product_statistic(
        metallicity_values=null_metallicity, valid_mask=null_valid
    )[0]

    noisy_null_flux = {
        name: null_flux[name] + line_error[name] * correlated_unit_noise(
            pixel_X.shape, noise_sigma_grid_pixel, selection_null_rng
        )
        for name in LINE_EXTENSIONS
    }
    noisy_null_metallicity, _, noisy_null_corrected = scal_metallicity(
        noisy_null_flux
    )
    noisy_null_valid = li21_selection_mask(
        noisy_null_flux, noisy_null_metallicity, noisy_null_corrected
    )
    selection_null_with_noise[null_index] = fft_pair_product_statistic(
        metallicity_values=noisy_null_metallicity,
        valid_mask=noisy_null_valid,
    )[0]


fixed_p16, fixed_p50, fixed_p84 = np.nanpercentile(
    fixed_selection_measurement_curves, [16.0, 50.0, 84.0], axis=0
)
dynamic_p16, dynamic_p50, dynamic_p84 = np.nanpercentile(
    dynamic_selection_measurement_curves, [16.0, 50.0, 84.0], axis=0
)
threshold_p16, threshold_p84 = np.nanpercentile(
    threshold_curves, [16.0, 84.0], axis=0
)
null_noiseless_p16, null_noiseless_p50, null_noiseless_p84 = np.nanpercentile(
    selection_null_noiseless, [16.0, 50.0, 84.0], axis=0
)
null_noisy_p16, null_noisy_p50, null_noisy_p84 = np.nanpercentile(
    selection_null_with_noise, [16.0, 50.0, 84.0], axis=0
)

selection_bias_fit = selection_response_bias[fit_indices]
selection_bias_mahalanobis = float(np.sqrt(
    selection_bias_fit @ np.linalg.solve(fit_covariance, selection_bias_fit)
))
selection_null_mean_fit = np.nanmean(
    selection_null_with_noise[:, fit_indices], axis=0
)
selection_null_mahalanobis = float(np.sqrt(
    selection_null_mean_fit
    @ np.linalg.solve(fit_covariance, selection_null_mean_fit)
))

selection_response_table = Table({
    "r_kpc": separation_centres,
    "xi_observed": xi_observed,
    "fixed_mc_p16": fixed_p16,
    "fixed_mc_p50": fixed_p50,
    "fixed_mc_p84": fixed_p84,
    "dynamic_mc_p16": dynamic_p16,
    "dynamic_mc_p50": dynamic_p50,
    "dynamic_mc_p84": dynamic_p84,
    "dynamic_minus_fixed_mean": selection_response_bias,
    "threshold_p16": threshold_p16,
    "threshold_p84": threshold_p84,
    "null_noiseless_p16": null_noiseless_p16,
    "null_noiseless_p50": null_noiseless_p50,
    "null_noiseless_p84": null_noiseless_p84,
    "null_noisy_p16": null_noisy_p16,
    "null_noisy_p50": null_noisy_p50,
    "null_noisy_p84": null_noisy_p84,
})
selection_response_table.write(
    OUTPUT_DIR / f"{GALAXY_NAME.lower()}_selection_response.csv",
    format="ascii.csv", overwrite=True,
)
Table(
    rows=threshold_rows,
    names=("line_sn_min", "halpha_ew_min_angstrom", "n_pixels",
           "total_pairs", "max_abs_delta_xi"),
).write(
    OUTPUT_DIR / f"{GALAXY_NAME.lower()}_selection_threshold_grid.csv",
    format="ascii.csv", overwrite=True,
)

selection_summary = {
    "galaxy": GALAXY_NAME,
    "nominal_valid_pixels": int(np.sum(pixel_valid_grid)),
    "dynamic_valid_pixels_p16_p50_p84": [
        float(value) for value in np.percentile(
            selection_pixel_counts, [16.0, 50.0, 84.0]
        )
    ],
    "paired_selection_bias_mahalanobis": selection_bias_mahalanobis,
    "paired_selection_max_abs_delta_xi": float(
        np.nanmax(np.abs(selection_response_bias[fit_indices]))
    ),
    "threshold_grid_max_abs_delta_xi": float(np.nanmax(
        np.abs(threshold_curves[:, fit_indices] - xi_observed[fit_indices])
    )),
    "annular_shuffle_noisy_null_mahalanobis": selection_null_mahalanobis,
    "annular_shuffle_noisy_null_max_abs_median_xi": float(
        np.nanmax(np.abs(null_noisy_p50[fit_indices]))
    ),
    "null_realisations": N_SELECTION_NULL,
    "interpretation": (
        "diagnostic conditional on released PSF-matched line maps; "
        "not a native-cube selection correction"
    ),
}
(OUTPUT_DIR / f"{GALAXY_NAME.lower()}_selection_summary.json").write_text(
    json.dumps(selection_summary, indent=2), encoding="utf-8"
)

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
axes[0].fill_between(
    separation_centres, fixed_p16, fixed_p84, color="tab:blue", alpha=0.16
)
axes[0].plot(
    separation_centres, fixed_p50, color="tab:blue", ls="--",
    label="fixed selection MC"
)
axes[0].fill_between(
    separation_centres, dynamic_p16, dynamic_p84,
    color="tab:orange", alpha=0.22,
)
axes[0].plot(
    separation_centres, dynamic_p50, color="tab:orange",
    label="recomputed selection MC",
)
axes[0].plot(
    separation_centres, xi_observed, color="black", lw=1.0,
    label="observed selection",
)
axes[0].set_title("Threshold crossings")
axes[0].legend(fontsize=8)

for curve in threshold_curves:
    axes[1].plot(separation_centres, curve, color="seagreen", alpha=0.18)
axes[1].fill_between(
    separation_centres, threshold_p16, threshold_p84,
    color="seagreen", alpha=0.20, label="threshold grid 16--84%",
)
axes[1].plot(separation_centres, xi_observed, color="black", lw=1.2,
             label="fiducial")
axes[1].set_title(r"$S/N\times W_{\rm H\alpha}$ robustness")
axes[1].legend(fontsize=8)

axes[2].fill_between(
    separation_centres, null_noiseless_p16, null_noiseless_p84,
    color="tab:purple", alpha=0.16,
)
axes[2].plot(
    separation_centres, null_noiseless_p50, color="tab:purple", ls="--",
    label="ratio-shuffle, no added noise",
)
axes[2].fill_between(
    separation_centres, null_noisy_p16, null_noisy_p84,
    color="sienna", alpha=0.18,
)
axes[2].plot(
    separation_centres, null_noisy_p50, color="sienna",
    label="ratio-shuffle + correlated noise",
)
axes[2].set_title("Zero-correlation selection null")
axes[2].legend(fontsize=8)

for axis in axes:
    axis.axhline(0.0, color="0.55", lw=0.8)
    axis.set_xlim(0.0, 5.0)
    axis.set_xlabel("separation [kpc]")
axes[0].set_ylabel(r"$\xi_{3,{\rm pp}}(r)$")
fig.suptitle(
    f"{GALAXY_NAME}: data-dependent abundance-selection diagnostics",
    fontsize=13,
)
selection_figure_path = (
    OUTPUT_DIR / f"{GALAXY_NAME.lower()}_selection_response_diagnostic.png"
)
fig.savefig(selection_figure_path, bbox_inches="tight")
display_figure(fig)

display(Table(
    rows=[
        ("nominal valid pixels", int(np.sum(pixel_valid_grid))),
        ("dynamic valid pixels [p16, p50, p84]",
         str(selection_summary["dynamic_valid_pixels_p16_p50_p84"])),
        ("paired selection shift [Mahalanobis]",
         selection_bias_mahalanobis),
        ("paired max |delta xi|", selection_summary[
            "paired_selection_max_abs_delta_xi"]),
        ("threshold-grid max |delta xi|", selection_summary[
            "threshold_grid_max_abs_delta_xi"]),
        ("noisy zero-correlation null [Mahalanobis]",
         selection_null_mahalanobis),
        ("noisy null max |median xi|", selection_summary[
            "annular_shuffle_noisy_null_max_abs_median_xi"]),
    ],
    names=("selection diagnostic", "value"),
))
'''


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


for notebook_path in NOTEBOOKS:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    cell7 = source(cells[7])
    if "def li21_selection_mask(" not in cell7:
        cell7 = replace_once(
            cell7,
            "\n\n\ndef correlated_unit_noise(\n",
            SELECTION_HELPER + "\n\n\ndef correlated_unit_noise(\n",
            f"{notebook_path.name}: selection helper insertion",
        )
    if "selection_mc_grid = np.empty(" not in cell7:
        cell7 = replace_once(
            cell7, OLD_MC, NEW_MC,
            f"{notebook_path.name}: dynamic selection MC",
        )
    cells[7]["source"] = lines(cell7)

    cell11 = source(cells[11])
    if "valid_mask: np.ndarray | None = None" not in cell11:
        cell11 = replace_once(
            cell11, OLD_STAT_SIGNATURE, NEW_STAT_SIGNATURE,
            f"{notebook_path.name}: statistic signature",
        )
        cell11 = replace_once(
            cell11, OLD_STAT_VALID, NEW_STAT_VALID,
            f"{notebook_path.name}: statistic selection",
        )
    if "fixed_selection_measurement_curves" not in cell11:
        cell11 = replace_once(
            cell11, OLD_MEASUREMENT_BLOCK, NEW_MEASUREMENT_BLOCK,
            f"{notebook_path.name}: measurement covariance",
        )
    if "selection_bias_covariance" in cell11 and OLD_TOTAL_COVARIANCE in cell11:
        cell11 = replace_once(
            cell11, OLD_TOTAL_COVARIANCE, NEW_TOTAL_COVARIANCE,
            f"{notebook_path.name}: total covariance",
        )
    cells[11]["source"] = lines(cell11)

    tagged = [
        (index, cell) for index, cell in enumerate(cells)
        if "selection-response-diagnostics"
        in cell.get("metadata", {}).get("tags", [])
    ]
    if tagged:
        if len(tagged) != 2:
            raise RuntimeError(
                f"{notebook_path.name}: expected two diagnostic cells"
            )
        tagged[0][1]["source"] = lines(DIAGNOSTIC_MARKDOWN)
        tagged[1][1]["source"] = lines(DIAGNOSTIC_CODE)
    else:
        insertion_index = 12
        cells[insertion_index:insertion_index] = [
            {
                "id": "selection-response-method",
                "cell_type": "markdown",
                "metadata": {"tags": ["selection-response-diagnostics"]},
                "source": lines(DIAGNOSTIC_MARKDOWN),
            },
            {
                "id": "selection-response-run",
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"tags": ["selection-response-diagnostics"]},
                "outputs": [],
                "source": lines(DIAGNOSTIC_CODE),
            },
        ]

    for cell in cells:
        text = source(cell)
        text = text.replace(
            "- The fixed observation transfer combines Li21 Gaussian-beam broadening with\n"
            "  a 200-realisation line-flux Monte Carlo estimate of correlated noise.  The\n"
            "  1.12 error inflation and Gaussian noise correlation remain approximations to\n"
            "  covariance not supplied in the public MAPS product.\n",
            "- The observation transfer combines Li21 Gaussian-beam broadening with a\n"
            "  200-realisation line-flux Monte Carlo.  Every realisation re-evaluates the\n"
            "  line-S/N, BPT, and H-alpha-EW cuts; fixed-mask curves are retained as a\n"
            "  paired control.  The 1.12 error inflation and Gaussian noise correlation\n"
            "  remain approximations to covariance not supplied in the public MAPS product.\n",
        )
        text = text.replace(
            "- Repeating the analysis for EW thresholds of 3, 6, and 10 Angstrom, line-S/N\n"
            "  thresholds of 2, 3, and 5, and alternative target beams is required before\n"
            "  publication.\n",
            "- The executed robustness grid uses H-alpha-EW thresholds of 3, 6, and 12\n"
            "  Angstrom and line-S/N thresholds of 2, 3, and 5.  Annular line-ratio\n"
            "  shuffles additionally test correlation recovered solely by the released-map\n"
            "  selection.  Alternative target beams and native-cube injection remain future\n"
            "  robustness tests.\n",
        )
        cell["source"] = lines(text)

    notebook_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"updated {notebook_path.name}")
