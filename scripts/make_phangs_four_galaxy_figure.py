"""Build the manuscript comparison for the four PHANGS--MUSE targets.

The inputs are the machine-readable products written by the four PHANGS-mask,
direct two-dimensional H-alpha, and H-alpha harmonic calculations.  Run from
any directory with

    python scripts/make_phangs_four_galaxy_figure.py

The PDFs are used by ``KTZ_obs.tex``; the PNG files are convenient previews.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "figs" / "phangs_four_galaxy_implementation.pdf"
COMPONENT_FIGURE_PATH = (
    ROOT / "figs" / "phangs_four_galaxy_components.pdf"
)
PREVIEW_PATH = (
    ROOT / "outputs" / "implications" / "phangs_four_galaxy_implementation.png"
)
COMPONENT_PREVIEW_PATH = (
    ROOT / "outputs" / "implications" / "phangs_four_galaxy_components.png"
)

TARGETS = (
    (
        "NGC 628",
        ROOT / "outputs" / "implications"
        / "ngc628_muse_scal_metallicity_residual_pixel_map.npz",
        ROOT / "outputs" / "implications"
        / "ngc628_xi_observed_and_models.csv",
        ROOT / "outputs" / "implications" / "ngc628_morphology_summary.json",
        ROOT / "outputs" / "implications" / "NGC628_halpha_map_systematic"
        / "ngc628_muse_scal_metallicity_residual_pixel_map.npz",
        ROOT / "outputs" / "implications" / "NGC628_halpha_map_systematic"
        / "ngc628_xi_observed_and_models.csv",
        ROOT / "outputs" / "implications" / "NGC628_halpha_map_systematic"
        / "ngc628_morphology_summary.json",
        ROOT / "outputs" / "implications" / "NGC628_logspiral_systematic"
        / "ngc628_xi_observed_and_models.csv",
        ROOT / "outputs" / "implications" / "NGC628_logspiral_systematic"
        / "ngc628_morphology_summary.json",
    ),
    (
        "NGC 1566",
        ROOT / "outputs" / "implications" / "NGC1566"
        / "ngc1566_muse_scal_metallicity_residual_pixel_map.npz",
        ROOT / "outputs" / "implications" / "NGC1566"
        / "ngc1566_xi_observed_and_models.csv",
        ROOT / "outputs" / "implications" / "NGC1566"
        / "ngc1566_morphology_summary.json",
        ROOT / "outputs" / "implications" / "NGC1566_halpha_map_systematic"
        / "ngc1566_muse_scal_metallicity_residual_pixel_map.npz",
        ROOT / "outputs" / "implications" / "NGC1566_halpha_map_systematic"
        / "ngc1566_xi_observed_and_models.csv",
        ROOT / "outputs" / "implications" / "NGC1566_halpha_map_systematic"
        / "ngc1566_morphology_summary.json",
        ROOT / "outputs" / "implications" / "NGC1566_logspiral_systematic"
        / "ngc1566_xi_observed_and_models.csv",
        ROOT / "outputs" / "implications" / "NGC1566_logspiral_systematic"
        / "ngc1566_morphology_summary.json",
    ),
    (
        "NGC 4254",
        ROOT / "outputs" / "implications" / "NGC4254"
        / "ngc4254_muse_scal_metallicity_residual_pixel_map.npz",
        ROOT / "outputs" / "implications" / "NGC4254"
        / "ngc4254_xi_observed_and_models.csv",
        ROOT / "outputs" / "implications" / "NGC4254"
        / "ngc4254_morphology_summary.json",
        ROOT / "outputs" / "implications" / "NGC4254_halpha_map_systematic"
        / "ngc4254_muse_scal_metallicity_residual_pixel_map.npz",
        ROOT / "outputs" / "implications" / "NGC4254_halpha_map_systematic"
        / "ngc4254_xi_observed_and_models.csv",
        ROOT / "outputs" / "implications" / "NGC4254_halpha_map_systematic"
        / "ngc4254_morphology_summary.json",
        ROOT / "outputs" / "implications" / "NGC4254_logspiral_systematic"
        / "ngc4254_xi_observed_and_models.csv",
        ROOT / "outputs" / "implications" / "NGC4254_logspiral_systematic"
        / "ngc4254_morphology_summary.json",
    ),
    (
        "NGC 4303",
        ROOT / "outputs" / "implications" / "NGC4303"
        / "ngc4303_muse_scal_metallicity_residual_pixel_map.npz",
        ROOT / "outputs" / "implications" / "NGC4303"
        / "ngc4303_xi_observed_and_models.csv",
        ROOT / "outputs" / "implications" / "NGC4303"
        / "ngc4303_morphology_summary.json",
        ROOT / "outputs" / "implications" / "NGC4303_halpha_map_systematic"
        / "ngc4303_muse_scal_metallicity_residual_pixel_map.npz",
        ROOT / "outputs" / "implications" / "NGC4303_halpha_map_systematic"
        / "ngc4303_xi_observed_and_models.csv",
        ROOT / "outputs" / "implications" / "NGC4303_halpha_map_systematic"
        / "ngc4303_morphology_summary.json",
        ROOT / "outputs" / "implications" / "NGC4303_logspiral_systematic"
        / "ngc4303_xi_observed_and_models.csv",
        ROOT / "outputs" / "implications" / "NGC4303_logspiral_systematic"
        / "ngc4303_morphology_summary.json",
    ),
)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.2,
        }
    )
    figure, axes = plt.subplots(
        3, 4, figsize=(7.10, 6.35), constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 1.0, 0.92)},
    )

    map_image = None
    for column, (
        galaxy, map_path, correlation_path, _summary_path,
        halpha_map_path, halpha_correlation_path, halpha_summary_path,
        systematic_correlation_path, systematic_summary_path,
    ) in enumerate(TARGETS):
        with np.load(map_path, allow_pickle=False) as map_data:
            x_grid = map_data["x_grid_kpc"]
            y_grid = map_data["y_grid_kpc"]
            residual = map_data["residual_dex"]
            valid = map_data["valid_mask"].astype(bool)
            environment_label = map_data["environment_label"]

        with np.load(halpha_map_path, allow_pickle=False) as halpha_map_data:
            halpha_modulation = halpha_map_data["arm_source_modulation"]
            halpha_source_support = np.isfinite(halpha_map_data["halpha_flux"])

        source_axis = axes[0, column]
        with open(halpha_summary_path) as handle:
            halpha_summary = json.load(handle)
        smooth_support_weight = gaussian_filter(
            halpha_source_support.astype(float),
            halpha_summary["halpha_template_fwhm_kpc"] / 2.354820045 / 0.06,
            mode="constant", cval=0.0, truncate=4.0,
        )
        halpha_valid = (
            (np.hypot(x_grid, y_grid) >= halpha_summary["analysis_rmin_kpc"])
            & (np.hypot(x_grid, y_grid) <= halpha_summary["analysis_rmax_kpc"])
            & (smooth_support_weight > 0.05)
        )
        source_rms = float(halpha_summary["mask_contrast_amplitude"])
        displayed_source = halpha_modulation / source_rms
        source_limit = 4.0
        source_image = source_axis.pcolormesh(
            x_grid,
            y_grid,
            np.where(halpha_valid, displayed_source, np.nan),
            shading="nearest",
            cmap="PuOr_r",
            vmin=-source_limit,
            vmax=source_limit,
            rasterized=True,
        )
        source_axis.set_aspect("equal")
        source_axis.set_title(galaxy)
        source_axis.set_xlabel(r"$x$ [kpc]")
        if column == 0:
            source_axis.set_ylabel("H" + r"$\alpha$ template" + "\n" + r"$y$ [kpc]")
        else:
            source_axis.set_yticklabels([])

        map_axis = axes[1, column]
        map_image = map_axis.pcolormesh(
            x_grid,
            y_grid,
            np.where(valid, residual, np.nan),
            shading="nearest",
            cmap="RdBu_r",
            vmin=-0.13,
            vmax=0.13,
            rasterized=True,
        )
        map_axis.set_aspect("equal")
        with open(systematic_summary_path) as handle:
            systematic_summary = json.load(handle)
        arm_mask = np.isin(environment_label, (5, 6)).astype(float)
        map_axis.contourf(
            x_grid, y_grid, arm_mask, levels=[0.5, 1.5],
            colors=["#66bd63"], alpha=0.24, zorder=4,
        )
        map_axis.contour(
            x_grid, y_grid, arm_mask, levels=[0.5], colors="#1b7837",
            linestyles="--", linewidths=1.05, zorder=5,
        )
        spiral_radius = np.linspace(
            systematic_summary["analysis_rmin_kpc"],
            systematic_summary["analysis_rmax_kpc"], 1200,
        )
        arm_number = int(systematic_summary["spiral_arm_number"])
        spiral_alpha = float(systematic_summary["spiral_alpha"])
        theta0 = float(systematic_summary["spiral_theta0"])
        reference_radius = float(systematic_summary["r_ref_kpc"])
        for arm_index in range(arm_number):
            spiral_phi = (
                spiral_alpha * np.log(spiral_radius / reference_radius)
                + theta0 - 2.0 * np.pi * arm_index
            ) / arm_number
            map_axis.plot(
                spiral_radius * np.cos(spiral_phi),
                spiral_radius * np.sin(spiral_phi),
                color="black", linewidth=1.15, zorder=6,
            )
        map_axis.text(
            0.03,
            0.97,
            rf"$m={arm_number}$, "
            + rf"$p={systematic_summary['spiral_pitch_angle_deg']:.1f}^\circ$",
            transform=map_axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.3,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70,
                  "pad": 1.0},
            zorder=7,
        )
        map_axis.set_xlabel(r"$x$ [kpc]")
        if column == 0:
            map_axis.set_ylabel("metal residual\n" + r"$y$ [kpc]")
        else:
            map_axis.set_yticklabels([])

        correlation = np.genfromtxt(
            correlation_path, delimiter=",", names=True, dtype=None,
            encoding="utf-8",
        )
        systematic_correlation = np.genfromtxt(
            systematic_correlation_path, delimiter=",", names=True,
            dtype=None, encoding="utf-8",
        )
        halpha_correlation = np.genfromtxt(
            halpha_correlation_path, delimiter=",", names=True, dtype=None,
            encoding="utf-8",
        )
        if not np.allclose(
            correlation["r_kpc"], systematic_correlation["r_kpc"]
        ):
            raise ValueError(f"Separation grids differ for {galaxy}.")
        if not np.allclose(
            correlation["xi_observed"], systematic_correlation["xi_observed"],
            equal_nan=True,
        ):
            raise ValueError(f"Observed correlations differ for {galaxy}.")
        if not np.allclose(
            correlation["r_kpc"], halpha_correlation["r_kpc"]
        ) or not np.allclose(
            correlation["xi_observed"], halpha_correlation["xi_observed"],
            equal_nan=True,
        ):
            raise ValueError(f"Direct H-alpha data differ for {galaxy}.")
        r_kpc = correlation["r_kpc"]
        correlation_axis = axes[2, column]
        correlation_axis.fill_between(
            r_kpc,
            correlation["xi_KT18_p16"],
            correlation["xi_KT18_p84"],
            color="#1f77b4",
            alpha=0.14,
            linewidth=0,
        )
        correlation_axis.plot(
            r_kpc,
            correlation["xi_KT18_median"],
            color="#1f77b4",
            linewidth=1.25,
            label="KT18",
        )
        correlation_axis.plot(
            r_kpc,
            correlation["xi_KTZ_median"],
            color="0.38",
            linestyle=":",
            linewidth=1.20,
            label="KTZ: PHANGS mask",
        )
        correlation_axis.fill_between(
            r_kpc,
            halpha_correlation["xi_KTZ_p16"],
            halpha_correlation["xi_KTZ_p84"],
            color="#7b3294",
            alpha=0.14,
            linewidth=0,
        )
        correlation_axis.plot(
            r_kpc,
            halpha_correlation["xi_KTZ_median"],
            color="#7b3294",
            linewidth=1.45,
            label=r"KTZ: direct H$\alpha$",
        )
        correlation_axis.plot(
            r_kpc,
            systematic_correlation["xi_KTZ_median"],
            color="#d95f02",
            linestyle="-.",
            linewidth=1.20,
            label=r"KTZ: H$\alpha$ harmonic",
        )
        correlation_axis.errorbar(
            r_kpc,
            correlation["xi_observed"],
            yerr=correlation["xi_jackknife_error"],
            fmt="o",
            markersize=1.4,
            markeredgewidth=0,
            color="black",
            ecolor="0.55",
            elinewidth=0.45,
            capsize=0,
            label="PHANGS--MUSE",
            zorder=4,
        )
        correlation_axis.axhline(0.0, color="0.65", linewidth=0.55)
        correlation_axis.set_xlim(0.0, 5.0)
        correlation_axis.set_ylim(-0.20, 1.04)
        correlation_axis.set_xlabel(r"$r$ [kpc]")
        if column == 0:
            correlation_axis.set_ylabel(r"$\xi_{3,{\rm pp}}(r)$")
            correlation_axis.legend(frameon=False, loc="upper right")
        else:
            correlation_axis.set_yticklabels([])

    colorbar = figure.colorbar(
        map_image,
        ax=axes[1, :],
        orientation="horizontal",
        fraction=0.055,
        pad=0.02,
        aspect=45,
    )
    colorbar.set_label(r"$\Delta[12+\log({\rm O/H})]$ [dex]")
    source_colorbar = figure.colorbar(
        source_image,
        ax=axes[0, :],
        orientation="horizontal",
        fraction=0.055,
        pad=0.02,
        aspect=45,
    )
    source_colorbar.set_label(
        r"1-kpc-smoothed H$\alpha$ modulation $g_{{\rm H}\alpha}/A_{{\rm H}\alpha,{\rm rms}}$"
    )
    figure.legend(
        handles=[
            Patch(facecolor="#66bd63", edgecolor="#1b7837", alpha=0.40,
                  linestyle="--", label="PHANGS arm region"),
            Line2D([0], [0], color="black", linewidth=1.15,
                   label=r"H$\alpha$ log spiral"),
        ],
        frameon=False, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, 1.035), fontsize=6.4,
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    figure.savefig(PREVIEW_PATH, dpi=240, bbox_inches="tight")
    plt.close(figure)

    component_order = ("CP0", "CPN", "CM", "CNP")
    component_colours = {
        "CP0": "#7fc97f",
        "CPN": "#ffff99",
        "CM": "#fdc086",
        "CNP": "#beaed4",
    }
    component_labels = {
        "CP0": r"$C_{{\rm 2p},0}$",
        "CPN": r"$C_{{\rm 2p},N}$",
        "CM": r"$C_{\rm M}$",
        "CNP": r"$C_{\rm NP}$",
    }
    component_figure, component_axes = plt.subplots(
        2, 2, figsize=(7.10, 5.25), constrained_layout=True,
        sharex=True, sharey=True,
    )
    for axis, (
        galaxy, _, _, _, _, correlation_path, _, _, _
    ) in zip(
        component_axes.ravel(), TARGETS
    ):
        correlation = np.genfromtxt(
            correlation_path, delimiter=",", names=True, dtype=None,
            encoding="utf-8",
        )
        positive = ~correlation["is_fixed_normalisation_anchor"]
        separation = correlation["r_kpc"][positive]
        cumulative = np.zeros_like(separation, dtype=float)
        for component in component_order:
            values = correlation[
                f"xi_component_{component}_joint_representative"
            ][positive]
            next_cumulative = cumulative + values
            axis.fill_between(
                separation, cumulative, next_cumulative,
                color=component_colours[component],
                alpha=0.50,
                linewidth=0,
                label=component_labels[component],
            )
            cumulative = next_cumulative
        noise_bias = correlation["observation_noise_bias"]
        axis.fill_between(
            correlation["r_kpc"],
            correlation["xi_KTZ_p16"] - noise_bias,
            correlation["xi_KTZ_p84"] - noise_bias,
            color="black",
            alpha=0.10,
            linewidth=0,
            label=r"complete KTZ 16--84\%",
        )
        axis.plot(
            separation,
            cumulative,
            color="black",
            linestyle="--",
            linewidth=1.35,
            label=r"complete KTZ joint representative",
        )
        axis.errorbar(
            correlation["r_kpc"],
            correlation["xi_observed"],
            yerr=correlation["xi_jackknife_error"],
            fmt="o",
            markersize=1.5,
            markeredgewidth=0,
            color="black",
            ecolor="0.55",
            elinewidth=0.45,
            capsize=0,
            label="PHANGS--MUSE",
            zorder=4,
        )
        axis.axhline(0.0, color="0.65", linewidth=0.55)
        axis.set_xlim(0.0, 5.0)
        axis.set_ylim(-0.20, 1.04)
        axis.set_title(galaxy)
        axis.set_xlabel(r"$r$ [kpc]")
        axis.set_ylabel(r"contribution to $\xi_{3,{\rm pp}}$")
    component_axes[0, 0].legend(
        frameon=False, ncol=2, loc="upper right", fontsize=6.0,
    )
    component_figure.savefig(COMPONENT_FIGURE_PATH, bbox_inches="tight")
    component_figure.savefig(
        COMPONENT_PREVIEW_PATH, dpi=240, bbox_inches="tight"
    )
    plt.close(component_figure)

    print(FIGURE_PATH)
    print(PREVIEW_PATH)
    print(COMPONENT_FIGURE_PATH)
    print(COMPONENT_PREVIEW_PATH)


if __name__ == "__main__":
    main()
