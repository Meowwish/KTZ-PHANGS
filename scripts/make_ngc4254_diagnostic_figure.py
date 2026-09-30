"""Make a standalone four-panel NGC 4254 diagnostic figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[1]
MASK_OUTPUT = ROOT / "outputs" / "implications" / "NGC4254"
OUTPUT = ROOT / "outputs" / "implications" / "NGC4254_logspiral_systematic"
MAP_PATH = OUTPUT / "ngc4254_muse_scal_metallicity_residual_pixel_map.npz"
CURVE_PATH = OUTPUT / "ngc4254_xi_observed_and_models.csv"
MASK_CURVE_PATH = MASK_OUTPUT / "ngc4254_xi_observed_and_models.csv"
MORPHOLOGY_PATH = OUTPUT / "ngc4254_morphology_summary.json"
PDF_PATH = ROOT / "figs" / "ngc4254_halpha_metal_ktz_kt18_diagnostic.pdf"
PNG_PATH = MASK_OUTPUT / "ngc4254_halpha_metal_ktz_kt18_diagnostic.png"


def read_curve(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
        if name != "is_fixed_normalisation_anchor"
    }


with np.load(MAP_PATH) as payload:
    x_grid = payload["x_grid_kpc"]
    y_grid = payload["y_grid_kpc"]
    halpha_flux = payload["halpha_flux"]
    residual = payload["residual_dex"]
    valid = payload["valid_mask"]
    environment = payload["environment_label"]
curve = read_curve(CURVE_PATH)
mask_curve = read_curve(MASK_CURVE_PATH)
with MORPHOLOGY_PATH.open(encoding="utf-8") as handle:
    morphology = json.load(handle)

m_arms = int(morphology["spiral_arm_number"])
spiral_alpha = float(morphology["spiral_alpha"])
spiral_theta0 = float(morphology["spiral_theta0"])
r_ref_kpc = float(morphology["r_ref_kpc"])
spiral_radius = np.linspace(
    float(morphology["analysis_rmin_kpc"]),
    float(morphology["analysis_rmax_kpc"]),
    1200,
)


def overlay_source_geometries(axis) -> None:
    """Show the released arm region and the independently fitted H-alpha spiral."""
    axis.contour(
        x_grid,
        y_grid,
        arm_region.astype(float),
        levels=[0.5],
        colors="#1b7837",
        linestyles="--",
        linewidths=1.45,
    )
    for arm_index in range(m_arms):
        spiral_phi = (
            spiral_alpha * np.log(spiral_radius / r_ref_kpc)
            + spiral_theta0 - 2.0 * np.pi * arm_index
        ) / m_arms
        axis.plot(
            spiral_radius * np.cos(spiral_phi),
            spiral_radius * np.sin(spiral_phi),
            color="black",
            ls="-",
            lw=1.25,
        )

arm_region = np.isin(environment, [5, 6])
halpha_valid = np.isfinite(halpha_flux) & (halpha_flux > 0)
display_arm_region = arm_region & halpha_valid
halpha_log = np.full_like(halpha_flux, np.nan, dtype=float)
halpha_log[halpha_valid] = (
    np.log10(halpha_flux[halpha_valid])
    - np.median(np.log10(halpha_flux[halpha_valid]))
)
metal_residual = np.where(valid, residual, np.nan)

plt.rcParams.update({"font.size": 9.4})
fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), constrained_layout=True)

# H-alpha source tracer and the actual PHANGS arm-labelled pixels.
halpha_image = axes[0, 0].pcolormesh(
    x_grid,
    y_grid,
    halpha_log,
    shading="nearest",
    cmap="magma",
    norm=Normalize(vmin=-1.25, vmax=1.25),
)
axes[0, 0].contourf(
    x_grid,
    y_grid,
    display_arm_region.astype(float),
    levels=[0.5, 1.5],
    colors=["#66bd63"],
    alpha=0.40,
)
overlay_source_geometries(axes[0, 0])
axes[0, 0].legend(
    handles=[
        Line2D([0], [0], color="#1b7837", ls="--", lw=1.45,
               label="PHANGS labels 5 or 6"),
        Line2D([0], [0], color="black", ls="-", lw=1.2,
               label="H-alpha-fitted source spiral"),
    ],
    loc="lower left",
    frameon=True,
    framealpha=0.82,
    fontsize=8.2,
)
axes[0, 0].set_title(r"Dereddened H$\alpha$ and source geometries")
halpha_bar = fig.colorbar(halpha_image, ax=axes[0, 0], pad=0.02)
halpha_bar.set_label(r"$\log_{10}(F_{{\rm H}\alpha}/\mathrm{median})$")

# Exact abundance-residual pixels used by the exhaustive pair estimator.
residual_image = axes[0, 1].pcolormesh(
    x_grid,
    y_grid,
    metal_residual,
    shading="nearest",
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vmin=-0.13, vcenter=0.0, vmax=0.13),
)
overlay_source_geometries(axes[0, 1])
axes[0, 1].set_title("Metallicity fluctuation used for the correlation")
residual_bar = fig.colorbar(residual_image, ax=axes[0, 1], pad=0.02)
residual_bar.set_label(r"$\Delta[12+\log({\rm O/H})]$ [dex]")

for axis in axes[0]:
    axis.set_aspect("equal")
    axis.set_xlabel(r"$x$ [kpc]")
    axis.set_ylabel(r"$y$ [kpc]")

# Direct comparison of the two complete observable predictions.
r = curve["r_kpc"]
axes[1, 0].errorbar(
    r,
    curve["xi_observed"],
    yerr=curve["xi_jackknife_error"],
    fmt="o",
    ms=2.8,
    lw=0.8,
    color="black",
    ecolor="0.55",
    capsize=1.2,
    label="PHANGS--MUSE",
    zorder=5,
)
axes[1, 0].fill_between(
    r,
    curve["xi_KT18_p16"],
    curve["xi_KT18_p84"],
    color="#1f77b4",
    alpha=0.14,
    linewidth=0,
)
axes[1, 0].plot(
    r,
    curve["xi_KT18_median"],
    color="#1f77b4",
    lw=1.8,
    label="KT18 median",
)
axes[1, 0].plot(
    r,
    mask_curve["xi_KTZ_median"],
    color="0.45",
    ls=":",
    lw=1.5,
    label="PHANGS-mask KTZ median",
)
axes[1, 0].fill_between(
    r,
    curve["xi_KTZ_p16"],
    curve["xi_KTZ_p84"],
    color="0.45",
    alpha=0.18,
    linewidth=0,
)
axes[1, 0].plot(
    r,
    curve["xi_KTZ_median"],
    color="black",
    ls="--",
    lw=1.8,
    label=r"H$\alpha$-spiral KTZ median",
)
axes[1, 0].axhline(0.0, color="0.55", lw=0.7)
axes[1, 0].set(
    xlim=(0.0, 5.0),
    ylim=(-0.12, 1.03),
    xlabel=r"separation $r$ [kpc]",
    ylabel=r"$\xi_{3,\mathrm{pp}}(r)$",
    title=r"H$\alpha$-spiral KTZ versus KT18",
)
axes[1, 0].legend(frameon=False, fontsize=8.1, loc="upper right")

difference_axis = inset_axes(
    axes[1, 0], width="43%", height="30%", loc="center right", borderpad=1.0
)
difference_axis.fill_between(
    r,
    curve["delta_KTZ_minus_KT18_p16"],
    curve["delta_KTZ_minus_KT18_p84"],
    color="#9467bd",
    alpha=0.22,
    linewidth=0,
)
difference_axis.plot(
    r,
    curve["delta_KTZ_minus_KT18_median"],
    color="#6a3d9a",
    lw=1.1,
)
difference_axis.axhline(0.0, color="0.5", lw=0.6)
difference_axis.set_xlim(0.0, 5.0)
difference_axis.set_title(r"KTZ$-$KT18", fontsize=7.6)
difference_axis.tick_params(labelsize=6.8)

# Stack one joint posterior representative draw.  The observing-noise transfer
# is omitted from this physical-component decomposition.
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
positive = r > 0.0
cumulative = np.zeros(np.count_nonzero(positive), dtype=float)
for component in component_order:
    values = curve[
        f"xi_component_{component}_joint_representative"
    ][positive]
    next_cumulative = cumulative + values
    axes[1, 1].fill_between(
        r[positive],
        cumulative,
        next_cumulative,
        color=component_colours[component],
        alpha=0.50,
        linewidth=0,
        label=component_labels[component],
    )
    cumulative = next_cumulative
noise_term = curve["observation_noise_bias"]
axes[1, 1].fill_between(
    r,
    curve["xi_KTZ_p16"] - noise_term,
    curve["xi_KTZ_p84"] - noise_term,
    color="black",
    alpha=0.10,
    linewidth=0,
    label=r"KTZ + isotropic $C_{\rm NP}$ 16--84%",
)
axes[1, 1].plot(
    r[positive],
    cumulative,
    color="black",
    ls="--",
    lw=1.8,
    label="joint representative KTZ",
)
axes[1, 1].errorbar(
    r,
    curve["xi_observed"],
    yerr=curve["xi_jackknife_error"],
    fmt="o",
    ms=2.0,
    lw=0.6,
    color="black",
    ecolor="0.55",
    capsize=0,
    label="PHANGS--MUSE",
    zorder=5,
)
axes[1, 1].axhline(0.0, color="0.55", lw=0.7)
axes[1, 1].set(
    xlim=(0.0, 5.0),
    ylim=(-0.12, 1.03),
    xlabel=r"separation $r$ [kpc]",
    ylabel=r"contribution to $\xi_{3,\mathrm{pp}}(r)$",
    title=r"KTZ + isotropic $C_{\rm NP}$ components",
)
axes[1, 1].legend(
    frameon=False, fontsize=7.1, loc="upper right", ncol=2,
)

fig.suptitle("NGC 4254", fontsize=15)
PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(PDF_PATH, bbox_inches="tight")
fig.savefig(PNG_PATH, dpi=210, bbox_inches="tight")
plt.close(fig)
print(PDF_PATH)
print(PNG_PATH)
