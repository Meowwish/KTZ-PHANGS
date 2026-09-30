"""Make the NGC 4303 source-geometry systematic figure.

The two fits use the same abundance map, pair estimator, covariance, observing
transfer, and KTZ model with isotropic C_NP.  They differ only in the
externally specified source template:
the primary PHANGS environment mask versus a log spiral fitted independently
to the H-alpha-weighted, arm-labelled H II-region sample.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "outputs" / "implications" / "NGC4303"
SYSTEMATIC = ROOT / "outputs" / "implications" / "NGC4303_logspiral_systematic"
PDF_PATH = ROOT / "figs" / "ngc4303_source_geometry_systematic.pdf"
PNG_PATH = ROOT / "outputs" / "implications" / "ngc4303_source_geometry_systematic.png"


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
        if name != "is_fixed_normalisation_anchor"
    }


primary = read_csv(PRIMARY / "ngc4303_xi_observed_and_models.csv")
systematic = read_csv(SYSTEMATIC / "ngc4303_xi_observed_and_models.csv")
with np.load(
    SYSTEMATIC / "ngc4303_muse_scal_metallicity_residual_pixel_map.npz"
) as payload:
    x_grid = payload["x_grid_kpc"]
    y_grid = payload["y_grid_kpc"]
    halpha_flux = payload["halpha_flux"]
    environment_label = payload["environment_label"]
with (SYSTEMATIC / "ngc4303_morphology_summary.json").open(
    encoding="utf-8"
) as handle:
    morphology = json.load(handle)


valid_halpha = np.isfinite(halpha_flux) & (halpha_flux > 0)
halpha_log = np.full_like(halpha_flux, np.nan, dtype=float)
halpha_log[valid_halpha] = (
    np.log10(halpha_flux[valid_halpha])
    - np.median(np.log10(halpha_flux[valid_halpha]))
)
arm_mask = np.isin(environment_label, [5, 6])
display_arm_mask = arm_mask & valid_halpha

m_arms = int(morphology["spiral_arm_number"])
alpha = float(morphology["spiral_alpha"])
theta0 = float(morphology["spiral_theta0"])
r_ref = float(morphology["r_ref_kpc"])
r_min = float(morphology["analysis_rmin_kpc"])
r_max = float(morphology["analysis_rmax_kpc"])
pitch = float(morphology["spiral_pitch_angle_deg"])
spiral_radius = np.linspace(r_min, r_max, 1600)


plt.rcParams.update({"font.size": 9.2})
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.15), constrained_layout=True)

image = axes[0].pcolormesh(
    x_grid,
    y_grid,
    halpha_log,
    shading="nearest",
    cmap="magma",
    norm=Normalize(vmin=-1.25, vmax=1.25),
)
axes[0].contourf(
    x_grid,
    y_grid,
    display_arm_mask.astype(float),
    levels=[0.5, 1.5],
    colors=["#66bd63"],
    alpha=0.40,
)
axes[0].contour(
    x_grid,
    y_grid,
    arm_mask.astype(float),
    levels=[0.5],
    colors="#1b7837",
    linestyles="--",
    linewidths=1.45,
)
for arm_index in range(m_arms):
    spiral_phi = (
        alpha * np.log(spiral_radius / r_ref)
        + theta0
        - 2.0 * np.pi * arm_index
    ) / m_arms
    axes[0].plot(
        spiral_radius * np.cos(spiral_phi),
        spiral_radius * np.sin(spiral_phi),
        color="black",
        lw=1.25,
    )
axes[0].set(
    xlabel=r"$x$ [kpc]",
    ylabel=r"$y$ [kpc]",
    title=fr"H$\alpha$ source geometry: $m={m_arms}$, $i={pitch:.1f}^\circ$",
    aspect="equal",
)
axes[0].legend(
    handles=[
        Line2D([0], [0], color="#1b7837", ls="--", lw=1.45,
               label="PHANGS labels 5 or 6"),
        Line2D([0], [0], color="black", ls="-", lw=1.25,
               label=r"H$\alpha$-fitted source spiral"),
    ],
    loc="lower left",
    frameon=True,
    framealpha=0.86,
    fontsize=7.7,
)
colour_bar = fig.colorbar(image, ax=axes[0], pad=0.02)
colour_bar.set_label(r"$log_{10} F_{{\rm H}\alpha}/\mathrm{median}$")

r = primary["r_kpc"]
axes[1].errorbar(
    r,
    primary["xi_observed"],
    yerr=primary["xi_jackknife_error"],
    fmt="o",
    ms=2.7,
    lw=0.8,
    color="black",
    ecolor="0.55",
    capsize=1.2,
    label="PHANGS--MUSE residuals",
    zorder=4,
)
axes[1].fill_between(
    r,
    primary["xi_KTZ_p16"],
    primary["xi_KTZ_p84"],
    color="0.65",
    alpha=0.24,
    linewidth=0,
)
axes[1].plot(
    r,
    primary["xi_KTZ_median"],
    color="0.35",
    ls=":",
    lw=1.8,
    label="environment-mask KTZ",
)
axes[1].fill_between(
    r,
    systematic["xi_KTZ_p16"],
    systematic["xi_KTZ_p84"],
    color="#d95f02",
    alpha=0.20,
    linewidth=0,
)
axes[1].plot(
    r,
    systematic["xi_KTZ_median"],
    color="#d95f02",
    lw=1.8,
    label=r"H$\alpha$ log-spiral KTZ",
)
axes[1].axhline(0.0, color="0.5", lw=0.7)
axes[1].set(
    xlim=(0.0, 5.0),
    ylim=(-0.12, 1.03),
    xlabel=r"separation $r$ [kpc]",
    ylabel=r"$\xi_{3,\mathrm{pp}}(r)$",
    title="NGC 4303 source-geometry systematic",
)
axes[1].legend(frameon=False, fontsize=8.3, loc="upper right")

PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(PDF_PATH, bbox_inches="tight")
fig.savefig(PNG_PATH, dpi=210, bbox_inches="tight")
plt.close(fig)
print(PDF_PATH)
print(PNG_PATH)
