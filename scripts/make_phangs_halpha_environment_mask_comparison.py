#!/usr/bin/env python3
"""Make a diagnostic H-alpha/metallicity comparison for the four PHANGS galaxies.

The H-alpha grids are reconstructed by executing only the data preparation cells
of the corresponding analysis notebooks.  Transport emulators and MCMC cells are
not run.  Two flux-Monte-Carlo realisations are sufficient here because the
Monte-Carlo metallicity uncertainty is not used in this diagnostic figure.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GALAXIES = ("NGC628", "NGC1566", "NGC4254", "NGC4303")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def execute_preparation_cells(galaxy: str) -> dict:
    """Execute notebook cells through construction of the direct source grid."""
    notebook_path = ROOT / f"{galaxy}_PHANGS_implications.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": f"_{galaxy.lower()}_halpha_diagnostic"}

    # Imports/configuration, catalogue geometry, and direct line-map processing.
    exec("".join(notebook["cells"][1]["source"]), namespace)
    namespace["N_FLUX_MC"] = 2
    exec("".join(notebook["cells"][5]["source"]), namespace)
    preparation = "".join(notebook["cells"][7]["source"])
    preparation = preparation.split("np.savez_compressed(", 1)[0]
    exec(preparation, namespace)
    return namespace


def draw_arm_boundaries(
    axis: plt.Axes, x_grid: np.ndarray, y_grid: np.ndarray,
    environment_labels: np.ndarray,
) -> None:
    arm_mask = np.isin(environment_labels, (5, 6)).astype(float)
    axis.contour(
        x_grid, y_grid, arm_mask, levels=[0.5], colors="black",
        linestyles="-", linewidths=1.1, zorder=5,
    )


def main() -> None:
    output_dir = ROOT / "outputs" / "implications"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2, 4, figsize=(13.8, 7.0), sharex="col", sharey="col",
        constrained_layout=True,
    )
    halpha_images = []
    residual_images = []

    for column, galaxy in enumerate(GALAXIES):
        namespace = execute_preparation_cells(galaxy)
        x_grid = namespace["pixel_X"]
        y_grid = namespace["pixel_Y"]
        source_mask = namespace["source_valid_grid"]
        halpha = np.full(x_grid.shape, np.nan)
        log_halpha = np.log10(namespace["source_halpha_flux"])
        halpha[source_mask] = log_halpha - np.nanmedian(log_halpha)

        galaxy_output = output_dir if galaxy == "NGC628" else output_dir / galaxy
        residual_path = galaxy_output / (
            f"{galaxy.lower()}_muse_scal_metallicity_residual_pixel_map.npz"
        )
        with np.load(residual_path) as residual_data:
            residual = residual_data["residual_dex"]
        environment_labels = namespace["environment_label_grid"]

        halpha_image = axes[0, column].pcolormesh(
            x_grid, y_grid, halpha, shading="nearest", cmap="magma",
            vmin=-1.25, vmax=1.25, rasterized=True,
        )
        residual_image = axes[1, column].pcolormesh(
            x_grid, y_grid, residual, shading="nearest", cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-0.13, vcenter=0.0, vmax=0.13),
            rasterized=True,
        )
        draw_arm_boundaries(
            axes[0, column], x_grid, y_grid, environment_labels
        )
        draw_arm_boundaries(
            axes[1, column], x_grid, y_grid, environment_labels
        )
        axes[0, column].set_title(galaxy.replace("NGC", "NGC "))
        axes[1, column].set_xlabel(r"$x$ [kpc]")
        for row in range(2):
            axes[row, column].set_aspect("equal")
        halpha_images.append(halpha_image)
        residual_images.append(residual_image)

        del namespace
        gc.collect()

    axes[0, 0].set_ylabel(r"H$\alpha$: $y$ [kpc]")
    axes[1, 0].set_ylabel(r"$\Delta[12+\log({\rm O/H})]$: $y$ [kpc]")
    figure.colorbar(
        halpha_images[-1], ax=axes[0, :], shrink=0.88,
        label=r"log attenuation-corrected H$\alpha$ cell flux / median",
    )
    figure.colorbar(
        residual_images[-1], ax=axes[1, :], shrink=0.88,
        label=r"$\Delta[12+\log({\rm O/H})]$ [dex]",
    )
    figure.suptitle(
        "H-alpha source field and released PHANGS arm masks (top); "
        "metallicity fluctuations (bottom)",
        fontsize=14,
    )

    png_path = (
        output_dir / "phangs_four_galaxy_halpha_environment_mask_comparison.png"
    )
    pdf_path = (
        output_dir / "phangs_four_galaxy_halpha_environment_mask_comparison.pdf"
    )
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path)
    plt.close(figure)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
