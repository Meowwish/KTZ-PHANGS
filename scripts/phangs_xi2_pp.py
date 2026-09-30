"""Legacy exploratory phase split; not used by the observational manuscript.

Every unordered pair of finite abundance cells is visited once.  Pairs are
grouped by separation, midpoint radius, and the phase of the independently
fitted H-alpha logarithmic spiral.  No missing cell is filled and no pair is
sampled.  Twelve delete-one azimuth-sector replicates provide the displayed
sampling uncertainty.  The source-rich/source-poor threshold and rotation
score are not KTZ observables and are retained only to preserve the provenance
of an earlier diagnostic; the paper now reports only the unthresholded
xi_2,pp and xi_3,pp definitions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs" / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

plt.rcParams.update({
    "font.size": 14.0,
    "axes.titlesize": 15.0,
    "axes.labelsize": 14.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 10.5,
})


OUTPUT = ROOT / "outputs" / "implications" / "phase_resolved"
OUTPUT.mkdir(parents=True, exist_ok=True)

GALAXIES = {
    "NGC628": {"r25": 13.97},
    "NGC1566": {"r25": 18.54},
    "NGC4254": {"r25": 9.53},
    "NGC4303": {"r25": 16.81},
}
SEPARATION_EDGES = np.arange(0.12, 2.52 + 1.0e-9, 0.24)
SEPARATION_CENTRES = 0.5 * (SEPARATION_EDGES[:-1] + SEPARATION_EDGES[1:])
RADIAL_EDGES_R25 = np.asarray([0.10, 0.20, 0.30, 0.40])
N_PHASE = 24
N_SECTOR = 12
PHASE_EDGES = np.linspace(0.0, 2.0 * np.pi, N_PHASE + 1)
PHASE_CENTRES = 0.5 * (PHASE_EDGES[:-1] + PHASE_EDGES[1:])


def paired_slices(length: int, delta: int) -> tuple[slice, slice]:
    if delta >= 0:
        return slice(0, length - delta), slice(delta, length)
    return slice(-delta, length), slice(0, length + delta)


def add_bincount(target: np.ndarray, index: np.ndarray, weight=None) -> None:
    target += np.bincount(index, weights=weight, minlength=target.size)


def accumulate_pairs(
    residual: np.ndarray,
    valid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r25: float,
    spiral: dict,
) -> dict[str, np.ndarray]:
    """Accumulate data products and delete-sector contributions exactly."""
    height, width = valid.shape
    pixel_size = float(np.nanmedian(np.diff(x, axis=1)))
    n_sep = len(SEPARATION_CENTRES)
    n_rad = len(RADIAL_EDGES_R25) - 1
    n_group = N_PHASE * n_rad * n_sep

    total = {
        "count": np.zeros(n_group, dtype=np.int64),
        "product": np.zeros(n_group),
        "endpoint": np.zeros(n_group),
    }
    touched = {
        name: np.zeros((N_SECTOR, n_group), dtype=value.dtype)
        for name, value in total.items()
    }

    maximum_offset = int(np.floor(SEPARATION_EDGES[-1] / pixel_size))
    for delta_y in range(maximum_offset + 1):
        delta_x_min = 1 if delta_y == 0 else -maximum_offset
        for delta_x in range(delta_x_min, maximum_offset + 1):
            separation = abs(pixel_size) * np.hypot(delta_x, delta_y)
            separation_bin = (
                np.searchsorted(SEPARATION_EDGES, separation, side="right") - 1
            )
            if separation_bin < 0 or separation_bin >= n_sep:
                continue

            row_i, row_j = paired_slices(height, delta_y)
            col_i, col_j = paired_slices(width, delta_x)
            selected = valid[row_i, col_i] & valid[row_j, col_j]
            if not np.any(selected):
                continue

            x_i = x[row_i, col_i][selected]
            x_j = x[row_j, col_j][selected]
            y_i = y[row_i, col_i][selected]
            y_j = y[row_j, col_j][selected]
            midpoint_x = 0.5 * (x_i + x_j)
            midpoint_y = 0.5 * (y_i + y_j)
            midpoint_radius = np.hypot(midpoint_x, midpoint_y)
            radial_bin = np.searchsorted(
                RADIAL_EDGES_R25, midpoint_radius / r25, side="right"
            ) - 1
            in_range = (radial_bin >= 0) & (radial_bin < n_rad)
            if not np.any(in_range):
                continue

            x_i = x_i[in_range]
            x_j = x_j[in_range]
            y_i = y_i[in_range]
            y_j = y_j[in_range]
            midpoint_x = midpoint_x[in_range]
            midpoint_y = midpoint_y[in_range]
            midpoint_radius = midpoint_radius[in_range]
            radial_bin = radial_bin[in_range]
            midpoint_azimuth = np.mod(
                np.arctan2(midpoint_y, midpoint_x), 2.0 * np.pi
            )
            theta = np.mod(
                spiral["alpha"]
                * np.log(np.maximum(midpoint_radius, 0.5 * abs(pixel_size))
                         / spiral["r_ref"])
                - spiral["m"] * midpoint_azimuth
                + spiral["theta0"],
                2.0 * np.pi,
            )
            phase_bin = np.minimum(
                np.floor(theta / (2.0 * np.pi) * N_PHASE).astype(int),
                N_PHASE - 1,
            )
            group = (
                (phase_bin * n_rad + radial_bin) * n_sep + separation_bin
            )

            values_i = residual[row_i, col_i][selected][in_range]
            values_j = residual[row_j, col_j][selected][in_range]
            pair_product = values_i * values_j
            endpoint_power = 0.5 * (values_i**2 + values_j**2)
            add_bincount(total["count"], group)
            add_bincount(total["product"], group, pair_product)
            add_bincount(total["endpoint"], group, endpoint_power)

            sector_i = np.minimum(
                np.floor(
                    np.mod(np.arctan2(y_i, x_i), 2.0 * np.pi)
                    / (2.0 * np.pi) * N_SECTOR
                ).astype(int),
                N_SECTOR - 1,
            )
            sector_j = np.minimum(
                np.floor(
                    np.mod(np.arctan2(y_j, x_j), 2.0 * np.pi)
                    / (2.0 * np.pi) * N_SECTOR
                ).astype(int),
                N_SECTOR - 1,
            )
            for sector, keep in (
                (sector_i, np.ones_like(sector_i, dtype=bool)),
                (sector_j, sector_j != sector_i),
            ):
                if not np.any(keep):
                    continue
                combined = sector[keep] * n_group + group[keep]
                touched["count"].flat[:] += np.bincount(
                    combined, minlength=N_SECTOR * n_group
                )
                touched["product"].flat[:] += np.bincount(
                    combined, weights=pair_product[keep],
                    minlength=N_SECTOR * n_group,
                )
                touched["endpoint"].flat[:] += np.bincount(
                    combined, weights=endpoint_power[keep],
                    minlength=N_SECTOR * n_group,
                )

    shape = (N_PHASE, n_rad, n_sep)
    return {
        **{name: value.reshape(shape) for name, value in total.items()},
        **{
            f"touch_{name}": value.reshape((N_SECTOR,) + shape)
            for name, value in touched.items()
        },
    }


def ratio(product: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
    return np.divide(
        product, endpoint, out=np.full(product.shape, np.nan),
        where=endpoint > 0.0,
    )


def aggregate_phase(
    arrays: dict[str, np.ndarray], selected_phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = arrays["count"][selected_phase].sum(axis=0)
    product = arrays["product"][selected_phase].sum(axis=0)
    endpoint = arrays["endpoint"][selected_phase].sum(axis=0)
    return ratio(product, endpoint), count, endpoint


def delete_sector_curves(
    arrays: dict[str, np.ndarray], selected_phase: np.ndarray,
) -> np.ndarray:
    curves = []
    for sector in range(N_SECTOR):
        product = (
            arrays["product"] - arrays["touch_product"][sector]
        )[selected_phase].sum(axis=0)
        endpoint = (
            arrays["endpoint"] - arrays["touch_endpoint"][sector]
        )[selected_phase].sum(axis=0)
        curves.append(ratio(product, endpoint))
    return np.asarray(curves)


def jackknife_error(replicates: np.ndarray) -> np.ndarray:
    centre = np.nanmean(replicates, axis=0)
    return np.sqrt(
        (len(replicates) - 1.0) / len(replicates)
        * np.nansum((replicates - centre) ** 2, axis=0)
    )


def source_rich_phase(spiral: dict, rotation: float = 0.0) -> np.ndarray:
    theta = PHASE_CENTRES + rotation
    modulation = np.zeros_like(theta)
    for order, (amplitude, phase) in enumerate(
        zip(spiral["amplitudes"], spiral["phases"]), start=1
    ):
        modulation += amplitude * np.cos(order * theta + phase)
    return modulation >= 0.0


def analyse_galaxy(galaxy: str, config: dict) -> dict:
    directory = ROOT / "outputs" / "implications" / f"{galaxy}_logspiral_systematic"
    stem = galaxy.lower()
    map_path = directory / f"{stem}_muse_scal_metallicity_residual_pixel_map.npz"
    summary_path = directory / f"{stem}_morphology_summary.json"
    with np.load(map_path, allow_pickle=False) as data:
        residual = np.asarray(data["residual_dex"], dtype=float)
        valid = np.asarray(data["valid_mask"], dtype=bool) & np.isfinite(residual)
        x = np.asarray(data["x_grid_kpc"], dtype=float)
        y = np.asarray(data["y_grid_kpc"], dtype=float)
    summary = json.loads(summary_path.read_text())
    spiral = {
        "m": int(summary["spiral_arm_number"]),
        "alpha": float(summary["spiral_alpha"]),
        "theta0": float(summary["spiral_theta0"]),
        "r_ref": float(summary["r_ref_kpc"]),
        "amplitudes": np.asarray(summary["harmonic_amplitudes"], dtype=float),
        "phases": np.asarray(summary["harmonic_phases"], dtype=float),
    }

    arrays = accumulate_pairs(
        residual, valid, x, y, float(config["r25"]), spiral
    )
    rich = source_rich_phase(spiral)
    poor = ~rich
    all_phase = np.ones(N_PHASE, dtype=bool)
    xi_all, count_all, _ = aggregate_phase(arrays, all_phase)
    xi_rich, count_rich, _ = aggregate_phase(arrays, rich)
    xi_poor, count_poor, _ = aggregate_phase(arrays, poor)
    contrast = xi_rich - xi_poor

    jk_all = delete_sector_curves(arrays, all_phase)
    jk_rich = delete_sector_curves(arrays, rich)
    jk_poor = delete_sector_curves(arrays, poor)
    jk_contrast = jk_rich - jk_poor
    error_all = jackknife_error(jk_all)
    error_contrast = jackknife_error(jk_contrast)

    rotations = 2.0 * np.pi * np.arange(N_PHASE) / N_PHASE
    rotation_rms = []
    for rotation in rotations:
        selected = source_rich_phase(spiral, rotation)
        xi_hi, _, _ = aggregate_phase(arrays, selected)
        xi_lo, _, _ = aggregate_phase(arrays, ~selected)
        delta = xi_hi - xi_lo
        finite = np.isfinite(delta) & (count_all >= 1000)
        rotation_rms.append(float(np.sqrt(np.mean(delta[finite] ** 2))))
    rotation_rms = np.asarray(rotation_rms)
    true_rms = float(rotation_rms[0])
    percentile = float(
        100.0 * (np.sum(rotation_rms < true_rms) + 0.5 * np.sum(rotation_rms == true_rms))
        / len(rotation_rms)
    )

    np.savez_compressed(
        OUTPUT / f"{stem}_xi2_pp_phase_resolved.npz",
        separation_edges_kpc=SEPARATION_EDGES,
        separation_centres_kpc=SEPARATION_CENTRES,
        radial_edges_r25=RADIAL_EDGES_R25,
        phase_edges_rad=PHASE_EDGES,
        pair_count=arrays["count"],
        pair_product_sum=arrays["product"],
        endpoint_second_moment_sum=arrays["endpoint"],
        xi2_pp=xi_all,
        xi2_pp_source_rich=xi_rich,
        xi2_pp_source_poor=xi_poor,
        delta_xi2_pp=contrast,
        xi2_pp_jackknife_sigma=error_all,
        delta_xi2_pp_jackknife_sigma=error_contrast,
        source_rich_phase_bins=rich,
        phase_rotation_rad=rotations,
        phase_rotation_delta_rms=rotation_rms,
    )

    return {
        "galaxy": galaxy,
        "xi": xi_all,
        "xi_error": error_all,
        "contrast": contrast,
        "contrast_error": error_contrast,
        "count": count_all,
        "count_rich": count_rich,
        "count_poor": count_poor,
        "true_rms": true_rms,
        "rotation_percentile": percentile,
        "n_pairs": int(arrays["count"].sum()),
    }


def write_summary(results: list[dict]) -> None:
    path = OUTPUT / "phangs_xi2_pp_summary.csv"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "galaxy,n_pairs_0p12_2p52,true_phase_delta_rms,"
            "phase_rotation_percentile\n"
        )
        for result in results:
            stream.write(
                f"{result['galaxy']},{result['n_pairs']},"
                f"{result['true_rms']:.8f},"
                f"{result['rotation_percentile']:.3f}\n"
            )


def make_figure(results: list[dict]) -> None:
    fig, axes = plt.subplots(
        2, 4, figsize=(16.4, 7.4), constrained_layout=True,
        sharex="row", sharey="row",
    )
    radial_labels = (
        r"$0.1<R_m/R_{25}<0.2$",
        r"$0.2<R_m/R_{25}<0.3$",
        r"$0.3<R_m/R_{25}<0.4$",
    )
    radial_colours = ("#1b9e77", "#d95f02", "#7570b3")
    for column, result in enumerate(results):
        axis = axes[0, column]
        image = axis.imshow(
            result["xi"], origin="lower", aspect="auto",
            extent=(SEPARATION_EDGES[0], SEPARATION_EDGES[-1], 0.1, 0.4),
            cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.15, vcenter=0.0, vmax=0.60),
        )
        axis.set_title(result["galaxy"].replace("NGC", "NGC "))
        axis.axvline(0.12, color="0.25", lw=0.8, ls=":")

        axis = axes[1, column]
        for radial_index, (label, colour) in enumerate(
            zip(radial_labels, radial_colours)
        ):
            axis.fill_between(
                SEPARATION_CENTRES,
                result["contrast"][radial_index]
                - result["contrast_error"][radial_index],
                result["contrast"][radial_index]
                + result["contrast_error"][radial_index],
                color=colour, alpha=0.14, linewidth=0,
            )
            axis.plot(
                SEPARATION_CENTRES, result["contrast"][radial_index],
                color=colour, lw=2.0, label=label,
            )
        axis.axhline(0.0, color="0.35", lw=0.8)
        axis.set_xlim(SEPARATION_EDGES[0], SEPARATION_EDGES[-1])
        axis.text(
            0.97, 0.95,
            f"phase-rotation percentile = {result['rotation_percentile']:.0f}%",
            transform=axis.transAxes, ha="right", va="top", fontsize=10.5,
        )
        if column == 0:
            axis.legend(frameon=False, loc="lower right")

    axes[0, 0].set_ylabel(r"midpoint radius $R_m/R_{25}$")
    axes[1, 0].set_ylabel(
        r"$\Delta\xi_{2,{\rm pp}}=\xi^{\rm rich}_{2,{\rm pp}}"
        r"-\xi^{\rm poor}_{2,{\rm pp}}$"
    )
    for axis in axes[1]:
        axis.set_xlabel(r"separation $r$ [kpc]")
    colourbar = fig.colorbar(image, ax=axes[0], shrink=0.92, pad=0.015)
    colourbar.set_label(r"$\xi_{2,{\rm pp}}(R_m,r)$")
    fig.savefig(OUTPUT / "phangs_xi2_pp_phase_resolved.pdf", bbox_inches="tight")
    fig.savefig(
        OUTPUT / "phangs_xi2_pp_phase_resolved.png", dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    results = []
    for galaxy, config in GALAXIES.items():
        print(f"Accumulating all pairs for {galaxy}...", flush=True)
        result = analyse_galaxy(galaxy, config)
        results.append(result)
        print(
            f"  {result['n_pairs']:,} pairs; "
            f"RMS(delta xi2,pp)={result['true_rms']:.4f}; "
            f"rotation percentile={result['rotation_percentile']:.1f}",
            flush=True,
        )
    write_summary(results)
    make_figure(results)


if __name__ == "__main__":
    main()
