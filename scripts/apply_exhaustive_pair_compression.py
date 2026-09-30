#!/usr/bin/env python3
"""Replace explicit PHANGS pair arrays with exhaustive compressed moments."""

from __future__ import annotations

from pathlib import Path

import nbformat


WORKSPACE = Path(__file__).resolve().parents[1]
TEMPLATE = WORKSPACE / "NGC1566_PHANGS_implications.ipynb"


PAIR_CELL = r'''# Compress the exact unordered pair geometry into fine deterministic cells.
# Every valid pair is visited exactly once; no pair sampling is used.  The
# compression replaces separation and midpoint radius only by fine-cell
# centres, while retaining exact pair counts and trigonometric moment sums.
from scripts.phangs_pair_geometry import compress_regular_grid_pairs

n_pixels = len(r_kpc)
window_signature = hashlib.sha256(np.column_stack((
    x_kpc, y_kpc, metal_residual_centered
)).tobytes()).hexdigest()
pair_separation_subbins = 12
pair_midpoint_radial_edges = np.linspace(0.0, R_MAX_KPC, 257)
theta_grid_for_pairs = np.full(pixel_X.shape, np.nan)
theta_grid_for_pairs[pixel_valid_grid] = theta

pair_geometry_signature = hashlib.sha256(json.dumps({
    "version": 1,
    "window": window_signature,
    "pixel_size_kpc": PIXEL_SIZE_KPC,
    "separation_edges": separation_edges.tolist(),
    "separation_subbins": pair_separation_subbins,
    "midpoint_radial_edges": pair_midpoint_radial_edges.tolist(),
    "harmonic_orders": harmonic_orders.tolist(),
    "harmonic_phases": harmonic_phases.tolist(),
    "spiral_alpha": spiral_alpha,
    "spiral_m": m_arms,
    "spiral_theta0": theta0,
    "spiral_reference_radius": r_ref_kpc,
}, sort_keys=True).encode()).hexdigest()
pair_geometry_path = (
    OUTPUT_DIR / "ngc1566_muse_scal_exhaustive_pair_geometry.npz"
)

pair_geometry_loaded = False
if pair_geometry_path.exists():
    with np.load(pair_geometry_path, allow_pickle=False) as cached:
        if str(cached["signature"].item()) == pair_geometry_signature:
            pair_geometry = {
                name: cached[name] for name in cached.files
                if name != "signature"
            }
            pair_geometry_loaded = True

if not pair_geometry_loaded:
    pair_geometry = compress_regular_grid_pairs(
        valid=pixel_valid_grid,
        x_grid=pixel_X,
        y_grid=pixel_Y,
        theta_grid=theta_grid_for_pairs,
        harmonic_orders=harmonic_orders,
        harmonic_phases=harmonic_phases,
        spiral_alpha=spiral_alpha,
        spiral_m=m_arms,
        spiral_theta0=theta0,
        spiral_reference_radius=r_ref_kpc,
        separation_edges=separation_edges,
        radial_edges=pair_midpoint_radial_edges,
        separation_subbins=pair_separation_subbins,
    )
    np.savez_compressed(
        pair_geometry_path,
        signature=np.asarray(pair_geometry_signature),
        **pair_geometry,
    )
else:
    print(f"Loaded matching exhaustive pair geometry: {pair_geometry_path.name}")

pair_counts_geometry = np.rint(np.bincount(
    pair_geometry["output_bin"],
    weights=pair_geometry["counts"],
    minlength=n_bins,
)).astype(np.int64)
if not np.array_equal(pair_counts_geometry, pair_counts):
    raise RuntimeError(
        "FFT and exhaustive compressed pair counts disagree; check bin edges."
    )
pair_count_total = int(np.sum(pair_geometry["counts"]))

print(f"Face-on analysis pixels: {n_pixels:,}")
print(f"All unordered pairs with r < 5 kpc: {pair_count_total:,}")
print(
    "Occupied fine pair cells: "
    f"{len(pair_geometry['counts']):,} "
    f"({pair_separation_subbins} separation subbins and "
    f"{len(pair_midpoint_radial_edges) - 1} midpoint-radius bins)"
)
print(f"MCMC fit bins (0 < r < 5 kpc): {len(fit_indices)}")
print(
    f"Residual mean: {residual_mean:+.6f} +/- {residual_mean_error:.6f} dex"
)
print(
    "Median measurement-noise signal fraction: "
    f"{np.nanmedian(observation_signal_fraction[fit_indices]):.3f}"
)
print(
    f"Primary raw fit-covariance rank: {np.linalg.matrix_rank(fit_covariance_raw)}"
    f"/{len(fit_indices)}"
)
'''


PAIR_BASIS = r'''def ktz_pair_basis(
    ell_value: float,
    epsilon_value: float,
    rate_log_slope_values: float | np.ndarray,
    quadrature_nodes: int = 20,
) -> dict[str, np.ndarray]:
    """Average KTZ bases over deterministic moments of every valid pair."""
    rate_log_slope_array = np.atleast_1d(np.asarray(rate_log_slope_values, dtype=float))
    scalar_rate_log_slope = np.ndim(rate_log_slope_values) == 0
    legendre_x, legendre_w = np.polynomial.legendre.leggauss(quadrature_nodes)
    log_u_min = 2.0 * np.log(epsilon_value)
    log_u = 0.5 * (0.0 - log_u_min) * legendre_x + 0.5 * log_u_min
    log_u_weight = 0.5 * (0.0 - log_u_min) * legendre_w
    u_nodes = np.exp(log_u)
    i0_zero = -np.log(epsilon_value)
    q_factor = np.sqrt(spiral_alpha**2 + m_arms**2)

    names = (
        "numerator_uniform", "numerator_harmonic", "numerator_mean",
        "variance_i_uniform", "variance_i_harmonic", "variance_i_mean",
        "variance_j_uniform", "variance_j_harmonic", "variance_j_mean",
    )
    sums = {name: np.zeros((len(rate_log_slope_array), n_bins)) for name in names}

    counts_cell = pair_geometry["counts"].astype(float)
    separation = pair_geometry["separation"]
    midpoint_radius = pair_geometry["midpoint_radius"]
    bins = pair_geometry["output_bin"]
    midpoint_cosine = pair_geometry["midpoint_cosine"]
    endpoint_cosine = pair_geometry["endpoint_cosine"]
    endpoint_cross = pair_geometry["endpoint_cross"]
    endpoint_self = pair_geometry["endpoint_self"]

    rho = separation / ell_value
    connected_uniform = 0.5 * np.sum(
        log_u_weight[None, :]
        * np.exp(-rho[:, None]**2 / (4.0 * u_nodes[None, :])),
        axis=1,
    )
    connected_harmonic_sum = np.zeros_like(separation)
    variance_harmonic_sum = np.zeros_like(separation)
    tau_by_harmonic = np.empty((len(harmonic_orders), len(separation)))

    for harmonic_index, (order, amplitude) in enumerate(zip(
        harmonic_orders, harmonic_amplitudes
    )):
        q_n = order * q_factor * ell_value / midpoint_radius
        i_n = 0.5 * np.sum(
            log_u_weight[None, :]
            * np.exp(
                -rho[:, None]**2 / (4.0 * u_nodes[None, :])
                -q_n[:, None]**2 * u_nodes[None, :] / 4.0
            ),
            axis=1,
        )
        i_n_zero = 0.5 * (
            exp1(q_n**2 * epsilon_value**2 / 4.0)
            - exp1(q_n**2 / 4.0)
        )
        tau_denominator = 0.5 * (1.0 - epsilon_value**2) * q_n**2
        tau_by_harmonic[harmonic_index] = (
            np.exp(-0.5 * epsilon_value**2 * q_n**2)
            * (-np.expm1(-tau_denominator)) / tau_denominator
        )
        connected_harmonic_sum += (
            amplitude * i_n * midpoint_cosine[harmonic_index]
        )
        variance_harmonic_sum += (
            amplitude * i_n_zero * endpoint_cosine[harmonic_index]
        )

    mean_cross_sum = np.zeros_like(separation)
    mean_self_sum = np.zeros_like(separation)
    for first, amplitude_first in enumerate(harmonic_amplitudes):
        for second, amplitude_second in enumerate(harmonic_amplitudes):
            coefficient = (
                amplitude_first * amplitude_second
                * tau_by_harmonic[first] * tau_by_harmonic[second]
            )
            mean_cross_sum += coefficient * endpoint_cross[first, second]
            mean_self_sum += coefficient * endpoint_self[first, second]

    rate_relative = np.exp(
        rate_log_slope_array[:, None] * (midpoint_radius[None, :] - r_ref_kpc)
    )
    for rate_log_slope_index, local_rate in enumerate(rate_relative):
        local_rate_squared = local_rate**2
        values = {
            "numerator_uniform": (
                local_rate * counts_cell * connected_uniform
            ),
            "numerator_harmonic": local_rate * connected_harmonic_sum,
            "numerator_mean": local_rate_squared * mean_cross_sum,
            "variance_i_uniform": local_rate * counts_cell * i0_zero,
            "variance_i_harmonic": local_rate * variance_harmonic_sum,
            "variance_i_mean": local_rate_squared * mean_self_sum,
            "variance_j_uniform": local_rate * counts_cell * i0_zero,
            "variance_j_harmonic": local_rate * variance_harmonic_sum,
            "variance_j_mean": local_rate_squared * mean_self_sum,
        }
        for name, value in values.items():
            sums[name][rate_log_slope_index] = np.bincount(
                bins, weights=value, minlength=n_bins
            )

    counts = pair_counts_geometry.astype(float)
    for name in names:
        sums[name] = np.divide(
            sums[name], counts[None, :],
            out=np.full((len(rate_log_slope_array), n_bins), np.nan),
            where=counts[None, :] > 0,
        )
        if scalar_rate_log_slope:
            sums[name] = sums[name][0]
    sums["counts"] = counts
    return sums


'''


CNP_EXACT = r'''def ktz_cnp_pair_basis_exact(
    ell_value: float, epsilon_value: float, rate_log_slope_value: float,
    source_length: float, quadrature_nodes: int = 20,
) -> dict[str, np.ndarray]:
    """Evaluate appendix C_NP on deterministic moments of every valid pair."""
    kernel = gaussian_source_cnp_kernel(
        pair_geometry["separation"], ell_value, epsilon_value, source_length,
        quadrature_nodes=quadrature_nodes,
    )
    local_rate_squared = np.exp(
        2.0 * rate_log_slope_value
        * (pair_geometry["midpoint_radius"] - r_ref_kpc)
    )
    bins = pair_geometry["output_bin"]
    counts_cell = pair_geometry["counts"]
    numerator = np.divide(
        np.bincount(
            bins, weights=counts_cell * local_rate_squared * kernel,
            minlength=n_bins,
        ),
        pair_counts,
        out=np.full(n_bins, np.nan),
        where=pair_counts > 0,
    )
    mean_rate_squared = np.divide(
        np.bincount(
            bins, weights=counts_cell * local_rate_squared,
            minlength=n_bins,
        ),
        pair_counts,
        out=np.full(n_bins, np.nan),
        where=pair_counts > 0,
    )
    zero_lag = gaussian_source_cnp_kernel(
        np.asarray([0.0]), ell_value, epsilon_value, source_length,
        quadrature_nodes=quadrature_nodes,
    )[0]
    variance = zero_lag * mean_rate_squared
    return {
        "numerator": numerator,
        "variance_i": variance,
        "variance_j": variance.copy(),
    }


'''


CNP_GEOMETRY = r'''# Collapse the exhaustive geometry over midpoint radius for C_NP.  The
# separation cells are the same deterministic subbins used by the other KTZ
# components, and every observed pair contributes its exact count.
cnp_fine_bins_per_output_bin = pair_separation_subbins
cnp_fine_centres = pair_geometry["fine_separation_centres"]
cnp_weighted_fine_counts = np.empty((
    len(rate_log_slope_emulator_grid), len(cnp_fine_centres)
))
for rate_log_slope_index, rate_log_slope_value in enumerate(rate_log_slope_emulator_grid):
    rate_squared = np.exp(
        2.0 * rate_log_slope_value
        * (pair_geometry["midpoint_radius"] - r_ref_kpc)
    )
    cnp_weighted_fine_counts[rate_log_slope_index] = np.bincount(
        pair_geometry["fine_bin"],
        weights=pair_geometry["counts"] * rate_squared,
        minlength=len(cnp_fine_centres),
    )
cnp_mean_rate_squared = np.divide(
    cnp_weighted_fine_counts.reshape(
        len(rate_log_slope_emulator_grid), n_bins, cnp_fine_bins_per_output_bin
    ).sum(axis=2),
    pair_counts[None, :],
    out=np.full((len(rate_log_slope_emulator_grid), n_bins), np.nan),
    where=pair_counts[None, :] > 0,
)

'''


def replace_between(source: str, start: str, end: str, replacement: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[:start_index] + replacement + source[end_index:]


def main() -> None:
    notebook = nbformat.read(TEMPLATE, as_version=4)
    notebook.cells[12].source = PAIR_CELL

    basis_source = notebook.cells[14].source
    basis_source = replace_between(
        basis_source, "def ktz_pair_basis(", "component_order =", PAIR_BASIS
    )
    basis_source = replace_between(
        basis_source,
        "def ktz_cnp_pair_basis_exact(",
        "def correlation_components_from_basis(",
        CNP_EXACT,
    )
    notebook.cells[14].source = basis_source

    emulator_source = notebook.cells[16].source
    emulator_source = emulator_source.replace('"version": 3,', '"version": 4,')
    emulator_source = emulator_source.replace(
        "pair_count=np.asarray(len(pair_i)),",
        "pair_count=np.asarray(pair_count_total),",
    )
    if "valid_pair =" in emulator_source:
        emulator_source = replace_between(
            emulator_source,
            "valid_pair =",
            "source_length_emulator_grid =",
            CNP_GEOMETRY,
        )
    notebook.cells[16].source = emulator_source

    interpretation = notebook.cells[26].source
    compression_note = (
        "- The finite-window KTZ quadrature visits every valid unordered pixel "
        "pair exactly once; it does not sample pairs.  To keep the 60 pc Li21 "
        "grid tractable, exact counts and spiral-phase moments are accumulated "
        "in 12 separation subbins per reported bin and 256 midpoint-radius "
        "bins.  The cached compression is checked bin by bin against the exact "
        "FFT pair counts.\n"
    )
    if compression_note not in interpretation:
        anchor = "- The likelihood uses all available positive-separation bins"
        location = interpretation.index(anchor)
        interpretation = (
            interpretation[:location] + compression_note
            + interpretation[location:]
        )
    notebook.cells[26].source = interpretation

    for index in (12, 14, 16):
        compile(notebook.cells[index].source, f"cell-{index}", "exec")
        notebook.cells[index].outputs = []
        notebook.cells[index].execution_count = None
    nbformat.validate(notebook)
    nbformat.write(notebook, TEMPLATE)
    print(f"Updated {TEMPLATE}")


if __name__ == "__main__":
    main()
