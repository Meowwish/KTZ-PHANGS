"""Finite-window KTZ bases for an arbitrary PHANGS source-rate template."""

from __future__ import annotations

import numpy as np
from scipy.special import exp1


def template_pair_basis(
    pair_geometry: dict[str, np.ndarray],
    pair_counts: np.ndarray,
    n_bins: int,
    reference_radius_kpc: float,
    ell_value: float,
    epsilon_value: float,
    rate_log_slope_values: float | np.ndarray,
) -> dict[str, np.ndarray]:
    """Average the local-mask KTZ bases over every observed pair.

    The released arm masks are 1--2 kpc wide, whereas the inferred transport
    lengths in this application are about 0.1--0.3 kpc.  We consequently use
    the local-source (WKBJ) limit: the transport kernel is evaluated exactly,
    while the arbitrary source modulation is sampled at the pair midpoint for
    the connected term and at both endpoints for the mean-field term.
    """
    rate_log_slope_array = np.atleast_1d(np.asarray(rate_log_slope_values, dtype=float))
    scalar_rate_log_slope = np.ndim(rate_log_slope_values) == 0
    counts_cell = pair_geometry["counts"].astype(float)
    separation = pair_geometry["separation"]
    midpoint_radius = pair_geometry["midpoint_radius"]
    bins = pair_geometry["output_bin"]
    midpoint_modulation = pair_geometry["midpoint_modulation"]
    endpoint_modulation = pair_geometry["endpoint_modulation"]
    endpoint_cross = pair_geometry["endpoint_modulation_cross"]
    endpoint_squared = pair_geometry["endpoint_modulation_squared"]

    rho = separation / ell_value
    connected_kernel = 0.5 * (
        exp1(rho**2 / 4.0)
        - exp1(rho**2 / (4.0 * epsilon_value**2))
    )
    zero_lag = -np.log(epsilon_value)
    names = (
        "numerator_uniform", "numerator_harmonic", "numerator_mean",
        "variance_i_uniform", "variance_i_harmonic", "variance_i_mean",
        "variance_j_uniform", "variance_j_harmonic", "variance_j_mean",
    )
    sums = {name: np.zeros((len(rate_log_slope_array), n_bins)) for name in names}

    relative_rate = np.exp(
        rate_log_slope_array[:, None]
        * (midpoint_radius[None, :] - reference_radius_kpc)
    )
    for rate_log_slope_index, local_rate in enumerate(relative_rate):
        local_rate_squared = local_rate**2
        values = {
            "numerator_uniform": local_rate * counts_cell * connected_kernel,
            "numerator_harmonic": (
                local_rate * midpoint_modulation * connected_kernel
            ),
            "numerator_mean": local_rate_squared * endpoint_cross,
            "variance_i_uniform": local_rate * counts_cell * zero_lag,
            "variance_i_harmonic": (
                local_rate * endpoint_modulation * zero_lag
            ),
            "variance_i_mean": local_rate_squared * endpoint_squared,
            "variance_j_uniform": local_rate * counts_cell * zero_lag,
            "variance_j_harmonic": (
                local_rate * endpoint_modulation * zero_lag
            ),
            "variance_j_mean": local_rate_squared * endpoint_squared,
        }
        for name, value in values.items():
            sums[name][rate_log_slope_index] = np.bincount(
                bins, weights=value, minlength=n_bins
            )

    counts = np.asarray(pair_counts, dtype=float)
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


def gaussian_source_cnp_kernel(
    separation: np.ndarray,
    ell_value: float,
    epsilon_value: float,
    source_length: float,
    quadrature_nodes: int = 14,
) -> np.ndarray:
    """Appendix C_NP kernel for unit Gaussian source-pair excess amplitude."""
    legendre_x, legendre_w = np.polynomial.legendre.leggauss(quadrature_nodes)
    age_fraction = 0.5 * (legendre_x + 1.0)
    age_weight = 0.5 * legendre_w
    width_fraction = (
        epsilon_value**2
        + (1.0 - epsilon_value**2) * age_fraction
    )
    denominator = (
        source_length**2
        + ell_value**2 * (
            width_fraction[:, None] + width_fraction[None, :]
        )
    ).ravel()
    pair_weight = (
        age_weight[:, None] * age_weight[None, :]
        * source_length**2
        / denominator.reshape(quadrature_nodes, quadrature_nodes)
    ).ravel()
    separation = np.atleast_1d(np.asarray(separation, dtype=float))
    return np.sum(
        pair_weight[None, :]
        * np.exp(-separation[:, None]**2 / (2.0 * denominator[None, :])),
        axis=1,
    )


def template_cnp_pair_basis_exact(
    pair_geometry: dict[str, np.ndarray],
    pair_counts: np.ndarray,
    n_bins: int,
    reference_radius_kpc: float,
    ell_value: float,
    epsilon_value: float,
    rate_log_slope_value: float,
    source_length: float,
    quadrature_nodes: int = 20,
) -> dict[str, np.ndarray]:
    """Evaluate the radial C_NP basis and legacy template moments.

    The observational likelihood uses only the ``uniform`` entries, which
    contain the locally varying axisymmetric rate squared.  The template and
    template-squared entries are retained solely so that existing emulator
    caches remain readable; coupling the residual distinct-event term to the
    non-axisymmetric source template is outside the validated C_NP theory.
    """
    kernel = gaussian_source_cnp_kernel(
        pair_geometry["separation"], ell_value, epsilon_value, source_length,
        quadrature_nodes=quadrature_nodes,
    )
    rate_squared = np.exp(
        2.0 * rate_log_slope_value
        * (pair_geometry["midpoint_radius"] - reference_radius_kpc)
    )
    bins = pair_geometry["output_bin"]
    numerator_moments = (
        pair_geometry["counts"],
        2.0 * pair_geometry["midpoint_modulation"],
        pair_geometry["midpoint_modulation_squared"],
    )
    variance_moments = (
        pair_geometry["counts"],
        2.0 * pair_geometry["endpoint_modulation"],
        pair_geometry["endpoint_modulation_squared"],
    )
    zero_lag = gaussian_source_cnp_kernel(
        np.asarray([0.0]), ell_value, epsilon_value, source_length,
        quadrature_nodes=quadrature_nodes,
    )[0]
    result: dict[str, np.ndarray] = {}
    suffixes = ("uniform", "template", "template_squared")
    for suffix, numerator_moment, variance_moment in zip(
        suffixes, numerator_moments, variance_moments
    ):
        numerator = np.bincount(
            bins, weights=rate_squared * numerator_moment * kernel,
            minlength=n_bins,
        )
        variance = np.bincount(
            bins, weights=rate_squared * variance_moment,
            minlength=n_bins,
        )
        result[f"numerator_{suffix}"] = np.divide(
            numerator, pair_counts, out=np.full(n_bins, np.nan),
            where=pair_counts > 0,
        )
        variance = zero_lag * np.divide(
            variance, pair_counts, out=np.full(n_bins, np.nan),
            where=pair_counts > 0,
        )
        result[f"variance_i_{suffix}"] = variance
        result[f"variance_j_{suffix}"] = variance.copy()
    return result


COMPONENT_ORDER = ("CP0", "CPN", "CM", "CNP")


def correlation_components_from_template_basis(
    basis: dict[str, np.ndarray],
    beta: float,
    eta: float = 1.0,
    source_excess_amplitude: float = 0.0,
    cnp_basis: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Return KTZ components with the optional isotropic C_NP extension."""
    numerator_parts = {
        "CP0": basis["numerator_uniform"],
        "CPN": eta * basis["numerator_harmonic"],
        "CM": beta * eta**2 * basis["numerator_mean"],
        "CNP": np.zeros_like(basis["numerator_uniform"]),
    }
    variance_i = (
        basis["variance_i_uniform"]
        + eta * basis["variance_i_harmonic"]
        + beta * eta**2 * basis["variance_i_mean"]
    )
    variance_j = (
        basis["variance_j_uniform"]
        + eta * basis["variance_j_harmonic"]
        + beta * eta**2 * basis["variance_j_mean"]
    )
    if source_excess_amplitude > 0.0:
        if cnp_basis is None:
            raise ValueError("cnp_basis is required for non-zero C_NP")
        cnp_scale = beta * source_excess_amplitude
        # Paper III validates only the isotropic, zeroth-order residual
        # distinct-event covariance C_NP.  Its radial normalisation follows
        # lambda_0(R)^2, but it must not inherit the arm template through
        # [1 + eta g(x)]^2.  In particular, eta intentionally does not
        # enter any C_NP contribution below.
        cnp_numerator = cnp_basis["numerator_uniform"]
        cnp_variance_i = cnp_basis["variance_i_uniform"]
        cnp_variance_j = cnp_basis["variance_j_uniform"]
        numerator_parts["CNP"] = cnp_scale * cnp_numerator
        variance_i = variance_i + cnp_scale * cnp_variance_i
        variance_j = variance_j + cnp_scale * cnp_variance_j
    denominator = np.sqrt(variance_i * variance_j)
    return {
        name: numerator_parts[name] / denominator for name in COMPONENT_ORDER
    }


def correlation_from_template_basis(*args, **kwargs) -> np.ndarray:
    return sum(correlation_components_from_template_basis(*args, **kwargs).values())


PAIR_BASIS_NAMES = (
    "numerator_uniform", "numerator_harmonic", "numerator_mean",
    "variance_i_uniform", "variance_i_harmonic", "variance_i_mean",
    "variance_j_uniform", "variance_j_harmonic", "variance_j_mean",
)

CNP_BASIS_NAMES = (
    "numerator_uniform", "numerator_template", "numerator_template_squared",
    "variance_i_uniform", "variance_i_template",
    "variance_i_template_squared", "variance_j_uniform",
    "variance_j_template", "variance_j_template_squared",
)


def build_template_pair_emulator(
    pair_geometry: dict[str, np.ndarray],
    pair_counts: np.ndarray,
    n_bins: int,
    reference_radius_kpc: float,
    ell_grid: np.ndarray,
    epsilon_grid: np.ndarray,
    rate_log_slope_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate the arbitrary-template Poisson/mean bases on a regular grid."""
    basis_grid = np.empty((
        len(ell_grid), len(epsilon_grid), len(rate_log_slope_grid),
        len(PAIR_BASIS_NAMES), n_bins,
    ))
    for ell_index, ell_value in enumerate(ell_grid):
        print(f"Template emulator row {ell_index + 1}/{len(ell_grid)}")
        for epsilon_index, epsilon_value in enumerate(epsilon_grid):
            basis = template_pair_basis(
                pair_geometry, pair_counts, n_bins, reference_radius_kpc,
                ell_value, epsilon_value, rate_log_slope_grid,
            )
            basis_grid[ell_index, epsilon_index] = np.transpose(
                np.asarray([basis[name] for name in PAIR_BASIS_NAMES]),
                (1, 0, 2),
            )
    return basis_grid


def build_template_cnp_emulator(
    pair_geometry: dict[str, np.ndarray],
    pair_counts: np.ndarray,
    n_bins: int,
    reference_radius_kpc: float,
    ell_grid: np.ndarray,
    epsilon_grid: np.ndarray,
    rate_log_slope_grid: np.ndarray,
    source_length_grid: np.ndarray,
    fine_bins_per_output_bin: int,
    quadrature_nodes: int = 14,
) -> np.ndarray:
    """Tabulate C_NP plus legacy template moments for cache compatibility.

    Only the ``uniform`` numerator and endpoint variances are consumed by the
    likelihood.  The other columns are retained until the existing notebook
    caches are migrated to a smaller C_NP-only format.
    """
    fine_centres = pair_geometry["fine_separation_centres"]
    fine_moments = (
        pair_geometry["counts"],
        2.0 * pair_geometry["midpoint_modulation"],
        pair_geometry["midpoint_modulation_squared"],
    )
    variance_moments = (
        pair_geometry["counts"],
        2.0 * pair_geometry["endpoint_modulation"],
        pair_geometry["endpoint_modulation_squared"],
    )
    weighted_fine = np.empty((len(rate_log_slope_grid), 3, len(fine_centres)))
    weighted_output = np.empty((len(rate_log_slope_grid), 3, n_bins))
    for rate_log_slope_index, rate_log_slope_value in enumerate(rate_log_slope_grid):
        rate_squared = np.exp(
            2.0 * rate_log_slope_value
            * (pair_geometry["midpoint_radius"] - reference_radius_kpc)
        )
        for moment_index, (fine_moment, variance_moment) in enumerate(zip(
            fine_moments, variance_moments
        )):
            weighted_fine[rate_log_slope_index, moment_index] = np.bincount(
                pair_geometry["fine_bin"],
                weights=rate_squared * fine_moment,
                minlength=len(fine_centres),
            )
            weighted_output[rate_log_slope_index, moment_index] = np.bincount(
                pair_geometry["output_bin"],
                weights=rate_squared * variance_moment,
                minlength=n_bins,
            )

    grid = np.empty((
        len(ell_grid), len(epsilon_grid), len(rate_log_slope_grid),
        len(source_length_grid), len(CNP_BASIS_NAMES), n_bins,
    ))
    counts = np.asarray(pair_counts, dtype=float)
    for ell_index, ell_value in enumerate(ell_grid):
        print(f"Template C_NP row {ell_index + 1}/{len(ell_grid)}")
        for epsilon_index, epsilon_value in enumerate(epsilon_grid):
            for length_index, source_length in enumerate(source_length_grid):
                kernel = gaussian_source_cnp_kernel(
                    fine_centres, ell_value, epsilon_value, source_length,
                    quadrature_nodes=quadrature_nodes,
                )
                numerator = np.divide(
                    (weighted_fine * kernel[None, None, :]).reshape(
                        len(rate_log_slope_grid), 3, n_bins,
                        fine_bins_per_output_bin,
                    ).sum(axis=3),
                    counts[None, None, :],
                    out=np.full((len(rate_log_slope_grid), 3, n_bins), np.nan),
                    where=counts[None, None, :] > 0,
                )
                zero_lag = gaussian_source_cnp_kernel(
                    np.asarray([0.0]), ell_value, epsilon_value,
                    source_length, quadrature_nodes=quadrature_nodes,
                )[0]
                variance = zero_lag * np.divide(
                    weighted_output, counts[None, None, :],
                    out=np.full((len(rate_log_slope_grid), 3, n_bins), np.nan),
                    where=counts[None, None, :] > 0,
                )
                # Duplicate the symmetric endpoint variance for i and j.
                grid[ell_index, epsilon_index, :, length_index] = np.concatenate(
                    (numerator, variance, variance), axis=1
                )
    return grid


def interpolate_regular_basis(
    grids: tuple[np.ndarray, ...],
    values: tuple[float, ...],
    basis_grid: np.ndarray,
    basis_names: tuple[str, ...],
    logarithmic_axes: tuple[bool, ...],
) -> dict[str, np.ndarray] | None:
    """Multilinearly interpolate a basis grid over linear/log coordinates."""
    brackets = []
    for grid, value, logarithmic in zip(grids, values, logarithmic_axes):
        coordinate_grid = np.log(grid) if logarithmic else np.asarray(grid)
        coordinate_value = np.log(value) if logarithmic else value
        if coordinate_value < coordinate_grid[0] or coordinate_value > coordinate_grid[-1]:
            return None
        index = min(np.searchsorted(coordinate_grid, coordinate_value) - 1,
                    len(coordinate_grid) - 2)
        index = max(index, 0)
        fraction = (
            (coordinate_value - coordinate_grid[index])
            / (coordinate_grid[index + 1] - coordinate_grid[index])
        )
        brackets.append(((index, 1.0 - fraction), (index + 1, fraction)))

    interpolated = np.zeros(basis_grid.shape[-2:])
    choices = [([], 1.0)]
    for bracket in brackets:
        choices = [
            (indices + [index], weight * node_weight)
            for indices, weight in choices for index, node_weight in bracket
        ]
    for indices, weight in choices:
        interpolated += weight * basis_grid[tuple(indices)]
    return {name: interpolated[index] for index, name in enumerate(basis_names)}
