"""Memory-bounded exhaustive pair compression for PHANGS face-on grids."""

from __future__ import annotations

import numpy as np


def _paired_slices(length: int, delta: int) -> tuple[slice, slice]:
    if delta >= 0:
        return slice(0, length - delta), slice(delta, length)
    return slice(-delta, length), slice(0, length + delta)


def compress_regular_grid_pairs(
    valid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    theta_grid: np.ndarray,
    harmonic_orders: np.ndarray,
    harmonic_phases: np.ndarray,
    spiral_alpha: float,
    spiral_m: int,
    spiral_theta0: float,
    spiral_reference_radius: float,
    separation_edges: np.ndarray,
    radial_edges: np.ndarray,
    separation_subbins: int = 12,
) -> dict[str, np.ndarray]:
    """Visit every unordered grid pair and accumulate finite-window moments.

    The compression is deterministic, not a pair sample.  Separation and
    midpoint radius are represented by fine-cell centres; all counts and all
    trigonometric moment sums include every valid pair exactly once.
    """
    valid = np.asarray(valid, dtype=bool)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    theta_grid = np.asarray(theta_grid, dtype=float)
    harmonic_orders = np.asarray(harmonic_orders, dtype=int)
    harmonic_phases = np.asarray(harmonic_phases, dtype=float)
    separation_edges = np.asarray(separation_edges, dtype=float)
    radial_edges = np.asarray(radial_edges, dtype=float)

    if not (valid.shape == x_grid.shape == y_grid.shape == theta_grid.shape):
        raise ValueError("All grid inputs must have identical shapes.")
    if len(harmonic_orders) != len(harmonic_phases):
        raise ValueError("Harmonic orders and phases must have equal lengths.")
    if separation_subbins < 1:
        raise ValueError("separation_subbins must be positive.")

    height, width = valid.shape
    pixel_x = float(np.nanmedian(np.diff(x_grid, axis=1)))
    pixel_y = float(np.nanmedian(np.diff(y_grid, axis=0)))
    pixel_size = 0.5 * (abs(pixel_x) + abs(pixel_y))
    if not np.isclose(abs(pixel_x), abs(pixel_y), rtol=1.0e-6):
        raise ValueError("Pair compression requires a square regular grid.")

    n_output_bins = len(separation_edges) - 1
    fine_edges = np.linspace(
        separation_edges[0], separation_edges[-1],
        n_output_bins * separation_subbins + 1,
    )
    fine_centres = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    radial_centres = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    n_fine = len(fine_centres)
    n_radial = len(radial_centres)
    n_cells = n_fine * n_radial
    n_harmonics = len(harmonic_orders)

    counts = np.zeros(n_cells, dtype=np.int64)
    midpoint_cosine = np.zeros((n_harmonics, n_cells), dtype=float)
    endpoint_cosine = np.zeros((n_harmonics, n_cells), dtype=float)
    endpoint_cross = np.zeros(
        (n_harmonics, n_harmonics, n_cells), dtype=float
    )
    endpoint_self = np.zeros(
        (n_harmonics, n_harmonics, n_cells), dtype=float
    )

    endpoint_phase_cosine = np.asarray([
        np.cos(order * theta_grid + phase)
        for order, phase in zip(harmonic_orders, harmonic_phases)
    ])

    maximum_offset = int(np.floor(separation_edges[-1] / pixel_size))
    pair_total = 0
    for delta_y in range(maximum_offset + 1):
        delta_x_min = 1 if delta_y == 0 else -maximum_offset
        for delta_x in range(delta_x_min, maximum_offset + 1):
            separation = pixel_size * np.hypot(delta_x, delta_y)
            fine_bin = np.searchsorted(fine_edges, separation, side="right") - 1
            if fine_bin < 0 or fine_bin >= n_fine:
                continue

            y_i, y_j = _paired_slices(height, delta_y)
            x_i, x_j = _paired_slices(width, delta_x)
            selected = valid[y_i, x_i] & valid[y_j, x_j]
            if not np.any(selected):
                continue

            midpoint_x = 0.5 * (
                x_grid[y_i, x_i][selected] + x_grid[y_j, x_j][selected]
            )
            midpoint_y = 0.5 * (
                y_grid[y_i, x_i][selected] + y_grid[y_j, x_j][selected]
            )
            midpoint_radius = np.hypot(midpoint_x, midpoint_y)
            radial_bin = (
                np.searchsorted(radial_edges, midpoint_radius, side="right") - 1
            )
            in_radial_range = (radial_bin >= 0) & (radial_bin < n_radial)
            if not np.any(in_radial_range):
                continue

            midpoint_x = midpoint_x[in_radial_range]
            midpoint_y = midpoint_y[in_radial_range]
            radial_bin = radial_bin[in_radial_range]
            cell = fine_bin * n_radial + radial_bin
            pair_total += len(cell)
            counts += np.bincount(cell, minlength=n_cells)

            midpoint_radius = np.hypot(midpoint_x, midpoint_y)
            midpoint_azimuth = np.arctan2(midpoint_y, midpoint_x)
            midpoint_theta = np.zeros_like(midpoint_radius)
            nonzero_midpoint = midpoint_radius > 0.0
            midpoint_theta[nonzero_midpoint] = (
                spiral_alpha * np.log(
                    midpoint_radius[nonzero_midpoint]
                    / spiral_reference_radius
                )
                - spiral_m * midpoint_azimuth[nonzero_midpoint]
                + spiral_theta0
            )
            endpoint_i = endpoint_phase_cosine[:, y_i, x_i][:, selected]
            endpoint_j = endpoint_phase_cosine[:, y_j, x_j][:, selected]
            endpoint_i = endpoint_i[:, in_radial_range]
            endpoint_j = endpoint_j[:, in_radial_range]
            midpoint_values = np.asarray([
                np.cos(order * midpoint_theta + phase)
                for order, phase in zip(harmonic_orders, harmonic_phases)
            ])

            for first in range(n_harmonics):
                midpoint_cosine[first] += np.bincount(
                    cell, weights=midpoint_values[first], minlength=n_cells
                )
                endpoint_cosine[first] += np.bincount(
                    cell,
                    weights=0.5 * (endpoint_i[first] + endpoint_j[first]),
                    minlength=n_cells,
                )
                for second in range(n_harmonics):
                    endpoint_cross[first, second] += np.bincount(
                        cell,
                        weights=0.5 * (
                            endpoint_i[first] * endpoint_j[second]
                            + endpoint_j[first] * endpoint_i[second]
                        ),
                        minlength=n_cells,
                    )
                    endpoint_self[first, second] += np.bincount(
                        cell,
                        weights=0.5 * (
                            endpoint_i[first] * endpoint_i[second]
                            + endpoint_j[first] * endpoint_j[second]
                        ),
                        minlength=n_cells,
                    )

    occupied = counts > 0
    occupied_index = np.flatnonzero(occupied)
    fine_bin = occupied_index // n_radial
    radial_bin = occupied_index % n_radial
    output_bin = fine_bin // separation_subbins
    compressed_counts = counts[occupied]
    if int(np.sum(compressed_counts)) != pair_total:
        raise RuntimeError("Compressed pair-count bookkeeping failed.")

    return {
        "counts": compressed_counts,
        "separation": fine_centres[fine_bin],
        "midpoint_radius": radial_centres[radial_bin],
        "fine_bin": fine_bin.astype(np.int32),
        "output_bin": output_bin.astype(np.int16),
        "midpoint_cosine": midpoint_cosine[:, occupied],
        "endpoint_cosine": endpoint_cosine[:, occupied],
        "endpoint_cross": endpoint_cross[:, :, occupied],
        "endpoint_self": endpoint_self[:, :, occupied],
        "fine_separation_centres": fine_centres,
        "radial_centres": radial_centres,
        "separation_subbins": np.asarray(separation_subbins),
    }


def compress_regular_grid_template_pairs(
    valid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    source_modulation_grid: np.ndarray,
    midpoint_modulation_grid: np.ndarray,
    separation_edges: np.ndarray,
    radial_edges: np.ndarray,
    separation_subbins: int = 12,
) -> dict[str, np.ndarray]:
    """Compress every valid pair for an arbitrary two-dimensional template.

    ``source_modulation_grid`` is sampled at the observed cell centres, while
    ``midpoint_modulation_grid`` has twice the linear sampling minus one cell
    in each direction.  Thus pair midpoints, including half-pixel locations,
    are read directly rather than reconstructed from a fitted spiral phase.
    The returned sums retain all terms through second order in the template
    amplitude, which is sufficient for the connected-Poisson, mean-field, and
    non-Poisson pieces of KTZ.
    """
    valid = np.asarray(valid, dtype=bool)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    source_modulation_grid = np.asarray(source_modulation_grid, dtype=float)
    midpoint_modulation_grid = np.asarray(
        midpoint_modulation_grid, dtype=float
    )
    separation_edges = np.asarray(separation_edges, dtype=float)
    radial_edges = np.asarray(radial_edges, dtype=float)

    if not (
        valid.shape == x_grid.shape == y_grid.shape
        == source_modulation_grid.shape
    ):
        raise ValueError("Endpoint grid inputs must have identical shapes.")
    expected_midpoint_shape = (2 * valid.shape[0] - 1, 2 * valid.shape[1] - 1)
    if midpoint_modulation_grid.shape != expected_midpoint_shape:
        raise ValueError(
            "Midpoint template must have shape " f"{expected_midpoint_shape}."
        )
    if separation_subbins < 1:
        raise ValueError("separation_subbins must be positive.")

    height, width = valid.shape
    pixel_x = float(np.nanmedian(np.diff(x_grid, axis=1)))
    pixel_y = float(np.nanmedian(np.diff(y_grid, axis=0)))
    pixel_size = 0.5 * (abs(pixel_x) + abs(pixel_y))
    if not np.isclose(abs(pixel_x), abs(pixel_y), rtol=1.0e-6):
        raise ValueError("Pair compression requires a square regular grid.")

    n_output_bins = len(separation_edges) - 1
    fine_edges = np.linspace(
        separation_edges[0], separation_edges[-1],
        n_output_bins * separation_subbins + 1,
    )
    fine_centres = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    radial_centres = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    n_fine = len(fine_centres)
    n_radial = len(radial_centres)
    n_cells = n_fine * n_radial

    names = (
        "counts",
        "midpoint_modulation",
        "midpoint_modulation_squared",
        "endpoint_modulation",
        "endpoint_modulation_squared",
        "endpoint_modulation_cross",
    )
    sums = {name: np.zeros(n_cells, dtype=float) for name in names}
    sums["counts"] = np.zeros(n_cells, dtype=np.int64)

    maximum_offset = int(np.floor(separation_edges[-1] / pixel_size))
    pair_total = 0
    for delta_y in range(maximum_offset + 1):
        delta_x_min = 1 if delta_y == 0 else -maximum_offset
        for delta_x in range(delta_x_min, maximum_offset + 1):
            separation = pixel_size * np.hypot(delta_x, delta_y)
            fine_bin = np.searchsorted(fine_edges, separation, side="right") - 1
            if fine_bin < 0 or fine_bin >= n_fine:
                continue

            y_i, y_j = _paired_slices(height, delta_y)
            x_i, x_j = _paired_slices(width, delta_x)
            selected = valid[y_i, x_i] & valid[y_j, x_j]
            if not np.any(selected):
                continue

            local_row, local_column = np.nonzero(selected)
            endpoint_i_row = local_row + (y_i.start or 0)
            endpoint_i_column = local_column + (x_i.start or 0)
            midpoint_row = 2 * endpoint_i_row + delta_y
            midpoint_column = 2 * endpoint_i_column + delta_x

            midpoint_x = 0.5 * (
                x_grid[y_i, x_i][selected] + x_grid[y_j, x_j][selected]
            )
            midpoint_y = 0.5 * (
                y_grid[y_i, x_i][selected] + y_grid[y_j, x_j][selected]
            )
            midpoint_radius = np.hypot(midpoint_x, midpoint_y)
            radial_bin = (
                np.searchsorted(radial_edges, midpoint_radius, side="right") - 1
            )
            in_radial_range = (radial_bin >= 0) & (radial_bin < n_radial)
            if not np.any(in_radial_range):
                continue

            radial_bin = radial_bin[in_radial_range]
            cell = fine_bin * n_radial + radial_bin
            pair_total += len(cell)
            sums["counts"] += np.bincount(cell, minlength=n_cells)

            midpoint_value = midpoint_modulation_grid[
                midpoint_row[in_radial_range], midpoint_column[in_radial_range]
            ]
            endpoint_i = source_modulation_grid[y_i, x_i][selected][
                in_radial_range
            ]
            endpoint_j = source_modulation_grid[y_j, x_j][selected][
                in_radial_range
            ]
            weights = {
                "midpoint_modulation": midpoint_value,
                "midpoint_modulation_squared": midpoint_value**2,
                "endpoint_modulation": 0.5 * (endpoint_i + endpoint_j),
                "endpoint_modulation_squared": 0.5 * (
                    endpoint_i**2 + endpoint_j**2
                ),
                "endpoint_modulation_cross": endpoint_i * endpoint_j,
            }
            for name, value in weights.items():
                sums[name] += np.bincount(
                    cell, weights=value, minlength=n_cells
                )

    occupied = sums["counts"] > 0
    occupied_index = np.flatnonzero(occupied)
    fine_bin = occupied_index // n_radial
    radial_bin = occupied_index % n_radial
    compressed_counts = sums["counts"][occupied]
    if int(np.sum(compressed_counts)) != pair_total:
        raise RuntimeError("Compressed pair-count bookkeeping failed.")

    result = {
        name: value[occupied] for name, value in sums.items()
    }
    result.update({
        "separation": fine_centres[fine_bin],
        "midpoint_radius": radial_centres[radial_bin],
        "fine_bin": fine_bin.astype(np.int32),
        "output_bin": (fine_bin // separation_subbins).astype(np.int16),
        "fine_separation_centres": fine_centres,
        "radial_centres": radial_centres,
        "separation_subbins": np.asarray(separation_subbins),
    })
    return result
