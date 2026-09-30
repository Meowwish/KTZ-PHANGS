"""Create H-alpha-only source-geometry systematics from the primary notebooks.

The scientific calculation is unchanged except for the source template.  The
PHANGS environment labels select arm-associated catalogue regions, a global
logarithmic spiral is fitted to their H-alpha-weighted positions, and the
direct H-alpha grid fixes the first three harmonics before the metallicity
likelihood is evaluated.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("NGC628", "NGC1566")


def as_source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement + text[end_index:]


def source_geometry_block(galaxy_number: int) -> str:
    return f'''# Build the source field independently of the abundance residuals.  PHANGS
# arm-labelled H II regions fix a global logarithmic spiral, and the direct
# dereddened H-alpha grid fixes its harmonics and radial rate.
source_valid_grid = (
    geometry_mask & bpt_star_forming & halpha_ew_quality
    & np.isfinite(corrected_line_flux["ha"])
    & (corrected_line_flux["ha"] > 0)
    & (line_signal_to_noise["ha"] >= LINE_SN_MIN)
    & (line_signal_to_noise["hb"] >= LINE_SN_MIN)
)
native_pixel_area_kpc2 = (
    (native_pixel_scale_arcsec / 206265.0 * DISTANCE_MPC * 1.0e3)**2
)
projected_target_area_kpc2 = PIXEL_SIZE_KPC**2 * np.cos(
    np.deg2rad(INCLINATION_DEG)
)
target_to_native_pixel_area = projected_target_area_kpc2 / native_pixel_area_kpc2
source_halpha_flux = (
    corrected_line_flux["ha"][source_valid_grid] * target_to_native_pixel_area
)
source_halpha_grid = np.full(pixel_X.shape, np.nan)
source_halpha_grid[source_valid_grid] = source_halpha_flux
source_r_kpc = pixel_R[source_valid_grid]
source_phi = pixel_PHI[source_valid_grid]

from scripts.phangs_environment_template import build_environment_template

# Retain the released mask only for an independent morphology overlay.  The
# KTZ source template below is instead fixed entirely by the H-alpha source
# distribution before the metallicity residual is inspected.
environment_template = build_environment_template(
    ROOT, GALAXY_NAME, pixel_X, pixel_Y, pixel_R, pixel_PHI,
    project_face_on_to_sky, source_valid_grid, source_halpha_flux,
    ANNULUS_WIDTH_KPC, R_MAX_KPC, seed={galaxy_number},
)
environment_label_grid = environment_template["labels"]

source_arm_selection = np.isin(source_environment_catalog, [5, 6])
alpha_grid = np.linspace(-15.0, 15.0, 6001)


def fit_log_spiral(arm_number):
    log_radius = np.log(
        source_r_kpc_catalog[source_arm_selection] / r_ref_kpc
    )
    arm_phi = source_phi_catalog[source_arm_selection]
    arm_weight = source_halpha_catalog[source_arm_selection]
    coherence = np.asarray([
        np.abs(np.sum(arm_weight * np.exp(
            1j * (alpha * log_radius - arm_number * arm_phi)
        ))) / np.sum(arm_weight)
        for alpha in alpha_grid
    ])
    best = int(np.argmax(coherence))
    alpha = float(alpha_grid[best])
    raw_phase = alpha * log_radius - arm_number * arm_phi
    theta0_value = float(-np.angle(np.sum(
        arm_weight * np.exp(1j * raw_phase)
    )))
    return {{
        "m": int(arm_number), "alpha": alpha, "theta0": theta0_value,
        "pitch_deg": float(np.degrees(np.arctan2(arm_number, abs(alpha)))),
        "coherence": float(coherence[best]), "coherence_curve": coherence,
    }}


spiral_trials = [fit_log_spiral(order) for order in range(1, 5)]
spiral_fit = max(spiral_trials, key=lambda trial: trial["coherence"])
m_arms = int(spiral_fit["m"])
spiral_alpha = float(spiral_fit["alpha"])
theta0 = float(spiral_fit["theta0"])
pitch_angle_deg = float(spiral_fit["pitch_deg"])

source_theta = (
    spiral_alpha * np.log(source_r_kpc / r_ref_kpc)
    - m_arms * source_phi + theta0
)
source_weights = source_halpha_flux / np.sum(source_halpha_flux)
harmonic_orders = np.arange(1, 4, dtype=int)
harmonic_complex = np.asarray([
    2.0 * np.sum(source_weights * np.exp(1j * order * source_theta))
    for order in harmonic_orders
])
harmonic_amplitudes = np.abs(harmonic_complex)
harmonic_phases = -np.angle(harmonic_complex)

source_azimuth_block = np.floor(source_phi / (2.0 * np.pi) * 12).astype(int)
source_radial_edges = np.sqrt(np.linspace(R_MIN_KPC**2, R_MAX_KPC**2, 5))
source_radial_block = np.clip(
    np.digitize(source_r_kpc, source_radial_edges) - 1, 0, 3
)
source_block_id = 12 * source_radial_block + source_azimuth_block
unique_source_blocks = np.unique(source_block_id)
block_flux = np.asarray([
    np.sum(source_halpha_flux[source_block_id == block])
    for block in unique_source_blocks
])
block_complex = np.asarray([[
    np.sum(source_halpha_flux[source_block_id == block]
           * np.exp(1j * order * source_theta[source_block_id == block]))
    for order in harmonic_orders
] for block in unique_source_blocks])
harmonic_rng = np.random.default_rng({galaxy_number})
harmonic_bootstrap = np.empty((400, len(harmonic_orders)))
for bootstrap_index in range(len(harmonic_bootstrap)):
    draw = harmonic_rng.integers(0, len(unique_source_blocks),
                                 len(unique_source_blocks))
    harmonic_bootstrap[bootstrap_index] = np.abs(
        2.0 * np.sum(block_complex[draw], axis=0) / np.sum(block_flux[draw])
    )
harmonic_amplitude_error = np.std(harmonic_bootstrap, axis=0, ddof=1)


def evaluate_spiral_modulation(radius, azimuth):
    radius = np.asarray(radius, dtype=float)
    safe_radius = np.maximum(radius, 0.5 * PIXEL_SIZE_KPC)
    phase = (
        spiral_alpha * np.log(safe_radius / r_ref_kpc)
        - m_arms * np.asarray(azimuth) + theta0
    )
    modulation = sum(
        amplitude * np.cos(order * phase + harmonic_phase)
        for order, amplitude, harmonic_phase in zip(
            harmonic_orders, harmonic_amplitudes, harmonic_phases
        )
    )
    return np.where(
        (radius >= R_MIN_KPC) & (radius <= R_MAX_KPC), modulation, 0.0
    )


source_modulation_grid = evaluate_spiral_modulation(pixel_R, pixel_PHI)
midpoint_axis = np.linspace(
    float(pixel_X[0, 0]), float(pixel_X[0, -1]),
    2 * pixel_X.shape[1] - 1,
)
midpoint_x, midpoint_y = np.meshgrid(midpoint_axis, midpoint_axis)
midpoint_radius = np.hypot(midpoint_x, midpoint_y)
midpoint_phi = np.mod(np.arctan2(midpoint_y, midpoint_x), 2.0 * np.pi)
midpoint_source_modulation_grid = evaluate_spiral_modulation(
    midpoint_radius, midpoint_phi
)

theta_dense = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
rate_profile = 1.0 + sum(
    amplitude * np.cos(order * theta_dense + harmonic_phase)
    for order, amplitude, harmonic_phase in zip(
        harmonic_orders, harmonic_amplitudes, harmonic_phases
    )
)
unit_arm_rate_profile_min = float(np.min(rate_profile))
minimum_modulation = unit_arm_rate_profile_min - 1.0
unit_arm_positive_limit = (
    None if minimum_modulation >= 0.0 else 0.98 / abs(minimum_modulation)
)
mask_contrast_amplitude = float(harmonic_amplitudes[0])
mask_contrast_amplitude_error = float(harmonic_amplitude_error[0])
source_geometry_payload = {{
    "m": m_arms, "alpha": spiral_alpha, "theta0": theta0,
    "pitch_deg": pitch_angle_deg,
    "harmonic_amplitudes": harmonic_amplitudes.tolist(),
    "harmonic_phases": harmonic_phases.tolist(),
}}
environment_mask_hash = hashlib.sha256(json.dumps(
    source_geometry_payload, sort_keys=True
).encode()).hexdigest()
print("H-alpha log-spiral source geometry:", source_geometry_payload)
print("phase coherence:", spiral_fit["coherence"])

'''


def transform(galaxy: str) -> None:
    galaxy_number = int(galaxy[3:])
    spaced_name = f"NGC {galaxy_number}"
    input_path = ROOT / f"{galaxy}_PHANGS_implications.ipynb"
    output_path = ROOT / f"{galaxy}_PHANGS_logspiral_systematic.ipynb"
    notebook = json.loads(input_path.read_text())

    cell = "".join(notebook["cells"][0]["source"])
    cell = cell.replace(
        f"# {spaced_name} as a PHANGS-conditioned implication of KTZ",
        f"# {spaced_name} source-geometry systematic: H-alpha log spiral",
    )
    cell = cell.replace("The primary KTZ fit", "The systematic KTZ fit")
    cell = cell.replace(
        "estimate of the correlated measurement-noise contribution.  The same\n"
        "annular-mean projector used for the data is applied to every model curve\n"
        "on the actual finite mask.  The displayed\n",
        "estimate of the correlated measurement-noise contribution.  Its source\n"
        "geometry is fitted to H-alpha-weighted, arm-labelled HII regions before\n"
        "the metallicity map is inspected.  The same annular-mean projector used\n"
        "for the data is applied to every model curve on the actual finite mask.\n"
        "The displayed\n",
    )
    notebook["cells"][0]["source"] = as_source(cell)

    cell = "".join(notebook["cells"][1]["source"])
    old_output = "OUTPUT_DIR = OUTPUT_ROOT" + (
        "\n" if galaxy == "NGC628" else f' / "{galaxy}"\n'
    )
    cell = cell.replace(
        old_output,
        f'OUTPUT_DIR = OUTPUT_ROOT / "{galaxy}_logspiral_systematic"\n',
    )
    notebook["cells"][1]["source"] = as_source(cell)

    notebook["cells"][4]["source"] = as_source(
        "## 2. Establish the face-on geometry and an independent H-alpha spiral template\n\n"
        "The catalogue supplies the HII-region classifications and the coordinates used\n"
        "to fit a robust affine sky-to-face-on transformation. The non-axisymmetric\n"
        "source geometry is never fitted to the metallicity map. Instead, labels 5 and 6\n"
        "in the released Querejeta et al. (2021) environment classification select an\n"
        "H-alpha-weighted HII-region sample. We fit its arm number, logarithmic pitch,\n"
        "and phase before evaluating the abundance statistic. The first three source\n"
        "harmonics and the radial event-rate profile are then measured from the direct,\n"
        "attenuation-corrected H-alpha grid.\n"
    )

    cell = "".join(notebook["cells"][7]["source"])
    cell = replace_between(
        cell, "# Build the source field independently", "rate_edges_r25",
        source_geometry_block(galaxy_number),
    )
    cell = cell.replace(
        '    "environment_mask_sha256": environment_mask_hash,\n'
        '    "environment_mask_kind": "PHANGS simple pixel-level mask",\n'
        '    "arm_labels": [5, 6],\n'
        '    "interarm_labels": [4, 7],\n',
        '    "source_geometry_sha256": environment_mask_hash,\n'
        '    "source_geometry_kind": "H-alpha-fitted logarithmic spiral",\n'
        '    "spiral_arm_selection_labels": [5, 6],\n'
        '    "spiral_arm_number": m_arms,\n'
        '    "spiral_alpha": spiral_alpha,\n'
        '    "spiral_theta0": theta0,\n'
        '    "spiral_pitch_angle_deg": pitch_angle_deg,\n'
        '    "spiral_phase_coherence": spiral_fit["coherence"],\n'
        '    "harmonic_amplitudes": harmonic_amplitudes.tolist(),\n'
        '    "harmonic_phases": harmonic_phases.tolist(),\n',
    )
    cell = cell.replace(
        '        ("PHANGS arm-mask cells", int(np.sum(np.isin(environment_label_grid, [5, 6])))),\n'
        '        ("Halpha-calibrated mask contrast", mask_contrast_amplitude),\n',
        '        ("PHANGS arm-labelled grid cells (geometry diagnostic)", int(np.sum(np.isin(environment_label_grid, [5, 6])))),\n'
        '        ("Halpha log-spiral arm number", m_arms),\n'
        '        ("Halpha log-spiral pitch angle [deg]", pitch_angle_deg),\n'
        '        ("Halpha first-harmonic amplitude", mask_contrast_amplitude),\n',
    )
    notebook["cells"][7]["source"] = as_source(cell)

    cell = "".join(notebook["cells"][8]["source"])
    old = '''arm_mask_grid = np.isin(environment_label_grid, [5, 6])
for axis in axes[0]:
    axis.contour(
        pixel_X, pixel_Y, arm_mask_grid.astype(float), levels=[0.5],
        colors="black", linestyles="-", linewidths=0.9,
    )
axes[0, 1].set_title("Direct dereddened H-alpha; PHANGS arm boundaries")
'''
    new = '''arm_mask_grid = np.isin(environment_label_grid, [5, 6])
spiral_radius = np.linspace(R_MIN_KPC, R_MAX_KPC, 1200)
for axis in axes[0]:
    axis.contourf(
        pixel_X, pixel_Y, arm_mask_grid.astype(float), levels=[0.5, 1.5],
        colors=["#00b4d8"], alpha=0.22, zorder=4,
    )
    axis.contour(
        pixel_X, pixel_Y, arm_mask_grid.astype(float), levels=[0.5],
        colors="#006d77", linestyles="--", linewidths=1.4, zorder=5,
    )
    for arm_index in range(m_arms):
        spiral_phi = (
            spiral_alpha * np.log(spiral_radius / r_ref_kpc)
            + theta0 - 2.0 * np.pi * arm_index
        ) / m_arms
        axis.plot(
            spiral_radius * np.cos(spiral_phi),
            spiral_radius * np.sin(spiral_phi),
            color="black", ls="-", lw=1.25, zorder=6,
        )
axes[0, 1].set_title("Direct dereddened H-alpha; fitted source spiral")
'''
    cell = cell.replace(old, new)
    start = 'radial_edges = environment_template["radial_edges_kpc"]'
    end = '\n\nfor axis in axes.ravel():'
    coherence_plot = '''axes[1, 1].plot(
    alpha_grid, spiral_trials[m_arms - 1]["coherence_curve"], lw=1.2,
)
axes[1, 1].axvline(spiral_alpha, color="black", ls="--", lw=1.0)
axes[1, 1].set_xlim(alpha_grid[0], alpha_grid[-1])
axes[1, 1].set_ylim(0.0, max(0.5, 1.05 * spiral_fit["coherence"]))
axes[1, 1].set_xlabel(r"log-spiral radial wavenumber $\\alpha$")
axes[1, 1].set_ylabel("H-alpha-weighted phase coherence")
axes[1, 1].set_title(
    fr"fixed source geometry: $m={m_arms}$, $i={pitch_angle_deg:.1f}^\\circ$"
)
'''
    cell = replace_between(cell, start, end, coherence_plot)
    notebook["cells"][8]["source"] = as_source(cell)

    notebook["cells"][15]["source"] = as_source(
        "### 5.1 Full-pair KTZ emulator for the independent H-alpha spiral template\n\n"
        "The connected term samples the H-alpha-derived spiral modulation at every\n"
        "pair midpoint, while the mean-field term uses its values at both endpoints.\n"
        "The fitted geometry and its first three harmonics are fixed before the\n"
        "metallicity likelihood is evaluated.  The source template is treated in the\n"
        "local-source limit, while the transport and C_NP kernels retain their full\n"
        "parameter dependence.\n"
    )
    cell = "".join(notebook["cells"][17]["source"])
    split = cell.index("\n\n")
    notebook["cells"][17]["source"] = as_source(
        cell[:split] + "\n\n"
        "The arm-amplitude parameter $\eta$ multiplies the fixed H-alpha logarithmic-spiral\n"
        "template.  The likelihood forward-models the annular detrending applied to\n"
        "the abundance map; it does not fit an additive negative-tail offset.\n"
    )
    notebook["cells"][23]["source"] = as_source(
        "## 6. Multi-element extension\n\n"
        "This notebook tests the source geometry only in the single-element\n"
        "correlation.  A multi-element calculation would additionally have to evolve\n"
        "the H-alpha-derived source harmonics through the element-dependent delay\n"
        "channels, which is outside the scope of this systematic check.\n"
    )
    cell = "".join(notebook["cells"][26]["source"])
    cell = cell.replace("The primary KTZ fit", "The systematic KTZ fit")
    cell = cell.replace("mask-template moments", "source-template moments")
    notebook["cells"][26]["source"] = as_source(cell)

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    output_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(output_path)


if __name__ == "__main__":
    for target in TARGETS:
        transform(target)
