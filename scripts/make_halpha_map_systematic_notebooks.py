"""Create direct two-dimensional H-alpha source-template calculations.

The 120-pc H-alpha map contains both galaxy-scale source geometry and compact
H II-region clustering.  To avoid assigning the same compact structure to the
deterministic source template and to C_NP, the central calculation smooths the
attenuation-corrected H-alpha map to a pre-specified 1-kpc FWHM and normalises
it in 0.2-kpc annuli.  Fixed 0.5- and 1.5-kpc calculations quantify smoothing
sensitivity without selecting the smoothing scale from the metallicity fit.
The resulting two-dimensional modulation is fixed before the metallicity
correlation is evaluated; only its global scale remains a KTZ likelihood
parameter.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("NGC628", "NGC1566", "NGC4254", "NGC4303")
FWHM_VALUES_KPC = (1.0, 0.5, 1.5)


def as_source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement + text[end_index:]


def fwhm_tag(fwhm_kpc: float) -> str:
    return f"{fwhm_kpc:.1f}".replace(".", "p")


def variant_name(fwhm_kpc: float) -> str:
    if fwhm_kpc == 1.0:
        return "halpha_map_systematic"
    return f"halpha_map_fwhm{fwhm_tag(fwhm_kpc)}_systematic"


def source_geometry_block(galaxy_number: int, fwhm_kpc: float) -> str:
    return f'''# Build a deterministic source field independently of the abundance residuals.
# The direct dereddened H-alpha map is smoothed to a {fwhm_kpc:.1f}-kpc FWHM so that this
# template retains arm-scale structure and limits overlap with compact
# source clustering represented by C_NP; separation is not guaranteed.
# It is then normalised separately in
# every 0.2-kpc annulus.
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

# The released environment mask is retained only as an external overlay.
environment_template = build_environment_template(
    ROOT, GALAXY_NAME, pixel_X, pixel_Y, pixel_R, pixel_PHI,
    project_face_on_to_sky, source_valid_grid, source_halpha_flux,
    ANNULUS_WIDTH_KPC, R_MAX_KPC, seed={galaxy_number},
)
environment_label_grid = environment_template["labels"]

HALPHA_TEMPLATE_FWHM_KPC = {fwhm_kpc:.1f}
template_sigma_pixel = (
    HALPHA_TEMPLATE_FWHM_KPC / 2.354820045 / PIXEL_SIZE_KPC
)
template_weight = gaussian_filter(
    source_valid_grid.astype(float), template_sigma_pixel,
    mode="constant", cval=0.0, truncate=4.0,
)
template_flux = gaussian_filter(
    np.where(source_valid_grid, source_halpha_grid, 0.0),
    template_sigma_pixel, mode="constant", cval=0.0, truncate=4.0,
)
template_surface_density = np.divide(
    template_flux, template_weight,
    out=np.full(pixel_X.shape, np.nan), where=template_weight > 0.05,
)

source_modulation_grid = np.zeros(pixel_X.shape, dtype=float)
template_radial_edges = np.arange(
    0.0, R_MAX_KPC + ANNULUS_WIDTH_KPC, ANNULUS_WIDTH_KPC,
)
template_radial_bin = np.digitize(pixel_R, template_radial_edges) - 1
for template_bin in range(len(template_radial_edges) - 1):
    selected = (
        geometry_mask & (template_radial_bin == template_bin)
        & np.isfinite(template_surface_density)
    )
    if np.count_nonzero(selected) < MIN_PIXELS_PER_ANNULUS:
        continue
    annular_halpha_mean = float(np.mean(template_surface_density[selected]))
    source_modulation_grid[selected] = (
        template_surface_density[selected] / annular_halpha_mean - 1.0
    )
source_modulation_grid = np.where(
    geometry_mask, source_modulation_grid, 0.0,
)

# Pair midpoints lie on the half-pixel grid.  Bilinear interpolation of the
# already-smoothed template retains its measured non-logarithmic branches.
midpoint_axis = np.linspace(
    float(pixel_X[0, 0]), float(pixel_X[0, -1]),
    2 * pixel_X.shape[1] - 1,
)
midpoint_x, midpoint_y = np.meshgrid(midpoint_axis, midpoint_axis)
midpoint_coordinates = np.vstack((
    ((midpoint_y - pixel_centres[0]) / PIXEL_SIZE_KPC).ravel(),
    ((midpoint_x - pixel_centres[0]) / PIXEL_SIZE_KPC).ravel(),
))
midpoint_source_modulation_grid = map_coordinates(
    source_modulation_grid, midpoint_coordinates, order=1,
    mode="constant", cval=0.0,
).reshape(midpoint_x.shape)

template_values = source_modulation_grid[geometry_mask]
minimum_modulation = float(np.min(template_values))
unit_arm_rate_profile_min = 1.0 + minimum_modulation
unit_arm_positive_limit = (
    None if minimum_modulation >= 0.0 else 0.98 / abs(minimum_modulation)
)
# The modulation is already dimensionless.  Its RMS is recorded as a
# descriptive contrast; the 0.35 uncertainty encodes tracer-to-enrichment
# mismatch and matches the minimum width of the other morphology priors.
mask_contrast_amplitude = float(np.sqrt(np.mean(template_values**2)))
mask_contrast_amplitude_error = 0.35 * mask_contrast_amplitude
source_geometry_payload = {{
    "kind": "annular-normalised {{:.1f}}-kpc-smoothed H-alpha map".format(
        HALPHA_TEMPLATE_FWHM_KPC
    ),
    "fwhm_kpc": HALPHA_TEMPLATE_FWHM_KPC,
    "modulation_sha256": hashlib.sha256(
        np.ascontiguousarray(source_modulation_grid).tobytes()
    ).hexdigest(),
}}
environment_mask_hash = hashlib.sha256(json.dumps(
    source_geometry_payload, sort_keys=True
).encode()).hexdigest()
print("Direct H-alpha source template:", source_geometry_payload)
print(
    "Template RMS/minimum modulation/positive eta limit:",
    mask_contrast_amplitude, minimum_modulation, unit_arm_positive_limit,
)

'''


def morphology_figure_cell(galaxy: str, fwhm_kpc: float) -> str:
    stem = galaxy.lower()
    return f'''fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2), constrained_layout=True)

residual_image = axes[0, 0].pcolormesh(
    pixel_X, pixel_Y,
    np.where(pixel_valid_grid, pixel_residual_centered_grid, np.nan),
    shading="nearest", cmap="RdBu_r",
    norm=TwoSlopeNorm(vmin=-0.13, vcenter=0.0, vmax=0.13),
)
axes[0, 0].set_title(r"Direct MUSE S-calibration residual used for $\\xi$")
fig.colorbar(
    residual_image, ax=axes[0, 0],
    label=r"$\\Delta[12+\\log({{\\rm O/H}})]$ [dex]",
)

halpha_plot = np.full(pixel_X.shape, np.nan)
halpha_plot[source_valid_grid] = (
    np.log10(source_halpha_flux) - np.median(np.log10(source_halpha_flux))
)
halpha_image = axes[0, 1].pcolormesh(
    pixel_X, pixel_Y, halpha_plot, shading="nearest", cmap="magma",
    vmin=-1.25, vmax=1.25,
)
axes[0, 1].set_title("Direct dereddened H-alpha")
fig.colorbar(
    halpha_image, ax=axes[0, 1], label="log H-alpha cell flux / median",
)

arm_mask_grid = np.isin(environment_label_grid, [5, 6])
for axis in axes[0]:
    axis.contourf(
        pixel_X, pixel_Y, arm_mask_grid.astype(float), levels=[0.5, 1.5],
        colors=["#2ca25f"], alpha=0.30, zorder=4,
    )
    axis.contour(
        pixel_X, pixel_Y, arm_mask_grid.astype(float), levels=[0.5],
        colors="#006d2c", linestyles="--", linewidths=1.5, zorder=5,
    )

template_image = axes[1, 0].pcolormesh(
    pixel_X, pixel_Y, np.where(geometry_mask, source_modulation_grid, np.nan),
    shading="nearest", cmap="RdBu_r", norm=TwoSlopeNorm(
        vmin=min(-1.0, float(np.nanpercentile(source_modulation_grid[geometry_mask], 2))),
        vcenter=0.0,
        vmax=max(1.0, float(np.nanpercentile(source_modulation_grid[geometry_mask], 98))),
    ),
)
axes[1, 0].contour(
    pixel_X, pixel_Y, source_modulation_grid, levels=[0.0],
    colors="black", linewidths=1.3,
)
axes[1, 0].set_title("Fixed {fwhm_kpc:.1f}-kpc-smoothed H-alpha source modulation")
fig.colorbar(template_image, ax=axes[1, 0], label=r"$g_{{\\rm H\\alpha}}(\\boldsymbol{{x}})$")

radial_centres = 0.5 * (
    template_radial_edges[:-1] + template_radial_edges[1:]
)
radial_template_mean = np.asarray([
    np.mean(source_modulation_grid[
        geometry_mask & (template_radial_bin == bin_index)
    ]) if np.any(geometry_mask & (template_radial_bin == bin_index)) else np.nan
    for bin_index in range(len(radial_centres))
])
axes[1, 1].plot(radial_centres, radial_template_mean, "o-", lw=1.2)
axes[1, 1].axhline(0.0, color="black", ls="--", lw=0.9)
axes[1, 1].set_xlim(R_MIN_KPC, R_MAX_KPC)
axes[1, 1].set_xlabel("face-on radius [kpc]")
axes[1, 1].set_ylabel("annular mean modulation")
axes[1, 1].set_title("Numerical check of annular normalisation")

for axis in axes.ravel():
    if axis is not axes[1, 1]:
        axis.set_aspect("equal")
        axis.set_xlabel("x [kpc]")
        axis.set_ylabel("y [kpc]")

fig.savefig(OUTPUT_DIR / "{stem}_phangs_morphology.png", bbox_inches="tight")
display_figure(fig)
'''


def refresh_markdown(notebook: dict, fwhm_kpc: float) -> None:
    """Keep source-template descriptions consistent without touching results."""
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            source = "".join(cell.get("source", []))
            source = source.replace(
                "# template contains arm-scale structure but does not duplicate the compact\n"
                "# source clustering represented by C_NP.  It is then normalised separately in\n",
                "# template retains arm-scale structure and limits overlap with compact\n"
                "# source clustering represented by C_NP; separation is not guaranteed.\n"
                "# It is then normalised separately in\n",
            )
            cell["source"] = source.splitlines(keepends=True)
            continue
        source = "".join(cell["source"])
        source = source.replace("the optional isotropic appendix", "the isotropic appendix")
        source = source.replace("KTZ + isotropic C_NP", "complete KTZ")
        source = source.replace(
            "The PHANGS-mask KTZ control", "The direct-H-alpha KTZ fit"
        )
        source = source.replace("mask-template moments", "source-template moments")
        source = source.replace(
            "determine the large-scale arm geometry; they do not select abundance pixels.",
            "determine the disc-coordinate transformation; they do not select abundance pixels.",
        )
        source = source.replace(
            "while separating it from the compact residual source clustering",
            "while limiting overlap with the compact residual source clustering",
        )
        cell["source"] = as_source(source)
    notebook["cells"][15]["source"] = as_source(
        "### 5.1 Full-pair KTZ emulator for the direct two-dimensional H-alpha template\n\n"
        "The connected term samples the fixed smoothed H-alpha modulation at every "
        "pair midpoint, while the mean-field term uses its values at both endpoints. "
        "The template retains non-logarithmic branches and asymmetry. "
        f"Its {fwhm_kpc:.1f}-kpc low-pass FWHM is not the 120-pc observing beam or a "
        "fitted diffusion scale. Mask-aware smoothing limits, but does not guarantee "
        "the removal of, compact structure that could overlap with $C_{\\rm NP}$. "
        "The 0.5/1.0/1.5-kpc comparison tests this allocation of power.\n"
    )
    notebook["cells"][17]["source"] = as_source(
        "### 5.1.1 Isotropic distinct-event kernel\n\n"
        "$C_{\\rm NP}$ is filtered by injection and diffusion and enters the same "
        "numerator and endpoint moments as the other terms. It uses only the squared "
        "axisymmetric rate, not the non-axisymmetric H-alpha template.\n"
    )
    notebook["cells"][19]["source"] = as_source(
        "### 5.2 Seven-parameter reparameterised KTZ and KT18 ensemble MCMC\n\n"
        "The source-scale coordinate multiplies the fixed annulus-normalised "
        "H-alpha intensity map. No arm number, pitch angle, phase, width, branch, "
        "or individual template pixel is fitted to the abundance correlation. "
        "All four physical components belong to one complete KTZ likelihood. "
        "The likelihood forward-models annular detrending and has no additive "
        "negative-tail offset. The split-Rhat rule is 1.05; a chain that fails "
        "at the declared length cap remains a diagnostic, not a formal constraint.\n"
    )
    notebook["cells"][23]["source"] = as_source(
        "### 5.4 Posterior visualisation\n\n"
        "Total curves are pointwise posterior medians. A single joint posterior "
        "draw is used for exactly additive component stacks; marginal component "
        "medians need not add to the median total.\n"
    )
    notebook["cells"][25]["source"] = as_source(
        "## 6. Multi-element scope\n\n"
        "This calculation tests only the oxygen correlation. A consistent "
        "multi-element application would require the same source-template moments "
        "and channel-dependent time histories; the former single-wavenumber "
        "forecast is not evaluated for this arbitrary two-dimensional map.\n"
    )
    source = "".join(notebook["cells"][28]["source"])
    stale = "- The multi-element panel remains a galaxy-conditioned forecast:"
    if stale in source:
        source = source[:source.index(stale)]
    additions_start = "- The H-alpha template and abundance map share line noise,"
    if additions_start in source:
        source = source[:source.index(additions_start)]
    source += (
        "- The H-alpha template and abundance map share line noise, attenuation "
        "correction, and selection. Not using abundance residual values to build "
        "the template does not make the input data statistically independent.\n"
        "- Template smoothing and geometry uncertainty are not included in the "
        "conditional posterior bands; the supplied sensitivity runs measure this "
        "additional systematic.\n"
    )
    notebook["cells"][28]["source"] = as_source(source)


def transform(galaxy: str, fwhm_kpc: float) -> None:
    galaxy_number = int(galaxy[3:])
    spaced_name = f"NGC {galaxy_number}"
    variant = variant_name(fwhm_kpc)
    role = (
        "primary direct H-alpha source map"
        if fwhm_kpc == 1.0 else "pre-specified smoothing sensitivity"
    )
    input_path = ROOT / f"{galaxy}_PHANGS_implications.ipynb"
    output_path = ROOT / f"{galaxy}_PHANGS_{variant}.ipynb"
    notebook = json.loads(input_path.read_text(encoding="utf-8"))

    cell = "".join(notebook["cells"][0]["source"])
    cell = cell.replace(
        f"# {spaced_name} as a PHANGS-conditioned implication of KTZ",
        f"# {spaced_name}: direct H-alpha map ({fwhm_kpc:.1f}-kpc FWHM)",
    )
    cell = cell.replace(
        "The primary KTZ fit with isotropic C_NP",
        f"The direct-H-alpha-template KTZ fit ({fwhm_kpc:.1f}-kpc FWHM) with isotropic C_NP",
    )
    notebook["cells"][0]["source"] = as_source(cell)

    cell = "".join(notebook["cells"][1]["source"])
    old_output = f'OUTPUT_DIR = OUTPUT_ROOT / "{galaxy}"\n'
    if old_output not in cell:
        old_output = "OUTPUT_DIR = OUTPUT_ROOT\n"
    if old_output not in cell:
        raise ValueError(f"Cannot locate OUTPUT_DIR assignment in {input_path}")
    cell = cell.replace(
        old_output,
        f'OUTPUT_DIR = OUTPUT_ROOT / "{galaxy}_{variant}"\n',
    )
    notebook["cells"][1]["source"] = as_source(cell)

    notebook["cells"][4]["source"] = as_source(
        "## 2. Establish the face-on geometry and an independent two-dimensional H-alpha template\n\n"
        "The non-axisymmetric source field is not fitted to the metallicity map. "
        "The direct attenuation-corrected H-alpha grid is smoothed to a fixed "
        f"{fwhm_kpc:.1f} kpc FWHM to retain arm-scale morphology while separating it from the "
        "compact residual source clustering represented by $C_{\\rm NP}$. The "
        "smoothed field is normalised independently in every 0.2 kpc annulus. "
        f"This is the {role}; the released PHANGS arm labels are used only as an external overlay.\n"
    )

    cell = "".join(notebook["cells"][7]["source"])
    cell = replace_between(
        cell, "# Build the source field independently", "rate_edges_r25",
        source_geometry_block(galaxy_number, fwhm_kpc),
    )
    cell = cell.replace(
        '    "environment_mask_sha256": environment_mask_hash,\n'
        '    "environment_mask_kind": "PHANGS simple pixel-level mask",\n'
        '    "arm_labels": [5, 6],\n'
        '    "interarm_labels": [4, 7],\n',
        '    "source_geometry_sha256": environment_mask_hash,\n'
        '    "source_geometry_kind": source_geometry_payload["kind"],\n'
        '    "halpha_template_fwhm_kpc": HALPHA_TEMPLATE_FWHM_KPC,\n'
        '    "environment_mask_role": "external morphology overlay only",\n'
        '    "arm_labels": [5, 6],\n'
        '    "interarm_labels": [4, 7],\n',
    )
    cell = cell.replace(
        '        ("PHANGS arm-mask cells", int(np.sum(np.isin(environment_label_grid, [5, 6])))),\n'
        '        ("Halpha-calibrated mask contrast", mask_contrast_amplitude),\n',
        '        ("PHANGS arm-mask cells (overlay only)", int(np.sum(np.isin(environment_label_grid, [5, 6])))),\n'
        '        ("Halpha-template RMS contrast", mask_contrast_amplitude),\n'
        '        ("Halpha-template smoothing FWHM [kpc]", HALPHA_TEMPLATE_FWHM_KPC),\n',
    )
    notebook["cells"][7]["source"] = as_source(cell)
    notebook["cells"][8]["source"] = as_source(
        morphology_figure_cell(galaxy, fwhm_kpc)
    )

    notebook["cells"][15]["source"] = as_source(
        "### 5.1 Full-pair KTZ emulator for the direct two-dimensional H-alpha template\n\n"
        "The connected term samples the fixed smoothed H-alpha modulation at "
        "every pair midpoint, while the mean-field term uses its values at both "
        "endpoints. The template retains non-logarithmic branches and asymmetry. "
        f"Its {fwhm_kpc:.1f}-kpc smoothing limits, but does not guarantee the removal of, "
        "compact H II-region structure that could overlap with $C_{\\rm NP}$. "
        "The 0.5/1.0/1.5-kpc comparison therefore tests the allocation of power "
        "between the deterministic template and residual association.\n"
    )
    cell = "".join(notebook["cells"][17]["source"])
    split = cell.index("\n\n")
    notebook["cells"][17]["source"] = as_source(
        cell[:split] + "\n\n"
        "The arm-amplitude parameter $\eta$ multiplies the fixed annular-normalised H-alpha "
        "map; no arm number, pitch angle, phase, width, or branch is fitted to "
        "the abundance correlation. The likelihood does not include an additive "
        "negative-tail offset.\n"
    )
    notebook["cells"][23]["cell_type"] = "markdown"
    notebook["cells"][23].pop("execution_count", None)
    notebook["cells"][23].pop("outputs", None)
    notebook["cells"][23]["source"] = as_source(
        "## 6. Multi-element extension\n\n"
        "This systematic tests the direct H-alpha source template only in the "
        "single-element correlation. Delayed channels would require a time-dependent "
        "source tracer and are outside this check.\n"
    )
    cell = "".join(notebook["cells"][26]["source"])
    cell = cell.replace(
        "The primary KTZ fit with isotropic C_NP",
        f"The direct-H-alpha-template KTZ fit ({fwhm_kpc:.1f}-kpc FWHM) with isotropic C_NP",
    )
    cell = cell.replace("mask-template moments", "source-template moments")
    notebook["cells"][26]["source"] = as_source(cell)

    # Physical-validation cells belong to the primary result, not this new fit.
    notebook["cells"] = [
        cell for cell in notebook["cells"]
        if "ktz-physical-validation" not in cell.get("metadata", {}).get("tags", [])
    ]
    refresh_markdown(notebook, fwhm_kpc)
    for notebook_cell in notebook["cells"]:
        if notebook_cell["cell_type"] == "code":
            notebook_cell["execution_count"] = None
            notebook_cell["outputs"] = []
    output_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output_path)


def main() -> None:
    for target in TARGETS:
        for fwhm_kpc in FWHM_VALUES_KPC:
            transform(target, fwhm_kpc)


if __name__ == "__main__":
    main()
