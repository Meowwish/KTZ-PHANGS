#!/usr/bin/env python3
"""Convert the NGC 1566 template to the Li et al. (2021) spaxel selection."""

from __future__ import annotations

import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
NOTEBOOK = WORKSPACE / "NGC1566_PHANGS_implications.ipynb"


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence, found {count}: {old[:100]!r}")
    return source.replace(old, new)


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text())
    if "HALPHA_EW_MIN_ANGSTROM" in "".join(nb["cells"][1]["source"]):
        print(f"Already converted: {NOTEBOOK}")
        return

    set_source(nb["cells"][0], r"""# NGC 1566 as a PHANGS-conditioned implication of KTZ

This notebook constructs a Li et al. (2021)-style two-dimensional metallicity
fluctuation map directly from the public PHANGS--MUSE emission-line maps for
NGC 1566.  It does **not** use the Williams et al. Gaussian-process metallicity
reconstruction and does not paint one integrated abundance over each nebular
footprint.

The calculation has three observation and inference layers:

1. The official `copt` PHANGS--MUSE MAPS product supplies per-spaxel H-beta,
   [O III], H-alpha, [N II], and [S II] fluxes and formal errors.  The line maps
   are brought from the released 0.80 arcsec Gaussian PSF to a 120 pc Gaussian
   PSF, sampled on a 60 pc face-on grid, attenuation-corrected, and converted
   to $12+\log({\rm O/H})$ with the Pilyugin & Grebel (2016) $S$ calibration.
   A matched `copt` SDSS-$r$ image provides the continuum-density proxy used
   for the Li21 $\mathrm{EW}({\rm H}\alpha)>6$ Angstrom cut.  The proxy is
   calibrated in each galaxy against the released region-integrated H-alpha
   equivalent widths.
2. The abundance field uses every spaxel in the analysis window that passes
   the Li21 selection: all required lines have $S/N\geq2$, the line ratios lie
   below the Kewley star-forming boundary, and
   $\mathrm{EW}({\rm H}\alpha)>6$ Angstrom.  Membership in a segmented HII
   region is **not** required.  An independent mean is then subtracted in every
   0.2 kpc annulus.  Missing cells remain `NaN`; neither pixels nor pairs are
   randomly subsampled.
3. The KTZ fit retains seven free parameters and includes the optional
   isotropic appendix term $C_{\rm NP}$.  The fixed observational transfer
   includes the Li21 Gaussian-beam broadening and a line-flux Monte Carlo
   estimate of the correlated measurement-noise contribution.  The displayed
   $r=0$ point is fixed to unity and is excluded from the likelihood.

Data sources:

- [PHANGS--MUSE DR1 release](https://www.eso.org/rm/api/v1/public/releaseDescriptions/184)
- [PHANGS--MUSE survey](https://doi.org/10.1051/0004-6361/202141727)
- [PHANGS--MUSE Nebula Catalogue DR1.1](https://www.eso.org/qi/catalog/show/436)
- [Pilyugin & Grebel (2016)](https://doi.org/10.1093/mnras/stw238)
""")

    source = "".join(nb["cells"][1]["source"])
    source = replace_once(
        source,
        'MUSE_MAPS_PATH = DATA_DIR / "NGC1566_MAPS_copt_0.80asec.fits"\n',
        'MUSE_MAPS_PATH = DATA_DIR / "NGC1566_MAPS_copt_0.80asec.fits"\n'
        'R_BAND_PATH = DATA_DIR / "NGC1566_PHANGS_IMAGE_SDSS_r_copt_0.80asec.fits"\n',
    )
    source = replace_once(source, "LINE_SN_MIN = 5.0", "LINE_SN_MIN = 2.0")
    source = replace_once(
        source,
        "LINE_COVERAGE_MIN = 0.80\nMETALLICITY_ERROR_MAX_DEX = 0.08",
        "LINE_COVERAGE_MIN = 0.80\nHALPHA_EW_MIN_ANGSTROM = 6.0\n"
        "METALLICITY_ERROR_MAX_DEX = None",
    )
    set_source(nb["cells"][1], source)

    set_source(nb["cells"][2], r"""## 1. Download the PHANGS--MUSE line maps, continuum image, and masks

The primary abundance input is the official 0.80 arcsec `copt` MAPS file, not
a reconstructed metallicity field.  The matched synthetic SDSS-$r$ image has
flux-density units and supplies the local continuum proxy for the H-alpha
equivalent-width cut.  The nebular catalogue and mask calibrate that proxy and
determine the large-scale arm geometry; they do not select abundance pixels.
The MAPS and SDSS-$r$ downloads are approximately 431 MiB and 9 MiB.

Set `FORCE_DOWNLOAD=True` only when the archived products should be refreshed.
""")

    set_source(nb["cells"][3], '''CATALOG_URL = (
    "https://dataportal.eso.org/dataPortal/file/"
    "ADP.2025-05-14T09:12:30.227"
)
NEBULA_MASK_URL = (
    "https://dataportal.eso.org/dataPortal/file/"
    "ADP.2025-05-14T09:12:30.228"
)
MUSE_MAPS_URL = (
    "https://dataportal.eso.org/dataPortal/file/"
    "ADP.2021-07-16T10:20:56.422"
)
R_BAND_URL = (
    "https://dataportal.eso.org/dataPortal/file/"
    "ADP.2021-07-16T10:20:56.420"
)
MUSE_MAPS_PRODUCT_ID = "ADP.2021-07-16T10:20:56.422"
R_BAND_PRODUCT_ID = "ADP.2021-07-16T10:20:56.420"


def download_product(url: str, path: Path, force: bool = False) -> Path:
    """Download one public product atomically."""
    if path.exists() and not force:
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=300) as response:
        with temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    temporary.replace(path)
    return path


def sha256_file(path: Path, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


FORCE_DOWNLOAD = False
for product_url, product_path in (
    (CATALOG_URL, CATALOG_PATH),
    (NEBULA_MASK_URL, NEBULA_MASK_PATH),
    (MUSE_MAPS_URL, MUSE_MAPS_PATH),
    (R_BAND_URL, R_BAND_PATH),
):
    download_product(product_url, product_path, force=FORCE_DOWNLOAD)

catalog_hash = sha256_file(CATALOG_PATH)
nebula_mask_hash = sha256_file(NEBULA_MASK_PATH)
muse_maps_hash = sha256_file(MUSE_MAPS_PATH)
r_band_hash = sha256_file(R_BAND_PATH)
for product_name, product_path, product_hash in (
    ("PHANGS nebular catalogue", CATALOG_PATH, catalog_hash),
    ("PHANGS nebular mask", NEBULA_MASK_PATH, nebula_mask_hash),
    ("PHANGS--MUSE copt MAPS", MUSE_MAPS_PATH, muse_maps_hash),
    ("PHANGS--MUSE copt SDSS-r continuum", R_BAND_PATH, r_band_hash),
):
    print(f"{product_name}: {product_path}")
    print(f"Size: {product_path.stat().st_size / 1024**2:.2f} MiB")
    print(f"SHA256: {product_hash}")
''')

    set_source(nb["cells"][6], r"""## 3. Construct the Li21-style spaxel metallicity fluctuation map

All available line fluxes and the SDSS-$r$ continuum-density image are first
brought to a common 120 pc Gaussian PSF and then sampled on a 60 pc face-on
grid.  Ratios and abundances are calculated only after this linear processing.
Internal attenuation is inferred from the Balmer decrement using an O'Donnell
$R_V=3.1$ curve.  The Pilyugin & Grebel (2016) $S$ calibration uses $R_3$,
$N_2$, and $S_2$; its upper/lower branch is selected independently in every
cell.

Following Li21, pixels are retained when every line needed by the diagnostic
has $S/N\geq2$, the [N II] BPT ratios lie below the Kewley boundary, and
$\mathrm{EW}({\rm H}\alpha)>6$ Angstrom.  The public MAPS product does not
contain a continuum extension, so the matched synthetic SDSS-$r$ flux density
is used for the continuum.  Its H-alpha/continuum ratio is calibrated against
the catalogue's fitted region-integrated H-alpha equivalent widths; the
calibration scatter is reported below.  The nebular segmentation never enters
the abundance-pixel mask.  Missing cells are `NaN`, never physical zeros.
""")

    source = "".join(nb["cells"][7]["source"])
    source = replace_once(
        source,
        '''        line_error[short_name] = MUSE_ERROR_INFLATION * error_grid
        line_coverage[short_name] = coverage_grid


with fits.open(NEBULA_MASK_PATH, memmap=True) as mask_hdul:
''',
        '''        line_error[short_name] = MUSE_ERROR_INFLATION * error_grid
        line_coverage[short_name] = coverage_grid


with fits.open(R_BAND_PATH, memmap=True) as r_band_hdul:
    r_band_wcs = WCS(r_band_hdul["DATA"].header).celestial
    r_band_x, r_band_y = r_band_wcs.world_to_pixel_values(target_ra, target_dec)
    r_band_coordinates = np.vstack((r_band_y, r_band_x))
    r_band_native = np.asarray(r_band_hdul["DATA"].data, dtype=np.float32)
    r_band_variance_native = np.asarray(
        r_band_hdul["STAT"].data, dtype=np.float32
    )
    r_band_dq_native = np.asarray(r_band_hdul["DQ"].data)
    r_band_good_native = (
        (r_band_dq_native == 0) & np.isfinite(r_band_native)
        & np.isfinite(r_band_variance_native) & (r_band_variance_native > 0)
    )
    r_band_flux, r_band_error, r_band_coverage = smooth_and_sample_line(
        np.where(r_band_good_native, r_band_native, np.nan),
        np.where(r_band_good_native, np.sqrt(r_band_variance_native), np.nan),
        r_band_coordinates, kernel_sigma_pixel,
    )


with fits.open(NEBULA_MASK_PATH, memmap=True) as mask_hdul:
''',
    )
    source = replace_once(
        source,
        ''').reshape(pixel_X.shape).astype(int)


def scal_metallicity(flux: dict[str, np.ndarray]):
''',
        ''').reshape(pixel_X.shape).astype(int)


# The synthetic r-band image is in flux-density units.  Calibrate the direct
# H-alpha/r-band ratio against the catalogue's fitted integrated EWs so that
# the Li21 6-Angstrom cut is tied to the released PHANGS measurements rather
# than to an arbitrary surface-brightness threshold.
with np.errstate(divide="ignore", invalid="ignore"):
    raw_halpha_ew_grid = line_flux["ha"] / r_band_flux
catalog_region_id = numeric_column("REGION_ID")
catalog_halpha_ew = numeric_column("EW_HA6562_FIT")
ew_catalog_valid = (
    catalog_hii_quality & np.isfinite(catalog_region_id)
    & np.isfinite(catalog_halpha_ew) & (catalog_halpha_ew > 0)
)
ew_sum_valid = (
    (region_id_grid >= 0) & np.isfinite(line_flux["ha"])
    & np.isfinite(r_band_flux) & (line_flux["ha"] > 0) & (r_band_flux > 0)
)
ew_ids = region_id_grid[ew_sum_valid].astype(int)
ew_halpha_sum = np.bincount(
    ew_ids, weights=line_flux["ha"][ew_sum_valid]
)
ew_continuum_sum = np.bincount(
    ew_ids, weights=r_band_flux[ew_sum_valid], minlength=len(ew_halpha_sum)
)
ew_region_proxy = np.divide(
    ew_halpha_sum, ew_continuum_sum,
    out=np.full_like(ew_halpha_sum, np.nan), where=ew_continuum_sum > 0,
)
ew_catalog_ids = catalog_region_id[ew_catalog_valid].astype(int)
ew_catalog_values = catalog_halpha_ew[ew_catalog_valid]
ew_match = (
    (ew_catalog_ids >= 0) & (ew_catalog_ids < len(ew_region_proxy))
)
ew_scale_samples = np.divide(
    ew_catalog_values[ew_match], ew_region_proxy[ew_catalog_ids[ew_match]],
    out=np.full(np.count_nonzero(ew_match), np.nan),
    where=ew_region_proxy[ew_catalog_ids[ew_match]] > 0,
)
ew_scale_samples = ew_scale_samples[
    np.isfinite(ew_scale_samples) & (ew_scale_samples > 0)
]
if len(ew_scale_samples) < 30:
    raise RuntimeError("Too few PHANGS regions to calibrate the H-alpha EW map.")
ew_scale_log = np.log(ew_scale_samples)
ew_scale_log_median = np.median(ew_scale_log)
ew_scale_log_mad = np.median(np.abs(ew_scale_log - ew_scale_log_median))
if ew_scale_log_mad > 0:
    ew_scale_samples = ew_scale_samples[
        np.abs(np.log(ew_scale_samples) - ew_scale_log_median)
        < 5.0 * 1.4826 * ew_scale_log_mad
    ]
halpha_ew_scale = float(np.median(ew_scale_samples))
halpha_ew_scale_p16, halpha_ew_scale_p84 = np.percentile(
    ew_scale_samples, [16.0, 84.0]
)
halpha_ew_grid = halpha_ew_scale * raw_halpha_ew_grid


def scal_metallicity(flux: dict[str, np.ndarray]):
''',
    )
    source = replace_once(
        source,
        '''hii_grid = np.isin(region_id_grid, hii_region_ids)
line_signal_to_noise = {
''',
        '''hii_grid = np.isin(region_id_grid, hii_region_ids)  # diagnostic only
line_signal_to_noise = {
''',
    )
    source = replace_once(
        source,
        '''preliminary_valid_grid = (
    geometry_mask & hii_grid & line_quality & bpt_star_forming
    & np.isfinite(metallicity_grid)
)
''',
        '''halpha_ew_quality = (
    np.isfinite(halpha_ew_grid) & (halpha_ew_grid >= HALPHA_EW_MIN_ANGSTROM)
    & (r_band_coverage >= LINE_COVERAGE_MIN)
)
preliminary_valid_grid = (
    geometry_mask & line_quality & bpt_star_forming & halpha_ew_quality
    & np.isfinite(metallicity_grid)
)
''',
    )
    source = replace_once(
        source,
        '''pixel_valid_grid = (
    preliminary_valid_grid & np.isfinite(metallicity_error_grid)
    & (metallicity_error_grid <= METALLICITY_ERROR_MAX_DEX)
)
''',
        '''# Li21 applies the line-S/N cut but no additional metallicity-error cut.
pixel_valid_grid = preliminary_valid_grid & np.isfinite(metallicity_error_grid)
''',
    )
    source = replace_once(
        source,
        '''# Measure the source harmonics independently from the direct, dereddened
# H-alpha grid.  Catalogue labels provide only the HII/DIG classification.
source_valid_grid = (
    geometry_mask & hii_grid
    & np.isfinite(corrected_line_flux["ha"])
    & (corrected_line_flux["ha"] > 0)
    & (line_signal_to_noise["ha"] >= LINE_SN_MIN)
    & (line_signal_to_noise["hb"] >= LINE_SN_MIN)
)
''',
        '''# Measure source harmonics from the same Li21-selected star-forming
# spaxels, independently of the abundance residual field.
source_valid_grid = (
    geometry_mask & bpt_star_forming & halpha_ew_quality
    & np.isfinite(corrected_line_flux["ha"])
    & (corrected_line_flux["ha"] > 0)
    & (line_signal_to_noise["ha"] >= LINE_SN_MIN)
    & (line_signal_to_noise["hb"] >= LINE_SN_MIN)
)
''',
    )
    source = replace_once(
        source,
        '''source_region_id = region_id_grid[source_valid_grid]
source_theta = (
''',
        '''source_theta = (
''',
    )
    source = replace_once(
        source,
        '''# Region bootstrap preserves the strong within-beam and within-nebula
# dependence between neighbouring output cells.
unique_source_regions = np.unique(source_region_id)
region_flux = np.array([
    np.sum(source_halpha_flux[source_region_id == region_id])
    for region_id in unique_source_regions
])
region_complex = np.array([
    [
        np.sum(source_halpha_flux[source_region_id == region_id]
               * np.exp(1j * order * source_theta[source_region_id == region_id]))
        for order in harmonic_orders
    ]
    for region_id in unique_source_regions
])
harmonic_bootstrap_rng = np.random.default_rng(1566)
n_harmonic_bootstrap = 400
harmonic_bootstrap = np.empty((n_harmonic_bootstrap, len(harmonic_orders)))
for bootstrap_index in range(n_harmonic_bootstrap):
    draw = harmonic_bootstrap_rng.integers(
        0, len(unique_source_regions), len(unique_source_regions)
    )
    total_flux = np.sum(region_flux[draw])
    harmonic_bootstrap[bootstrap_index] = np.abs(
        2.0 * np.sum(region_complex[draw], axis=0) / total_flux
    )
''',
        '''# A 48-block spatial bootstrap replaces the region bootstrap because the
# Li21 field includes star-forming spaxels outside segmented nebulae.
source_azimuth_block = np.floor(source_phi / (2.0 * np.pi) * 12).astype(int)
source_radial_edges = np.sqrt(np.linspace(R_MIN_KPC**2, R_MAX_KPC**2, 5))
source_radial_block = np.clip(
    np.digitize(source_r_kpc, source_radial_edges) - 1, 0, 3
)
source_block_id = 12 * source_radial_block + source_azimuth_block
unique_source_blocks = np.unique(source_block_id)
block_flux = np.array([
    np.sum(source_halpha_flux[source_block_id == block_id])
    for block_id in unique_source_blocks
])
block_complex = np.array([
    [
        np.sum(source_halpha_flux[source_block_id == block_id]
               * np.exp(1j * order * source_theta[source_block_id == block_id]))
        for order in harmonic_orders
    ]
    for block_id in unique_source_blocks
])
harmonic_bootstrap_rng = np.random.default_rng(1566)
n_harmonic_bootstrap = 400
harmonic_bootstrap = np.empty((n_harmonic_bootstrap, len(harmonic_orders)))
for bootstrap_index in range(n_harmonic_bootstrap):
    draw = harmonic_bootstrap_rng.integers(
        0, len(unique_source_blocks), len(unique_source_blocks)
    )
    total_flux = np.sum(block_flux[draw])
    harmonic_bootstrap[bootstrap_index] = np.abs(
        2.0 * np.sum(block_complex[draw], axis=0) / total_flux
    )
''',
    )
    source = replace_once(
        source,
        '''    residual_dex=np.where(pixel_valid_grid, pixel_residual_centered_grid, np.nan),
    valid_mask=pixel_valid_grid,
''',
        '''    residual_dex=np.where(pixel_valid_grid, pixel_residual_centered_grid, np.nan),
    halpha_equivalent_width_angstrom=np.where(
        geometry_mask, halpha_ew_grid, np.nan
    ),
    valid_mask=pixel_valid_grid,
''',
    )
    source = replace_once(
        source,
        '''    "muse_maps_sha256": muse_maps_hash,
    "muse_maps_product_id": MUSE_MAPS_PRODUCT_ID,
''',
        '''    "muse_maps_sha256": muse_maps_hash,
    "muse_maps_product_id": MUSE_MAPS_PRODUCT_ID,
    "r_band_sha256": r_band_hash,
    "r_band_product_id": R_BAND_PRODUCT_ID,
''',
    )
    source = replace_once(
        source,
        '''    "metallicity_error_maximum_dex": METALLICITY_ERROR_MAX_DEX,
    "n_face_on_analysis_cells": int(np.sum(pixel_valid_grid)),
''',
        '''    "metallicity_error_maximum_dex": METALLICITY_ERROR_MAX_DEX,
    "halpha_ew_minimum_angstrom": HALPHA_EW_MIN_ANGSTROM,
    "halpha_ew_scale": halpha_ew_scale,
    "halpha_ew_scale_p16_p84": [
        float(halpha_ew_scale_p16), float(halpha_ew_scale_p84)
    ],
    "n_annulus_grid_cells": int(np.sum(geometry_mask)),
    "n_li21_ew_cells": int(np.sum(geometry_mask & halpha_ew_quality)),
    "n_li21_bpt_cells": int(np.sum(geometry_mask & bpt_star_forming)),
    "n_face_on_analysis_cells": int(np.sum(pixel_valid_grid)),
''',
    )
    source = replace_once(
        source,
        '''        ("preliminary qualified cells", int(np.sum(preliminary_valid_grid))),
        ("final metallicity cells", int(np.sum(pixel_valid_grid))),
''',
        '''        ("annulus grid cells", int(np.sum(geometry_mask))),
        ("Li21 EW-qualified cells", int(np.sum(geometry_mask & halpha_ew_quality))),
        ("Li21 BPT-qualified cells", int(np.sum(geometry_mask & bpt_star_forming))),
        ("EW calibration scale [p16, p50, p84]", str([
            float(halpha_ew_scale_p16), halpha_ew_scale,
            float(halpha_ew_scale_p84)
        ])),
        ("preliminary qualified cells", int(np.sum(preliminary_valid_grid))),
        ("final metallicity cells", int(np.sum(pixel_valid_grid))),
''',
    )
    set_source(nb["cells"][7], source)

    source = "".join(nb["cells"][16]["source"])
    source = source.replace(
        '"muse_maps": muse_maps_hash,\n    "window": window_signature,',
        '"muse_maps": muse_maps_hash,\n    "r_band": r_band_hash,\n'
        '    "halpha_ew_min_angstrom": HALPHA_EW_MIN_ANGSTROM,\n'
        '    "window": window_signature,',
    )
    if source.count('"r_band": r_band_hash') != 2:
        raise RuntimeError("Both KTZ emulator signatures must contain the r-band hash.")
    set_source(nb["cells"][16], source)

    set_source(nb["cells"][26], r"""## 7. Interpretation and limits

- The metallicity statistic now follows the Li21 spaxel-selection logic:
  every diagnostic line has $S/N\geq2$, the [N II] BPT position lies below
  the Kewley boundary, and $\mathrm{EW}({\rm H}\alpha)>6$ Angstrom.  The
  PHANGS nebular mask does not select abundance pixels.
- PHANGS MAPS does not contain a continuum extension.  The EW map therefore
  uses the PSF-matched synthetic SDSS-$r$ flux density and is tied to the
  catalogue's fitted region-integrated H-alpha EWs.  Its per-galaxy calibration
  interval is recorded in the morphology summary and should be propagated as
  a selection sensitivity test.
- The KTZ fit with optional isotropic $C_{\rm NP}$ has seven free parameters:
  $\ell_*$, $\epsilon$, $\beta_{\rm src}$, $\eta$, $h_R^{-1}$,
  $\beta_{\rm src}{\cal A}_{\rm src}$, and $\ell_{\rm src}$.  The direct
  H-alpha source grid supplies informative priors on $\eta$ and $h_R^{-1}$.
- The likelihood uses all available positive-separation bins from the 60 pc
  grid out to 5 kpc.  The displayed $r=0$ observation and model are fixed to
  unity and do not enter the likelihood.
- Fluxes are matched to a 120 pc Gaussian PSF before ratios are formed.  The
  output is Nyquist-sampled at 60 pc, but independent information remains
  limited by the 120 pc beam.
- Missing or failed-selection cells remain `NaN`.  The FFT numerator is divided
  by an independently transformed binary mask, so internal zero filling never
  treats a missing pixel as zero metallicity fluctuation.
- A separate arithmetic mean is removed in every 0.2 kpc annulus, following
  Li21.  This guards against nonlinear radial calibration structure but also
  removes genuine axisymmetric radial modes.
- The fixed observation transfer combines Li21 Gaussian-beam broadening with
  a 200-realisation line-flux Monte Carlo estimate of correlated noise.  The
  1.12 error inflation and Gaussian noise correlation remain approximations to
  covariance not supplied in the public MAPS product.
- The optional $C_{\rm NP}$ term is the isotropic appendix prediction for a
  Gaussian pre-mixing source-pair excess filtered by injection and diffusion.
  It follows the radial rate $\lambda_0^2$ and is not modulated by the arm
  template or added as a free post-mixing Gaussian correlation.
- Repeating the analysis for EW thresholds of 3, 6, and 10 Angstrom, line-S/N
  thresholds of 2, 3, and 5, and alternative target beams is required before
  publication.
- The multi-element panel remains a galaxy-conditioned forecast: MUSE does not
  provide independent abundance maps for all plotted elements, and a frozen
  spiral is a major systematic for delayed channels.
""")

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"Updated: {NOTEBOOK}")


if __name__ == "__main__":
    main()
