"""Prepare padded H-alpha sources without confusing coverage with SF selection.

Inputs are the released MAPS/r-band products and the existing notebook's
geometry/EW calibration. Unknown source cells are never asserted to be zero.
Low-EW cells are zero only under an explicitly recorded tracer assumption.
This does not impute metallicities, nor correct all source incompleteness.
"""

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.stats import linregress

from scripts.phangs_source_integral import beam_covariance_face_on


def sample_line(flux, error, coordinates, sigma, shape):
    good = np.isfinite(flux)&np.isfinite(error)&(error > 0)
    weight = gaussian_filter(good.astype(np.float32), sigma, mode='constant', truncate=4)
    numerator = gaussian_filter(np.where(good, flux, 0).astype(np.float32), sigma,
                                mode='constant', truncate=4)
    radius = int(4*sigma+.5)
    k = np.exp(-.5*(np.arange(-radius, radius+1)/sigma)**2)
    k /= k.sum()
    squared_sum = np.sum(k*k)**2
    variance = squared_sum*gaussian_filter(np.where(good, error*error, 0).astype(np.float32),
                                          sigma/np.sqrt(2), mode='constant', truncate=4)
    smoothed = np.divide(numerator, weight, out=np.full_like(numerator, np.nan), where=weight > 0)
    propagated = np.divide(np.sqrt(np.maximum(variance, 0)), weight,
                           out=np.full_like(variance, np.nan), where=weight > 0)
    sample = lambda a, missing: map_coordinates(a, coordinates, order=1, mode='constant',
                                                cval=missing).reshape(shape)
    return sample(smoothed, np.nan), sample(propagated, np.nan), sample(weight, 0)


def prepare_sources(ns, output_dir, padding_kpc=1.62, template_fwhm_kpc=1., bootstrap=400):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step = ns['PIXEL_SIZE_KPC']
    half = int(np.ceil((ns['R_MAX_KPC']+padding_kpc)/step))
    axis = np.arange(-half, half+1)*step
    y, x = np.meshgrid(axis, axis, indexing='ij')
    radius, phi = np.hypot(x, y), np.arctan2(y, x) % (2*np.pi)
    ra, dec = ns['project_face_on_to_sky'](x, y)
    flux, error, coverage = {}, {}, {}
    names = ('ha', 'hb', 'nii', 'oiii')
    with fits.open(ns['MUSE_MAPS_PATH'], memmap=True) as hdul:
        wcs = WCS(hdul['HA6562_FLUX'].header).celestial
        px, py = wcs.world_to_pixel_values(ra, dec)
        coordinates = np.vstack([py, px])
        for name in names:
            extension = ns['LINE_EXTENSIONS'][name]
            f, e, c = sample_line(np.asarray(hdul[extension+'_FLUX'].data, np.float32),
                                  np.asarray(hdul[extension+'_FLUX_ERR'].data, np.float32),
                                  coordinates, ns['kernel_sigma_pixel'], x.shape)
            flux[name], error[name], coverage[name] = f, ns['MUSE_ERROR_INFLATION']*e, c
    with fits.open(ns['R_BAND_PATH'], memmap=True) as hdul:
        wcs = WCS(hdul['DATA'].header).celestial
        px, py = wcs.world_to_pixel_values(ra, dec)
        native = np.asarray(hdul['DATA'].data, np.float32)
        variance = np.asarray(hdul['STAT'].data, np.float32)
        good = (hdul['DQ'].data == 0)&np.isfinite(native)&np.isfinite(variance)&(variance > 0)
        continuum, continuum_error, continuum_coverage = sample_line(
            np.where(good, native, np.nan), np.where(good, np.sqrt(np.maximum(variance, 0)), np.nan),
            np.vstack([py, px]), ns['kernel_sigma_pixel'], x.shape)
    geometric = continuum_coverage >= ns['LINE_COVERAGE_MIN']
    for name in names:
        geometric &= (coverage[name] >= ns['LINE_COVERAGE_MIN'])&np.isfinite(flux[name])&np.isfinite(error[name])
    with np.errstate(divide='ignore', invalid='ignore'):
        ew = ns['halpha_ew_scale']*flux['ha']/continuum
        ew_error = ns['halpha_ew_scale']*np.sqrt((error['ha']/continuum)**2+
                                               (flux['ha']*continuum_error/continuum**2)**2)
        ew_proxy_scatter = (ns['halpha_ew_scale_p84']-ns['halpha_ew_scale_p16'])/(2*ns['halpha_ew_scale'])
        ew_error = np.sqrt(ew_error**2+(ew*ew_proxy_scatter)**2)
        ebv = np.clip(2.5/(ns['extinction_k']['hb']-ns['extinction_k']['ha'])*
                      np.log10((flux['ha']/flux['hb'])/2.86), 0, 2)
        corrected = {name: flux[name]*10**(.4*ns['extinction_k'][name]*ebv) for name in names}
        bx = np.log10(corrected['nii']/corrected['ha'])
        by = np.log10(corrected['oiii']/corrected['hb'])
        bpt = np.isfinite(bx)&np.isfinite(by)&(bx < .4)&(by < .61/(bx-.47)+1.19)
        sn = {name: flux[name]/error[name] for name in names}
    detected = geometric&bpt&(ew >= ns['HALPHA_EW_MIN_ANGSTROM'])
    for name in names:
        detected &= sn[name] >= ns['LINE_SN_MIN']
    detected &= np.isfinite(corrected['ha'])&(corrected['ha'] > 0)
    quiet = (geometric&np.isfinite(ew)&np.isfinite(ew_error)&(continuum > 0)&
             (continuum/continuum_error >= 3)&(ew+2*ew_error < ns['HALPHA_EW_MIN_ANGSTROM']))
    quiet &= ~detected
    known, unknown = detected|quiet, geometric&~(detected|quiet)
    native_area = ns['native_pixel_area_kpc2']
    projected_area = step**2*abs(np.linalg.det(np.linalg.inv(ns['face_on_affine'][:2])))
    pixel_flux = corrected['ha']*projected_area/native_area
    tracer = np.where(detected, pixel_flux, 0.)
    sigma = template_fwhm_kpc/2.354820045/step
    smoothing_weight = gaussian_filter(known.astype(float), sigma, mode='constant', truncate=4)
    numerator = gaussian_filter(tracer, sigma, mode='constant', truncate=4)
    intensity = np.divide(numerator, smoothing_weight, out=np.full(x.shape, np.nan),
                           where=smoothing_weight > .05)
    # The smoothing denominator is the supported source area, including
    # explicitly quiet cells, not just the selected bright-source area.
    modulation = np.zeros(x.shape)
    support = np.zeros(x.shape, bool)
    labels = np.floor(radius/ns['ANNULUS_WIDTH_KPC']).astype(int)
    for a in np.unique(labels[np.isfinite(intensity)]):
        selected = (labels == a)&np.isfinite(intensity)
        if selected.sum() < ns['MIN_PIXELS_PER_ANNULUS']:
            continue
        value = intensity[selected].mean()
        if value > 0:
            modulation[selected] = intensity[selected]/value-1
            support[selected] = True
    edges = np.arange(ns['R_MIN_R25'], ns['R_MAX_R25']+.001, .05)*ns['R25_KPC']
    bin_index = np.searchsorted(edges, radius, side='right')-1
    annulus = (radius >= ns['R_MIN_KPC'])&(radius <= ns['R_MAX_KPC'])
    nb = len(edges)-1
    total_area = np.pi*(edges[1:]**2-edges[:-1]**2)
    centers = .5*(edges[1:]+edges[:-1])
    sector = (phi/(2*np.pi)*12).astype(int)
    flux_sector, area_sector = np.zeros((12, nb)), np.zeros((12, nb))
    rows = []
    for b in range(nb):
        selected = (bin_index == b)&annulus
        for s in range(12):
            region = selected&(sector == s)
            flux_sector[s, b] = tracer[region].sum()
            area_sector[s, b] = (geometric&region).sum()*step**2
        covered_area = area_sector[:, b].sum()
        rows.append([centers[b], total_area[b], covered_area,
                     flux_sector[:, b].sum(), int((detected&selected).sum()),
                     int((quiet&selected).sum()), int((unknown&selected).sum())])
    profile = np.asarray(rows)
    surface = profile[:, 3]/profile[:, 2]
    positive = (surface > 0)&np.isfinite(surface)
    fit = linregress(centers[positive], np.log(surface[positive]))
    old_fit = linregress(centers[positive], np.log((profile[:, 3]/total_area)[positive]))
    rng = np.random.default_rng(628773)
    slopes = []
    for _ in range(bootstrap):
        indices = rng.integers(0, 12, size=12)
        f, a = flux_sector[indices].sum(axis=0), area_sector[indices].sum(axis=0)
        selected = (f > 0)&(a > 0)
        if selected.sum() >= 3:
            slopes.append(linregress(centers[selected], np.log(f[selected]/a[selected])).slope)
    slope_uncertainty = max(float(fit.stderr), float(np.std(slopes, ddof=1)))
    # A sensitivity scenario, not a confidence limit: possible H-alpha in
    # unclassified covered cells, with the local median Balmer attenuation.
    scenario = np.zeros(nb)
    for b in range(nb):
        region = (bin_index == b)&annulus
        dust = np.nanmedian(ebv[detected&region]) if np.any(detected&region) else 0.
        u = unknown&region
        upper_flux = np.maximum(flux['ha'][u]+2*error['ha'][u], 0)*10**(.4*ns['extinction_k']['ha']*dust)
        scenario[b] = upper_flux.sum()*projected_area/native_area/profile[b, 2]
    luminosity_factor = 1e-20*4*np.pi*(ns['DISTANCE_MPC']*1e6*3.085677581491367e18)**2
    measured_sfr = 5.5e-42*profile[:, 3].sum()*luminosity_factor
    corrected_sfr = 5.5e-42*np.sum(surface*total_area)*luminosity_factor
    min_g = float(modulation.min())
    summary = {
        'galaxy': ns['GALAXY_NAME'], 'padding_kpc': padding_kpc,
        'source_step_kpc': step, 'template_fwhm_kpc': template_fwhm_kpc,
        'rate_log_slope_covered_area_kpc_inv': float(fit.slope),
        'rate_log_slope_full_area_kpc_inv': float(old_fit.slope),
        'rate_log_slope_bootstrap_or_regression_error': slope_uncertainty,
        'covered_area_fraction_per_ring': (profile[:, 2]/total_area).tolist(),
        'measured_qualified_Halpha_SFR': float(measured_sfr),
        'covered_area_extrapolated_Halpha_SFR': float(corrected_sfr),
        'minimum_modulation': min_g, 'maximum_positive_eta': .98/abs(min_g) if min_g < 0 else None,
        'reference_radius_kpc': float(ns['r_ref_kpc']),
        'beam_covariance_face_on_kpc2': beam_covariance_face_on(ns['face_on_affine'], ns['TARGET_FWHM_KPC']).tolist(),
        'analysis_source_cells': {'detected_SF': int((detected&annulus).sum()),
                                  'low_EW_assumed_quiet': int((quiet&annulus).sum()),
                                  'unclassified_covered': int((unknown&annulus).sum()),
                                  'not_covered': int((~geometric&annulus).sum())},
        'assumptions': [
            'Low EW with EW+2 propagated/proxy-scatter error <6 A is treated as zero in the chosen SF tracer, not proven physically source-free.',
            'Unclassified covered cells and unobserved cells remain unknown; mask-aware smoothing interpolates from supported cells. This is a fixed source-completion assumption, not new metallicity data.',
            'Annular source modulation is zero outside supported H-alpha; axisymmetric radial rate is extrapolated there. Boundary/completion sensitivity must be quantified.',
            'Coverage-area correction assumes observed sectors represent missing sectors; angular bootstrap is conditional and cannot remove informative selection.',
            'SF-selected H-alpha can miss faint sources and includes shared line-measurement errors with metallicity. No claim of statistical independence.',
            'The unknown-flux scenario is not an upper confidence bound; median detected Balmer attenuation is assumed.',
            'Sky PSF transforms elliptically to face-on. Linear-field convolution remains an approximation to flux convolution followed by nonlinear abundance calibration.',
        ],
    }
    np.savez_compressed(output_dir/'ngc628_padded_source_inputs.npz',
        source_axis_kpc=axis, source_modulation=modulation,
        geometric_coverage=geometric, detected_SF=detected, assumed_quiet=quiet,
        unclassified_covered=unknown, smoothing_support=known,
        modulation_support=support, smoothing_weight=smoothing_weight,
        selected_Halpha_flux=np.where(detected, pixel_flux, np.nan),
        smoothed_source_intensity=intensity,
        beam_covariance_kpc2=np.asarray(summary['beam_covariance_face_on_kpc2']),
        face_on_affine=ns['face_on_affine'])
    np.savetxt(output_dir/'ngc628_source_rate_coverage.csv',
               np.column_stack([profile, surface, scenario]), delimiter=',',
               header='radius_kpc,full_area_kpc2,covered_area_kpc2,qualified_Halpha_flux,SF_cells,quiet_cells,unknown_cells,covered_area_surface_density,unknown_flux_sensitivity_scenario',
               comments='')
    (output_dir/'ngc628_source_inputs_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
