"""Build and independently check the CP/mean age-response library.

This is a real-data full-window precomputation, not a galaxy fit. CNP stays
in the global integrator and is not implemented/replaced by this table.
"""

import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import expit

from scripts.phangs_source_integral import PairWindow, combine_moments
from scripts.phangs_source_fft import FFTSourceIntegral
from scripts.phangs_age_response import SingleAgeResponse, AgeResponseLibrary
from scripts.phangs_window_projection import mean_pair_moments


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(root, output):
    baseline = root/'outputs/implications/NGC628_window_projection_refit'
    with np.load(baseline/'ngc628_muse_scal_metallicity_residual_pixel_map.npz') as z:
        window = PairWindow(z['x_grid_kpc'], z['y_grid_kpc'], z['valid_mask'],
                             np.arange(.03, 5.031, .06), .2)
    with np.load(output/'ngc628_padded_source_inputs.npz') as z:
        axis, g, beam = z['source_axis_kpc'].copy(), z['source_modulation'].copy(), z['beam_covariance_kpc2'].copy()
    info = json.loads((output/'ngc628_source_inputs_summary.json').read_text())
    return window, axis, g, beam, info


def padded_source(window, axis, g, beam, variance, truncate=6):
    step = axis[1]-axis[0]
    old_half = (len(axis)-1)//2
    np.testing.assert_allclose(axis, np.arange(-old_half, old_half+1)*step, atol=1e-9)
    sigma = np.sqrt(variance+np.diag(beam).max())
    observed_half = int(np.ceil(max(abs(window.x).max(), abs(window.y).max())/step))
    half = max(old_half, observed_half+int(np.ceil(truncate*sigma/step))+1)
    if half == old_half:
        return axis, g
    extended = np.zeros((2*half+1, 2*half+1))
    offset = half-old_half
    extended[offset:offset+len(axis), offset:offset+len(axis)] = g
    return np.arange(-half, half+1)*step, extended


def mean_moments(window, fields):
    maps = []
    for field in fields:
        values = np.zeros(window.valid.shape)
        values[window.valid] = field
        maps.append(values)
    n0, d0, _ = mean_pair_moments(maps[0], window.valid, window.step, window.edges)
    n1, d1, _ = mean_pair_moments(maps[1], window.valid, window.step, window.edges)
    nt, dt, _ = mean_pair_moments(maps[0]+maps[1], window.valid, window.step, window.edges)
    return np.asarray([[n0, d0], [nt-n0-n1, dt-d0-d1], [n1, d1]])


def run_fft_benchmark(root, output):
    root, output = Path(root), Path(output)
    info_path = output/'ngc628_fft_real_map_benchmark.json'
    code_signature = file_hash(root/'scripts/phangs_source_fft.py')
    signature = hashlib.sha256((code_signature+file_hash(output/'ngc628_padded_source_inputs.npz')+
        file_hash(output/'ngc628_global_source_reference_point.npz')).encode()).hexdigest()
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if info['fingerprint'] != signature:
            raise RuntimeError('FFT benchmark inputs/code changed; use a new version/output directory.')
        return info
    start = time.time()
    window, axis, g, beam, source_info = load_inputs(root, output)
    with np.load(output/'ngc628_global_source_reference_point.npz') as z:
        coordinate = z['coordinates'].copy()
        reference = {k: z[k].copy() for k in ('CP', 'mean', 'CNP')}
    ell, eps, rate_log_slope, length = np.exp(coordinate[0]), expit(coordinate[1]), coordinate[4], np.exp(coordinate[6])
    model = FFTSourceIntegral(window, axis, g, source_info['reference_radius_kpc'], beam,
        progress_path=output/'ngc628_fft_benchmark_progress.json')
    response = model.moments(ell, eps, rate_log_slope, length)
    errors = {k: float(np.max(abs(response[k]-reference[k]))) for k in reference}
    old_curve, _ = combine_moments(reference, *np.exp(coordinate[[2, 3, 5]]))
    curve, _ = combine_moments(response, *np.exp(coordinate[[2, 3, 5]]))
    curve_error = float(np.max(abs(curve-old_curve)))
    if curve_error > 5e-10:
        raise RuntimeError('FFT implementation fails real-map equivalence; do not build/fit a surrogate.')
    old_info = json.loads((output/'ngc628_global_source_reference_point.json').read_text())
    info = {'status': 'real-map convolution equivalence passed', 'fingerprint': signature,
        'raw_moment_max_absolute_errors': errors, 'complete_KTZ_curve_max_error': curve_error,
        'new_end_to_end_seconds': time.time()-start, 'new_integral_seconds': response['elapsed_seconds'],
        'previous_reference_end_to_end_seconds': old_info['elapsed_seconds'],
        'valid_pixels': window.n, 'ordered_pairs_in_positive_bins': int(window.counts.sum()),
        'scope': 'Same fixed seven-coordinate point and source/beam/window/noise model. Not a new fit or full-prior emulator.'}
    np.savez_compressed(output/'ngc628_fft_real_map_response.npz', CP=response['CP'],
                        mean=response['mean'], CNP=response['CNP'], coordinates=coordinate,
                        fingerprint=np.asarray(signature))
    info_path.write_text(json.dumps(info, indent=2)+'\n')
    return info


def build_age_library(root, output, variance_nodes=65, slope_nodes=9):
    root, output = Path(root), Path(output)
    # Do not start the table unless the same real-map integral passed.
    benchmark = json.loads((output/'ngc628_fft_real_map_benchmark.json').read_text())
    if benchmark['complete_KTZ_curve_max_error'] > 5e-10:
        raise RuntimeError('Convolution equivalence gate not passed.')
    window, axis, g, beam, info = load_inputs(root, output)
    variances = np.geomspace(.015**2, 3.2**2, variance_nodes)
    slopes = np.linspace(-.35, .35, slope_nodes)
    implementation_files = ['phangs_age_response.py', 'phangs_source_fft.py', 'phangs_source_integral.py']
    manifest = {'source_hash': file_hash(output/'ngc628_padded_source_inputs.npz'),
        'observed_hash': file_hash(output/'ngc628_observed_abundance_control.npz'),
        'source_info_hash': file_hash(output/'ngc628_source_inputs_summary.json'),
        'implementation_hashes': {f: file_hash(root/'scripts'/f) for f in implementation_files},
        'variance_nodes': variances.tolist(), 'slope_nodes': slopes.tolist(),
        'source_grid_kpc': .06, 'truncate': 6, 'annulus_width_kpc': .2,
        'source_completion': 'unchanged saved g; g=0 outside measured support and exponential axisymmetric extrapolation',
        'role': 'CP/transported-mean only; NOT a complete KTZ likelihood or CNP emulator'}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    folder = output/'single_age_response_nodes'
    folder.mkdir(exist_ok=True)
    progress_path = output/'ngc628_age_library_progress.json'
    cp, fields = [], []
    elapsed_nodes = []
    for i, variance in enumerate(variances):
        path = folder/f'age_response_{i:03d}.npz'
        start = time.time()
        if path.exists():
            with np.load(path) as z:
                if str(z['signature']) != signature or not np.isclose(z['variance'], variance):
                    raise RuntimeError('Node fingerprint differs; use a new version/output directory.')
                response, field = z['CP_density'].copy(), z['projected_mean_fields'].copy()
                elapsed = float(z['elapsed_seconds'])
        else:
            extended_axis, extended_g = padded_source(window, axis, g, beam, variance)
            model = SingleAgeResponse(window, extended_axis, extended_g, info['reference_radius_kpc'], beam)
            response, field = model.single_age(variance, slopes)
            elapsed = time.time()-start
            np.savez_compressed(path, signature=np.asarray(signature), variance=variance,
                CP_density=response, projected_mean_fields=field, elapsed_seconds=elapsed)
            del model, extended_g
        cp.append(response)
        fields.append(field)
        elapsed_nodes.append(elapsed)
        progress = {'stage': 'single-age CP/mean response library', 'completed_width_nodes': i+1,
            'total_width_nodes': len(variances), 'slopes_per_width': len(slopes),
            'intrinsic_sigma_kpc': float(np.sqrt(variance)), 'last_node_seconds': elapsed,
            'not_a_new_posterior': True}
        progress_path.write_text(json.dumps(progress, indent=2)+'\n')
        print(progress, flush=True)
    cp, fields = np.asarray(cp), np.asarray(fields)
    path = output/'ngc628_single_age_response_library.npz'
    np.savez_compressed(path, signature=np.asarray(signature), variance_grid=variances,
        slope_grid=slopes, CP_density=cp, projected_mean_fields=fields)
    (output/'ngc628_age_library_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    del fields, cp
    library = AgeResponseLibrary(path)
    with np.load(output/'ngc628_global_source_reference_point.npz') as z:
        coordinate = z['coordinates'].copy()
        reference = {k: z[k].copy() for k in ('CP', 'mean', 'CNP')}
    ell, eps, rate_log_slope = np.exp(coordinate[0]), expit(coordinate[1]), coordinate[4]
    test_cp, test_fields = library.integrate(ell, eps, rate_log_slope)
    test_mean = mean_moments(window, test_fields)
    test = {'CP': test_cp, 'mean': test_mean, 'CNP': reference['CNP']}
    true_curve, _ = combine_moments(reference, *np.exp(coordinate[[2, 3, 5]]))
    table_curve, _ = combine_moments(test, *np.exp(coordinate[[2, 3, 5]]))
    curve_error = float(np.max(abs(table_curve-true_curve)))
    # Independent direct single-age points are not table knots.
    withheld = []
    for variance, rate_log_slope in [(.0007, -.299), (.012, .103), (.22, .319), (3., .103)]:
        extended_axis, extended_g = padded_source(window, axis, g, beam, variance)
        model = SingleAgeResponse(window, extended_axis, extended_g, info['reference_radius_kpc'], beam)
        direct, direct_fields = model.single_age(variance, [rate_log_slope])
        interpolated = CubicSpline(library.slopes, CubicSpline(library.variance, library.cp, axis=0)(variance), axis=0)(rate_log_slope)
        interpolated_fields = CubicSpline(library.slopes, CubicSpline(library.variance, library.fields, axis=0)(variance), axis=0)(rate_log_slope)
        def probe_curve(cp, field):
            m = mean_moments(window, field)
            # A same-age response test, not the age-integrated galaxy model.
            projected_mean = m[0]+.8*m[1]+.8**2*m[2]
            total = (4*np.pi*.32*variance)*(cp[0]+.8*cp[1])+projected_mean
            return total[0]/total[1]
        difference = probe_curve(interpolated, interpolated_fields)-probe_curve(direct[0], direct_fields[0])
        withheld.append({'intrinsic_variance_kpc2': variance, 'rate_log_slope_kpc_inv': rate_log_slope,
            'single_age_probe_curve_max_error': float(np.max(abs(difference)))})
        del model, direct, direct_fields, extended_g
    maximum_probe_error = max(p['single_age_probe_curve_max_error'] for p in withheld)
    summary = {'status': 'CP/mean response library completed; CNP emulation and MCMC not completed',
        'variance_nodes': variance_nodes, 'slope_nodes': slope_nodes,
        'intrinsic_sigma_range_kpc': [float(np.sqrt(variances[0])), float(np.sqrt(variances[-1]))],
        'rate_log_slope_range_kpc_inv': [float(slopes[0]), float(slopes[-1])],
        'training_node_seconds_sum': float(sum(elapsed_nodes)),
        'reference_age_integrated_curve_error_CNP_held_exact': curve_error,
        'withheld_single_age_points': withheld,
        'tested_CP_mean_precision_gate_passed': bool(curve_error < .002 and maximum_probe_error < .002),
        'all_pixel_pairs_retained': True, 'valid_pixels': window.n,
        'interpretation': 'CNP is frozen at its exact known value ONLY for the reference-point test. This library must not be used as a complete variable-CNP KTZ likelihood.',
        'remaining': 'Additional prior-wide/refinement/source-boundary validation; nonlocal CNP surrogate; native-flux recovery; new independent-ensemble KTZ/KT18 fitting.'}
    (output/'ngc628_age_library_validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    r = .5*(window.edges[1:]+window.edges[:-1])
    ax.plot(r, table_curve-true_curve, color='#7b3294')
    ax.axhline(0, color='0.5', lw=.8)
    ax.set(xlabel='separation [kpc]', ylabel='library minus direct reference',
        title='Age-integrated CP/mean interpolation check (CNP held exact)')
    fig.savefig(output/'ngc628_age_library_validation.png', dpi=180)
    plt.close(fig)
    progress_path.write_text(json.dumps({'stage': 'CP/mean library and tested-point validation completed',
        'status': 'complete', 'tested_CP_mean_precision_gate_passed': summary['tested_CP_mean_precision_gate_passed']}, indent=2)+'\n')
    return summary
