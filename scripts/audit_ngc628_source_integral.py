"""Real-map age/source-grid refinement at the fixed reference coordinate.

Every valid observed pixel/pair is retained. Fine-grid source modulation is
bilinear refinement of the SAME saved smoothed tracer, not newly observed data.
This checks numerical integration at one point, not source-systematic errors
or precision throughout the physical prior.
"""

import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import expit

from scripts.phangs_source_integral import PairWindow, SourceIntegral, combine_moments


def run_precision_audit(root, output_dir):
    root, output_dir = Path(root), Path(output_dir)
    baseline = root/'outputs/implications/NGC628_window_projection_refit'
    reference = json.loads((output_dir/'ngc628_global_source_reference_point.json').read_text())
    info = json.loads((output_dir/'ngc628_source_inputs_summary.json').read_text())
    with np.load(output_dir/'ngc628_global_source_reference_point.npz') as z:
        coordinate, coarse = z['coordinates'].copy(), z['full_observed_response'].copy()
    source_file = output_dir/'ngc628_padded_source_inputs.npz'
    source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    with np.load(source_file) as z:
        axis, g, beam = z['source_axis_kpc'].copy(), z['source_modulation'].copy(), z['beam_covariance_kpc2'].copy()
    with np.load(baseline/'ngc628_muse_scal_metallicity_residual_pixel_map.npz') as z:
        window = PairWindow(z['x_grid_kpc'], z['y_grid_kpc'], z['valid_mask'],
                             np.arange(.03, 5.031, .06), .2)
    data = np.genfromtxt(baseline/'ngc628_residual_covariance_diagnostics.csv', delimiter=',', names=True)
    ell, eps, beta, arm, rate_log_slope, betaA, length = (
        np.exp(coordinate[0]), expit(coordinate[1]), np.exp(coordinate[2]), np.exp(coordinate[3]),
        coordinate[4], np.exp(coordinate[5]), np.exp(coordinate[6]))
    fine_axis = np.linspace(axis[0], axis[-1], 2*len(axis)-1)
    fy, fx = np.meshgrid(np.arange(len(fine_axis))/2, np.arange(len(fine_axis))/2, indexing='ij')
    fine_g = map_coordinates(g, [fy, fx], order=1, mode='nearest')
    variants = [('age64_source60pc', axis, g, 64), ('age32_source30pc', fine_axis, fine_g, 32)]
    results = {}
    curves = {}
    for tag, source_axis, modulation, nodes in variants:
        path = output_dir/f'ngc628_source_precision_{tag}.npz'
        signature = hashlib.sha256(json.dumps({'reference': reference['fingerprint'],
            'source_hash': source_hash, 'tag': tag, 'version': 1}, sort_keys=True).encode()).hexdigest()
        start = time.time()
        if path.exists():
            with np.load(path) as z:
                if str(z['fingerprint']) != signature:
                    raise RuntimeError('Precision cache fingerprint mismatch; choose a new output directory.')
                curve, elapsed = z['curve'].copy(), float(z['elapsed_seconds'])
        else:
            model = SourceIntegral(window, source_axis, modulation, info['reference_radius_kpc'],
                beam, age_nodes=nodes, progress_path=output_dir/'ngc628_source_precision_progress.json')
            response = model.moments(ell, eps, rate_log_slope, length)
            curve, _ = combine_moments(response, beta, arm, betaA,
                                       data['noise_signal_fraction'], data['noise_additive_bias'])
            elapsed = time.time()-start
            np.savez_compressed(path, fingerprint=np.asarray(signature), CP=response['CP'],
                                mean=response['mean'], CNP=response['CNP'], curve=curve, elapsed_seconds=elapsed)
            del model, response
        curves[tag] = curve
        results[tag] = {'max_absolute_curve_change': float(np.max(np.abs(curve-coarse))),
                        'elapsed_seconds': elapsed}
        (output_dir/'ngc628_source_precision_partial.json').write_text(json.dumps(results, indent=2)+'\n')
    age_passed = results['age64_source60pc']['max_absolute_curve_change'] < .0005
    grid_passed = results['age32_source30pc']['max_absolute_curve_change'] < .002
    summary = {'status': 'completed fixed-coordinate real-map numerical refinement',
        'age_32_vs_64': results['age64_source60pc'], 'source_grid_60_vs_30_pc': results['age32_source30pc'],
        'point_precision_gate_passed': bool(age_passed and grid_passed),
        'thresholds_absolute_xi': {'age': .0005, 'source_grid': .002},
        'all_observed_pixels': window.n, 'ordered_pixel_pairs_in_positive_bins': int(window.counts.sum()),
        'scope': 'One fixed joint point, unchanged source completion and frozen measurement response. Not a new fit or full-prior precision gate.',
        'remaining': 'Prior-wide beam/grid/age/Gaussian-cutoff/boundary/source-completion tests, validated fast emulator, native-flux recovery, independent MCMC.'}
    (output_dir/'ngc628_source_precision_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    for tag, curve in curves.items():
        ax.plot(data['r_kpc'], curve-coarse, label=tag)
    ax.axhline(0, color='0.5', lw=.8)
    ax.set(xlabel='separation [kpc]', ylabel='refined minus reference correlation',
           title='Real-map numerical precision at one fixed joint draw')
    ax.legend(frameon=False)
    fig.savefig(output_dir/'ngc628_source_precision_audit.png', dpi=180)
    plt.close(fig)
    (output_dir/'ngc628_source_precision_progress.json').write_text(json.dumps(
        {'stage': 'point numerical precision completed', 'status': 'complete',
         'point_precision_gate_passed': summary['point_precision_gate_passed']}, indent=2)+'\n')
    return summary
