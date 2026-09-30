"""Bounded source-response refinement after the failed first CNP pilot.

Preserve the pilot. Reuse its exact source integrals as development data,
including its former holdouts (explicitly no longer independent validation).
Each stage freezes a CV-selected representation before evaluating NEW
holdouts. All pixel pairs remain exhaustive; no astronomical prior changes.
"""

import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from scripts.build_ngc628_age_library import file_hash, load_inputs, padded_source
from scripts.build_ngc628_cnp_pilot import (physical_shape, shape_points,
    source_transfer, load_control, LOW, HIGH)
from scripts.phangs_cnp_emulator import (PositiveCNPEmulator, features,
    mixture_error_bound, select_numerical_settings, test_mixture_bound)
from scripts.phangs_source_fft import FFTSourceIntegral


FOLDER = 'cnp_refinement_v1'
TOLERANCE = .002


def save_json(path, value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def prepare(root, output):
    root, output = Path(root), Path(output)
    folder = output/FOLDER
    folder.mkdir(exist_ok=True)
    old_manifest = json.loads((output/'ngc628_cnp_pilot_manifest.json').read_text())
    for name, expected in old_manifest['implementation_hashes'].items():
        if file_hash(root/'scripts'/name) != expected:
            raise RuntimeError('Legacy teacher code changed; do not silently reuse its source integrals.')
    checks = {'source_hash': output/'ngc628_padded_source_inputs.npz',
              'observed_hash': output/'ngc628_observed_abundance_control.npz',
              'control_hash': output/'ngc628_cnp_gaussian_control_library.npz'}
    for key, path in checks.items():
        if file_hash(path) != old_manifest[key]:
            raise RuntimeError('Legacy source/control fingerprint changed.')
    age_manifest = json.loads((output/'ngc628_age_library_manifest.json').read_text())
    source_info_hash = file_hash(output/'ngc628_source_inputs_summary.json')
    if source_info_hash != age_manifest['source_info_hash']:
        raise RuntimeError('Radial source definition changed; a new teacher version is required.')
    implementation = ['refine_ngc628_cnp_response.py', 'phangs_cnp_emulator.py',
        'phangs_source_fft.py', 'phangs_source_integral.py', 'build_ngc628_age_library.py',
        'build_ngc628_cnp_pilot.py', 'phangs_cnp_baseline.py']
    manifest = {'legacy_pilot_hash': file_hash(output/'ngc628_nonlocal_cnp_pilot.npz'),
        'source_hash': old_manifest['source_hash'], 'observed_hash': old_manifest['observed_hash'],
        'control_hash': old_manifest['control_hash'], 'source_info_hash': source_info_hash,
        'implementation_hashes': {f: file_hash(root/'scripts'/f) for f in implementation},
        'stages': 2, 'new_training_points_per_stage': 128, 'new_random_holdouts_per_stage': 16,
        'limit_stress_points_per_stage': 8, 'age_nodes': 32, 'gaussian_cutoff_sigma': 6,
        'source_step_kpc': .06, 'numerical_xi_tolerance': TOLERANCE,
        'source_completion': 'unchanged exponential radial extrapolation; zero arm modulation outside support',
        'physical_scope': 'ONE original nonlocal radial-rate CNP; no arm coupling or extra free parameter',
        'validation': 'old 16 holdouts are development data; new holdouts never select settings within their stage',
        'schedule': 'stage 2 runs only if stage 1 fails; failed stage-1 tests may then become development data',
        'inference_ready': False}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = folder/'manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise RuntimeError('Refinement fingerprint changed; use a new refinement version, not old caches.')
    else:
        save_json(manifest_path, manifest)
    return root, output, folder, signature


def legacy_development(output):
    with np.load(output/'ngc628_nonlocal_cnp_pilot.npz') as z:
        return z['coordinate'].copy(), z['CNP'].copy(), z['baseline'].copy()


def space_filling_additions(existing, n, stage):
    """Maximin numerical design, not posterior draws or a new physical prior."""
    candidates = shape_points(1024, 6281950+stage)
    metric = features(candidates, 'variance', .25)
    distances = cdist(metric, features(existing, 'variance', .25)).min(axis=1)
    chosen = []
    for _ in range(n):
        index = int(np.argmax(distances))
        if distances[index] < 1e-8:
            raise RuntimeError('Duplicate/exhausted numerical source-response design.')
        chosen.append(index)
        distances = np.minimum(distances, cdist(metric, metric[index:index+1])[:, 0])
    return candidates[chosen]


def limit_stress_points(stage):
    specs = [(.13, .2, 0., .05), (.7, .08, 0., .5), (2.1, .06, 0., 2.),
        (.18, .2, -.35, .02), (.18, .2, .35, 1.1),
        (1.8, .07, -.35, .005), (1.8, .07, .35, 2.5), (2.8, .06, -.35, 2.5)]
    output = []
    for ell, eps, rate_log_slope, length in specs:
        ell *= 1+.025*stage
        sigma = ell*eps
        kt = .5*ell*ell*(1-eps*eps)
        x = np.array([np.log(kt), np.log(sigma), rate_log_slope, np.log(length)])
        if np.any(x < LOW-1e-12) or np.any(x > HIGH+1e-12):
            raise RuntimeError('Invalid prespecified limit stress point.')
        output.append(x)
    return np.asarray(output)


def exact_points(root, output, folder, signature, coordinates, role, stage):
    window, axis, g, beam, info = load_inputs(root, output)
    control = load_control(output/'ngc628_cnp_gaussian_control_library.npz')
    nodes = folder/'source_nodes'
    nodes.mkdir(exist_ok=True)
    responses, baselines = [], []
    start = time.time()
    for i, x in enumerate(coordinates):
        point_key = hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest()
        path = nodes/f'{point_key}.npz'
        kt, sigma, rate_log_slope, length, ell, eps = physical_shape(x)
        if path.exists():
            with np.load(path) as z:
                if str(z['signature']) != signature or not np.array_equal(x, z['coordinate']):
                    raise RuntimeError('Point-level source-integral fingerprint mismatch.')
                response = z['CNP'].copy()
        else:
            extended_axis, extended_g = padded_source(window, axis, g, beam, ell*ell)
            model = FFTSourceIntegral(window, extended_axis, extended_g,
                info['reference_radius_kpc'], beam, age_nodes=32, distinct_block=2,
                progress_path=folder/'integral_progress.json')
            transfer = source_transfer(model, ell, eps)
            rate = np.exp(rate_log_slope*(model.radius-info['reference_radius_kpc']))
            response = model.distinct_from_kernel(transfer, rate, length)
            np.savez_compressed(path, signature=np.asarray(signature), coordinate=x, CNP=response)
            del model, transfer, rate, extended_g
        responses.append(response)
        baselines.append(control.moments(kt, sigma, rate_log_slope, length, info['reference_radius_kpc']))
        state = {'stage': stage+1, 'phase': 'exact nonlocal source integrals', 'role': role,
            'completed_points_in_role': i+1, 'points_in_role': len(coordinates),
            'ell_star_kpc': float(ell), 'rate_log_slope_kpc_inv': float(rate_log_slope), 'L_src_kpc': float(length),
            'role_elapsed_seconds': time.time()-start, 'all_pixels_and_pairs_retained': True,
            'not_a_galaxy_fit': True}
        save_json(folder/'progress.json', state)
        print(state, flush=True)
    return np.asarray(responses), np.asarray(baselines)


def run_refinement_stage(root, output, stage=0):
    if stage not in (0, 1):
        raise ValueError('The bounded refinement has two stages only.')
    root, output, folder, signature = prepare(root, output)
    final_path = folder/f'stage_{stage+1}_validation.json'
    if final_path.exists():
        return json.loads(final_path.read_text())
    x, truth, baseline = legacy_development(output)
    if stage:
        previous_summary = json.loads((folder/'stage_1_validation.json').read_text())
        if previous_summary['tested_CNP_precision_gate_passed']:
            return {'status': 'Second stage unnecessary: first-stage fresh tests passed',
                    'tested_CNP_precision_gate_passed': True, 'inference_ready': False}
        with np.load(folder/'stage_1_response.npz') as z:
            # Failed tests are expressly reclassified as development data.
            x, truth, baseline = z['all_coordinates'], z['all_CNP'], z['all_baseline']
    additions = space_filling_additions(x, 128, stage)
    added_truth, added_baseline = exact_points(root, output, folder, signature,
        additions, 'new training (source integrals only)', stage)
    x = np.vstack((x, additions))
    truth, baseline = np.vstack((truth, added_truth)), np.vstack((baseline, added_baseline))
    control = load_control(output/'ngc628_cnp_gaussian_control_library.npz')
    info = json.loads((output/'ngc628_source_inputs_summary.json').read_text())
    settings, cv_scores = select_numerical_settings(x, truth, baseline, control,
        info['reference_radius_kpc'], seed=6281904+stage)
    save_json(folder/f'stage_{stage+1}_cross_validation.json', {
        'selected_settings': settings, 'scores': cv_scores, 'development_points': len(x),
        'fresh_validation_evaluated_during_selection': False,
        'selection_uses_observed_metallicity_values': False})
    np.savez_compressed(folder/f'stage_{stage+1}_training.npz', signature=np.asarray(signature),
        coordinates=x, CNP=truth, baseline=baseline, settings=np.asarray(json.dumps(settings)))
    model = PositiveCNPEmulator(x, truth, baseline, control, info['reference_radius_kpc'], **settings)
    # Freeze the stage representation before computing independent validation.
    random_tests = shape_points(16, 6281990+stage)
    tests = np.vstack((random_tests, limit_stress_points(stage)))
    if cdist(tests, x).min() < 1e-9:
        raise RuntimeError('A validation coordinate duplicates development data.')
    actual, test_baseline = exact_points(root, output, folder, signature, tests,
        'fresh random/limit validation, not training', stage)
    approximation = model.predict(tests, test_baseline)
    bounds = mixture_error_bound(actual, approximation)
    true_shape, fast_shape = actual[:, 0]/actual[:, 1], approximation[:, 0]/approximation[:, 1]
    physical = (np.max(abs(true_shape), axis=1) <= 1+1e-8)&(np.max(abs(fast_shape), axis=1) <= 1+1e-8)
    passed = bool(np.all(physical)&np.all(bounds < TOLERANCE))
    diagnostics = []
    for i, point in enumerate(tests):
        _, _, rate_log_slope, length, ell, eps = physical_shape(point)
        diagnostics.append({'index': i, 'role': 'new random holdout' if i < 16 else 'new limit stress point',
            'ell_star_kpc': float(ell), 'epsilon': float(eps), 'rate_log_slope_kpc_inv': float(rate_log_slope),
            'L_src_kpc': float(length), 'physical_shape': bool(physical[i]),
            'maximum_CNP_shape_error': float(np.max(abs(true_shape[i]-fast_shape[i]))),
            'maximum_log_second_moment_error': float(np.max(abs(np.log(approximation[i, 1]/actual[i, 1])))),
            'maximum_any_positive_mixture_error_bound': float(bounds[i].max())})
    summary = {'status': 'Numerical CNP refinement stage completed; no galaxy posterior sampled',
        'stage': stage+1, 'development_points': len(x), 'new_independent_random_holdouts': 16,
        'new_limit_stress_points': 8, 'selected_numerical_settings': settings,
        'absolute_xi_tolerance': TOLERANCE, 'tested_CNP_precision_gate_passed': passed,
        'maximum_CNP_shape_error': float(np.max(abs(true_shape-fast_shape))),
        'maximum_any_positive_mixture_error_bound': float(bounds.max()), 'diagnostics': diagnostics,
        'all_pixels_and_pairs_retained': True, 'free_physical_parameters_added': 0,
        'physical_components': 'one original CNP; positive-moment encoding is numerical only',
        'old_pilot_holdouts_reclassified_as_development': True,
        'inference_ready': False,
        'remaining': 'prior-wide/refinement/source-boundary and native-flux recovery; independent galaxy MCMC'}
    np.savez_compressed(folder/f'stage_{stage+1}_response.npz', signature=np.asarray(signature),
        training_coordinate=x, training_CNP=truth, training_baseline=baseline,
        test_coordinate=tests, actual_CNP=actual, predicted_CNP=approximation, error_bound=bounds,
        all_coordinates=np.vstack((x, tests)), all_CNP=np.vstack((truth, actual)),
        all_baseline=np.vstack((baseline, test_baseline)))
    worst = np.argsort(bounds.max(axis=1))[-4:][::-1]
    r = .5*(np.arange(.03, 5.031, .06)[1:]+np.arange(.03, 5.031, .06)[:-1])
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
    for ax, i in zip(axes.ravel(), worst):
        ax.plot(r, true_shape[i], 'k--', label='direct nonlocal source integral')
        ax.plot(r, fast_shape[i], color='#7b3294', label='fresh-test numerical response')
        ax.axhline(0, color='0.5', lw=.8)
        ax.set(title=f'Test {i}: shape error {diagnostics[i]["maximum_CNP_shape_error"]:.4g}',
               xlabel='separation [kpc]', ylabel='projected CNP shape (not observed xi)')
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(folder/f'stage_{stage+1}_worst_shape_tests.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    ax.plot(np.arange(len(tests)), bounds.max(axis=1), 'o', color='#7b3294',
            label='bound including variance-allocation error')
    ax.axhline(TOLERANCE, color='k', ls='--', label='predefined 0.002 tolerance')
    ax.axvline(15.5, color='0.6', lw=.8)
    ax.set(xlabel='fresh numerical test index (16 random, then 8 limit tests)',
           ylabel='maximum absolute xi error bound', title=f'CNP refinement stage {stage+1}: {"PASS" if passed else "FAIL"}')
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(folder/f'stage_{stage+1}_mixture_error_tests.png', dpi=180)
    plt.close(fig)
    save_json(final_path, summary)
    save_json(folder/'progress.json', {'stage': stage+1, 'phase': 'stage complete',
        'tested_CNP_precision_gate_passed': passed, 'inference_ready': False})
    return summary


def development_audit(root, output):
    root, output, folder, _ = prepare(root, output)
    algebra = test_mixture_bound()
    save_json(folder/'mixture_bound_algebra.json', algebra)
    x, truth, baseline = legacy_development(output)
    control = load_control(output/'ngc628_cnp_gaussian_control_library.npz')
    info = json.loads((output/'ngc628_source_inputs_summary.json').read_text())
    settings, scores = select_numerical_settings(x, truth, baseline, control, info['reference_radius_kpc'])
    summary = {'status': 'legacy teacher cross-validation; NOT new independent validation',
        'development_points': len(x), 'selected_numerical_settings': settings, 'scores': scores,
        'original_pilot': 'failed; its former holdouts now belong to development',
        'all_pixels_and_pairs_retained': True, 'inference_ready': False}
    save_json(folder/'legacy_development_cross_validation.json', summary)
    return {'status': summary['status'], 'development_points': len(x),
            'selected_numerical_settings': settings, 'mixture_bound_algebra': algebra,
            'inference_ready': False}


def notebook_cells():
    import nbformat
    cells = [nbformat.v4.new_markdown_cell(r'''## 14. Failed CNP pilot and bounded refinement

The first 65-training/16-holdout pilot **failed** the predefined 0.002 absolute
correlation tolerance (largest pure-CNP shape error 0.0255). Its products remain
unchanged. All 81 exact source responses are now development data, including
the old holdouts: those 16 points are **not** independent validation any more.

The refined numerical representation models the correction to the normalised
CNP shape and the logarithm of its projected second moment, then reconstructs
one $(N,D)$ CNP response **before** combining KTZ components. This enforces a
positive second moment, not a positive correlation: signed projected tails
remain allowed. The control/correction are not separate physical terms and
add no fit parameter. Numerical settings use fixed five-fold source-response
cross-validation, never observed metallicity values. An optional radial-slope
factor imposes the homogeneous continuum control at zero slope; fresh zero-
slope tests explicitly check its discretisation error.

Each of at most two stages adds 128 maximin source-parameter training points
and freezes the numerical representation before evaluating 16 NEW random
holdouts and 8 NEW limit tests. If stage 1 fails, its tests may become stage-2
development data, but stage 2 uses new tests. Observing pixels/pairs are never
sampled. Physical priors, source completion, age quadrature and the CNP
definition remain unchanged.

Besides shape error, validation bounds the error for **any positive allocation**
to the other KTZ terms: with $h=D_{
m fast}/D_{
m true}$, it is bounded by
$|\xi_{
m fast}-\xi_{
m true}|+2\tanh(|\ln h|/4)$. Thus a good normalised
CNP curve cannot conceal an incorrect component amplitude. The 0.002 bound
is a numerical gate, not an extra observational noise floor. Native-flux
recovery and the remaining prior/source-boundary tests are still required;
this job does not start MCMC or replace manuscript results.''')]
    cells[0]['id'] = 'source-integral-cnp-refine-method'
    init = nbformat.v4.new_code_cell('''from pathlib import Path
import json
from IPython.display import Image, display
from scripts.refine_ngc628_cnp_response import development_audit, run_refinement_stage
refine_root = Path.cwd()
if not (refine_root/'scripts').is_dir():
    raise RuntimeError('Run from the manuscript workspace root.')
refine_output = refine_root/'outputs/implications/NGC628_source_integral_refit'
refine_folder = refine_output/'cnp_refinement_v1'
print(json.dumps(development_audit(refine_root, refine_output), indent=2))''')
    init['id'] = 'source-integral-cnp-refine-audit'
    cells.append(init)
    for stage in (0, 1):
        code = f'''refine_stage_{stage+1} = run_refinement_stage(refine_root, refine_output, stage={stage})
print(json.dumps({{k:v for k,v in refine_stage_{stage+1}.items() if k != 'diagnostics'}}, indent=2))
for suffix in ('worst_shape_tests', 'mixture_error_tests'):
    image_path = refine_folder/f'stage_{stage+1}_{{suffix}}.png'
    if image_path.exists():
        display(Image(filename=str(image_path)))'''
        cell = nbformat.v4.new_code_cell(code)
        cell['id'] = f'source-integral-cnp-refine-stage-{stage+1}'
        cells.append(cell)
    return cells
