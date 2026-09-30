"""Numerical nonlocal CNP surrogate pilot; never trained on metallicity data.

One CNP is retained. The analytic endpoint-rate control and its learned
source-integral residual are numerical representations of that single term,
not two physical components or an extra fit amplitude. Holdouts can fail; no
galaxy MCMC is authorised by table construction alone.
"""

import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.special import expit
from scipy.stats import qmc

from scripts.build_ngc628_age_library import load_inputs, padded_source, file_hash, mean_moments
from scripts.phangs_age_response import AgeResponseLibrary
from scripts.phangs_cnp_baseline import gaussian_endpoint_responses, StationaryCNPControl
from scripts.phangs_source_fft import FFTSourceIntegral
from scripts.phangs_source_integral import age_quadrature


LOW = np.array([np.log(.0025), np.log(.015), -.35, np.log(.005)])
HIGH = np.array([np.log(4.5), np.log(.25), .35, np.log(2.5)])


def physical_shape(coordinate):
    kt, sigma, rate_log_slope, length = np.exp(coordinate[0]), np.exp(coordinate[1]), coordinate[2], np.exp(coordinate[3])
    ell = np.sqrt(sigma*sigma+2*kt)
    return kt, sigma, rate_log_slope, length, ell, sigma/ell


def shape_points(n, seed):
    proposals = qmc.Sobol(4, scramble=True, seed=seed).random_base2(12)
    coordinates = LOW+proposals*(HIGH-LOW)
    points = []
    for x in coordinates:
        _, _, _, _, ell, eps = physical_shape(x)
        # Existing transport-grid limits plus the product feasibility of the
        # unchanged kappa/time/sigma priors. These are training coordinates,
        # not posterior draws or a newly restricted galaxy prior.
        if .08 <= ell <= 3.2 and .015 <= eps <= .9:
            points.append(x)
            if len(points) == n:
                return np.asarray(points)
    raise RuntimeError('Insufficient valid physical pilot coordinates.')


def uniform_cnp_scale(ell, eps, length, beam):
    u, w = age_quadrature(eps, 32)
    eigen = np.linalg.eigvalsh(beam)
    isotropic = length*length+ell*ell*(u[:, None]+u[None])
    return float(np.sum(w[:, None]*w[None]*length**2/
                        np.sqrt((isotropic+2*eigen[0])*(isotropic+2*eigen[1]))))


def source_transfer(model, ell, eps):
    u, weights = age_quadrature(eps, model.age_nodes)
    transfer = np.zeros_like(model.kernel(np.eye(2)*ell*ell*u.max()+model.beam))
    cy, cx = np.asarray(transfer.shape)//2
    for width, weight in zip(u, weights):
        kernel = model.kernel(np.eye(2)*ell*ell*width+model.beam)
        ry, rx = np.asarray(kernel.shape)//2
        transfer[cy-ry:cy+ry+1, cx-rx:cx+rx+1] += weight*kernel
    return transfer


def build_gaussian_control(root, output, variance_nodes=81, slope_nodes=9):
    root, output = Path(root), Path(output)
    window, _, _, beam, _ = load_inputs(root, output)
    variances = np.geomspace(.00047, 25., variance_nodes)
    slopes = np.linspace(-.35, .35, slope_nodes)
    manifest = {'observed_hash': file_hash(output/'ngc628_observed_abundance_control.npz'),
        'source_hash': file_hash(output/'ngc628_padded_source_inputs.npz'),
        'code_hash': file_hash(root/'scripts/phangs_cnp_baseline.py'),
        'variance_grid': variances.tolist(), 'slope_grid': slopes.tolist(),
        'role': 'stationary endpoint control only; NOT a replacement for nonlocal CNP'}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    folder = output/'cnp_gaussian_control_nodes'
    folder.mkdir(exist_ok=True)
    responses = []
    reference = None
    for i, variance in enumerate(variances):
        node = folder/f'gaussian_control_{i:03d}.npz'
        if node.exists():
            with np.load(node) as z:
                if str(z['signature']) != signature:
                    raise RuntimeError('Gaussian-control cache changed; choose a new version/output directory.')
                response, reference = z['response'].copy(), float(z['reference_radius'])
        else:
            response, reference = gaussian_endpoint_responses(window, variance, beam, slopes)
            np.savez_compressed(node, signature=np.asarray(signature), response=response,
                                reference_radius=reference, variance=variance)
        responses.append(response)
        progress = {'stage': 'Gaussian CNP regression control (not full theory)',
                    'completed_width_nodes': i+1, 'total_width_nodes': variance_nodes}
        (output/'ngc628_cnp_control_progress.json').write_text(json.dumps(progress, indent=2)+'\n')
        print(progress, flush=True)
    path = output/'ngc628_cnp_gaussian_control_library.npz'
    np.savez_compressed(path, signature=np.asarray(signature), variance_grid=variances,
        slope_grid=slopes, response=np.asarray(responses), reference_radius=reference)
    (output/'ngc628_cnp_control_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return {'status': 'Gaussian regression control completed; not the nonlocal CNP likelihood',
            'variance_nodes': variance_nodes, 'slope_nodes': slope_nodes,
            'next': 'Train deterministic nonlocal source-integral differences and validate held-out points.'}


def load_control(path):
    with np.load(path) as z:
        return StationaryCNPControl(z['variance_grid'], z['slope_grid'], z['response'], float(z['reference_radius']))


def run_cnp_pilot(root, output, training_points=64, holdout_points=16):
    root, output = Path(root), Path(output)
    window, axis, g, beam, source_info = load_inputs(root, output)
    control_file = output/'ngc628_cnp_gaussian_control_library.npz'
    control = load_control(control_file)
    training = shape_points(training_points, 6281901)
    holdouts = shape_points(holdout_points, 6281902)
    with np.load(output/'ngc628_global_source_reference_point.npz') as z:
        old = z['coordinates']
        ell, eps = np.exp(old[0]), expit(old[1])
        reference_coordinate = np.array([np.log(.5*ell*ell*(1-eps*eps)), np.log(ell*eps), old[4], old[6]])
        reference_np = z['CNP'].copy()
    training = np.vstack([training, reference_coordinate])
    coordinates = np.vstack([training, holdouts])
    roles = ['training']*len(training)+['heldout']*len(holdouts)
    implementation = ['phangs_source_fft.py', 'phangs_source_integral.py',
                       'phangs_cnp_baseline.py', 'build_ngc628_cnp_pilot.py', 'build_ngc628_age_library.py']
    manifest = {'source_hash': file_hash(output/'ngc628_padded_source_inputs.npz'),
        'observed_hash': file_hash(output/'ngc628_observed_abundance_control.npz'),
        'control_hash': file_hash(control_file),
        'implementation_hashes': {f: file_hash(root/'scripts'/f) for f in implementation},
        'coordinates': coordinates.tolist(), 'roles': roles, 'age_nodes': 32, 'truncate': 6,
        'role': 'nonlocal CNP source-integral interpolation; no metallicity curve used in training'}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    folder = output/'cnp_nonlocal_pilot_nodes'
    folder.mkdir(exist_ok=True)
    truth, baselines, scales = [], [], []
    start_all = time.time()
    for i, (x, role) in enumerate(zip(coordinates, roles)):
        kt, sigma, rate_log_slope, length, ell, eps = physical_shape(x)
        path = folder/f'nonlocal_cnp_{i:03d}.npz'
        if path.exists():
            with np.load(path) as z:
                if str(z['signature']) != signature:
                    raise RuntimeError('Nonlocal CNP pilot fingerprint changed; choose a new version/output directory.')
                response = z['CNP'].copy()
        elif i == len(training)-1:
            response = reference_np
            np.savez_compressed(path, signature=np.asarray(signature), coordinate=x, CNP=response,
                                role=np.asarray(role), reused_exact_reference=np.asarray(True))
        else:
            extended_axis, extended_g = padded_source(window, axis, g, beam, ell*ell)
            model = FFTSourceIntegral(window, extended_axis, extended_g,
                source_info['reference_radius_kpc'], beam, age_nodes=32,
                progress_path=output/'ngc628_cnp_pilot_integral_progress.json', distinct_block=2)
            transfer = source_transfer(model, ell, eps)
            rate = np.exp(rate_log_slope*(model.radius-source_info['reference_radius_kpc']))
            response = model.distinct_from_kernel(transfer, rate, length)
            np.savez_compressed(path, signature=np.asarray(signature), coordinate=x,
                                CNP=response, role=np.asarray(role))
            del model, transfer, rate, extended_g
        truth.append(response)
        baselines.append(control.moments(kt, sigma, rate_log_slope, length, source_info['reference_radius_kpc']))
        scales.append(uniform_cnp_scale(ell, eps, length, beam))
        progress = {'stage': 'nonlocal CNP source-integral pilot', 'completed_points': i+1,
            'total_points': len(coordinates), 'role': role, 'ell_star_kpc': float(ell),
            'epsilon': float(eps), 'rate_log_slope_kpc_inv': float(rate_log_slope), 'source_length_kpc': float(length),
            'elapsed_seconds': time.time()-start_all, 'not_a_galaxy_fit': True}
        (output/'ngc628_cnp_pilot_progress.json').write_text(json.dumps(progress, indent=2)+'\n')
        print(progress, flush=True)
    truth, baselines, scales = np.asarray(truth), np.asarray(baselines), np.asarray(scales)
    normalized = (coordinates-LOW)/(HIGH-LOW)
    residual = (truth-baselines)/scales[:, None, None]
    # Kernel choices are numerical interpolation settings, not additional
    # astrophysical parameters or quantities optimised on the observed xi.
    interpolator = RBFInterpolator(normalized[:len(training)], residual[:len(training)].reshape(len(training), -1),
                                   kernel='thin_plate_spline', degree=1, smoothing=1e-12)
    predicted = baselines[len(training):]+scales[len(training):, None, None]*interpolator(
        normalized[len(training):]).reshape(holdout_points, 2, window.nb)
    age_library = AgeResponseLibrary(output/'ngc628_single_age_response_library.npz')
    diagnostics = []
    arm_max = min(1., float(source_info['maximum_positive_eta']))
    for k, (x, actual, approximate) in enumerate(zip(holdouts, truth[len(training):], predicted)):
        _, _, rate_log_slope, _, ell, eps = physical_shape(x)
        positive = bool(np.all(approximate[1] > 0))
        pure_error = float(np.max(abs(actual[0]/actual[1]-approximate[0]/approximate[1]))) if positive else None
        cp, fields = age_library.integrate(ell, eps, rate_log_slope)
        mean = mean_moments(window, fields)
        maximum_error = 0.
        # Numerical mixture tests cover limiting allocation, not galaxy
        # parameter choices, optimisation or an extra Poisson KTZ fit.
        if positive:
            for beta in (.01, 1., 100.):
                for arm in (.1, arm_max):
                    no_np = cp[0]+arm*cp[1]+beta*(mean[0]+arm*mean[1]+arm*arm*mean[2])
                    for A in (.01, .1, 1., 10., 100., 1000.):
                        true = no_np+beta*A*actual
                        fast = no_np+beta*A*approximate
                        if np.any(fast[1] <= 0) or np.any(true[1] <= 0):
                            positive = False
                            break
                        maximum_error = max(maximum_error, float(np.max(abs(true[0]/true[1]-fast[0]/fast[1]))))
        diagnostics.append({'heldout_index': k, 'positive_second_moments': positive,
            'CNP_only_shape_error': pure_error, 'maximum_mixture_probe_curve_error': maximum_error if positive else None,
            'ell_star_kpc': float(ell), 'epsilon': float(eps), 'rate_log_slope_kpc_inv': float(rate_log_slope)})
    passed = all(d['positive_second_moments'] and d['maximum_mixture_probe_curve_error'] < .002
                  and d['CNP_only_shape_error'] < .002 for d in diagnostics)
    summary = {'status': 'nonlocal CNP pilot completed; no new galaxy posterior sampled',
        'training_points': len(training), 'independent_heldout_points': holdout_points,
        'tested_CNP_precision_gate_passed': bool(passed), 'absolute_xi_tolerance': .002,
        'heldout_diagnostics': diagnostics,
        'scope': 'Fixed-parameter source-integral interpolation, conditioned on stated radial extrapolation; not observational support or native-flux validation.',
        'physical_components': 'one CNP only; regression baseline + deterministic numerical residual are not extra fitted physical terms',
        'next': 'Refine failed heldouts/domain tests; native-flux recovery and independent-ensemble KTZ/KT18 fits only after complete response is validated.'}
    np.savez_compressed(output/'ngc628_nonlocal_cnp_pilot.npz', signature=np.asarray(signature),
        coordinate=coordinates, roles=np.asarray(roles), CNP=truth, baseline=baselines,
        uniform_scale=scales, normalized_coordinate=normalized, predicted_heldout_CNP=predicted)
    (output/'ngc628_cnp_pilot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'ngc628_cnp_pilot_validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    r = .5*(window.edges[1:]+window.edges[:-1])
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for k, ax in enumerate(axes.ravel()):
        actual, approximate = truth[len(training)+k], predicted[k]
        ax.plot(r, actual[0]/actual[1], 'k--', label='direct nonlocal source integral')
        if np.all(approximate[1] > 0):
            ax.plot(r, approximate[0]/approximate[1], color='#7b3294', label='held-out numerical surrogate')
        ax.set(title=f'Held-out source-shape point {k}', xlabel='separation [kpc]', ylabel='projected CNP shape (not observed xi)')
        ax.axhline(0, color='0.5', lw=.8)
        ax.legend(fontsize=7, frameon=False)
    fig.savefig(output/'ngc628_cnp_pilot_holdouts.png', dpi=180)
    plt.close(fig)
    (output/'ngc628_cnp_pilot_progress.json').write_text(json.dumps({'stage': 'CNP pilot complete',
        'status': 'complete', 'tested_CNP_precision_gate_passed': bool(passed),
        'not_a_galaxy_fit': True}, indent=2)+'\n')
    return summary
