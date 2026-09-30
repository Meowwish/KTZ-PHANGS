"""Real-data reference points, not a new posterior or a replacement fit.

All measured pixel pairs are retained. Seven coordinates stay fixed at a
joint draw from the previous posterior so numerical/input corrections can
be assessed without fitting them away. Cached products are fingerprinted.
"""

import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import exp1, expit

from scripts.phangs_source_integral import PairWindow, SourceIntegral, combine_moments
from scripts.phangs_template_ktz import gaussian_source_cnp_kernel


def legacy_point(inputs, x, beam_fraction=.5):
    """Direct radial kernel in the old midpoint/raw-window implementation."""
    with np.load(inputs/'ngc628_full_window_second_moment_response.npz') as z:
        radius = z['kernel_radius']
        response = RegularGridInterpolator((z['rate_log_slope_grid'],), z['response'])([[x[4]]])[0]
        mean = RegularGridInterpolator((z['rate_log_slope_grid'],), z['mean_response'])([[x[4]]])[0]
    ell, eps = np.exp(x[0]), expit(x[1])
    beam2 = beam_fraction*(.12/2.354820045)**2
    ell_obs = np.sqrt(ell*ell+beam2)
    eps_obs = np.sqrt((ell*eps)**2+beam2)/ell_obs
    K = np.r_[-np.log(eps_obs), .5*(exp1(radius[1:]**2/(4*ell_obs**2))-
                                    exp1(radius[1:]**2/(4*(ell_obs*eps_obs)**2)))]
    KN = gaussian_source_cnp_kernel(radius, ell_obs, eps_obs, np.exp(x[6]), quadrature_nodes=64)
    beta, scale, betaA = np.exp(x[[2, 3, 5]])
    terms = np.asarray([response[0]@K, scale*(response[1]@K),
                        beta*(mean[0]+scale*mean[1]+scale*scale*mean[2]),
                        betaA*(response[2]@KN)])
    return terms[:, 0].sum(axis=0)/terms[:, 1].sum(axis=0)


def joint_reference_coordinate(inputs):
    with np.load(inputs/'ngc628_transport_chain.npz') as z:
        chain = z['internal_chain'][int(z['burn_steps']):]
        draws = chain[::100].reshape(-1, 7)
    center = np.median(draws, axis=0)
    scale = np.std(draws, axis=0)
    return draws[np.argmin(np.sum(((draws-center)/scale)**2, axis=1))].copy()


def run_reference_point(root, output_dir, age_nodes=32):
    root, output_dir = Path(root), Path(output_dir)
    inputs = root/'outputs/implications/NGC628_window_projection_refit'
    source_path = output_dir/'ngc628_padded_source_inputs.npz'
    source_summary = json.loads((output_dir/'ngc628_source_inputs_summary.json').read_text())
    coord = joint_reference_coordinate(inputs)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps({'source': source_hash, 'coordinate': coord.tolist(),
        'age_nodes': age_nodes, 'truncate': 6, 'integrator_version': 1}, sort_keys=True).encode()).hexdigest()
    cache = output_dir/'ngc628_global_source_reference_point.npz'
    info_path = output_dir/'ngc628_global_source_reference_point.json'
    if cache.exists():
        with np.load(cache) as z:
            if str(z['fingerprint']) != fingerprint:
                raise RuntimeError('Reference-point cache differs from current inputs; choose a new output directory.')
            moments = {k: z[k] for k in ('CP', 'mean', 'CNP')}
        print('Using validated-identity reference-point cache.', flush=True)
        return json.loads(info_path.read_text())
    progress_path = output_dir/'ngc628_source_integral_progress.json'
    progress_path.write_text(json.dumps({'stage': 'full pair geometry', 'status': 'running'})+'\n')
    start = time.time()
    with np.load(inputs/'ngc628_muse_scal_metallicity_residual_pixel_map.npz') as z:
        window = PairWindow(z['x_grid_kpc'], z['y_grid_kpc'], z['valid_mask'],
                             np.arange(.03, 5.031, .06), .2)
    data = np.genfromtxt(inputs/'ngc628_residual_covariance_diagnostics.csv', delimiter=',', names=True)
    np.testing.assert_allclose(window.counts/2, data['pair_count'], atol=.1)
    with np.load(source_path) as z:
        model = SourceIntegral(window, z['source_axis_kpc'], z['source_modulation'],
                               source_summary['reference_radius_kpc'], z['beam_covariance_kpc2'],
                               age_nodes=age_nodes, progress_path=progress_path)
    ell, eps, beta, arm, rate_log_slope, betaA, length = (
        np.exp(coord[0]), expit(coord[1]), np.exp(coord[2]), np.exp(coord[3]),
        coord[4], np.exp(coord[5]), np.exp(coord[6]))
    if np.min(1+arm*model.modulation) <= 0:
        raise ValueError('The saved old draw is not a positive source rate with the revised source template.')
    moments = model.moments(ell, eps, rate_log_slope, length)
    full, _ = combine_moments(moments, beta, arm, betaA,
                              data['noise_signal_fraction'], data['noise_additive_bias'])
    latent, components = combine_moments(moments, beta, arm, betaA)
    legacy = legacy_point(inputs, coord)
    legacy_fullbeam = legacy_point(inputs, coord, beam_fraction=1.)
    observed = data['xi_centered']
    old = np.genfromtxt(inputs/'ngc628_xi_observed_and_models.csv', delimiter=',', names=True)
    shoulder = (data['r_kpc'] >= .6)&(data['r_kpc'] <= 2.5)
    with np.load(inputs/'ngc628_fit_covariance.npz') as z:
        residual = np.linalg.solve(np.linalg.cholesky(z['covariance']), full-observed)
    summary = {
        'status': 'completed fixed-coordinate real-data reference calculation; NOT refitted',
        'fingerprint': fingerprint, 'elapsed_seconds': time.time()-start,
        'coordinates': dict(zip(['log_ell', 'logit_epsilon', 'log_beta', 'log_eta',
                                 'rate_log_slope', 'log_betaA', 'log_source_length'], coord.tolist())),
        'physical_shape_parameters': {'ell_star_kpc': float(ell), 'epsilon': float(eps),
            'beta': float(beta), 'eta': float(arm), 'rate_log_slope_kpc_inv': float(rate_log_slope),
            'source_excess_amplitude': float(betaA/beta), 'source_length_kpc': float(length)},
        'reference_coordinate_selection': 'one actual prior-run joint draw nearest the coordinatewise median, not a new pointwise posterior median',
        'ordered_pixel_pairs_in_positive_bins': int(window.counts.sum()), 'valid_pixels': window.n,
        'source_grid_step_kpc': model.step, 'age_nodes': age_nodes,
        'fixed_coordinate_chi2_with_old_covariance': float(residual@residual),
        'fixed_coordinate_shoulder_RMSE': float(np.sqrt(np.mean((full[shoulder]-observed[shoulder])**2))),
        'latent_component_sum_max_error': float(np.max(np.abs(latent-components.sum(axis=0)))),
        'source_integral_scope': moments['CNP_scope'],
        'limitations': [
            'Not a newly sampled posterior, not maximum likelihood, and not a claim of improved fit.',
            'Uses the previous correlated measurement response and covariance as a frozen control; new native-flux injection/recovery is still required.',
            'Only this joint point is integrated. Full-prior grid/age/boundary precision and emulator validation must precede a new seven-parameter MCMC.',
            'Source completion and coverage-area corrections remain assumptions; see source-input summary.',
            'Gaussian beam applied to a linear abundance fluctuation is not exact nonlinear multi-line PSF propagation.',
            'Only NGC628; other galaxies/geometry controls and manuscript comparison figures remain unchanged.',
        ],
    }
    np.savez_compressed(cache, fingerprint=np.asarray(fingerprint), CP=moments['CP'], mean=moments['mean'],
                        CNP=moments['CNP'], coordinates=coord, full_observed_response=full,
                        latent_components=components, latent_total=latent)
    np.savetxt(output_dir/'ngc628_global_source_reference_curves.csv',
        np.column_stack([np.r_[0, data['r_kpc']], np.r_[1, observed], np.r_[1, full],
                         np.r_[1, data['noise_signal_fraction']*legacy+data['noise_additive_bias']],
                         np.r_[1, data['noise_signal_fraction']*legacy_fullbeam+data['noise_additive_bias']],
                         np.r_[1, latent], *[np.r_[np.nan, c] for c in components]]), delimiter=',', comments='',
        header='r_kpc,observed,global_fixed_coordinate,midpoint_same_coordinate,midpoint_full_beam_same_coordinate,global_latent,CP0,CPN,CM,CNP')
    (info_path).write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    r = data['r_kpc']
    axes[0].errorbar(r, observed, yerr=data['total_error'], fmt='.', color='k', alpha=.65, label='observed full-pair statistic')
    axes[0].plot(r, old['xi_KTZ_median'][1:], color='0.4', ls='--', label='previous posterior median (old model)')
    axes[0].plot(r, data['noise_signal_fraction']*legacy+data['noise_additive_bias'], color='#e66101',
                 ls=':', label='old model, same fixed joint draw')
    axes[0].plot(r, full, color='#7b3294', label='global integral, fixed joint draw (NOT posterior)')
    axes[0].plot(r, old['xi_KT18_median'][1:], color='#1f78b4', ls='--', label='previous KT18 posterior median')
    axes[0].set(title='Reference calculation before new inference', xlabel='separation [kpc]', ylabel=r'$\xi_{3,\rm pp}$')
    axes[0].legend(fontsize=7, frameon=False)
    base = np.zeros(len(r))
    for component, color, label in zip(components, ['#7fc97f', '#ffff99', '#fdc086', '#beaed4'],
                                      [r'$C_{\rm 2p,0}$', r'$C_{\rm 2p,N}$', r'$C_{\rm M}$', r'$C_{\rm NP}$']):
        axes[1].fill_between(r, base, base+component, color=color, alpha=.7, label=label)
        base += component
    axes[1].plot(r, latent, color='k', ls='--', label='latent total: exact sum of this draw')
    axes[1].scatter(r, (observed-data['noise_additive_bias'])/data['noise_signal_fraction'],
                    s=5, color='k', label='noise-adjusted observed statistic')
    axes[1].set(title='Complete KTZ components at one fixed draw', xlabel='separation [kpc]', ylabel='latent pair statistic')
    axes[1].legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.axhline(0, color='0.5', lw=.8)
        ax.set_xlim(0, 5)
    fig.savefig(output_dir/'ngc628_global_source_reference_point.png', dpi=180)
    plt.close(fig)
    progress_path.write_text(json.dumps({'stage': 'reference point completed', 'status': 'complete',
                                        'elapsed_seconds': summary['elapsed_seconds']}, indent=2)+'\n')
    return summary
