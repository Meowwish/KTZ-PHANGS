"""Audit the completed refit without changing priors or posterior summaries.

Likelihood optimisation is a diagnostic only, never a replacement for the
requested posterior-median curves. The shoulder was selected after viewing
the data: all shoulder checks are exploratory.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.optimize import minimize
from scipy.special import expit, logit

from scripts.refit_ngc628_window_projection import (
    ProjectedKTZ, kernel_point_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'outputs/implications/NGC628_halpha_map_systematic'
OUTPUT = ROOT/'outputs/implications/NGC628_window_projection_refit'


def audit_completed_refit():
    with np.load(OUTPUT/'ngc628_fit_covariance.npz') as archive:
        model = ProjectedKTZ(
            INPUT, OUTPUT, archive['covariance'], archive['raw_covariance'],
        )
    with np.load(OUTPUT/'ngc628_transport_chain.npz') as archive:
        burn = int(archive['burn_steps'])
        chain = archive['internal_chain'][burn:]
        density = archive['log_probability'][burn:]
    samples = chain[::20].reshape(-1, 7)
    prior = model.prior(samples)
    eps = expit(samples[:, 1])
    original_prior = (
        np.log(2)-np.log1p(eps)
        - .5*(samples[:, 3]/model.eta_prior_sigma)**2
        - .5*((samples[:, 4]-model.metadata['event_rate_log_slope_kpc_inv'])
              /model.metadata['event_rate_log_slope_error_kpc_inv'])**2
    )
    if not np.all(np.isfinite(prior)):
        raise RuntimeError('A retained posterior draw violates physical bounds.')
    np.testing.assert_allclose(prior, original_prior, rtol=1e-13, atol=1e-13)

    best_chi2, best_retained = np.inf, None
    # All retained chain states, not just the thinned diagnostic sample.
    for start in range(0, len(chain), 2000):
        states = chain[start:start+2000].reshape(-1, 7)
        likelihood_chi2 = -2*(density[start:start+2000].ravel()
                              - model.prior(states))
        index = int(np.argmin(likelihood_chi2))
        if likelihood_chi2[index] < best_chi2:
            best_chi2 = float(likelihood_chi2[index])
            best_retained = states[index].copy()

    rng = np.random.default_rng(907628)
    draws = samples[rng.choice(len(samples), 4000, replace=False)]
    curves = model.predict(draws)
    if not np.all(np.isfinite(curves)):
        raise RuntimeError('Non-finite retained posterior predictive curve.')
    checks = draws[rng.choice(len(draws), 80, replace=False)]
    point_errors = []
    for x in checks:
        direct = kernel_point_prediction(model, x)
        error = np.abs(model.predict(x)[0]-direct)
        if not np.all(np.isfinite(error)):
            raise RuntimeError('Non-finite posterior kernel point check.')
        point_errors.append(float(error.max()))
    maximum_point_error = float(max(point_errors))
    if maximum_point_error > .01:
        raise RuntimeError('Posterior kernel point checks exceed 0.01.')

    def chi2(x):
        prediction = model.predict(x)[0]
        if not np.all(np.isfinite(prediction)):
            return 1e10
        residual = (prediction-model.data['xi_centered'])@model.precision_root.T
        return float(residual@residual)

    bounds = np.asarray([
        np.log(model.ell_grid[[0, -1]]),
        logit(model.eps_grid[[0, -1]]),
        np.log([1e-4, 1e4]),
        np.log([.1, model.metadata['maximum_positive_eta']]),
        [-.35, .35], np.log([1e-6, 1e7]),
        np.log(model.src_grid[[0, -1]]),
    ])

    def physical_constraints(x):
        values = model.derived(x)
        constraints = []
        for index, lower, upper in (
            (4, .05, 1.5), (5, .05, 3), (6, .015, .25), (7, .01, 1e3),
        ):
            value = float(values[index][0])
            constraints.extend([np.log(value/lower), np.log(upper/value)])
        coordinates = model.kernel_coordinates(np.atleast_2d(x))[0]
        constraints.extend([
            coordinates[0]-np.log(model.ell_grid[0]),
            np.log(model.ell_grid[-1])-coordinates[0],
            coordinates[1]-np.log(model.eps_grid[0]),
            np.log(model.eps_grid[-1])-coordinates[1],
        ])
        return np.asarray(constraints)

    drawn_chi2 = np.sum(
        ((curves-model.data['xi_centered'])@model.precision_root.T)**2, axis=1,
    )
    starts = np.vstack((
        best_retained, draws[np.argsort(drawn_chi2)[:6]],
        draws[rng.choice(len(draws), 5, replace=False)],
    ))
    best_fit = best_retained.copy()
    optimisation_records = []
    for i, start in enumerate(starts):
        result = minimize(
            chi2, start, method='SLSQP', bounds=bounds,
            constraints={'type': 'ineq', 'fun': physical_constraints},
            options={'maxiter': 250, 'ftol': 1e-8},
        )
        accepted = (np.isfinite(model.prior(result.x)[0])
                    and np.min(physical_constraints(result.x)) >= -1e-9)
        record = {
            'start': i, 'solver_success': bool(result.success),
            'physical_constraints_satisfied': bool(accepted),
            'chi2': float(result.fun), 'message': str(result.message),
        }
        optimisation_records.append(record)
        print(record, flush=True)
        if accepted and result.fun < chi2(best_fit):
            best_fit = result.x.copy()

    best_curve = model.predict(best_fit)[0]
    direct_best = kernel_point_prediction(model, best_fit)
    best_point_error = float(np.max(np.abs(direct_best-best_curve)))
    if not np.isfinite(best_point_error) or best_point_error > .01:
        raise RuntimeError('Optimised diagnostic state fails kernel point check.')
    new_table = np.genfromtxt(
        OUTPUT/'ngc628_xi_observed_and_models.csv', delimiter=',', names=True,
    )
    old_table = np.genfromtxt(
        INPUT/'ngc628_xi_observed_and_models.csv', delimiter=',', names=True,
    )
    shoulder = (model.data['r_kpc'] >= .6)&(model.data['r_kpc'] <= 2.5)
    weight = shoulder.astype(float)/shoulder.sum()
    observed_mean = float(weight@model.data['xi_centered'])
    shoulder_means = curves@weight
    shoulder_sigma = float(np.sqrt(weight@model.covariance@weight))
    median_curve = new_table['xi_KTZ_median'][1:]
    old_curve = old_table['xi_KTZ_median'][1:]
    long = np.exp(draws[:, 6]) > 1
    ranks = [float(np.mean(samples[:, j] <= best_fit[j])) for j in range(7)]
    old_rmse = float(np.sqrt(np.mean(
        (old_curve[shoulder]-model.data['xi_centered'][shoulder])**2,
    )))
    new_rmse = float(np.sqrt(np.mean(
        (median_curve[shoulder]-model.data['xi_centered'][shoulder])**2,
    )))
    parameters = list(model.derived(samples)) + [samples[:, 4]]
    names = [
        'ell_star', 'epsilon', 'beta', 'eta', 'kappa', 't_star',
        'sigma', 'source_excess_amplitude', 'source_clustering_length', 'rate_log_slope',
    ]
    intervals = np.percentile(np.column_stack(parameters), [16, 50, 84], axis=0).T
    Table({
        'parameter': names, 'p16': intervals[:, 0],
        'median': intervals[:, 1], 'p84': intervals[:, 2],
    }).write(OUTPUT/'ngc628_transport_posterior.csv', format='ascii.csv', overwrite=True)
    diagnostic = {
        'scope': 'NGC628 only; unchanged data, source template, prior and seven fitted coordinates',
        'posterior_curve_draws_checked': len(curves),
        'posterior_kernel_point_checks': len(checks),
        'posterior_kernel_point_check_max_error': maximum_point_error,
        'optimised_state_kernel_point_check_max_error': best_point_error,
        'kernel_check_limitations': 'rate_log_slope interpolation, full-window radial representation and local-source approximation held fixed',
        'prior_max_absolute_difference_from_original': float(np.max(np.abs(prior-original_prior))),
        'highest_likelihood_retained_chain_chi2': best_chi2,
        'multistart_constrained_likelihood_chi2': chi2(best_fit),
        'optimisation_records': optimisation_records,
        'likelihood_state_is_diagnostic_not_posterior_summary': True,
        'optimised_internal_coordinates': best_fit.tolist(),
        'optimised_coordinate_posterior_percentile_ranks': ranks,
        'optimised_derived_parameters': dict(zip(
            names, [float(v[0]) for v in model.derived(best_fit)]+[float(best_fit[4])],
        )),
        'observed_shoulder_mean': observed_mean,
        'old_pointwise_median_curve_shoulder_mean': float(weight@old_curve),
        'corrected_pointwise_median_curve_shoulder_mean': float(weight@median_curve),
        'posterior_of_shoulder_mean_p16_median_p84': np.percentile(shoulder_means, [16, 50, 84]).tolist(),
        'shoulder_covariance_standard_error': shoulder_sigma,
        'median_curve_shoulder_discrepancy_in_covariance_sigma': float((observed_mean-weight@median_curve)/shoulder_sigma),
        'maximum_line_flux_mc_fraction_of_diagonal_shoulder_variance': float(np.max(
            model.data['line_flux_mc_error'][shoulder]**2
            /model.data['total_error'][shoulder]**2,
        )),
        'optimised_rate_log_slope_distance_from_halpha_prior_mean_in_sigma': float(
            (best_fit[4]-model.metadata['event_rate_log_slope_kpc_inv'])
            /model.metadata['event_rate_log_slope_error_kpc_inv'],
        ),
        'optimised_curve_shoulder_mean': float(weight@best_curve),
        'optimised_curve_shoulder_RMSE': float(np.sqrt(np.mean((best_curve[shoulder]-model.data['xi_centered'][shoulder])**2))),
        'old_shoulder_RMSE': old_rmse, 'corrected_shoulder_RMSE': new_rmse,
        'shoulder_RMSE_fractional_reduction': float(1-new_rmse/old_rmse),
        'posterior_probability_source_length_above_1_kpc': float(np.mean(np.exp(samples[:, 6]) > 1)),
        'conditional_shoulder_mean_given_source_length_above_1_kpc_p16_median_p84': np.percentile(shoulder_means[long], [16, 50, 84]).tolist(),
        'shoulder_definition_is_exploratory': True,
        'posterior_interval_is_mean_curve_not_replicated_observation': True,
        'all_galaxy_manuscript_results_replaced': False,
    }
    (OUTPUT/'ngc628_window_refit_diagnostics.json').write_text(
        json.dumps(diagnostic, indent=2)+'\n',
    )
    Table({
        'r_kpc': np.r_[0, model.data['r_kpc']],
        'xi_constrained_likelihood_diagnostic': np.r_[1, best_curve],
        'xi_direct_kernel_diagnostic': np.r_[1, direct_best],
    }).write(OUTPUT/'ngc628_likelihood_shape_diagnostic.csv', format='ascii.csv', overwrite=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax in axes:
        ax.fill_between(
            new_table['r_kpc'], new_table['xi_KTZ_p16'], new_table['xi_KTZ_p84'],
            color='#7b3294', alpha=.16, label='complete KTZ posterior 16–84%',
        )
        ax.plot(new_table['r_kpc'], new_table['xi_KTZ_median'],
                color='#7b3294', label='complete KTZ posterior median')
        ax.plot(np.r_[0, model.data['r_kpc']], np.r_[1, best_curve],
                color='.35', ls='--', lw=1.3,
                label='constrained likelihood: shape diagnostic only')
        ax.plot(new_table['r_kpc'], new_table['xi_KT18_median'],
                color='#1f77b4', label='KT18 posterior median')
        ax.errorbar(model.data['r_kpc'], model.data['xi_centered'],
                    yerr=model.data['total_error'], fmt='o', ms=2,
                    color='black', ecolor='.65', lw=.7, label='PHANGS-MUSE')
        ax.axhline(0, color='.6', lw=.6)
        ax.set_xlabel('separation r [kpc]')
        ax.set_ylabel(r'$\xi_{3,\mathrm{pp}}$')
    axes[0].set(xlim=(0, 5), ylim=(-.2, 1.04), title='NGC 628: median versus likelihood support')
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set(xlim=(.5, 2.6), ylim=(-.04, .25), title='Exploratory shoulder shape diagnostic')
    fig.savefig(OUTPUT/'ngc628_likelihood_shape_diagnostic.png', dpi=200)
    plt.close(fig)
    print(json.dumps({k:v for k,v in diagnostic.items() if k != 'optimisation_records'}, indent=2), flush=True)
    return diagnostic


if __name__ == '__main__':
    audit_completed_refit()
