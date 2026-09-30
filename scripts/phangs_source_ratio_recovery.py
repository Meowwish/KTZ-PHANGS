"""Covariance-level recovery of the random finite-window pair normalisation.

This small-map Gaussian experiment uses the corrected source/beam operators.
It is NOT native-flux injection/recovery, a physical clustered-event simulator,
SBC of a fitted posterior, or a significance test for a galaxy.
"""

import contextlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfftn

from scripts.phangs_source_integral import SourceIntegral, combine_moments
from scripts.test_phangs_source_integral import small_window


def run_ratio_recovery(output_dir, realisations=4000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window = small_window()
    axis = np.arange(-17, 18)*.04
    y, x = np.meshgrid(axis, axis, indexing='ij')
    model = SourceIntegral(window, axis, .35*np.cos(11*x)+.2*np.sin(7*y), .15,
                           np.array([[.04**2, .0002], [.0002, .042**2]]),
                           truncate=7, age_nodes=20)
    ell, eps, rate_log_slope, length, beta, scale, betaA = .055, .6, .25, .08, .9, .7, .25
    with contextlib.redirect_stdout(io.StringIO()):
        result = model.moments(ell, eps, rate_log_slope, length, return_operators=True)
    rate = (result['rate0']+scale*result['rate1']).ravel()
    connected = np.zeros((window.n, window.n))
    for w, G in result['age_operators']:
        connected += 4*np.pi*(.5*ell**2*(1-eps**2))*w/model.area*(G.multiply(rate)@G.T).toarray()
    T = result['transfer'].toarray()
    padded, spectrum = model.source_excess_spectrum(length)
    kernel = irfftn(spectrum, s=padded)
    iy, ix = np.indices(model.shape)
    iy, ix = iy.ravel(), ix.ravel()
    E = kernel[(iy[:, None]-iy) % padded[0], (ix[:, None]-ix) % padded[1]]
    W = T*result['rate0'].ravel()
    connected += betaA*(W@E@W.T)
    mean = np.sqrt(beta)*(T@rate)
    P = np.eye(window.n)
    for a in range(window.na):
        selected = np.flatnonzero(window.ann == a)
        P[np.ix_(selected, selected)] -= 1/len(selected)
    values, vectors = np.linalg.eigh(P@connected@P.T)
    root = vectors*np.sqrt(np.maximum(values, 0))
    rng = np.random.default_rng(628041)
    fields = rng.normal(size=(realisations, window.n))@root.T+(P@mean)
    _, bins, selected = window.block_geometry(0, window.n)
    numerators, denominators = np.empty((2, realisations, window.nb))
    for b in range(window.nb):
        i, j = np.nonzero(selected&(bins == b))
        numerators[:, b] = np.mean(fields[:, i]*fields[:, j], axis=1)
        denominators[:, b] = np.mean((fields[:, i]**2+fields[:, j]**2)/2, axis=1)
    ratios = numerators/denominators
    reference, _ = combine_moments(result, beta, scale, betaA)
    mean_ratio, mc_error = ratios.mean(axis=0), ratios.std(axis=0, ddof=1)/np.sqrt(realisations)
    ratio_expected = numerators.mean(axis=0)/denominators.mean(axis=0)
    summary = {
        'status': 'completed covariance-level small-map experiment',
        'realisations': realisations, 'observed_pixels': window.n,
        'ordered_pair_counts': window.counts.astype(int).tolist(),
        'mean_random_ratio': mean_ratio.tolist(),
        'ratio_of_expected_moments': reference.tolist(),
        'empirical_ratio_of_averaged_moments': ratio_expected.tolist(),
        'Monte_Carlo_standard_error_of_mean_ratio': mc_error.tolist(),
        'normalisation_bias_mean_ratio_minus_ratio_of_moments': (mean_ratio-reference).tolist(),
        'scope': 'Gaussian covariance-level recovery on a fixed irregular small mask; no pair sampling. Not a correction inferred for NGC628.',
        'remaining': 'Native multi-line PSF/calibration/selection injection-recovery and independent posterior calibration are not performed by this experiment.',
    }
    (output_dir/'ngc628_small_map_ratio_recovery.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savez_compressed(output_dir/'ngc628_small_map_ratio_recovery.npz',
                        ratios=ratios, numerator=numerators, denominator=denominators)
    radius = .5*(window.edges[1:]+window.edges[:-1])
    lo, med, hi = np.percentile(ratios, [16, 50, 84], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
    axes[0].fill_between(radius, lo, hi, color='0.7', alpha=.4, label='single-map 16–84% (not posterior)')
    axes[0].plot(radius, reference, color='#7b3294', label='ratio of expected moments')
    axes[0].errorbar(radius, mean_ratio, yerr=mc_error, fmt='o', color='k', label='mean exhaustive mock estimator')
    axes[0].plot(radius, med, color='0.4', ls=':', label='single-map estimator median')
    axes[0].set(xlabel='separation [kpc]', ylabel='pair statistic', title='Covariance-level normalisation recovery')
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].errorbar(radius, mean_ratio-reference, yerr=mc_error, fmt='o-', color='#7b3294')
    axes[1].axhline(0, color='0.5', lw=.8)
    axes[1].set(xlabel='separation [kpc]', ylabel=r'$E[N/D]-E[N]/E[D]$', title='Finite-map random-ratio bias')
    fig.savefig(output_dir/'ngc628_small_map_ratio_recovery.png', dpi=180)
    plt.close(fig)
    return summary
