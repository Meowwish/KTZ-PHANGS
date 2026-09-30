"""Independent ensemble restarts and target-function Monte Carlo diagnostics.

Walkers from one interacting ensemble are NEVER counted as independent chains
in Rhat. Instead, each Rhat comparison uses the same selected walker trajectory
from separately initialised, separately seeded ensembles. MCSE/ESS estimates
use each independent ensemble's time series of ensemble-average functions.
This utility does not validate any likelihood or source-response emulator.
"""

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def _basic_split_rhat(chains):
    half = chains.shape[1]//2
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = half*split.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt(((half-1)*within+between)/(half*within))


def _rank_split_rhat(chains):
    shape = chains.shape
    scores = np.empty(shape)
    folded = np.empty(shape)
    for k in range(shape[-1]):
        value = chains[:, :, k]
        for target, sample in [(scores, value), (folded, abs(value-np.median(value)))]:
            ranks = rankdata(sample.ravel(), method='average')
            target[:, :, k] = ndtri((ranks-.375)/(len(ranks)+.25)).reshape(shape[:2])
    return np.maximum(_basic_split_rhat(scores), _basic_split_rhat(folded))


def _positive_sequence_tau(values):
    """Conservative paired positive autocorrelation sequence of one series."""
    centered = values-values.mean()
    n = len(values)
    transform = np.fft.rfft(centered, n=2*n)
    acov = np.fft.irfft(transform*transform.conj(), n=2*n)[:n]/n
    if acov[0] <= 0:
        return np.inf
    rho = acov/acov[0]
    pairs = rho[1:1+2*((n-1)//2)].reshape(-1, 2).sum(axis=1)
    stopped = np.flatnonzero(pairs <= 0)
    pairs = pairs[:int(stopped[0])] if len(stopped) else pairs
    if len(pairs):
        pairs = np.minimum.accumulate(pairs)
    return max(1., float(1+2*pairs.sum()))


def independent_diagnostics(chains, target=None, target_names=None, minimum_ess=1000):
    """chains: (independent ensembles, retained steps, walkers, coordinates)."""
    chains = np.asarray(chains)
    if chains.ndim != 4 or chains.shape[0] < 4 or chains.shape[1] < 100:
        raise ValueError('Require >=4 independent ensembles and >=100 retained steps.')
    E, T, W, _ = chains.shape
    values = chains if target is None else np.asarray(target(chains.reshape(-1, chains.shape[-1]))).reshape(E, T, W, -1)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite target function.')
    selected_walkers = np.unique([0, W//4, W//2, 3*W//4])
    rhat = np.maximum.reduce([_rank_split_rhat(values[:, :, w]) for w in selected_walkers])
    series = values.mean(axis=2)
    tau = np.asarray([[_positive_sequence_tau(series[e, :, k])
                        for k in range(values.shape[-1])] for e in range(E)])
    # Var of the grand mean: sum_e Var(mean_e)/E^2. Within-ensemble
    # dependencies enter through the actual ensemble-mean time series.
    variance_means = series.var(axis=1, ddof=1)*tau/T
    mc_variance = variance_means.sum(axis=0)/E**2
    mcse = np.sqrt(mc_variance)
    ess = values.reshape(-1, values.shape[-1]).var(axis=0, ddof=1)/mc_variance
    enough_time = bool(T >= 50*np.max(tau))
    return {'independent_ensembles': E, 'retained_steps_per_ensemble': T,
        'walkers_per_ensemble': W, 'Rhat_walker_indices': selected_walkers.tolist(),
        'target_names': target_names or [f'target_{k}' for k in range(values.shape[-1])],
        'max_rank_folded_split_Rhat_across_selected_trajectories': rhat.tolist(),
        'ensemble_mean_autocorrelation_times': tau.tolist(),
        'target_function_MCSE': mcse.tolist(), 'target_function_ESS': ess.tolist(),
        'time_series_length_at_least_50_tau': enough_time,
        'numerical_sampling_gate_passed': bool(enough_time and np.max(rhat) < 1.01 and np.min(ess) >= minimum_ess),
        'scope': 'Sampling diagnostic conditional on a validated likelihood. Passing is not proof of astrophysical model correctness.'}


def sample_independent_ensembles(log_probability, starts, steps, burn, seeds):
    """Generic batched stretch updates with independent RNGs and starting sets.

    A fixed-length building block; the caller must checkpoint and extend until
    target diagnostics pass. Never adapt priors or use an old chain as output.
    """
    starts = np.asarray(starts, float)
    E, W, D = starts.shape
    if E < 4 or len(seeds) != E or len(set(seeds)) != E or W % 2 or W < 2*D:
        raise ValueError('Need >=4 unique seeds/ensembles and even W >= 2D.')
    if burn < 0 or steps <= burn:
        raise ValueError('Require steps > burn >= 0.')
    chains = np.empty((E, steps-burn, W, D))
    acceptance = np.zeros((E, W))
    for e, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        positions = starts[e].copy()
        density = np.asarray(log_probability(positions))
        if not np.isfinite(density).all():
            raise ValueError('All dispersed starting positions must have finite probability.')
        for step in range(steps):
            for parity in (0, 1):
                active, other = np.arange(parity, W, 2), np.arange(1-parity, W, 2)
                partners = positions[rng.choice(other, len(active))]
                stretch = (1+rng.random(len(active)))**2/2
                proposed = partners+stretch[:, None]*(positions[active]-partners)
                probability = np.asarray(log_probability(proposed))
                keep = np.log(rng.random(len(active))) < (D-1)*np.log(stretch)+probability-density[active]
                positions[active[keep]], density[active[keep]] = proposed[keep], probability[keep]
                acceptance[e, active] += keep
            if step >= burn:
                chains[e, step-burn] = positions
    return chains, acceptance/steps


def test_gaussian_inference():
    rng = np.random.default_rng(628101)
    starts = rng.normal(size=(4, 16, 2))+np.array([-2., -1., 1., 2.])[:, None, None]
    chains, accepted = sample_independent_ensembles(lambda x: -.5*np.sum(x*x, axis=1),
        starts, steps=20000, burn=2000, seeds=[901, 902, 903, 904])
    targets = lambda x: np.column_stack([x, np.sum(x*x, axis=1)])
    diagnostics = independent_diagnostics(chains, targets, ['x0', 'x1', 'squared_radius'])
    means = targets(chains.reshape(-1, 2)).mean(axis=0)
    expected = np.array([0., 0., 2.])
    assert np.all(abs(means-expected) < 6*np.asarray(diagnostics['target_function_MCSE']))
    assert accepted.min() > .4 and accepted.max() < .9
    assert diagnostics['numerical_sampling_gate_passed']
    shifted = chains.copy()
    shifted[0, :, :, 0] += 3
    bad = independent_diagnostics(shifted)
    assert not bad['numerical_sampling_gate_passed']
    diagnostics.update(gaussian_target_means=means.tolist(), expected_means=expected.tolist(),
                       shifted_restart_rejected=True,
                       scope='Closed-form Gaussian sampler test, NOT a new galaxy fit or SBC.')
    return diagnostics
