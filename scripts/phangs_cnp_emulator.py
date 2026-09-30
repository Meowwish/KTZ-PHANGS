"""Positive-moment numerical representation of ONE nonlocal CNP.

Training targets are source-integral outputs, never the observed metallicity
correlation. Shape and log projected second moment are represented separately
but reconstructed as one (N,D) response before combining physical components.
Interpolation settings are selected by numerical cross-validation only.
"""

import numpy as np
from scipy.interpolate import RBFInterpolator

from scripts.build_ngc628_cnp_pilot import LOW, HIGH, physical_shape


def features(coordinates, kind='original', sigma_weight=1.):
    x = np.atleast_2d(coordinates)
    if kind == 'original':
        result = (x-LOW)/(HIGH-LOW)
    elif kind == 'variance':
        kt, sigma = np.exp(x[:, 0]), np.exp(x[:, 1])
        variance = sigma*sigma+kt
        lo, hi = np.log(.015**2+.0025), np.log(.25**2+4.5)
        result = np.column_stack(((np.log(variance)-lo)/(hi-lo),
            sigma*sigma/variance, (x[:, 2]+.35)/.7,
            (x[:, 3]-LOW[3])/(HIGH[3]-LOW[3])))
    else:
        raise ValueError('Unknown numerical feature coordinates.')
    result = result.copy()
    result[:, 1] *= sigma_weight
    return result


def mixture_error_bound(actual, approximate):
    """Bound error for ANY positive allocation to the other physical terms.

    For a valid pooled pair statistic |xi_other|<=1. If h=D_fast/D_true,
    max_a |a_fast-a|=tanh(|log h|/4). Thus the returned binwise bound includes
    amplitude allocation errors, not just the normalised CNP shape. It is not
    an extra uncertainty/noise floor added to an astronomical likelihood.
    """
    actual, approximate = np.asarray(actual), np.asarray(approximate)
    if np.any(actual[..., 1, :] <= 0) or np.any(approximate[..., 1, :] <= 0):
        raise ValueError('Projected second moments must be positive.')
    shape_error = abs(actual[..., 0, :]/actual[..., 1, :]-
                      approximate[..., 0, :]/approximate[..., 1, :])
    log_ratio = abs(np.log(approximate[..., 1, :]/actual[..., 1, :]))
    return shape_error+2*np.tanh(log_ratio/4)


class PositiveCNPEmulator:
    """The control plus its deterministic integral correction is ONE CNP.

    An optional rate_log_slope factor imposes the homogeneous continuum control at rate_log_slope=0.
    It is only a candidate numerical representation: direct rate_log_slope=0 checks are
    required because discrete kernels/control interpolation need not match
    exactly. No division by zero, clipping or restriction of physical priors
    is used. The unfactored candidate does not impose this condition.
    """
    def __init__(self, coordinates, truth, baseline, control, reference_radius,
                 kind='original', sigma_weight=1., kernel='cubic', rate_slope_factor=False):
        x = np.asarray(coordinates)
        truth, baseline = np.asarray(truth), np.asarray(baseline)
        if not np.isfinite(truth).all() or not np.isfinite(baseline).all():
            raise ValueError('Non-finite training source moments.')
        if np.any(truth[:, 1] <= 0) or np.any(baseline[:, 1] <= 0):
            raise ValueError('Training projected second moments must be positive.')
        target = np.stack((truth[:, 0]/truth[:, 1]-baseline[:, 0]/baseline[:, 1],
                           np.log(truth[:, 1]/baseline[:, 1])), axis=1)
        self.control, self.reference_radius = control, reference_radius
        self.settings = {'kind': kind, 'sigma_weight': float(sigma_weight),
                         'kernel': kernel, 'rate_slope_factor': bool(rate_slope_factor)}
        selected = np.ones(len(x), dtype=bool)
        if rate_slope_factor:
            selected = abs(x[:, 2]) > 1e-6
            target = target[selected]/x[selected, 2, None, None]
        else:
            target = target[selected]
        self.nb = truth.shape[-1]
        self.interpolator = RBFInterpolator(features(x[selected], kind, sigma_weight),
            target.reshape(selected.sum(), -1), kernel=kernel,
            degree=2 if kernel == 'quintic' else 1, smoothing=1e-10)

    def baseline(self, coordinates):
        result = []
        for x in np.atleast_2d(coordinates):
            kt, sigma, rate_log_slope, length, _, _ = physical_shape(x)
            result.append(self.control.moments(kt, sigma, rate_log_slope, length, self.reference_radius))
        return np.asarray(result)

    def predict(self, coordinates, baseline=None):
        x = np.atleast_2d(coordinates)
        if np.any(x < LOW-1e-12) or np.any(x > HIGH+1e-12):
            raise ValueError('Outside numerical domain; never clip a physical prior.')
        baseline = self.baseline(x) if baseline is None else np.asarray(baseline)
        if np.any(baseline[:, 1] <= 0):
            raise ValueError('Control projected second moment is non-positive.')
        correction = self.interpolator(features(x, self.settings['kind'],
            self.settings['sigma_weight'])).reshape(len(x), 2, self.nb)
        if self.settings['rate_slope_factor']:
            correction *= x[:, 2, None, None]
        shape = baseline[:, 0]/baseline[:, 1]+correction[:, 0]
        with np.errstate(over='ignore', invalid='ignore'):
            denominator = baseline[:, 1]*np.exp(correction[:, 1])
        result = np.stack((shape*denominator, denominator), axis=1)
        if not np.isfinite(result).all() or np.any(denominator <= 0):
            raise ValueError('Numerical interpolation overflow/invalid variance; do not clip.')
        return result


def select_numerical_settings(coordinates, truth, baseline, control, reference_radius,
                              seed=6281904, folds=5):
    """Fixed five-fold protocol; no fresh validation or observed data enter."""
    x, truth, baseline = map(np.asarray, (coordinates, truth, baseline))
    groups = np.array_split(np.random.default_rng(seed).permutation(len(x)), folds)
    scores = []
    for kind in ('original', 'variance'):
        for sigma_weight in (.25, 1.):
            for kernel in ('thin_plate_spline', 'cubic', 'quintic'):
                for rate_slope_factor in (False, True):
                    settings = dict(kind=kind, sigma_weight=sigma_weight,
                                    kernel=kernel, rate_slope_factor=rate_slope_factor)
                    errors, failure = [], None
                    for test in groups:
                        train = np.setdiff1d(np.arange(len(x)), test)
                        try:
                            model = PositiveCNPEmulator(x[train], truth[train], baseline[train],
                                control, reference_radius, **settings)
                            prediction = model.predict(x[test], baseline[test])
                            errors.extend(mixture_error_bound(truth[test], prediction).max(axis=1).tolist())
                        except (ValueError, np.linalg.LinAlgError) as exc:
                            failure = str(exc)
                            break
                    scores.append({'settings': settings,
                        'maximum_cross_validation_bound': max(errors) if failure is None else None,
                        'mean_cross_validation_bound': float(np.mean(errors)) if failure is None else None,
                        'failure': failure})
    valid = [s for s in scores if s['failure'] is None]
    if not valid:
        raise RuntimeError('No candidate numerical representation passed algebra/positivity checks.')
    best = min(valid, key=lambda s: (s['maximum_cross_validation_bound'], s['mean_cross_validation_bound']))
    return best['settings'], scores


def test_mixture_bound():
    """Independent random allocations, including positive/negative shapes."""
    rng = np.random.default_rng(6281905)
    n = 10000
    shape_true, shape_fast, shape_other = rng.uniform(-1, 1, (3, n))
    d_true, d_fast = np.exp(rng.uniform(-6, 6, (2, n)))
    a = rng.uniform(0, 1, n)
    h = d_fast/d_true
    a_fast = a*h/(1-a+a*h)
    actual = np.stack((d_true*shape_true, d_true), axis=0)[None]
    approximate = np.stack((d_fast*shape_fast, d_fast), axis=0)[None]
    bound = mixture_error_bound(actual, approximate)[0]
    error = abs((1-a_fast)*shape_other+a_fast*shape_fast-
                ((1-a)*shape_other+a*shape_true))
    if np.any(error > bound+1e-12):
        raise AssertionError('Mixture-allocation bound is violated.')
    # The allocation part attains its analytic maximum at a=1/(1+sqrt(h)).
    optimum = 1/(1+np.sqrt(h))
    shifted = optimum*h/(1-optimum+optimum*h)
    np.testing.assert_allclose(abs(shifted-optimum), np.tanh(abs(np.log(h))/4), atol=1e-14)
    return {'status': 'positive-allocation error bound verified', 'random_allocations': n,
            'maximum_violation': float(np.max(error-bound)),
            'scope': 'Numerical bound only; no observed fit or physical validation.'}
