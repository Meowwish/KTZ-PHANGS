"""Algebra tests; not a source-response precision or galaxy-fit claim."""

import numpy as np

from scripts.build_ngc628_cnp_pilot import shape_points, physical_shape
from scripts.phangs_cnp_emulator import (PositiveCNPEmulator, mixture_error_bound,
                                        test_mixture_bound, features)


class AnalyticControl:
    def moments(self, kt, sigma, rate_log_slope, length, reference_radius):
        lag = np.linspace(.05, 3., 11)
        denominator = np.exp(.1*rate_log_slope)*(1+.05*lag)
        shape = .4*np.exp(-lag/(.5+length))-.02
        return np.stack((denominator*shape, denominator))


def run_tests():
    control = AnalyticControl()
    training = shape_points(48, 6281981)
    tests = shape_points(12, 6281982)
    test_zero = tests[0].copy()
    test_zero[2] = 0.
    tests = np.vstack((tests, test_zero))
    baseline = np.asarray([control.moments(*physical_shape(x)[:4], 0.) for x in training])
    actual_test_baseline = np.asarray([control.moments(*physical_shape(x)[:4], 0.) for x in tests])
    cases = []
    for rate_slope_factor in (False, True):
        for kind in ('original', 'variance'):
            def target(x, base):
                f = features(x, kind, .25)
                delta_shape = (.012+.02*f[:, 0]-.01*f[:, 3])[:, None]*np.ones((1, 11))
                log_amplitude = (.03-.02*f[:, 1]+.04*f[:, 2])[:, None]*np.ones((1, 11))
                if rate_slope_factor:
                    delta_shape *= x[:, 2, None]
                    log_amplitude *= x[:, 2, None]
                denominator = base[:, 1]*np.exp(log_amplitude)
                return np.stack((denominator*(base[:, 0]/base[:, 1]+delta_shape), denominator), axis=1)
            truth = target(training, baseline)
            model = PositiveCNPEmulator(training, truth, baseline, control, 0.,
                kind=kind, sigma_weight=.25, kernel='cubic', rate_slope_factor=rate_slope_factor)
            prediction = model.predict(tests)
            expected = target(tests, actual_test_baseline)
            np.testing.assert_allclose(prediction, expected, atol=2e-10)
            if rate_slope_factor:
                np.testing.assert_array_equal(prediction[-1], actual_test_baseline[-1])
            if np.any(prediction[:, 1] <= 0):
                raise AssertionError('Positive-moment reconstruction failed.')
            cases.append({'features': kind, 'rate_slope_factor': rate_slope_factor,
                'maximum_reconstruction_error': float(np.max(abs(prediction-expected)))})
    # Deliberately identical shapes but different second moments: the gate
    # must expose the hidden allocation error instead of reporting zero.
    shape = np.array([.6, -.08])
    actual = np.stack((shape, np.ones(2)))[None]
    approximate = 2*actual
    bound = mixture_error_bound(actual, approximate)
    np.testing.assert_allclose(bound, 2*np.tanh(np.log(2)/4))
    bad = tests.copy()
    bad[0, 2] = .4
    try:
        model.predict(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Out-of-domain radial rate was silently accepted/clipped.')
    return {'status': 'CNP encoding/zero-slope/domain/hidden-amplitude algebra verified',
        'reconstruction_cases': cases, 'mixture_bound': test_mixture_bound(),
        'scope': 'Synthetic algebra only; fresh real-source precision checks remain required.'}


if __name__ == '__main__':
    import json
    print(json.dumps(run_tests(), indent=2))
