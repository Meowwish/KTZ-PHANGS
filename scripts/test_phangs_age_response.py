"""Verify reusable midpoint contractions against dense single-age covariance."""

import numpy as np

from scripts.phangs_age_response import SingleAgeResponse
from scripts.phangs_source_integral import SourceIntegral
from scripts.phangs_window_projection import annular_project
from scripts.test_phangs_source_integral import small_window


def run_age_response_tests():
    window = small_window()
    axis = np.arange(-17, 18)*.04
    y, x = np.meshgrid(axis, axis, indexing='ij')
    g = .35*np.cos(11*x)+.2*np.sin(7*y)
    beam = np.array([[.04**2, .0002], [.0002, .042**2]])
    args = (window, axis, g, .15, beam)
    model = SingleAgeResponse(*args, truncate=7)
    original = SourceIntegral(*args, truncate=7)
    slopes = [-.35, 0., .27]
    variance = .055**2
    cp, fields = model.single_age(variance, slopes)
    G = original.gaussian_operator(np.eye(2)*variance+beam)
    errors = []
    for e, rate_log_slope in enumerate(slopes):
        rate0 = np.exp(rate_log_slope*(model.radius-.15))
        for c, rate in enumerate([rate0, rate0*g]):
            C = (G.multiply(rate.ravel())@G.T).toarray()/model.area
            expected = window.dense_moments(C)
            np.testing.assert_allclose(cp[e, c], expected, atol=2e-8, rtol=2e-9)
            field = np.zeros(window.valid.shape)
            field[window.valid] = G@rate.ravel()
            expected_field = annular_project(field, window.valid, window.radius,
                                              window.annulus_width)[window.valid]
            np.testing.assert_allclose(fields[e, c], expected_field, atol=2e-12)
            errors.append(float(np.max(abs(cp[e, c]-expected))))
    return {'status': 'single-age full-pair responses match direct source matrices',
            'max_absolute_density_response_errors': errors,
            'scope': 'single-age algebra only; no CNP emulator or galaxy fit supplied by this library'}
