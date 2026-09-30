"""Analytic stationary-CNP control for a nonlocal source-response surrogate.

Endpoint-rate weighting is a CONTROL/REGRESSION BASELINE, NOT the nonlocal
physical covariance. Any eventual emulator must learn its deterministic
difference from T D_lambda E D_lambda T^T using source-integral calculations,
not galaxy data. No extra astrophysical amplitude or fitted term is introduced.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve


def gaussian_endpoint_responses(window, intrinsic_variance, beam, slopes, annulus_block=4):
    """Exact full-window P D_f G D_f P^T Gaussian control contractions."""
    covariance = np.eye(2)*intrinsic_variance+2*np.asarray(beam)
    dy, dx = np.meshgrid(np.arange(1-window.valid.shape[0], window.valid.shape[0]),
                         np.arange(1-window.valid.shape[1], window.valid.shape[1]), indexing='ij')
    y, x = window.step*dy, window.step*dx
    inv = np.linalg.inv(covariance)
    kernel = np.exp(-.5*(inv[0, 0]*x*x+2*inv[0, 1]*x*y+inv[1, 1]*y*y))/(2*np.pi*np.sqrt(np.linalg.det(covariance)))
    distance = window.step*np.hypot(dx, dy)
    bins = np.searchsorted(window.edges, distance, side='right')-1
    selected = (distance > 0)&(bins >= 0)&(bins < window.nb)
    annulus_maps = np.zeros((window.na, *window.valid.shape))
    annulus_maps[window.ann, window.iy, window.ix] = 1/window.ann_counts[window.ann]
    # Arbitrary zero here is only a reference for numerical scaling. Supplying
    # Rref by normalising f later preserves the represented physical rate.
    output = []
    reference_radius = float(np.median(window.radius[window.valid]))
    for rate_log_slope in slopes:
        f = np.exp(rate_log_slope*(window.radius-reference_radius))
        footprint = np.where(window.valid, f, 0)
        autocorrelation = fftconvolve(footprint, footprint[::-1, ::-1], mode='full')
        raw = np.bincount(bins[selected], weights=(autocorrelation*kernel)[selected], minlength=window.nb)
        row_mean = np.empty((window.n, window.na))
        for first in range(0, window.na, annulus_block):
            last = min(first+annulus_block, window.na)
            values = fftconvolve(annulus_maps[first:last]*f, kernel[None],
                                 mode='same', axes=(-2, -1))
            row_mean[:, first:last] = f[window.valid, None]*values[:, window.valid].T
        diagonal = f[window.valid]**2/(2*np.pi*np.sqrt(np.linalg.det(covariance)))
        output.append(window.project_moments(raw, row_mean, diagonal))
    return np.asarray(output), reference_radius


class StationaryCNPControl:
    """Cheap known baseline; cannot replace nonlocal CNP in a likelihood."""
    def __init__(self, variance_grid, slopes, responses, reference_radius):
        self.variances, self.slopes = np.asarray(variance_grid), np.asarray(slopes)
        self.responses = np.asarray(responses)
        self.reference_radius = reference_radius
        spline = CubicSpline(self.variances, self.responses, axis=0)
        self.I0 = spline.antiderivative()
        # Multiply the local polynomial by v=(v-v_left)+v_left before
        # integrating; coefficient orders run from cubic to constant.
        c = spline.c
        vc = np.zeros((5, *c.shape[1:]))
        vc[:4] += c
        vc[1:] += c*self.variances[:-1, None, None, None]
        from scipy.interpolate import PPoly
        self.I1 = PPoly(vc, self.variances, extrapolate=False).antiderivative()

    def moments(self, kappa_t, sigma, rate_log_slope, length, physical_reference_radius):
        v0 = length*length+2*sigma*sigma
        v1, v2 = v0+2*kappa_t, v0+4*kappa_t
        if kappa_t <= 0 or v0 < self.variances[0] or v2 > self.variances[-1] or not self.slopes[0] <= rate_log_slope <= self.slopes[-1]:
            raise ValueError('Outside Gaussian-control table; do not clip physical priors.')
        lo = (self.I1(v1)-self.I1(v0))-v0*(self.I0(v1)-self.I0(v0))
        hi = v2*(self.I0(v2)-self.I0(v1))-(self.I1(v2)-self.I1(v1))
        by_slope = 2*np.pi*length*length*(lo+hi)/(4*kappa_t*kappa_t)
        result = CubicSpline(self.slopes, by_slope, axis=0)(rate_log_slope)
        return result*np.exp(2*rate_log_slope*(self.reference_radius-physical_reference_radius))


def test_gaussian_control():
    from scripts.test_phangs_source_integral import small_window
    window = small_window()
    beam = np.array([[.04**2, .0002], [.0002, .042**2]])
    slopes = [-.3, 0., .27]
    variance = .006
    response, reference = gaussian_endpoint_responses(window, variance, beam, slopes)
    dx = window.x[:, None]-window.x
    dy = window.y[:, None]-window.y
    covariance = np.eye(2)*variance+2*beam
    inv = np.linalg.inv(covariance)
    Gaussian = np.exp(-.5*(inv[0, 0]*dx*dx+2*inv[0, 1]*dx*dy+inv[1, 1]*dy*dy))/(2*np.pi*np.sqrt(np.linalg.det(covariance)))
    errors = []
    for i, rate_log_slope in enumerate(slopes):
        f = np.exp(rate_log_slope*(window.radius[window.valid]-reference))
        expected = window.dense_moments(f[:, None]*f*Gaussian)
        np.testing.assert_allclose(response[i], expected, atol=2e-12)
        errors.append(float(np.max(abs(response[i]-expected))))
    # Also test the triangular age-sum Jacobian and polynomial v-weighted
    # antiderivative independently of any Gaussian/source approximation.
    grid = np.geomspace(.0001, 2, 80)
    coeff = np.asarray([[[1., 2.], [3., 4.]]]*len(slopes))
    values = (grid[:, None, None, None]**2)*coeff
    control = StationaryCNPControl(grid, slopes, values, 0)
    kt, sigma, length = .12, .06, .2
    result = control.moments(kt, sigma, 0., length, 0.)
    v0 = length**2+2*sigma**2
    # For z=a+a', E z=1, E z^2=7/6.
    expected = 2*np.pi*length**2*(v0*v0+4*v0*kt+4*kt*kt*7/6)*coeff[1]
    np.testing.assert_allclose(result, expected, atol=2e-12)
    return {'status': 'Gaussian endpoint control and exact age-sum integral verified',
        'dense_covariance_errors': errors,
        'scope': 'This control is not the variable-rate nonlocal CNP theory; no galaxy fit uses it alone.'}
