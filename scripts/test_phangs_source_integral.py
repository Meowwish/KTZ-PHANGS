"""Independent dense and analytic checks for the global KTZ reference integral."""

import contextlib
import io
import json

import numpy as np
from scipy.fft import irfftn

from scripts.phangs_source_integral import (
    PairWindow, SourceIntegral, age_quadrature, beam_covariance_face_on,
    combine_moments,
)
from scripts.phangs_template_ktz import gaussian_source_cnp_kernel


def quiet_moments(model, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return model.moments(*args, **kwargs)


def small_window(step=.04):
    axis = np.arange(-3, 4)*step
    y, x = np.meshgrid(axis, axis, indexing='ij')
    valid = np.ones(x.shape, bool)
    valid[1, 2] = valid[4, 5] = valid[5, 1] = False
    return PairWindow(x, y, valid, [.02, .08, .14, .20, .26, .34], .08, block=7)


def test_dense():
    window = small_window()
    axis = np.arange(-17, 18)*.04
    y, x = np.meshgrid(axis, axis, indexing='ij')
    modulation = .35*np.cos(11*x)+.2*np.sin(7*y)
    ell, eps, rate_log_slope, length = .055, .6, .25, .08
    beam = np.array([[.04**2, .0002], [.0002, .042**2]])
    model = SourceIntegral(window, axis, modulation, .15, beam, truncate=7, age_nodes=20)
    result = quiet_moments(model, ell, eps, rate_log_slope, length, return_operators=True)
    extent = .5*ell**2*(1-eps**2)
    covariances = []
    for source in [result['rate0'], result['rate1']]:
        C = np.zeros((window.n, window.n))
        for w, G in result['age_operators']:
            C += 4*np.pi*extent*w/model.area*(G.multiply(source.ravel())@G.T).toarray()
        covariances.append(C)
    errors = {}
    for i, C in enumerate(covariances):
        expected = window.dense_moments(C)
        errors['dense_CP_'+str(i)] = float(np.max(np.abs(result['CP'][i]-expected)))
        # Gaussian-product numerator and sparse operator use identical source
        # samples; their only difference is the seven-sigma operator cutoff.
        np.testing.assert_allclose(result['CP'][i], expected, rtol=2e-8, atol=3e-10)
    T = result['transfer'].toarray()
    field0, field1 = T@result['rate0'].ravel(), T@result['rate1'].ravel()
    mean_cov = [np.outer(field0, field0),
                np.outer(field0, field1)+np.outer(field1, field0),
                np.outer(field1, field1)]
    for i, C in enumerate(mean_cov):
        expected = window.dense_moments(C)
        errors['dense_mean_'+str(i)] = float(np.max(np.abs(result['mean'][i]-expected)))
        np.testing.assert_allclose(result['mean'][i], expected, atol=2e-12)
    padded, spectrum = model.source_excess_spectrum(length)
    kernel = irfftn(spectrum, s=padded)
    iy, ix = np.indices(model.shape)
    iy, ix = iy.ravel(), ix.ravel()
    E = kernel[(iy[:, None]-iy) % padded[0], (ix[:, None]-ix) % padded[1]]
    W = T*result['rate0'].ravel()
    CNP = W@E@W.T
    expected = window.dense_moments(CNP)
    errors['dense_CNP'] = float(np.max(np.abs(result['CNP']-expected)))
    np.testing.assert_allclose(result['CNP'], expected, atol=2e-12)
    beta, scale, betaA = .9, .7, .25
    assert (result['rate0']+scale*result['rate1']).min() > 0
    C = covariances[0]+scale*covariances[1]+beta*np.outer(field0+scale*field1, field0+scale*field1)+betaA*CNP
    assert np.linalg.eigvalsh(C).min() > -1e-12
    expected = window.dense_moments(C)
    curve, components = combine_moments(result, beta, scale, betaA, .91, .001)
    np.testing.assert_allclose(curve, .91*expected[0]/expected[1]+.001, atol=2e-8)
    np.testing.assert_allclose(curve, components.sum(axis=0)+.001, atol=1e-14)
    return errors


def test_analytic():
    window = small_window()
    axis = np.arange(-38, 39)*.02
    y, x = np.meshgrid(axis, axis, indexing='ij')
    ell, eps, length, k = .07, .4, .12, 28.
    sigma_beam = .035
    model = SourceIntegral(window, axis, np.cos(k*x), .15,
                           np.eye(2)*sigma_beam**2, truncate=8, age_nodes=32)
    u, weights = age_quadrature(eps, 32)
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-13)
    T = np.zeros((window.n, len(axis)**2))
    cp = np.zeros((window.n, window.n))
    harmonic = cp.copy()
    for width, w in zip(u, weights):
        G = model.gaussian_operator(np.eye(2)*(ell**2*width+sigma_beam**2))
        T += w*G.toarray()
        cp += 4*np.pi*(.5*ell**2*(1-eps**2))*w/model.area*(G@G.T).toarray()
        harmonic += 4*np.pi*(.5*ell**2*(1-eps**2))*w/model.area*(G.multiply(np.cos(k*x).ravel())@G.T).toarray()
    expected_mean = np.sum(weights*np.exp(-.5*k*k*(ell**2*u+sigma_beam**2)))*np.cos(k*window.x)
    np.testing.assert_allclose(T@np.cos(k*x).ravel(), expected_mean, atol=1e-12)
    distance = np.hypot(window.x[:, None]-window.x, window.y[:, None]-window.y)
    v = ell**2*u+sigma_beam**2
    uniform = np.sum((.5*ell**2*(1-eps**2)*weights/v)[None, None, :]*
                     np.exp(-distance[:, :, None]**2/(4*v)), axis=2)
    attenuated = (np.sum((.5*ell**2*(1-eps**2)*weights/v)[None, None, :]*
                        np.exp(-distance[:, :, None]**2/(4*v)-k*k*v/4), axis=2)*
                        np.cos(k*(window.x[:, None]+window.x)/2))
    np.testing.assert_allclose(cp, uniform, atol=2e-10)
    np.testing.assert_allclose(harmonic, attenuated, atol=2e-10)
    # This is a non-small-q test: q=k ell=1.96. The harmonic must not be
    # replaced by its midpoint amplitude times the uniform overlap kernel.
    midpoint = uniform*np.cos(k*(window.x[:, None]+window.x)/2)
    missing_attenuation = float(np.max(np.abs(midpoint-attenuated)))
    assert missing_attenuation > .02
    padded, spectrum = model.source_excess_spectrum(length)
    transformed = np.fft.rfftn(T.reshape(window.n, *model.shape), s=padded, axes=(-2, -1))
    convolved = np.fft.irfftn(transformed*spectrum, s=padded, axes=(-2, -1))
    cnp = T@convolved[:, :len(axis), :len(axis)].reshape(window.n, -1).T
    observed_ell = np.sqrt(ell**2+sigma_beam**2)
    observed_eps = np.sqrt((ell*eps)**2+sigma_beam**2)/observed_ell
    analytic = gaussian_source_cnp_kernel(distance.ravel(), observed_ell, observed_eps,
                                         length, quadrature_nodes=64).reshape(distance.shape)
    np.testing.assert_allclose(cnp, analytic, atol=2e-10)
    affine = np.array([[1., 0.], [0., 1.5], [0., 0.]])
    np.testing.assert_allclose(beam_covariance_face_on(affine, .12),
                               (.12/2.354820045)**2*np.diag([1., 2.25]))
    return {'uniform_overlap_max_error': float(np.max(np.abs(cp-uniform))),
            'harmonic_overlap_max_error': float(np.max(np.abs(harmonic-attenuated))),
            'mean_transport_max_error': float(np.max(np.abs(T@np.cos(k*x).ravel()-expected_mean))),
            'CNP_uniform_analytic_max_error': float(np.max(np.abs(cnp-analytic))),
            'omitted_harmonic_attenuation_max_error': missing_attenuation,
            'harmonic_q': k*ell}


def run_tests():
    result = {'dense': test_dense(), 'analytic': test_analytic(),
              'quadrature_and_grid': test_convergence(),
              'scope': 'discrete global source-integral algebra and analytic continuum limits; not native-flux recovery or MCMC calibration'}
    print(json.dumps(result, indent=2))
    return result


def test_convergence():
    # Use NGC628-like physical widths, without claiming this small map tests
    # the real source-completion assumptions or the entire fitted parameter box.
    window = small_window(.06)
    curves = []
    for step, nodes, pad in [(.06, 16, 1.56), (.06, 32, 1.56),
                              (.03, 32, 1.56), (.03, 32, 1.74)]:
        half = round(pad/step)
        axis = np.arange(-half, half+1)*step
        y, x = np.meshgrid(axis, axis, indexing='ij')
        model = SourceIntegral(window, axis, .35*np.cos(8*x)+.15*np.sin(5*y),
                               .15, np.eye(2)*(.12/2.354820045)**2,
                               truncate=7, age_nodes=nodes)
        result = quiet_moments(model, .154, .123, .03, .82)
        curves.append(combine_moments(result, 1., .8, .18)[0])
    error = {'age_16_vs_32_curve_max_error': float(np.max(np.abs(curves[0]-curves[1]))),
             'source_60_vs_30_pc_curve_max_error': float(np.max(np.abs(curves[1]-curves[2]))),
             'padding_1p56_vs_1p74_curve_max_error': float(np.max(np.abs(curves[2]-curves[3]))),
             'scope': 'one NGC628-like point on a small map; not full-prior convergence'}
    assert error['age_16_vs_32_curve_max_error'] < 1e-5
    assert error['source_60_vs_30_pc_curve_max_error'] < .005
    assert error['padding_1p56_vs_1p74_curve_max_error'] < 1e-6
    return error


if __name__ == '__main__':
    run_tests()
