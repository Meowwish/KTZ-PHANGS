"""Convolution-vs-matrix algebra checks including unresolved source lengths."""

import contextlib
import io
import json

import numpy as np

from scripts.phangs_source_integral import SourceIntegral, combine_moments
from scripts.phangs_source_fft import FFTSourceIntegral
from scripts.test_phangs_source_integral import small_window


def run_fft_tests():
    window = small_window()
    axis = np.arange(-17, 18)*.04
    y, x = np.meshgrid(axis, axis, indexing='ij')
    g = .35*np.cos(11*x)+.2*np.sin(7*y)
    beam = np.array([[.04**2, .0002], [.0002, .042**2]])
    args = (window, axis, g, .15, beam)
    sparse = SourceIntegral(*args, truncate=7, age_nodes=20)
    fft = FFTSourceIntegral(*args, truncate=7, age_nodes=20, distinct_block=3)
    checks = []
    for rate_log_slope, length in [(.25, .08), (-.3, .005), (.35, .8)]:
        with contextlib.redirect_stdout(io.StringIO()):
            a = sparse.moments(.055, .6, rate_log_slope, length)
            b = fft.moments(.055, .6, rate_log_slope, length)
        error = {k: float(np.max(np.abs(a[k]-b[k]))) for k in ('CP', 'mean', 'CNP')}
        for k in error:
            np.testing.assert_allclose(a[k], b[k], atol=2e-12)
        pa, _ = combine_moments(a, .9, .7, .25)
        pb, _ = combine_moments(b, .9, .7, .25)
        np.testing.assert_allclose(pa, pb, atol=3e-11)
        error.update(rate_log_slope=rate_log_slope, source_length_kpc=length, curve_error=float(np.max(abs(pa-pb))))
        checks.append(error)
    result = {'status': 'convolution algebra verified against original discrete source operators',
        'checks': checks, 'scope': 'Independent irregular small-map exact discrete checks; not full-prior emulator validation.'}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_fft_tests()
