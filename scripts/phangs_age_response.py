"""Tabulate reusable single-age source responses BEFORE any normalisation.

CP and transported mean depend on scalar intrinsic endpoint variance v and
radial slope rate_log_slope. Their age integral can be reused for all ell/epsilon values.
CNP is NOT supplied by this library: no stationary replacement is smuggled in.
Every observing pixel/pair is represented in the raw pair contractions.
"""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from scripts.phangs_source_integral import gaussian_kernel
from scripts.phangs_source_fft import FFTSourceIntegral, convolve_same
from scripts.phangs_window_projection import annular_project


def overlap_weights(window, covariance, truncate):
    """Full lattice-pair midpoint weights for Gaussian product reduction."""
    W, h = window.valid, window.step
    height, width = W.shape
    weights = np.zeros((window.nb, 2*height-1, 2*width-1))
    inv = np.linalg.inv(covariance)
    prefactor = 1/(4*np.pi*np.sqrt(np.linalg.det(covariance)))
    limit = min(window.edges[-1], 2*truncate*np.sqrt(np.linalg.eigvalsh(covariance).max()))
    nlag = int(np.ceil(limit/h))
    for dy in range(min(height-1, nlag)+1):
        for dx in range(-min(width-1, nlag), min(width-1, nlag)+1):
            if dy == 0 and dx <= 0:
                continue
            r = h*np.hypot(dx, dy)
            b = np.searchsorted(window.edges, r, side='right')-1
            if b < 0 or b >= window.nb or r > limit:
                continue
            exponent = h*h*(inv[0, 0]*dx*dx+2*inv[0, 1]*dx*dy+inv[1, 1]*dy*dy)/4
            if exponent > truncate**2:
                continue
            y0, y1 = 0, height-dy
            x0, x1 = max(0, -dx), min(width, width-dx)
            mask = W[y0:y1, x0:x1]&W[y0+dy:y1+dy, x0+dx:x1+dx]
            weights[b, dy+2*y0:dy+2*y1:2, dx+2*x0:dx+2*x1:2] += 2*prefactor*np.exp(-exponent)*mask
    return weights.reshape(window.nb, -1)


class SingleAgeResponse(FFTSourceIntegral):
    def single_age(self, variance, slopes, annulus_block=4):
        covariance = np.eye(2)*variance+self.beam
        kernel = self.kernel(covariance)
        overlap = overlap_weights(self.window, covariance, self.truncate)
        rates = np.exp(np.asarray(slopes)[:, None, None]*(self.radius-self.reference_radius))
        output = np.zeros((len(slopes), 2, 2, self.window.nb))
        fields = np.zeros((len(slopes), 2, self.window.n))
        rows = np.zeros((len(slopes), 2, self.window.n, self.window.na))
        diagonal = np.zeros((len(slopes), 2, self.window.n))
        for first in range(0, self.window.na, annulus_block):
            last = min(first+annulus_block, self.window.na)
            source_annulus = convolve_same(self.annulus_maps[first:last], kernel)
            for e, rate0 in enumerate(rates):
                for c, source in enumerate([rate0, rate0*self.modulation]):
                    rows[e, c, :, first:last] = self.sample(convolve_same(source_annulus*source, kernel)).T/self.area
            del source_annulus
        # Midpoint smoothing has one common observing-grid lattice. Computing
        # its weights once avoids repeating an exhaustive lag traversal for
        # every radial slope/source component.
        yy, xx = np.meshgrid(np.arange(2*self.window.valid.shape[0]-1),
                             np.arange(2*self.window.valid.shape[1]-1), indexing='ij')
        half_y = 2*self.pad_y+self.ratio*yy
        half_x = 2*self.pad_x+self.ratio*xx
        smoothing_kernel = gaussian_kernel(covariance/2, self.step/2)
        for e, rate0 in enumerate(rates):
            for c, source in enumerate([rate0, rate0*self.modulation]):
                half = np.zeros((2*self.shape[0]-1, 2*self.shape[1]-1))
                half[::2, ::2] = source*self.area
                smoothed = fftconvolve(half, smoothing_kernel, mode='same')
                raw = overlap@smoothed[half_y, half_x].ravel()
                diagonal[e, c] = self.sample(convolve_same(source, kernel*kernel))/self.area
                output[e, c] = self.window.project_moments(raw, rows[e, c], diagonal[e, c])
                field = np.zeros(self.window.valid.shape)
                field[self.window.valid] = self.sample(convolve_same(source, kernel))
                fields[e, c] = annular_project(field, self.window.valid, self.window.radius,
                                                self.window.annulus_width)[self.window.valid]
        return output, fields


class AgeResponseLibrary:
    def __init__(self, path):
        with np.load(path) as z:
            self.variance, self.slopes = z['variance_grid'], z['slope_grid']
            self.cp, self.fields = z['CP_density'], z['projected_mean_fields']
            self.signature = str(z['signature'])
        # Integrate the spline in PHYSICAL variance, not log variance. Nodes
        # can be logarithmically spaced; dv remains the physical age Jacobian.
        self.cp_integral = CubicSpline(self.variance, self.cp, axis=0).antiderivative()
        self.field_integral = CubicSpline(self.variance, self.fields, axis=0).antiderivative()

    def integrate(self, ell, epsilon, slope):
        lo, hi = (ell*epsilon)**2, ell*ell
        if lo < self.variance[0]*(1-1e-12) or hi > self.variance[-1]*(1+1e-12) or not self.slopes[0] <= slope <= self.slopes[-1]:
            raise ValueError('Outside the library; evaluate/refine the physical response, never silently clip a prior.')
        cp_by_slope = 2*np.pi*(self.cp_integral(hi)-self.cp_integral(lo))
        # 4 pi kappa t/(hi-lo)=2 pi. Mean has the uniform-age average instead.
        fields_by_slope = (self.field_integral(hi)-self.field_integral(lo))/(hi-lo)
        cp = CubicSpline(self.slopes, cp_by_slope, axis=0)(slope)
        fields = CubicSpline(self.slopes, fields_by_slope, axis=0)(slope)
        return cp, fields
