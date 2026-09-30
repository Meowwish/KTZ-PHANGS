"""Memory-bounded convolution implementation of the SAME global source integrals.

The source operators are translation-invariant Gaussian convolutions; only
their source weights and observing window are inhomogeneous. No source-rate
commutation, midpoint approximation, stationary normalised-curve transfer,
pixel/pair subsampling or additional covariance term is introduced.
"""

import time

import numpy as np
from scipy.fft import rfftn, irfftn
from scipy.signal import fftconvolve

from scripts.phangs_source_integral import SourceIntegral, age_quadrature, gaussian_kernel
from scripts.phangs_window_projection import annular_project, mean_pair_moments


def convolve_same(values, kernel):
    if values.ndim == 3:
        kernel = kernel[None]
    return fftconvolve(values, kernel, mode='same', axes=(-2, -1))


class FFTSourceIntegral(SourceIntegral):
    def __init__(self, *args, distinct_block=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.distinct_block = distinct_block
        self.annulus_maps = np.zeros((self.window.na, *self.shape))
        self.annulus_maps[self.window.ann, self.obs_sy, self.obs_sx] = 1/self.window.ann_counts[self.window.ann]

    def kernel(self, covariance):
        kernel = gaussian_kernel(covariance, self.step, self.truncate)*self.area
        ry, rx = np.asarray(kernel.shape)//2
        yy, xx = np.meshgrid(np.arange(-ry, ry+1)*self.step,
                             np.arange(-rx, rx+1)*self.step, indexing='ij')
        inv = np.linalg.inv(covariance)
        exponent = inv[0, 0]*xx**2+2*inv[0, 1]*xx*yy+inv[1, 1]*yy**2
        kernel[exponent > self.truncate**2] = 0
        if (self.obs_sy.min() < ry or self.obs_sy.max()+ry >= self.shape[0]
                or self.obs_sx.min() < rx or self.obs_sx.max()+rx >= self.shape[1]):
            raise ValueError('Insufficient padded source grid; do not shrink the physical prior.')
        return kernel

    def sample(self, fields):
        return fields[..., self.obs_sy, self.obs_sx]

    def distinct_from_kernel(self, transfer_kernel, rate, length):
        """T D_rate E D_rate T^T via two source convolutions per endpoint row."""
        padded, spectrum = self.source_excess_spectrum(length)
        ry, rx = np.asarray(transfer_kernel.shape)//2
        raw = np.zeros(self.window.nb)
        row_mean = np.empty((self.window.n, self.window.na))
        diagonal = np.empty(self.window.n)
        for start in range(0, self.window.n, self.distinct_block):
            stop = min(start+self.distinct_block, self.window.n)
            # W_j(z)=T(x_j-z) lambda0(z); all injection cells in the kernel
            # support are included. Only endpoint rows are chunked, not sampled.
            W = np.zeros((stop-start, *self.shape))
            for k, j in enumerate(range(start, stop)):
                y, x = self.obs_sy[j], self.obs_sx[j]
                W[k, y-ry:y+ry+1, x-rx:x+rx+1] = transfer_kernel*rate[y-ry:y+ry+1, x-rx:x+rx+1]
            transformed = rfftn(W, s=padded, axes=(-2, -1), workers=1)
            del W
            E_W = irfftn(transformed*spectrum, s=padded, axes=(-2, -1), workers=1)
            del transformed
            E_W = E_W[:, :self.shape[0], :self.shape[1]]*rate
            # The second T is a convolution too, not a huge sparse matrix.
            covariance = self.sample(convolve_same(E_W, transfer_kernel))
            del E_W
            diagonal[start:stop] = covariance[np.arange(stop-start), np.arange(start, stop)]
            row_mean[start:stop] = covariance@self.window.averager
            _, bins, selected = self.window.block_geometry(start, stop)
            raw += np.bincount(bins[selected], weights=covariance[selected], minlength=self.window.nb)
            if start == 0 or stop == self.window.n or start//self.distinct_block % 200 == 0:
                self.progress('FFT nonlocal distinct-event integral', completed_pixels=stop,
                    total_pixels=self.window.n, source_length_kpc=float(length))
        return self.window.project_moments(raw, row_mean, diagonal)

    def same_event_and_mean(self, ell, epsilon, rate_log_slope, return_kernel=False):
        u, weights = age_quadrature(epsilon, self.age_nodes)
        extent = .5*ell*ell*(1-epsilon*epsilon)
        rate0 = np.exp(rate_log_slope*(self.radius-self.reference_radius))
        rate1 = rate0*self.modulation
        rates = [rate0, rate1]
        rows = np.zeros((2, self.window.n, self.window.na))
        diagonal = np.zeros((2, self.window.n))
        raw = np.zeros((2, self.window.nb))
        # Allocate the largest required kernel once. Smaller age kernels are
        # inserted without changing their discrete Gaussian truncation.
        transfer = np.zeros_like(self.kernel(np.eye(2)*ell**2*u.max()+self.beam))
        cy, cx = np.asarray(transfer.shape)//2
        for i, (width, weight) in enumerate(zip(u, weights)):
            covariance = np.eye(2)*ell*ell*width+self.beam
            kernel = self.kernel(covariance)
            ry, rx = np.asarray(kernel.shape)//2
            transfer[cy-ry:cy+ry+1, cx-rx:cx+rx+1] += weight*kernel
            source_annulus = convolve_same(self.annulus_maps, kernel)
            factor = 4*np.pi*extent*weight/self.area
            for j, rate in enumerate(rates):
                rows[j] += factor*self.sample(convolve_same(source_annulus*rate, kernel)).T
                diagonal[j] += factor*self.sample(convolve_same(rate, kernel*kernel))
                raw[j] += 4*np.pi*extent*weight*self.raw_overlap_numerator(rate, covariance)
            del source_annulus
            self.progress('FFT same-event source integral', completed_age_nodes=i+1,
                total_age_nodes=self.age_nodes, ell_star_kpc=float(ell))
        cp = np.asarray([self.window.project_moments(raw[j], rows[j], diagonal[j]) for j in range(2)])
        means = []
        for rate in rates:
            field = np.zeros(self.window.valid.shape)
            field[self.window.valid] = self.sample(convolve_same(rate, transfer))
            means.append(annular_project(field, self.window.valid, self.window.radius, self.window.annulus_width))
        n0, d0, _ = mean_pair_moments(means[0], self.window.valid, self.window.step, self.window.edges)
        n1, d1, _ = mean_pair_moments(means[1], self.window.valid, self.window.step, self.window.edges)
        nt, dt, _ = mean_pair_moments(means[0]+means[1], self.window.valid, self.window.step, self.window.edges)
        mean = np.asarray([[n0, d0], [nt-n0-n1, dt-d0-d1], [n1, d1]])
        result = {'CP': cp, 'mean': mean}
        if return_kernel:
            result.update(transfer_kernel=transfer, rate0=rate0, rate1=rate1)
        return result

    def moments(self, ell, epsilon, rate_log_slope, source_length, return_operators=False):
        if not np.isfinite([ell, epsilon, rate_log_slope, source_length]).all() or ell <= 0 or source_length <= 0:
            raise ValueError('Non-finite or non-positive physical lengths.')
        start = time.time()
        result = self.same_event_and_mean(ell, epsilon, rate_log_slope, return_kernel=True)
        distinct = self.distinct_from_kernel(result['transfer_kernel'], result['rate0'], source_length)
        result.update(CNP=distinct, elapsed_seconds=time.time()-start,
            source_step_kpc=self.step, age_nodes=self.age_nodes,
            minimum_source_modulation=float(self.modulation.min()),
            source_boundary='same explicit axisymmetric extrapolation outside measured modulation support',
            CNP_scope='same nonlocal radial-rate isotropic zeroth-order source covariance, evaluated by convolution; no arm coupling')
        if not return_operators:
            for key in ('transfer_kernel', 'rate0', 'rate1'):
                result.pop(key)
        self.progress('FFT source-integral point completed', elapsed_seconds=result['elapsed_seconds'])
        return result
