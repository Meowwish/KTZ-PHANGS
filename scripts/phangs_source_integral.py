"""Global, discretised source integrals for the frozen-pattern KTZ model.

No midpoint-rate replacement is made. Gaussian source operators include the
full (possibly elliptical) observing-beam covariance. The isotropic zeroth-
order distinct-event term uses the axisymmetric source rate only; it never
inherits the arm template. Projection and pair normalisation follow integration.

This is a reference integrator, not a fast MCMC emulator. Its source-grid,
age-quadrature, Gaussian truncation and source-boundary errors must be checked.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len, rfftn, irfftn
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix

from scripts.phangs_window_projection import annular_project, mean_pair_moments


def beam_covariance_face_on(affine, sky_fwhm_kpc):
    """The affine maps row vectors in sky-kpc to face-on-kpc coordinates."""
    linear = np.asarray(affine)[:2]
    return (sky_fwhm_kpc/2.354820045)**2*(linear.T@linear)


def age_quadrature(epsilon, nodes=32):
    """Uniform physical ages, integrated with Gauss nodes in log width."""
    if not 0 < epsilon < 1 or nodes < 2:
        raise ValueError('Require 0 < epsilon < 1 and at least two age nodes.')
    x, w = np.polynomial.legendre.leggauss(nodes)
    log_min = 2*np.log(epsilon)
    log_u = .5*((0-log_min)*x+log_min)
    u = np.exp(log_u)
    weight = .5*(-log_min)*w*u/(1-epsilon**2)
    return u, weight


def gaussian_kernel(covariance, step, truncate=8):
    covariance = np.asarray(covariance)
    ry, rx = np.ceil(truncate*np.sqrt(np.diag(covariance)[::-1])/step).astype(int)
    yy, xx = np.meshgrid(np.arange(-ry, ry+1)*step,
                         np.arange(-rx, rx+1)*step, indexing='ij')
    inv = np.linalg.inv(covariance)
    exponent = inv[0, 0]*xx**2+2*inv[0, 1]*xx*yy+inv[1, 1]*yy**2
    return np.exp(-exponent/2)/(2*np.pi*np.sqrt(np.linalg.det(covariance)))


class PairWindow:
    """Full ordered-pair geometry and exact annular projection contractions."""

    def __init__(self, x_grid, y_grid, valid, edges, annulus_width, block=96):
        self.x_grid, self.y_grid = np.asarray(x_grid), np.asarray(y_grid)
        self.valid, self.edges = np.asarray(valid, bool), np.asarray(edges)
        self.step = float(x_grid[0, 1]-x_grid[0, 0])
        self.iy, self.ix = np.nonzero(self.valid)
        self.x, self.y = x_grid[self.valid], y_grid[self.valid]
        self.radius = np.hypot(x_grid, y_grid)
        _, self.ann = np.unique(np.floor(self.radius[self.valid]/annulus_width),
                                return_inverse=True)
        self.n, self.na, self.nb = len(self.x), int(self.ann.max()+1), len(edges)-1
        self.annulus_width = annulus_width
        self.ann_counts = np.bincount(self.ann, minlength=self.na)
        self.averager = csr_matrix((1/self.ann_counts[self.ann],
                                   (np.arange(self.n), self.ann)),
                                  shape=(self.n, self.na))
        partner = np.zeros((self.n, self.na, self.nb), dtype=np.float32)
        for start in range(0, self.n, block):
            stop = min(start+block, self.n)
            distance, bins, selected = self.block_geometry(start, stop)
            rows = np.arange(stop-start)[:, None]*self.na+self.ann[None]
            partner[start:stop] = np.bincount(
                (rows*self.nb+bins)[selected],
                minlength=(stop-start)*self.na*self.nb,
            ).reshape(stop-start, self.na, self.nb)
        self.partner = partner.reshape(self.n*self.na, self.nb).T
        self.counts = partner.sum(axis=(0, 1), dtype=np.float64)
        self.endpoint_weight = partner.sum(axis=1, dtype=np.float64).T
        self.ann_pairs = np.asarray([partner[self.ann == a].sum(axis=0, dtype=np.float64)
                                     for a in range(self.na)])
        if np.any(self.counts == 0):
            raise ValueError('Empty positive-separation bin; use a narrower bin range.')

    def block_geometry(self, start, stop):
        # Use integer-grid lags everywhere. Subtracting floating coordinates
        # would put identical lattice separations on opposite sides of a bin
        # edge in the direct-pair and lag-reduction implementations.
        distance = self.step*np.hypot(self.ix[start:stop, None]-self.ix,
                                     self.iy[start:stop, None]-self.iy)
        bins = np.searchsorted(self.edges, distance, side='right')-1
        selected = (distance > 0)&(bins >= 0)&(bins < self.nb)
        return distance, bins, selected

    def project_moments(self, raw_numerator, row_mean, diagonal):
        ann_mean = np.asarray([row_mean[self.ann == a].mean(axis=0)
                               for a in range(self.na)])
        numerator = (raw_numerator-2*self.partner@row_mean.ravel()
                     +self.ann_pairs.reshape(self.na*self.na, self.nb).T@ann_mean.ravel())
        diag_projected = diagonal-2*row_mean[np.arange(self.n), self.ann]+ann_mean[self.ann, self.ann]
        denominator = self.endpoint_weight@diag_projected
        return np.asarray([numerator, denominator])/self.counts

    def dense_moments(self, covariance):
        """Independent small-map verifier only: explicit P C P^T."""
        P = np.eye(self.n)
        for a in range(self.na):
            indices = np.flatnonzero(self.ann == a)
            P[np.ix_(indices, indices)] -= 1/len(indices)
        C = P@covariance@P.T
        _, bins, selected = self.block_geometry(0, self.n)
        n = np.bincount(bins[selected], weights=C[selected], minlength=self.nb)
        d = np.bincount(bins[selected], weights=np.broadcast_to(np.diag(C)[:, None], C.shape)[selected],
                        minlength=self.nb)
        return np.asarray([n, d])/self.counts


class SourceIntegral:
    def __init__(self, window, source_axis, source_modulation, reference_radius,
                 beam_covariance, truncate=6, age_nodes=32, progress_path=None):
        self.window = window
        self.axis = np.asarray(source_axis)
        self.step = float(self.axis[1]-self.axis[0])
        self.area = self.step**2
        self.modulation = np.asarray(source_modulation)
        if not np.all(np.isfinite(self.modulation)):
            raise ValueError('Source completion must be explicit; source modulation contains NaN/Inf.')
        self.shape = (len(self.axis), len(self.axis))
        if self.modulation.shape != self.shape:
            raise ValueError('Source grid/modulation shapes disagree.')
        sy, sx = np.meshgrid(self.axis, self.axis, indexing='ij')
        self.radius = np.hypot(sx, sy)
        self.reference_radius = reference_radius
        self.beam = np.asarray(beam_covariance)
        if np.linalg.eigvalsh(self.beam).min() <= 0:
            raise ValueError('Beam covariance must be positive definite.')
        self.truncate, self.age_nodes = truncate, age_nodes
        self.progress_path = Path(progress_path) if progress_path else None
        self.obs_sy = np.rint((window.y-self.axis[0])/self.step).astype(int)
        self.obs_sx = np.rint((window.x-self.axis[0])/self.step).astype(int)
        np.testing.assert_allclose(self.axis[self.obs_sy], window.y, atol=1e-9)
        np.testing.assert_allclose(self.axis[self.obs_sx], window.x, atol=1e-9)
        self.ratio = round(window.step/self.step)
        if not np.isclose(window.step/self.step, self.ratio):
            raise ValueError('Observation/source grid spacings must have integer ratio.')
        self.pad_y = round((window.y_grid[0, 0]-self.axis[0])/self.step)
        self.pad_x = round((window.x_grid[0, 0]-self.axis[0])/self.step)

    def progress(self, stage, **values):
        record = {'stage': stage, **values}
        if self.progress_path:
            self.progress_path.write_text(json.dumps(record, indent=2)+'\n')
        print(record, flush=True)

    def gaussian_operator(self, covariance):
        ry, rx = np.ceil(self.truncate*np.sqrt(np.diag(covariance)[::-1])/self.step).astype(int)
        oy, ox = np.meshgrid(np.arange(-ry, ry+1), np.arange(-rx, rx+1), indexing='ij')
        inv = np.linalg.inv(covariance)
        exponent = self.area*(inv[0, 0]*ox**2+2*inv[0, 1]*ox*oy+inv[1, 1]*oy**2)
        use = exponent <= self.truncate**2
        oy, ox, exponent = oy[use], ox[use], exponent[use]
        ny, nx = self.shape
        if (self.obs_sy.min() < ry or self.obs_sy.max()+ry >= ny
                or self.obs_sx.min() < rx or self.obs_sx.max()+rx >= nx):
            raise ValueError('Insufficient source padding for this transport state.')
        columns = ((self.obs_sy[:, None]+oy)*nx+self.obs_sx[:, None]+ox).astype(np.int32)
        weights = self.area*np.exp(-exponent/2)/(2*np.pi*np.sqrt(np.linalg.det(covariance)))
        indptr = np.arange(self.window.n+1, dtype=np.int64)*len(weights)
        return csr_matrix((np.tile(weights, self.window.n), columns.ravel(), indptr),
                          shape=(self.window.n, ny*nx))

    def raw_overlap_numerator(self, source, covariance):
        """Gaussian product identity; every fitted lag and valid pair included."""
        half = np.zeros((2*self.shape[0]-1, 2*self.shape[1]-1))
        half[::2, ::2] = source*self.area
        smoothed = fftconvolve(half, gaussian_kernel(covariance/2, self.step/2), mode='same')
        inv = np.linalg.inv(covariance)
        prefactor = 1/(4*np.pi*np.sqrt(np.linalg.det(covariance)))
        W, h = self.window.valid, self.window.step
        height, width = W.shape
        raw = np.zeros(self.window.nb)
        limit = min(self.window.edges[-1],
                    2*self.truncate*np.sqrt(np.linalg.eigvalsh(covariance).max()))
        nlag = int(np.ceil(limit/h))
        for dy in range(0, min(height-1, nlag)+1):
            for dx in range(-min(width-1, nlag), min(width-1, nlag)+1):
                if dy == 0 and dx <= 0:
                    continue
                r = h*np.hypot(dx, dy)
                b = np.searchsorted(self.window.edges, r, side='right')-1
                if b < 0 or b >= self.window.nb or r > limit:
                    continue
                exponent = h*h*(inv[0, 0]*dx*dx+2*inv[0, 1]*dx*dy+inv[1, 1]*dy*dy)/4
                if exponent > self.truncate**2:
                    continue
                y0, y1 = 0, height-dy
                x0, x1 = max(0, -dx), min(width, width-dx)
                mask = W[y0:y1, x0:x1]&W[y0+dy:y1+dy, x0+dx:x1+dx]
                if not mask.any():
                    continue
                sy = 2*self.pad_y+self.ratio*(2*y0+dy)
                sx = 2*self.pad_x+self.ratio*(2*x0+dx)
                step = 2*self.ratio
                values = smoothed[sy:sy+step*(y1-y0):step, sx:sx+step*(x1-x0):step]
                raw[b] += 2*prefactor*np.exp(-exponent)*values[mask].sum()
        return raw

    def source_excess_spectrum(self, length):
        # At least eight correlation lengths to every nearest periodic image.
        padded = tuple(next_fast_len(max(2*n-1, n+int(np.ceil(8*length/self.step))))
                       for n in self.shape)
        ky = 2*np.pi*np.fft.fftfreq(padded[0], d=self.step)
        kx = 2*np.pi*np.fft.rfftfreq(padded[1], d=self.step)
        spectrum = 2*np.pi*length**2*np.exp(-.5*length**2*(ky[:, None]**2+kx[None, :]**2))/self.area
        return padded, spectrum

    def distinct_event_moments(self, transfer, rate, length, block=8):
        """Nonlocal T D_lambda E D_lambda T^T; complete pixel-pair reduction."""
        W = transfer.multiply(rate.ravel()[None]).tocsr()
        padded, spectrum = self.source_excess_spectrum(length)
        raw = np.zeros(self.window.nb)
        rows = np.empty((self.window.n, self.window.na))
        diagonal = np.empty(self.window.n)
        for start in range(0, self.window.n, block):
            stop = min(start+block, self.window.n)
            values = W[start:stop].toarray().reshape(stop-start, *self.shape)
            transformed = rfftn(values, s=padded, axes=(-2, -1), workers=1)
            smoothed = irfftn(transformed*spectrum, s=padded, axes=(-2, -1), workers=1)
            covariance = (W@smoothed[:, :self.shape[0], :self.shape[1]].reshape(stop-start, -1).T).T
            diagonal[start:stop] = covariance[np.arange(stop-start), np.arange(start, stop)]
            rows[start:stop] = covariance@self.window.averager
            _, bins, selected = self.window.block_geometry(start, stop)
            raw += np.bincount(bins[selected], weights=covariance[selected], minlength=self.window.nb)
            if start == 0 or stop == self.window.n or start//block % 100 == 0:
                self.progress('nonlocal distinct-event integral', completed_pixels=stop,
                              total_pixels=self.window.n, source_length_kpc=float(length))
        return self.window.project_moments(raw, rows, diagonal)

    def moments(self, ell, epsilon, rate_log_slope, source_length, return_operators=False):
        if not np.isfinite([ell, epsilon, rate_log_slope, source_length]).all() or ell <= 0 or source_length <= 0:
            raise ValueError('Non-finite or non-positive transport/source length.')
        start = time.time()
        rate0 = np.exp(rate_log_slope*(self.radius-self.reference_radius))
        rate1 = rate0*self.modulation
        rates = [rate0, rate1]
        width, weights = age_quadrature(epsilon, self.age_nodes)
        extent = .5*ell**2*(1-epsilon**2)
        rows = np.zeros((2, self.window.n, self.window.na))
        diagonal = np.zeros((2, self.window.n))
        raw = np.zeros((2, self.window.nb))
        transfer = csr_matrix((self.window.n, np.prod(self.shape)))
        kept = []
        for i, (u, w) in enumerate(zip(width, weights)):
            covariance = np.eye(2)*ell**2*u+self.beam
            G = self.gaussian_operator(covariance)
            transfer = transfer+w*G
            column_annulus = G.T@self.window.averager
            squared = G.multiply(G)
            for j, source in enumerate(rates):
                factor = 4*np.pi*extent*w/self.area
                rows[j] += factor*(G@column_annulus.multiply(source.ravel()[:, None])).toarray()
                diagonal[j] += factor*(squared@source.ravel())
                raw[j] += 4*np.pi*extent*w*self.raw_overlap_numerator(source, covariance)
            if return_operators:
                kept.append((w, G))
            self.progress('same-event source integral', completed_age_nodes=i+1,
                          total_age_nodes=self.age_nodes, ell_star_kpc=float(ell))
        cp = np.asarray([self.window.project_moments(raw[j], rows[j], diagonal[j])
                         for j in range(2)])
        means = []
        for source in rates:
            field = np.zeros(self.window.valid.shape)
            field[self.window.valid] = transfer@source.ravel()
            means.append(annular_project(field, self.window.valid,
                                         self.window.radius, self.window.annulus_width))
        n0, d0, _ = mean_pair_moments(means[0], self.window.valid, self.window.step, self.window.edges)
        n1, d1, _ = mean_pair_moments(means[1], self.window.valid, self.window.step, self.window.edges)
        nt, dt, _ = mean_pair_moments(means[0]+means[1], self.window.valid, self.window.step, self.window.edges)
        mean = np.asarray([[n0, d0], [nt-n0-n1, dt-d0-d1], [n1, d1]])
        # Do not retain the final large sparse age operator during the
        # different-event FFT contractions; the transfer already contains it.
        del G, column_annulus, squared
        distinct = self.distinct_event_moments(transfer, rate0, source_length)
        result = {'CP': cp, 'mean': mean, 'CNP': distinct,
                  'elapsed_seconds': time.time()-start,
                  'source_step_kpc': self.step, 'age_nodes': self.age_nodes,
                  'minimum_source_modulation': float(self.modulation.min()),
                  'source_boundary': 'axisymmetric exponential extrapolation outside measured modulation support',
                  'CNP_scope': 'nonlocal radial-rate numerical extension of the isotropic zeroth-order source covariance; no arm coupling'}
        if return_operators:
            result.update(transfer=transfer, age_operators=kept, rate0=rate0, rate1=rate1)
        self.progress('source-integral point completed', elapsed_seconds=result['elapsed_seconds'])
        return result


def combine_moments(moments, beta, eta, betaA, noise_fraction=1, noise_bias=0):
    if not np.isfinite([beta, eta, betaA]).all() or min(beta, eta, betaA) < 0:
        raise ValueError('Physical covariance amplitudes must be finite and non-negative.')
    if 1+eta*moments.get('minimum_source_modulation', 0.) <= 0:
        raise ValueError('Non-positive physical source rate for the chosen arm scale.')
    cp = moments['CP']
    mean = moments['mean']
    terms = np.asarray([
        cp[0], eta*cp[1],
        beta*(mean[0]+eta*mean[1]+eta**2*mean[2]),
        betaA*moments['CNP'],
    ])
    denominator = terms[:, 1].sum(axis=0)
    if np.any(denominator <= 0):
        raise ValueError('Non-positive total projected second moment.')
    components = noise_fraction*terms[:, 0]/denominator
    return components.sum(axis=0)+noise_bias, components
