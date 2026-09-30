"""Exact finite-window KTZ likelihood with a per-pointing zero-point nuisance term.

This module is the second inference stage of the PHANGS--MUSE analysis.  Stage 1
(``NGC<n>_PHANGS_halpha_map_systematic.ipynb``) builds the abundance-residual pixel
map, the all-pair statistic xi_3pp(r), its regularised covariance and the fixed
source template.  This stage replaces the stationary-equivalent annular response of
stage 1 by the exact projection  P C P^T  of every KTZ component on the actual
pixel mask (P = annular-mean operator), splits every pixel pair into same-pointing
and cross-pointing subsets, adds the rank-K pointing zero-point covariance
sigma_q^2 sum_p w_p w_p^T  (w_p = P 1_p) and fits the joint data vector
[xi_all(r), D(r) = xi_same - xi_cross] with the unchanged four KTZ components.

Notation follows the theory paper: eta is the arm-profile amplitude, h_R the
event-rate scale length (code coordinate ``rate_log_slope`` = -1/h_R), beta_src the
mean-field strength, (beta_src A_src, ell_src) the optional clustering term.

Internal coordinates x = [ln ell_star, logit eps, ln beta_src, ln eta, rate_log_slope,
ln(beta_src A_src), ln ell_src, (sigma_q)].
"""
from __future__ import annotations
import json, time, hashlib
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.special import exp1, expit
from scripts.phangs_template_ktz import gaussian_source_cnp_kernel

FINE_EDGES = np.arange(0.03, 5.031, 0.06)          # stage-1 separation bins [kpc]
ANNULUS_WIDTH = 0.2                                 # annular-mean detrending [kpc]
COARSE_EDGES = np.arange(0.0, 3.01, 0.3)            # D(r) compression bins [kpc]
BEAM_FWHM_KPC = 0.12                                # stage-1 target PSF
PARAMETER_NAMES = ("ell_star", "epsilon", "beta_src", "eta", "kappa", "t_star", "sigma", "A_src", "ell_src")


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def annular_project(values, valid, labels_grid):
    """Subtract the mean in every annulus (the operator P applied to a field)."""
    result = np.where(valid, values, 0.0).astype(float)
    for label in np.unique(labels_grid[valid]):
        selected = valid & (labels_grid == label)
        result[selected] -= np.mean(result[selected])
    return result


def detrend(z, valid, radius, width=ANNULUS_WIDTH):
    out = np.where(valid, z, 0.0).astype(float); lab = np.floor(radius/width).astype(int)
    for a in np.unique(lab[valid]):
        s = valid & (lab == a); out[s] -= out[s].mean()
    out[valid] -= out[valid].mean()
    return out


def file_sha256(path, n=16):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:n]


def _grid_key(npz, *names):
    for n in names:
        if n in npz.files:
            return npz[n]
    raise KeyError(names)


# ----------------------------------------------------------------------------------
# exact window responses restricted to all / same-pointing / cross-pointing pairs
# ----------------------------------------------------------------------------------
def build_window_response(pixel_map, pointing_labels, reference_radius, selection="all",
                          rate_log_slope_grid=np.linspace(-0.35, 0.35, 5), edges=FINE_EDGES,
                          annulus_width=ANNULUS_WIDTH, block_size=32, log=print):
    """Projected numerator/denominator responses of the KTZ components for one pair subset.

    Returns a dict with ``response`` (n_slope, 3 kernel components, 2, n_bins, n_kernel_radii),
    ``mean_response`` (n_slope, 3 mean-field pieces, 2, n_bins), ``delta_response`` (2, n_bins)
    = projected pointing-offset numerator/denominator for unit sigma_q^2, and pair counts.
    ``selection`` is 'all', 'same' or 'cross'; same + cross reproduces all exactly.
    """
    pm = pixel_map
    valid = pm["valid_mask"].astype(bool); x_grid, y_grid = pm["x_grid_kpc"], pm["y_grid_kpc"]
    modulation = pm["arm_source_modulation"]; pointing_grid = np.asarray(pointing_labels)
    iy, ix = np.nonzero(valid)
    x, y = x_grid[valid], y_grid[valid]; pnt = pointing_grid[valid]
    radius = np.hypot(x_grid, y_grid)
    ann_labels_grid = np.floor(radius/annulus_width).astype(int)
    label_values, ann = np.unique(ann_labels_grid[valid], return_inverse=True)
    npix, na, nb = len(x), len(label_values), len(edges)-1
    ann_counts = np.bincount(ann, minlength=na)
    maximum_distance = np.hypot(np.ptp(x), np.ptp(y))
    kernel_radius = np.concatenate((np.arange(0.0, 0.600001, 0.01), np.arange(0.66, maximum_distance+0.12, 0.06)))
    nk = len(kernel_radius)
    half_y, half_x = np.meshgrid(np.arange(2*valid.shape[0]-1)/2, np.arange(2*valid.shape[1]-1)/2, indexing="ij")
    half_modulation = map_coordinates(modulation, [half_y, half_x], order=1, mode="nearest")
    npt = int(pointing_grid.max())+1
    W = np.stack([annular_project((pointing_grid == p).astype(float), valid, ann_labels_grid)[valid] for p in range(npt)], axis=1)
    g = np.sum(W**2, axis=1)

    def block_geometry(start, stop):
        distance = np.hypot(x[start:stop, None]-x[None, :], y[start:stop, None]-y[None, :])
        midpoint_radius = np.hypot(0.5*(x[start:stop, None]+x[None, :]), 0.5*(y[start:stop, None]+y[None, :]))
        midpoint_modulation = half_modulation[iy[start:stop, None]+iy[None, :], ix[start:stop, None]+ix[None, :]]
        lower = np.clip(np.searchsorted(kernel_radius, distance, side="right")-1, 0, nk-2)
        fraction = (distance-kernel_radius[lower])/(kernel_radius[lower+1]-kernel_radius[lower])
        output_bin = np.searchsorted(edges, distance, side="right")-1
        same = pnt[start:stop, None] == pnt[None, :]
        fitted = (distance > 0) & (output_bin >= 0) & (output_bin < nb)
        if selection == "same":
            fitted &= same
        elif selection == "cross":
            fitted &= ~same
        return midpoint_radius, midpoint_modulation, lower, fraction, output_bin, fitted

    t0 = time.time()
    partner = np.zeros((npix, na, nb), dtype=np.float64)
    delta_num = np.zeros(nb); delta_den = np.zeros(nb)
    for start in range(0, npix, block_size):
        stop = min(start+block_size, npix)
        *_, bins, selected = block_geometry(start, stop)
        row_ann = np.arange(stop-start)[:, None]*na + ann[None, :]
        indices = row_ann*nb + bins
        partner[start:stop] = np.bincount(indices[selected], minlength=(stop-start)*na*nb).reshape(stop-start, na, nb)
        ww = W[start:stop] @ W.T
        delta_num += np.bincount(bins[selected], weights=ww[selected], minlength=nb)
        gg = 0.5*(g[start:stop, None]+g[None, :])
        delta_den += np.bincount(bins[selected], weights=gg[selected], minlength=nb)
    ordered_counts = partner.sum(axis=(0, 1))
    log(f"[{selection}] pair geometry {time.time()-t0:.0f}s; ordered pairs per bin min/max {ordered_counts.min():.0f}/{ordered_counts.max():.0f}")
    partner_flat = np.ascontiguousarray(partner.reshape(npix*na, nb).T)
    endpoint_weight = partner.sum(axis=1).T
    ann_pair_counts = np.asarray([partner[ann == a].sum(axis=0) for a in range(na)])
    response = np.full((len(rate_log_slope_grid), 3, 2, nb, nk), np.nan)
    mean_response = np.full((len(rate_log_slope_grid), 3, 2, nb), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        delta_response = np.asarray([delta_num, delta_den])/ordered_counts[None, :]

    for ei, slope in enumerate(rate_log_slope_grid):
        log(f"[{selection}] rate_log_slope node {ei+1}/{len(rate_log_slope_grid)}: {slope:+.4f}  ({time.time()-t0:.0f}s)")
        rate_grid = np.exp(slope*(radius-reference_radius))
        mean0 = annular_project(rate_grid, valid, ann_labels_grid)[valid]
        mean1 = annular_project(rate_grid*modulation, valid, ann_labels_grid)[valid]
        fields = np.stack([mean0, mean1, mean0+mean1], axis=1)
        mn = np.zeros((3, nb)); md = np.zeros((3, nb))
        for component in range(3):
            row_mean = np.zeros((npix, na, nk), dtype=float)
            raw_numerator = np.zeros((nb, nk), dtype=float)
            for start in range(0, npix, block_size):
                stop = min(start+block_size, npix)
                mr, gm, lo, fraction, bins, selected = block_geometry(start, stop)
                rate = np.exp(slope*(mr-reference_radius))
                factor = rate if component == 0 else rate*gm if component == 1 else rate**2
                row_ann = np.arange(stop-start)[:, None]*na + ann[None, :]
                left_weight, right_weight = factor*(1-fraction), factor*fraction
                histogram = (np.bincount((row_ann*nk+lo).ravel(), weights=left_weight.ravel(), minlength=(stop-start)*na*nk)
                             + np.bincount((row_ann*nk+lo+1).ravel(), weights=right_weight.ravel(), minlength=(stop-start)*na*nk))
                row_mean[start:stop] = histogram.reshape(stop-start, na, nk)/ann_counts[None, :, None]
                raw_numerator += (np.bincount((bins*nk+lo)[selected], weights=left_weight[selected], minlength=nb*nk)
                                  + np.bincount((bins*nk+lo+1)[selected], weights=right_weight[selected], minlength=nb*nk)).reshape(nb, nk)
                if component == 0:
                    for c in range(3):
                        f = fields[:, c]
                        prod = f[start:stop, None]*f[None, :]; mom = 0.5*(f[start:stop, None]**2+f[None, :]**2)
                        mn[c] += np.bincount(bins[selected], weights=prod[selected], minlength=nb)
                        md[c] += np.bincount(bins[selected], weights=mom[selected], minlength=nb)
            ann_mean = np.asarray([row_mean[ann == a].mean(axis=0) for a in range(na)])
            numerator = (raw_numerator - 2*(partner_flat @ row_mean.reshape(npix*na, nk))
                         + ann_pair_counts.reshape(na*na, nb).T @ ann_mean.reshape(na*na, nk))
            diagonal = -2*row_mean[np.arange(npix), ann] + ann_mean[ann, ann]
            factor_diag = rate_grid[valid]
            if component == 1:
                factor_diag = factor_diag*modulation[valid]
            elif component == 2:
                factor_diag = factor_diag**2
            diagonal[:, 0] += factor_diag
            denominator = endpoint_weight @ diagonal
            with np.errstate(invalid="ignore", divide="ignore"):
                response[ei, component] = np.asarray([numerator, denominator])/ordered_counts[None, :, None]
            del row_mean
        with np.errstate(invalid="ignore", divide="ignore"):
            n0, d0 = mn[0]/ordered_counts, md[0]/ordered_counts
            n1, d1 = mn[1]/ordered_counts, md[1]/ordered_counts
            nt, dt = mn[2]/ordered_counts, md[2]/ordered_counts
        mean_response[ei] = np.asarray([[n0, d0], [nt-n0-n1, dt-d0-d1], [n1, d1]])
    return dict(kernel_radius=kernel_radius, rate_log_slope_grid=np.asarray(rate_log_slope_grid), response=response,
                mean_response=mean_response, ordered_pair_counts=ordered_counts, delta_response=delta_response,
                selection=np.array(selection), n_pointings=np.array(npt))


# ----------------------------------------------------------------------------------
# observed same / cross statistics, D(r) and the 48-block jackknife covariance
# ----------------------------------------------------------------------------------
class PairStatistic:
    """FFT pair sums of a detrended field on the face-on grid (same estimator as stage 1)."""

    def __init__(self, valid, pixel=0.06, edges=FINE_EDGES):
        self.valid = valid; h, w = valid.shape
        self.edges = edges; self.nb = len(edges)-1; self.r = 0.5*(edges[:-1]+edges[1:])
        ly, lx = np.meshgrid(np.arange(1-h, h)*pixel, np.arange(1-w, w)*pixel, indexing="ij")
        lag = np.hypot(lx, ly).ravel(); lb = np.searchsorted(edges, lag, side="right")-1
        self.ok = (lb >= 0) & (lb < self.nb) & (lag > 0); self.b = lb[self.ok]

    def sums(self, z, mask):
        m = mask.astype(float); f = np.where(mask, z, 0.0); p = f**2
        cnt = fftconvolve(m, m[::-1, ::-1], mode="full").ravel()[self.ok]
        num = fftconvolve(f, f[::-1, ::-1], mode="full").ravel()[self.ok]
        end = 0.5*(fftconvolve(p, m[::-1, ::-1], mode="full")+fftconvolve(m, p[::-1, ::-1], mode="full")).ravel()[self.ok]
        bc = lambda w_: np.bincount(self.b, weights=w_, minlength=self.nb)
        return bc(num), bc(end), bc(cnt)

    def split(self, z, mask, labels):
        nA, eA, cA = self.sums(z, mask); nS = np.zeros(self.nb); eS = np.zeros(self.nb); cS = np.zeros(self.nb)
        for L in np.unique(labels[mask]):
            n_, e_, c_ = self.sums(z, mask & (labels == L)); nS += n_; eS += e_; cS += c_
        div = lambda a, b_: np.where(b_ > 0, a/np.where(b_ > 0, b_, 1), np.nan)
        return div(nA, eA), div(nS, eS), div(nA-nS, eA-eS), cS, cA-cS


def compress_difference(r, xs, xc, cS, cC, coarse_edges=COARSE_EDGES):
    nc = len(coarse_edges)-1; cb = np.clip(np.searchsorted(coarse_edges, r, side="right")-1, 0, nc)
    out = np.full(nc, np.nan)
    for k in range(nc):
        m = (cb == k) & np.isfinite(xs) & np.isfinite(xc) & (cS > 0) & (cC > 0)
        if m.any():
            out[k] = np.sum(xs[m]*cS[m])/np.sum(cS[m]) - np.sum(xc[m]*cC[m])/np.sum(cC[m])
    return out, cb


def split_statistics(pixel_map, pointing_labels, coarse_edges=COARSE_EDGES, n_radial=4, n_azimuth=12, log=print):
    """xi_all, xi_same, xi_cross, D(r) and the delete-one jackknife covariance of [xi_all, D]."""
    pm = pixel_map; valid0 = pm["valid_mask"].astype(bool); X, Y = pm["x_grid_kpc"], pm["y_grid_kpc"]
    resid0 = pm["residual_dex"]; labels = np.asarray(pointing_labels)
    R = np.hypot(X, Y); PHI = np.mod(np.arctan2(Y, X), 2*np.pi)
    ps = PairStatistic(valid0); nb = ps.nb

    def statistic(retained):
        valid = valid0 & retained
        z = detrend(resid0, valid, R)
        return ps.split(z, valid, labels)

    xiA, xiS, xiC, cS, cC = statistic(np.ones_like(valid0))
    D, cbin = compress_difference(ps.r, xiS, xiC, cS, cC, coarse_edges)
    rr = R[valid0]; radial_edges = np.sqrt(np.linspace(rr.min()**2, rr.max()**2+1e-9, n_radial+1))
    rb = np.clip(np.digitize(R, radial_edges)-1, 0, n_radial-1); ab = np.floor(PHI/(2*np.pi)*n_azimuth).astype(int)
    block = n_azimuth*rb+ab; blocks = np.unique(block[valid0])
    vecs = []; fine = []
    for L in blocks:
        a_, s_, c_, cs_, cc_ = statistic(block != L)
        vecs.append(np.r_[a_, compress_difference(ps.r, s_, c_, cs_, cc_, coarse_edges)[0]]); fine.append(np.c_[s_, c_])
    vecs = np.array(vecs); fine = np.array(fine)
    fac = (len(blocks)-1)/len(blocks)
    dv = np.nan_to_num(vecs-np.nanmean(vecs, 0)); cov_jk = fac*dv.T@dv
    dfine = np.nan_to_num(fine-np.nanmean(fine, 0))
    log(f"pointing split: {len(np.unique(labels[valid0]))} pointings, {len(blocks)} jackknife blocks; D = {np.round(D, 4)}")
    return dict(r=ps.r, xi_all=xiA, xi_same=xiS, xi_cross=xiC, pairs_same=cS, pairs_cross=cC, D=D, coarse_edges=coarse_edges, cbin=cbin,
                cov_jk=cov_jk, n_blocks=len(blocks), err_same_fine=np.sqrt(fac*np.sum(dfine[:, :, 0]**2, 0)),
                err_cross_fine=np.sqrt(fac*np.sum(dfine[:, :, 1]**2, 0)), D_err=np.sqrt(np.diag(cov_jk)[nb:]))


def joint_covariance(fit_covariance, cov_jk, nb, shrinkage=0.35, floor=0.01):
    """[xi_all, D] covariance: stage-1 regularised block for xi_all; jackknife-only D block with the same recipe.

    The line-flux Monte Carlo and selection-systematic pieces of the stage-1 covariance are common to the
    same- and cross-pointing subsets and cancel in D to first order."""
    nc = cov_jk.shape[0]-nb
    C = np.zeros((nb+nc, nb+nc)); C[:nb, :nb] = fit_covariance
    C[nb:, nb:] = (1-shrinkage)*cov_jk[nb:, nb:]+shrinkage*np.diag(np.diag(cov_jk[nb:, nb:]))+floor**2*np.eye(nc)
    C[:nb, nb:] = (1-shrinkage)*cov_jk[:nb, nb:]; C[nb:, :nb] = C[:nb, nb:].T
    return C


# ----------------------------------------------------------------------------------
# likelihood
# ----------------------------------------------------------------------------------
class ExactWindowKTZ:
    """All-pair exact-window KTZ likelihood (seven coordinates)."""

    def __init__(self, metadata, diagnostics, fit_covariance, response, emulator_grids, rate_log_slope_prior=None,
                 src_length_bounds=None, rate_log_slope_bound=0.35, eta_prior="flat", beam_fwhm_kpc=BEAM_FWHM_KPC):
        self.metadata = metadata
        self.data = {k: np.asarray(diagnostics[k]) for k in diagnostics.dtype.names}
        self.covariance = np.asarray(fit_covariance)
        self.precision_root = np.linalg.inv(np.linalg.cholesky(self.covariance))
        self.eta_prior_kind = eta_prior; self.beam_fwhm_kpc = beam_fwhm_kpc
        self.ell_grid, self.eps_grid, self.src_grid = (np.asarray(emulator_grids[k]) for k in ("ell_grid", "epsilon_grid", "source_length_grid"))
        refine = lambda g: np.exp(np.sort(np.r_[np.log(g), 0.5*(np.log(g[:-1])+np.log(g[1:]))]))
        self.ell_grid, self.eps_grid, self.src_grid = refine(self.ell_grid), refine(self.eps_grid), refine(self.src_grid)
        self.interp_all = self._interpolators(response)
        node, weight = np.polynomial.legendre.leggauss(32)
        rmin, rmax = metadata["analysis_rmin_kpc"], metadata["analysis_rmax_kpc"]
        self.area_radius = 0.5*((rmax-rmin)*node+rmax+rmin)
        self.area_weight = weight*self.area_radius/(rmax+rmin)
        norm = 1/(1/100-1/1e6)
        mean_mass, second_mass = norm*np.log(1e4), norm*(1e6-100)
        self.mass_factor = mean_mass**2/second_mass
        self.gamma_area = metadata["halpha_sfr_msun_per_yr"]/(np.pi*(rmax**2-rmin**2)*mean_mass)*1e6
        self.eta_prior_sigma = max(metadata["mask_contrast_amplitude_error"]/metadata["mask_contrast_amplitude"], 0.35)
        self.eta_positive_limit = metadata["maximum_positive_eta"]
        self.slope_mu = metadata["event_rate_log_slope_kpc_inv"] if rate_log_slope_prior is None else rate_log_slope_prior[0]
        self.slope_sd = metadata["event_rate_log_slope_error_kpc_inv"] if rate_log_slope_prior is None else rate_log_slope_prior[1]
        self.slope_bound = rate_log_slope_bound
        self.src_lo = self.src_grid[0] if src_length_bounds is None or src_length_bounds[0] is None else max(src_length_bounds[0], self.src_grid[0])
        self.src_hi = self.src_grid[-1] if src_length_bounds is None or src_length_bounds[1] is None else min(src_length_bounds[1], self.src_grid[-1])

    # kernels on the response kernel-radius grid
    def _kernels(self, kr):
        kp = np.empty((len(self.ell_grid), len(self.eps_grid), len(kr)))
        kn = np.empty((*kp.shape[:2], len(self.src_grid), len(kr)))
        for i, ell in enumerate(self.ell_grid):
            for j, eps in enumerate(self.eps_grid):
                kp[i, j, 0] = -np.log(eps)
                kp[i, j, 1:] = 0.5*(exp1(kr[1:]**2/(4*ell**2))-exp1(kr[1:]**2/(4*(eps*ell)**2)))
                for k, length in enumerate(self.src_grid):
                    kn[i, j, k] = gaussian_source_cnp_kernel(kr, ell, eps, length, quadrature_nodes=24)
        return kp, kn

    def _interpolators(self, response):
        R = {k: np.asarray(response[k]) for k in response.files} if hasattr(response, "files") else dict(response)
        kr = R["kernel_radius"]; kp, kn = self._kernels(kr)
        resp = np.nan_to_num(R["response"])
        pgrid = np.einsum("ecabk,ijk->ijecab", resp[:, :2], kp, optimize=True)
        ngrid = np.einsum("eabk,ijlk->ijelab", resp[:, 2], kn, optimize=True)
        eg = R["rate_log_slope_grid"] if "rate_log_slope_grid" in R else R["eta_grid"]
        return dict(pinterp=RegularGridInterpolator((np.log(self.ell_grid), np.log(self.eps_grid), eg), pgrid, bounds_error=False, fill_value=np.nan),
                    ninterp=RegularGridInterpolator((np.log(self.ell_grid), np.log(self.eps_grid), eg, np.log(self.src_grid)), ngrid, bounds_error=False, fill_value=np.nan),
                    minterp=RegularGridInterpolator((eg,), np.nan_to_num(R["mean_response"]), bounds_error=False, fill_value=np.nan),
                    delta=np.nan_to_num(R["delta_response"]) if "delta_response" in R else None,
                    counts=R["ordered_pair_counts"] if "ordered_pair_counts" in R else None)

    # ---- parameters ----
    def derived(self, x):
        x = np.atleast_2d(x)
        ell, eps, beta, eta = np.exp(x[:, 0]), expit(x[:, 1]), np.exp(x[:, 2]), np.exp(x[:, 3])
        extent = 0.5*ell**2*(1-eps**2)
        rel = np.exp(x[:, 4, None]*(self.area_radius-self.metadata["r_ref_kpc"])) @ self.area_weight
        gamma = self.gamma_area/rel
        t = beta/(4*np.pi*gamma*1e3*self.mass_factor*extent)
        return ell, eps, beta, eta, extent/t, t, eps*ell, np.exp(x[:, 5]-x[:, 2]), np.exp(x[:, 6])

    def kernel_coordinates(self, x):
        ell, eps = np.exp(x[:, 0]), expit(x[:, 1])
        beam2 = 0.5*(self.beam_fwhm_kpc/2.354820045)**2
        observed_ell = np.sqrt(ell**2+beam2)
        observed_eps = np.sqrt((ell*eps)**2+beam2)/observed_ell
        return np.column_stack((np.log(observed_ell), np.log(observed_eps), x[:, 4]))

    def raw_terms(self, x, interp=None):
        """Un-normalised projected numerator/denominator per component (n, 5, 2, nbins):
        C_2p,0 ; C_2p,N ; C_M ; C_NP ; pointing term (sigma_q^2, zero without the 8th coordinate)."""
        x = np.atleast_2d(x); s = self.interp_all if interp is None else interp
        coords = self.kernel_coordinates(x)
        p, n, m = s["pinterp"](coords), s["ninterp"](np.column_stack((coords, x[:, 6]))), s["minterp"](x[:, 4, None])
        beta, eta, betaA = np.exp(x[:, 2]), np.exp(x[:, 3]), np.exp(x[:, 5])
        mean = beta[:, None, None]*(m[:, 0]+eta[:, None, None]*m[:, 1]+eta[:, None, None]**2*m[:, 2])
        q = x[:, 7]**2 if x.shape[1] > 7 else np.zeros(len(x))
        pointing = q[:, None, None]*(s["delta"][None] if s["delta"] is not None else np.zeros((1, 2, p.shape[-1])))
        return np.stack((p[:, 0], eta[:, None, None]*p[:, 1], mean, betaA[:, None, None]*n, pointing), axis=1)

    def _ratio(self, t):
        den = t[:, :, 1].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            comp = t[:, :, 0]/den[:, None, :]
        comp[np.broadcast_to((den <= 0)[:, None, :], comp.shape)] = np.nan
        return self.data["noise_signal_fraction"][None, None, :]*comp

    def components(self, x):
        return self._ratio(self.raw_terms(x))

    def predict(self, x):
        return self.components(x).sum(axis=1)+self.data["noise_additive_bias"]

    def predict_kt18(self, x):
        x = np.atleast_2d(x); dummy = np.zeros((len(x), 7)); dummy[:, :2] = x
        p = self.interp_all["pinterp"](self.kernel_coordinates(dummy))[:, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.data["noise_signal_fraction"]*p[:, 0]/p[:, 1]+self.data["noise_additive_bias"]

    # ---- priors ----
    def prior(self, x):
        x = np.atleast_2d(x)
        ell, eps, beta, eta, kappa, t, sigma, A, length = self.derived(x[:, :7])
        eta_lo = 1e-3 if self.eta_prior_kind == "flat" else 0.10
        accepted = ((ell >= self.ell_grid[0]) & (ell <= self.ell_grid[-1]) & (eps >= self.eps_grid[0]) & (eps <= self.eps_grid[-1])
                    & (beta >= 1e-4) & (beta <= 1e4) & (eta >= eta_lo) & (eta <= self.eta_positive_limit)
                    & (np.abs(x[:, 4]) <= self.slope_bound) & (kappa >= 0.05) & (kappa <= 1.50) & (t >= 0.05) & (t <= 3.00)
                    & (sigma >= 0.015) & (sigma <= 0.25) & (A >= 1e-2) & (A <= 1e3)
                    & (np.exp(x[:, 5]) >= 1e-6) & (np.exp(x[:, 5]) <= 1e7) & (length >= self.src_lo) & (length <= self.src_hi))
        # eta prior: flat in eta (Jacobian of ln eta) or the stage-1 log-normal centred on the tracer contrast
        eta_term = x[:, 3] if self.eta_prior_kind == "flat" else -0.5*(x[:, 3]/self.eta_prior_sigma)**2
        prior = np.log(2)-np.log1p(eps)+eta_term-0.5*((x[:, 4]-self.slope_mu)/self.slope_sd)**2
        return np.where(accepted, prior, -np.inf)

    def prior_kt18(self, x):
        x = np.atleast_2d(x); ell, eps = np.exp(x[:, 0]), expit(x[:, 1])
        extent = 0.5*ell**2*(1-eps**2)
        lower, upper = np.maximum(np.log(.05), np.log(extent)-np.log(1.5)), np.minimum(np.log(3), np.log(extent)-np.log(.05))
        valid = ((ell >= self.ell_grid[0]) & (ell <= self.ell_grid[-1]) & (eps >= self.eps_grid[0]) & (eps <= self.eps_grid[-1])
                 & (ell*eps >= .015) & (ell*eps <= .25) & (upper > lower))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(valid, np.log(2)-np.log1p(eps)+np.log(upper-lower), -np.inf)

    def log_likelihood(self, x, kt18=False):
        x = np.atleast_2d(x); pred = self.predict_kt18(x) if kt18 else self.predict(x)
        residual = (pred-self.data["xi_centered"]) @ self.precision_root.T
        return -0.5*np.sum(residual**2, axis=1)

    def log_probability(self, x, kt18=False):
        x = np.atleast_2d(x); prior = self.prior_kt18(x) if kt18 else self.prior(x)
        sel = np.isfinite(prior); out = np.full(len(x), -np.inf)
        if sel.any():
            out[sel] = prior[sel]+self.log_likelihood(x[sel], kt18)
        return np.where(np.isfinite(out), out, -np.inf)

    def chi2(self, curve):
        return float(np.sum(((curve-self.data["xi_centered"]) @ self.precision_root.T)**2))


class SplitKTZ(ExactWindowKTZ):
    """Eight coordinates (KTZ + sigma_q), joint data vector [xi_all, D]."""

    def __init__(self, metadata, diagnostics, fit_covariance, response_all, response_same, response_cross,
                 split, covariance_joint, emulator_grids, use_D=True, sigma_q_max=3.0, **kw):
        super().__init__(metadata, diagnostics, fit_covariance, response_all, emulator_grids, **kw)
        self.split = {k: np.asarray(split[k]) for k in (split.files if hasattr(split, "files") else split)}
        self.use_D = use_D; self.sigma_q_max = sigma_q_max
        self.sub = {"same": self._interpolators(response_same), "cross": self._interpolators(response_cross)}
        self.counts_same, self.counts_cross = self.sub["same"]["counts"], self.sub["cross"]["counts"]
        self.counts_all = self.counts_same+self.counts_cross
        self.D_obs = self.split["D"]; self.nb = len(self.data["xi_centered"])
        self.y = np.r_[self.data["xi_centered"], self.D_obs] if use_D else self.data["xi_centered"]
        C = np.asarray(covariance_joint) if use_D else np.asarray(covariance_joint)[:self.nb, :self.nb]
        self.covariance_joint = C; self.precision_root_joint = np.linalg.inv(np.linalg.cholesky(C))

    def curves(self, x):
        """xi_all, xi_same, xi_cross (n, nbins) and the component split of xi_all (n, 5, nbins)."""
        ts, tc = self.raw_terms(x, self.sub["same"]), self.raw_terms(x, self.sub["cross"])
        ws, wc = self.counts_same/self.counts_all, self.counts_cross/self.counts_all
        ta = ts*ws+tc*wc; bias = self.data["noise_additive_bias"]
        ca, cs, cc = self._ratio(ta), self._ratio(ts), self._ratio(tc)
        return ca.sum(1)+bias, cs.sum(1)+bias, cc.sum(1)+bias, ca

    def compress_D(self, xs, xc):
        cb, cS, cC = self.split["cbin"], self.split["pairs_same"], self.split["pairs_cross"]
        nc = len(self.split["coarse_edges"])-1; out = np.empty((len(xs), nc))
        for k in range(nc):
            m = (cb == k); out[:, k] = (xs[:, m]*cS[m]).sum(1)/cS[m].sum()-(xc[:, m]*cC[m]).sum(1)/cC[m].sum()
        return out

    def predict_joint(self, x):
        xa, xs, xc, _ = self.curves(x)
        return np.c_[xa, self.compress_D(xs, xc)] if self.use_D else xa

    def prior(self, x):
        x = np.atleast_2d(x); base = super().prior(x[:, :7])
        if x.shape[1] > 7:
            base = np.where((x[:, 7] >= 0) & (x[:, 7] <= self.sigma_q_max), base, -np.inf)
        return base

    def log_probability(self, x, kt18=False):
        x = np.atleast_2d(x)
        if kt18:
            return super().log_probability(x, kt18=True)
        prior = self.prior(x); sel = np.isfinite(prior); out = np.full(len(x), -np.inf)
        if sel.any():
            res = (self.predict_joint(x[sel])-self.y) @ self.precision_root_joint.T
            out[sel] = prior[sel]-0.5*np.sum(res**2, axis=1)
        return np.where(np.isfinite(out), out, -np.inf)

    def chi2_joint(self, pred):
        return float(np.sum(((pred-self.y) @ self.precision_root_joint.T)**2))


# ----------------------------------------------------------------------------------
# sampler and summaries
# ----------------------------------------------------------------------------------
def split_rhat(chain):
    half = len(chain)//2
    split = np.concatenate((chain[:half].transpose(1, 0, 2), chain[-half:].transpose(1, 0, 2)))
    within = split.var(axis=1, ddof=1).mean(axis=0); between = half*split.mean(axis=1).var(axis=0, ddof=1)
    return np.sqrt(((half-1)*within/half+between/half)/within)


def run_ensemble(model, start, nsteps, kt18=False, seed=0, progress=None, thin=5, burn=None):
    """Affine-invariant stretch-move ensemble sampler (same algorithm as stage 1)."""
    rng = np.random.default_rng(seed)
    positions = np.array(start, float).copy(); nw, nd = positions.shape
    probability = model.log_probability(positions, kt18)
    assert np.all(np.isfinite(probability)), "non-finite starting probability"
    chain = np.empty((nsteps, nw, nd)); density = np.empty((nsteps, nw)); accepted = np.zeros(nw)
    for step in range(nsteps):
        for parity in (0, 1):
            active = np.arange(parity, nw, 2); complement = np.arange(1-parity, nw, 2)
            partners = positions[rng.choice(complement, len(active))]
            z = (1+rng.random(len(active)))**2/2
            proposals = partners+z[:, None]*(positions[active]-partners)
            proposed = model.log_probability(proposals, kt18)
            keep = np.log(rng.random(len(active))) < ((nd-1)*np.log(z)+proposed-probability[active])
            positions[active[keep]] = proposals[keep]; probability[active[keep]] = proposed[keep]
            accepted[active] += keep
        chain[step], density[step] = positions, probability
        if progress is not None and (step+1) % 1000 == 0:
            progress(step+1)
    burn = nsteps//5 if burn is None else burn
    return dict(chain=chain, log_probability=density, acceptance=accepted/nsteps, rhat=split_rhat(chain[burn:]), burn=burn,
                samples=chain[burn::thin].reshape(-1, nd), sample_lp=density[burn::thin].reshape(-1))


def summarise_joint(model, samples, n_draw=2000, seed=11):
    """Posterior summary of one joint run: curves, D, component split, parameter percentiles."""
    r = model.data["r_kpc"]; sh = (r >= .6) & (r <= 2.5); tl = (r >= 3.3) & (r <= 4.5)
    obs = model.data["xi_centered"]
    rng = np.random.default_rng(seed); d = samples[rng.choice(len(samples), min(n_draw, len(samples)), replace=False)]
    xa, xs, xc, comp = model.curves(d); Dm = model.compress_D(xs, xc)
    med = np.nanmedian(xa, 0); Dmed = np.median(Dm, 0)
    pointing_part = np.nanmedian(comp[:, 4, :], 0); ktz_part = np.nanmedian(comp[:, :4, :].sum(1), 0)
    der = np.asarray(model.derived(samples[:, :7])).T
    pct = np.percentile(der, [16, 50, 84], axis=0)
    params = {n: pct[:, i].tolist() for i, n in enumerate(PARAMETER_NAMES)}
    params["inv_h_R"] = np.percentile(-samples[:, 4], [16, 50, 84]).tolist()
    params["sigma_q"] = np.percentile(samples[:, 7], [16, 50, 84]).tolist() if samples.shape[1] > 7 else [0.0, 0.0, 0.0]
    summary = dict(
        shoulder_obs=float(np.nanmean(obs[sh])), shoulder_model=float(np.nanmean(med[sh])), shoulder_ktz=float(np.nanmean(ktz_part[sh])),
        shoulder_pointing=float(np.nanmean(pointing_part[sh])),
        tail_obs=float(np.nanmean(obs[tl])) if tl.any() else None, tail_model=float(np.nanmean(med[tl])) if tl.any() else None,
        chi2_all_median_curve=model.chi2(med), chi2_joint_median=model.chi2_joint(np.r_[med, Dmed]) if model.use_D else None,
        n_data=int(len(model.y)), pointing_variance_fraction=np.nanpercentile(comp[:, 4, 0], [16, 50, 84]).tolist() if samples.shape[1] > 7 else [0, 0, 0],
        D_model=Dmed.tolist(), D_model_p16=np.percentile(Dm, 16, 0).tolist(), D_model_p84=np.percentile(Dm, 84, 0).tolist(), D_obs=model.D_obs.tolist(),
        P_ell_src_gt_1kpc=float((der[:, 8] > 1).mean()), params=params)
    curves = dict(r=r, xi_median=med, xi_p16=np.nanpercentile(xa, 16, 0), xi_p84=np.nanpercentile(xa, 84, 0), pointing_part=pointing_part,
                  ktz_part=ktz_part, comp_median=np.nanmedian(comp, 0), D_median=Dmed, D_p16=np.percentile(Dm, 16, 0), D_p84=np.percentile(Dm, 84, 0))
    return summary, curves
