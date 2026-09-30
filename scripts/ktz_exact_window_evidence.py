"""Nested-model Bayesian evidence on the joint [xi_all, D] data vector (dynesty).

Hierarchy (coordinates x = [ln ell_star, logit eps, ln beta_src, ln eta, rate_log_slope, ln(beta_src A_src), ln ell_src, sigma_q]):
  L0   KT18 transport only          C_2p,0                     active {0, 1, 4}
  L1   + arm-modulated source       + C_2p,N + C_M             active {0, 1, 2, 3, 4}
  L1n  + isotropic source clustering  C_2p,0 + C_NP            active {0, 1, 4, 5, 6}
  L2   complete KTZ                 all four                   active {0..6}
Suffix q adds the pointing zero-point term (coordinate 7, uniform on [0, 3]).
Priors are those of the SplitKTZ model passed in (flat eta, Gaussian h_R^-1, ell_src bounds).
"""
from __future__ import annotations
import time
import numpy as np
from scipy.special import expit, logit
from scipy.stats import truncnorm

ACTIVE = {"L0": [0, 1, 4], "L1": [0, 1, 2, 3, 4], "L1n": [0, 1, 4, 5, 6], "L2": [0, 1, 2, 3, 4, 5, 6]}
COMPONENTS = {"L0": [0], "L1": [0, 1, 2], "L1n": [0, 3], "L2": [0, 1, 2, 3]}


class NestedModel:
    def __init__(self, model, name):
        self.m = model; self.name = name
        base = name.rstrip("q"); self.withq = name.endswith("q")
        self.active = list(ACTIVE[base]) + ([7] if self.withq else [])
        self.components = list(COMPONENTS[base]) + ([4] if self.withq else [])
        m = model
        eta_lo = 1e-3 if m.eta_prior_kind == "flat" else 0.10
        self.lo = {0: np.log(m.ell_grid[0]), 1: logit(m.eps_grid[0]), 2: np.log(1e-4), 3: np.log(eta_lo), 4: -m.slope_bound, 5: np.log(1e-6), 6: np.log(m.src_lo), 7: 0.0}
        self.hi = {0: np.log(m.ell_grid[-1]), 1: logit(m.eps_grid[-1]), 2: np.log(1e4), 3: np.log(m.eta_positive_limit), 4: m.slope_bound, 5: np.log(1e7), 6: np.log(m.src_hi), 7: m.sigma_q_max if m.sigma_q_max > 0 else 3.0}
        self.mask = np.ones(5, bool); self.mask[self.components] = False

    def fill(self, u):
        u = np.atleast_2d(u); x = np.zeros((len(u), 8)); m = self.m
        x[:, 3] = -np.inf; x[:, 5] = -np.inf; x[:, 6] = np.log(m.src_grid[len(m.src_grid)//2]); x[:, 7] = 0.0
        for j, i in enumerate(self.active):
            if i == 3:
                if m.eta_prior_kind == "flat":   # flat in eta
                    x[:, 3] = np.log(np.exp(self.lo[3]) + (np.exp(self.hi[3]) - np.exp(self.lo[3]))*u[:, j])
                else:
                    from scipy.special import erfinv
                    x[:, 3] = m.eta_prior_sigma*np.sqrt(2)*erfinv(2*u[:, j]-1)
            elif i == 4:
                a, b = (self.lo[4]-m.slope_mu)/m.slope_sd, (self.hi[4]-m.slope_mu)/m.slope_sd
                x[:, 4] = truncnorm.ppf(u[:, j], a, b, loc=m.slope_mu, scale=m.slope_sd)
            else:
                x[:, i] = self.lo[i] + (self.hi[i]-self.lo[i])*u[:, j]
        return x

    def bounds_ok(self, x):
        ell, eps, beta, eta, kappa, t, sigma, A, length = self.m.derived(x[:, :7])
        ok = (sigma >= 0.015) & (sigma <= 0.25)
        if 3 in self.active:
            ok &= (eta >= np.exp(self.lo[3])) & (eta <= self.m.eta_positive_limit)
        if 2 in self.active:
            ok &= (kappa >= 0.05) & (kappa <= 1.50) & (t >= 0.05) & (t <= 3.00)
        if 5 in self.active:
            ok &= (A >= 1e-2) & (A <= 1e3)
        return ok

    def predict_joint(self, x):
        m = self.m
        ts, tc = m.raw_terms(x, m.sub["same"]), m.raw_terms(x, m.sub["cross"])
        ts[:, self.mask] = 0.0; tc[:, self.mask] = 0.0
        ws, wc = m.counts_same/m.counts_all, m.counts_cross/m.counts_all
        bias = m.data["noise_additive_bias"]
        xa, xs, xc = m._ratio(ts*ws+tc*wc).sum(1)+bias, m._ratio(ts).sum(1)+bias, m._ratio(tc).sum(1)+bias
        return np.c_[xa, m.compress_D(xs, xc)] if m.use_D else xa, xa

    def loglike(self, u):
        x = self.fill(u[None]); ok = self.bounds_ok(x)
        if not ok[0]:
            return -np.inf
        pred, _ = self.predict_joint(x)
        res = (pred-self.m.y) @ self.m.precision_root_joint.T
        ll = -0.5*np.sum(res**2)+np.log(2)-np.log1p(expit(x[0, 1]))   # shared epsilon prior factor
        return float(ll) if np.isfinite(ll) else -np.inf


def evidence_table(model, models=("L0", "L0q", "L1", "L1q", "L1n", "L1nq", "L2", "L2q"), nlive=400, seed=2, log=print):
    import dynesty
    out = {}
    r = model.data["r_kpc"]; sh = (r >= .6) & (r <= 2.5); tl = (r >= 3.3) & (r <= 4.5)
    for name in models:
        nm = NestedModel(model, name); nd = len(nm.active); t0 = time.time()
        sampler = dynesty.NestedSampler(nm.loglike, lambda u: u, nd, nlive=nlive, rstate=np.random.default_rng(seed), bound="multi", sample="rwalk")
        sampler.run_nested(dlogz=0.05, print_progress=False); res = sampler.results
        w = np.exp(res.logwt-res.logz[-1]); w /= w.sum()
        X = nm.fill(res.samples[np.random.default_rng(seed).choice(len(w), 2000, p=w)])
        der = np.asarray(model.derived(X[:, :7])).T
        xmap = nm.fill(res.samples[np.argmax(res.logl)][None])[0]
        chi2 = float(-2*(res.logl.max()-np.log(2)+np.log1p(expit(xmap[1]))))
        pred, xa = nm.predict_joint(X)
        out[name] = dict(logZ=float(res.logz[-1]), logZ_err=float(res.logzerr[-1]), n_params=nd, chi2_min=chi2, BIC=chi2+nd*np.log(len(model.y)), n_data=int(len(model.y)),
                         ell_star=np.percentile(der[:, 0], [16, 50, 84]).tolist(), eta=np.percentile(der[:, 3], [16, 50, 84]).tolist() if 3 in nm.active else None,
                         sigma_q=np.percentile(X[:, 7], [16, 50, 84]).tolist() if nm.withq else None,
                         shoulder=float(np.nanmedian(xa[:, sh].mean(1))), tail=float(np.nanmedian(xa[:, tl].mean(1))) if tl.any() else None, time_s=time.time()-t0)
        log(f"{name}: logZ {out[name]['logZ']:.2f} +- {out[name]['logZ_err']:.2f}, chi2_min {chi2:.2f}, ell_star {out[name]['ell_star'][1]:.3f} [{out[name]['time_s']:.0f} s]")
    return out
