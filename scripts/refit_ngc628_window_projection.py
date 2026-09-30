"""NGC628 refit: unchanged seven-parameter physics/prior, corrected window.

Called by the companion notebook after the original, deterministic line-flux
MC pipeline supplies its full covariance. Old results are never overwritten.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import exp1, expit, logit

from scripts.phangs_template_ktz import gaussian_source_cnp_kernel
from scripts.phangs_window_projection import load_or_build_window_responses, VERSION


class ProjectedKTZ:
    def __init__(self, inputs: Path, output: Path, covariance, raw_covariance):
        self.inputs, self.output = inputs, output
        output.mkdir(parents=True, exist_ok=True)
        self.metadata = json.loads((inputs/"ngc628_morphology_summary.json").read_text())
        self.data = np.genfromtxt(inputs/"ngc628_residual_covariance_diagnostics.csv", delimiter=",", names=True)
        self.covariance = np.asarray(covariance)
        self.raw_covariance = np.asarray(raw_covariance)
        self.precision_root = np.linalg.inv(np.linalg.cholesky(self.covariance))
        with np.load(inputs/"ngc628_muse_scal_metallicity_residual_pixel_map.npz") as a:
            self.response = load_or_build_window_responses(
                output/"ngc628_full_window_second_moment_response.npz",
                valid=a["valid_mask"].astype(bool), x_grid=a["x_grid_kpc"],
                y_grid=a["y_grid_kpc"], modulation=a["arm_source_modulation"],
                pixel_size=0.06, edges=np.arange(0.03, 5.031, 0.06),
                annulus_width=0.2, reference_radius=self.metadata["r_ref_kpc"],
                rate_log_slope_grid=np.linspace(-0.35, 0.35, 5),
            )
        np.testing.assert_allclose(self.response["ordered_pair_counts"]/2, self.data["pair_count"], atol=0.1)
        with np.load(inputs/"ngc628_muse_scal_7p_ktz_full_pair_emulator.npz") as a:
            self.ell_grid, self.eps_grid = a["ell_grid"], a["epsilon_grid"]
        with np.load(inputs/"ngc628_muse_scal_7p_ktz_cnp_full_pair_emulator.npz") as a:
            self.src_grid = a["source_length_grid"]
        # Refinement changes numerical resolution, never prior bounds.
        refine=lambda grid: np.exp(np.sort(np.r_[np.log(grid),0.5*(np.log(grid[:-1])+np.log(grid[1:]))]))
        self.ell_grid=refine(self.ell_grid)
        self.eps_grid=refine(self.eps_grid)
        self.src_grid=refine(self.src_grid)
        kr = self.response["kernel_radius"]
        kp = np.empty((len(self.ell_grid), len(self.eps_grid), len(kr)))
        kn = np.empty((*kp.shape[:2], len(self.src_grid), len(kr)))
        for i, ell in enumerate(self.ell_grid):
            for j, eps in enumerate(self.eps_grid):
                kp[i, j, 0] = -np.log(eps)
                kp[i, j, 1:] = 0.5*(exp1(kr[1:]**2/(4*ell**2))-exp1(kr[1:]**2/(4*(eps*ell)**2)))
                for k, length in enumerate(self.src_grid):
                    kn[i, j, k] = gaussian_source_cnp_kernel(kr, ell, eps, length, quadrature_nodes=24)
        R = self.response["response"]
        pgrid = np.einsum("ecabk,ijk->ijecab", R[:, :2], kp, optimize=True)
        ngrid = np.einsum("eabk,ijlk->ijelab", R[:, 2], kn, optimize=True)
        self.pinterp = RegularGridInterpolator(
            (np.log(self.ell_grid), np.log(self.eps_grid), self.response["rate_log_slope_grid"]),
            pgrid, bounds_error=False, fill_value=np.nan,
        )
        self.ninterp = RegularGridInterpolator(
            (np.log(self.ell_grid), np.log(self.eps_grid), self.response["rate_log_slope_grid"], np.log(self.src_grid)),
            ngrid, bounds_error=False, fill_value=np.nan,
        )
        self.minterp = RegularGridInterpolator(
            (self.response["rate_log_slope_grid"],), self.response["mean_response"],
            bounds_error=False, fill_value=np.nan,
        )
        node, weight = np.polynomial.legendre.leggauss(32)
        rmin, rmax = self.metadata["analysis_rmin_kpc"], self.metadata["analysis_rmax_kpc"]
        self.area_radius = 0.5*((rmax-rmin)*node+rmax+rmin)
        self.area_weight = weight*self.area_radius/(rmax+rmin)
        norm = 1/(1/100-1/1e6)
        mean_mass, second_mass = norm*np.log(1e4), norm*(1e6-100)
        self.mass_factor = mean_mass**2/second_mass
        self.gamma_area = self.metadata["halpha_sfr_msun_per_yr"]/(np.pi*(rmax**2-rmin**2)*mean_mass)*1e6
        self.eta_prior_sigma = max(
            self.metadata["mask_contrast_amplitude_error"]
            / self.metadata["mask_contrast_amplitude"], 0.35,
        )
        np.savez_compressed(output/"ngc628_fit_covariance.npz", covariance=covariance, raw_covariance=raw_covariance)

    def derived(self, x):
        x = np.atleast_2d(x)
        ell, eps, beta, scale = np.exp(x[:, 0]), expit(x[:, 1]), np.exp(x[:, 2]), np.exp(x[:, 3])
        extent = 0.5*ell**2*(1-eps**2)
        rel = np.exp(x[:, 4, None]*(self.area_radius-self.metadata["r_ref_kpc"])) @ self.area_weight
        gamma = self.gamma_area/rel
        t = beta/(4*np.pi*gamma*1e3*self.mass_factor*extent)
        return ell, eps, beta, scale, extent/t, t, eps*ell, np.exp(x[:, 5]-x[:, 2]), np.exp(x[:, 6])

    def kernel_coordinates(self, x):
        ell, eps = np.exp(x[:, 0]), expit(x[:, 1])
        beam2 = 0.5*(0.12/2.354820045)**2
        observed_ell = np.sqrt(ell**2+beam2)
        observed_eps = np.sqrt((ell*eps)**2+beam2)/observed_ell
        return np.column_stack((np.log(observed_ell), np.log(observed_eps), x[:, 4]))

    def components(self, x):
        x = np.atleast_2d(x)
        coords = self.kernel_coordinates(x)
        p, n, m = self.pinterp(coords), self.ninterp(np.column_stack((coords,x[:, 6]))), self.minterp(x[:, 4, None])
        beta, scale, betaA = np.exp(x[:, 2]), np.exp(x[:, 3]), np.exp(x[:, 5])
        mean = beta[:, None, None]*(m[:, 0]+scale[:, None, None]*m[:, 1]+scale[:, None, None]**2*m[:, 2])
        terms = np.stack((p[:, 0], scale[:, None, None]*p[:, 1], mean, betaA[:, None, None]*n), axis=1)
        denominator = terms[:, :, 1].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = terms[:, :, 0]/denominator[:, None, :]
        result[denominator.min(axis=1) <= 0] = np.nan
        return self.data["noise_signal_fraction"][None, None, :]*result

    def predict(self, x):
        return self.components(x).sum(axis=1)+self.data["noise_additive_bias"]

    def predict_kt18(self, x):
        x = np.atleast_2d(x)
        dummy = np.zeros((len(x), 7)); dummy[:, :2] = x
        p = self.pinterp(self.kernel_coordinates(dummy))[:, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.data["noise_signal_fraction"]*p[:, 0]/p[:, 1]+self.data["noise_additive_bias"]

    def prior(self, x):
        x = np.atleast_2d(x)
        ell, eps, beta, scale, kappa, t, sigma, A, length = self.derived(x)
        accepted = (
            (ell >= self.ell_grid[0]) & (ell <= self.ell_grid[-1])
            & (eps >= self.eps_grid[0]) & (eps <= self.eps_grid[-1])
            & (beta >= 1e-4) & (beta <= 1e4)
            & (scale >= 0.10) & (scale <= self.metadata["maximum_positive_eta"])
            & (np.abs(x[:, 4]) <= 0.35)
            & (kappa >= 0.05) & (kappa <= 1.50) & (t >= 0.05) & (t <= 3.00)
            & (sigma >= 0.015) & (sigma <= 0.25) & (A >= 1e-2) & (A <= 1e3)
            & (np.exp(x[:, 5]) >= 1e-6) & (np.exp(x[:, 5]) <= 1e7)
            & (length >= self.src_grid[0]) & (length <= self.src_grid[-1])
        )
        prior = (np.log(2)-np.log1p(eps)-0.5*(x[:, 3]/self.eta_prior_sigma)**2
                 -0.5*((x[:, 4]-self.metadata["event_rate_log_slope_kpc_inv"])
                        /self.metadata["event_rate_log_slope_error_kpc_inv"])**2)
        return np.where(accepted, prior, -np.inf)

    def prior_kt18(self, x):
        x = np.atleast_2d(x)
        ell, eps = np.exp(x[:, 0]), expit(x[:, 1])
        extent = 0.5*ell**2*(1-eps**2)
        lower, upper = np.maximum(np.log(.05), np.log(extent)-np.log(1.5)), np.minimum(np.log(3), np.log(extent)-np.log(.05))
        valid = ((ell >= self.ell_grid[0]) & (ell <= self.ell_grid[-1])
                 & (eps >= self.eps_grid[0]) & (eps <= self.eps_grid[-1])
                 & (ell*eps >= .015) & (ell*eps <= .25) & (upper > lower))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(valid, np.log(2)-np.log1p(eps)+np.log(upper-lower), -np.inf)

    def log_probability(self, x, kt18=False):
        x = np.atleast_2d(x)
        prior = self.prior_kt18(x) if kt18 else self.prior(x)
        selected = np.isfinite(prior)
        result = np.full(len(x), -np.inf)
        if selected.any():
            pred = self.predict_kt18(x[selected]) if kt18 else self.predict(x[selected])
            residual = (pred-self.data["xi_centered"]) @ self.precision_root.T
            result[selected] = prior[selected]-0.5*np.sum(residual**2, axis=1)
        return np.where(np.isfinite(result), result, -np.inf)


def split_rhat(chain):
    half = len(chain)//2
    split = np.concatenate((chain[:half].transpose(1,0,2),chain[-half:].transpose(1,0,2)))
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = half*split.mean(axis=1).var(axis=0, ddof=1)
    return np.sqrt(((half-1)*within/half+between/half)/within)


def sample(model, kt18=False, seed=718628):
    rng = np.random.default_rng(seed)
    inputs, output = model.inputs, model.output
    filename = "ngc628_kt18_transport_chain.npz" if kt18 else "ngc628_transport_chain.npz"
    with np.load(inputs/filename) as archive:
        old = archive["internal_chain"][int(archive["burn_steps"]):][::20].reshape(-1, 2 if kt18 else 7)
    old = old[rng.choice(len(old),min(5000,len(old)),replace=False)]
    nw, burn, minimum, maximum, block = (32,4000,15000,15000,15000) if kt18 else (64,10000,50000,150000,25000)
    # Dispersed old-chain draws are initialisation only, never reweighted into
    # the new result. Their broad and compact source-pair states both survive.
    probability = model.log_probability(old, kt18)
    candidates = old[np.isfinite(probability)]
    positions = candidates[rng.choice(len(candidates), nw, replace=False)].copy()
    probability = model.log_probability(positions, kt18)
    chains, densities = [], []
    accepted = np.zeros(nw)
    completed = 0
    while completed < maximum:
        nsteps = minimum if completed == 0 else min(block,maximum-completed)
        chain = np.empty((nsteps,nw,positions.shape[1])); density = np.empty((nsteps,nw))
        for step in range(nsteps):
            for parity in (0,1):
                active = np.arange(parity,nw,2); complement=np.arange(1-parity,nw,2)
                partners=positions[rng.choice(complement,len(active))]
                z=(1+rng.random(len(active)))**2/2
                proposals=partners+z[:,None]*(positions[active]-partners)
                proposed=model.log_probability(proposals,kt18)
                keep=np.log(rng.random(len(active))) < ((positions.shape[1]-1)*np.log(z)+proposed-probability[active])
                positions[active[keep]]=proposals[keep]; probability[active[keep]]=proposed[keep]
                accepted[active]+=keep
            chain[step],density[step]=positions,probability
            if (step+1)%1000 == 0:
                progress={"stage":"KT18 MCMC" if kt18 else "complete KTZ MCMC","steps":completed+step+1,"target_minimum_steps":minimum}
                (output/"run_progress.json").write_text(json.dumps(progress,indent=2)+"\n")
                print(progress,flush=True)
        chains.append(chain); densities.append(density); completed+=nsteps
        combined=np.concatenate(chains); combined_density=np.concatenate(densities)
        rhat=split_rhat(combined[burn:])
        np.savez_compressed(output/filename, internal_chain=combined,log_probability=combined_density,burn_steps=burn,rhat=rhat,acceptance_fraction=accepted/completed,window_response_version=VERSION)
        print("split Rhat",rhat,flush=True)
        if np.max(rhat) <= 1.05:
            break
    return combined[burn::5].reshape(-1,combined.shape[-1]),rhat


def kernel_point_prediction(model, x, quadrature_nodes=32):
    """Check kernels directly, holding represented window/rate_log_slope responses fixed.

    This does not validate rate_log_slope interpolation, the midpoint approximation, or
    the radial representation of the full pixel covariance.
    """
    x = np.asarray(x)
    coords = model.kernel_coordinates(x[None])[0]
    ell, eps = np.exp(coords[:2])
    kr = model.response["kernel_radius"]
    kernel = np.r_[
        -np.log(eps),
        0.5*(exp1(kr[1:]**2/(4*ell**2))
             - exp1(kr[1:]**2/(4*(eps*ell)**2))),
    ]
    cnp = gaussian_source_cnp_kernel(
        kr, ell, eps, np.exp(x[6]), quadrature_nodes=quadrature_nodes,
    )
    e = model.response['rate_log_slope_grid']
    i = np.clip(np.searchsorted(e, x[4])-1, 0, len(e)-2)
    fraction = (x[4]-e[i])/(e[i+1]-e[i])
    response = ((1-fraction)*model.response['response'][i]
                + fraction*model.response['response'][i+1])
    nd, cnpnd = response[:2]@kernel, response[2]@cnp
    mean = model.minterp([[x[4]]])[0]
    beta, scale, betaA = np.exp(x[[2, 3, 5]])
    total = (nd[0]+scale*nd[1]
             + beta*(mean[0]+scale*mean[1]+scale**2*mean[2])
             + betaA*cnpnd)
    if np.any(total[1] <= 0):
        raise ValueError('Non-positive projected endpoint second moment.')
    result = (model.data['noise_signal_fraction']*total[0]/total[1]
              + model.data['noise_additive_bias'])
    if not np.all(np.isfinite(result)):
        raise ValueError('Non-finite direct-kernel point prediction.')
    return result


def run_refit(inputs, output, covariance, raw_covariance):
    from astropy.table import Table
    import matplotlib.pyplot as plt

    start=time.time(); inputs,output=Path(inputs),Path(output)
    model=ProjectedKTZ(inputs,output,covariance,raw_covariance)
    # Check interpolation against analytic radial kernels at retained old
    # states before paying for a new chain. This holds rate_log_slope interpolation fixed;
    # it is not a global bound or a test of the local-source approximation.
    with np.load(inputs/'ngc628_transport_chain.npz') as a:
        old=a['internal_chain'][int(a['burn_steps'])::1000].reshape(-1,7)
    check_rng=np.random.default_rng(829628)
    checks=old[check_rng.choice(len(old),40,replace=False)]
    maximum_error=0.0
    for x in checks:
        exact = kernel_point_prediction(model, x)
        difference = np.abs(model.predict(x)[0]-exact)
        if not np.all(np.isfinite(difference)):
            raise RuntimeError('Non-finite window-emulator point check.')
        maximum_error=max(maximum_error,float(np.max(difference)))
    (output/'ngc628_window_emulator_check.json').write_text(json.dumps({'maximum_absolute_correlation_error_over_40_old_states':maximum_error,'scope':'kernel-grid and age-quadrature point check; rate_log_slope interpolation, radial representation and local-source assumptions held fixed'},indent=2)+'\n')
    print('Window emulator point-check max error',maximum_error,flush=True)
    if maximum_error>0.01:
        raise RuntimeError('Kernel emulator point checks exceed 0.01; refine before fitting.')
    samples,rhat=sample(model)
    samples18,rhat18=sample(model,kt18=True,seed=718629)
    rng=np.random.default_rng(908628)
    draws=samples[rng.choice(len(samples),min(1500,len(samples)),replace=False)]
    draws18=samples18[rng.choice(len(samples18),min(1500,len(samples18)),replace=False)]
    curves=model.predict(draws); curves18=model.predict_kt18(draws18)
    intervals=np.percentile(curves,[16,50,84],axis=0); intervals18=np.percentile(curves18,[16,50,84],axis=0)
    components=model.components(draws)
    whiten=(curves-intervals[1]) @ model.precision_root.T
    representative=np.argmin(np.sum(whiten**2,axis=1))
    Q=lambda curve: float(np.sum(((curve-model.data['xi_centered']) @ model.precision_root.T)**2))
    shoulder=(model.data['r_kpc']>=.6)&(model.data['r_kpc']<=2.5)
    weight=np.zeros(len(shoulder));weight[shoulder]=1/shoulder.sum()
    # Replicated-observation uncertainty, not merely uncertainty in a mean curve.
    observed_shoulder=float(weight@model.data['xi_centered'])
    noise_sigma=np.sqrt(weight@model.covariance@weight)
    predicted_shoulder=curves@weight
    replicated=predicted_shoulder+rng.normal(0,noise_sigma,len(curves))
    mean_shoulder_p=float(np.mean(replicated>=observed_shoulder))
    table={"r_kpc":np.r_[0,model.data['r_kpc']],"xi_observed":np.r_[1,model.data['xi_centered']],"xi_jackknife_error":np.r_[0,model.data['total_error']],"is_fixed_normalisation_anchor":np.r_[True,np.zeros(len(shoulder),bool)],"observation_noise_bias":np.r_[0,model.data['noise_additive_bias']]}
    for prefix,inter in [('KTZ',intervals),('KT18',intervals18)]:
        for label,array in zip(['p16','median','p84'],inter):table[f'xi_{prefix}_{label}']=np.r_[1,array]
    table['xi_KTZ_joint_representative']=np.r_[1,curves[representative]]
    for k,name in enumerate(['CP0','CPN','CM','CNP']):
        table[f'xi_component_{name}_joint_representative']=np.r_[0,components[representative,k]]
        for label,array in zip(['p16','median','p84'],np.percentile(components[:,k],[16,50,84],axis=0)):
            table[f'xi_component_{name}_{label}']=np.r_[0,array]
    Table(table).write(output/'ngc628_xi_observed_and_models.csv',format='ascii.csv',overwrite=True)
    parameter_names=['ell_star','epsilon','beta','eta','kappa','t_star','sigma','source_excess_amplitude','source_clustering_length']
    param=np.asarray(model.derived(samples)).T
    param_inter=np.percentile(param,[16,50,84],axis=0).T
    Table({'parameter':parameter_names,'p16':param_inter[:,0],'median':param_inter[:,1],'p84':param_inter[:,2]}).write(output/'ngc628_transport_posterior.csv',format='ascii.csv',overwrite=True)
    summary={
        'status':'converged' if max(np.max(rhat),np.max(rhat18))<=1.05 else 'not_converged_diagnostic_only',
        'projection':'raw midpoint local covariance projected on actual finite mask before common normalisation; no 5-kpc covariance cutoff',
        'source_template_fwhm_kpc':1.0,'physics_and_physical_prior':'unchanged seven coordinates; no extra fit parameters',
        'local_source_approximation_retained':True,'mcmc_max_rhat':float(np.max(rhat)),'kt18_max_rhat':float(np.max(rhat18)),
        'KTZ_median_curve_chi2':Q(intervals[1]),'KT18_median_curve_chi2':Q(intervals18[1]),
        'regularised_chi2_reference_mean':float(np.trace(np.linalg.solve(model.covariance,model.raw_covariance))),
        'observed_shoulder_mean':observed_shoulder,'model_shoulder_mean_p16_median_p84':np.percentile(predicted_shoulder,[16,50,84]).tolist(),
        'shoulder_covariance_standard_error':float(noise_sigma),'shoulder_posterior_predictive_upper_tail_fraction':mean_shoulder_p,
        'shoulder_RMSE':float(np.sqrt(np.mean((intervals[1,shoulder]-model.data['xi_centered'][shoulder])**2))),
        'posterior_intervals_p16_median_p84':dict(zip(parameter_names,param_inter.tolist())),
        'elapsed_seconds':time.time()-start,
        'window_emulator_point_check_max_error':maximum_error,
        'limitations':'Exact annular projection only for the adopted local covariance and interpolated kernel/rate_log_slope grids. Not a global source-integral solution. Shoulder chosen after inspection: exploratory, not a confirmatory p-value.',
    }
    (output/'ngc628_window_refit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    old=np.genfromtxt(inputs/'ngc628_xi_observed_and_models.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for ax in axes:
        ax.fill_between(model.data['r_kpc'],intervals[0],intervals[2],color='#7b3294',alpha=.16,label='corrected complete KTZ 16–84%')
        ax.plot(model.data['r_kpc'],intervals[1],color='#7b3294',label='corrected KTZ posterior median')
        ax.plot(old['r_kpc'],old['xi_KTZ_median'],color='.45',ls=':',label='previous stationary-equivalent median')
        ax.plot(model.data['r_kpc'],intervals18[1],color='#1f77b4',label='KT18 posterior median')
        ax.errorbar(model.data['r_kpc'],model.data['xi_centered'],yerr=model.data['total_error'],fmt='o',ms=2,color='black',ecolor='.6',lw=.7,label='PHANGS-MUSE')
        ax.axhline(0,color='.6',lw=.6);ax.set_xlabel('separation r [kpc]');ax.set_ylabel(r'$\xi_{3,\mathrm{pp}}$')
    axes[0].set(xlim=(0,5),ylim=(-.2,1.04),title='NGC 628: corrected finite-window fit');axes[0].legend(fontsize=8,frameon=False)
    axes[1].set(xlim=(.5,2.6),ylim=(-.04,.25),title='Resolved positive-correlation shoulder')
    fig.savefig(output/'ngc628_window_projection_comparison.png',dpi=200)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.8),constrained_layout=True)
    accumulated=np.zeros(len(shoulder))
    for values,color,label in zip(components[representative],['#7fc97f','#ffff99','#fdc086','#beaed4'],[r'$C_{\rm 2p,0}$',r'$C_{\rm 2p,N}$',r'$C_{\rm M}$',r'$C_{\rm NP}$']):
        ax.fill_between(model.data['r_kpc'],accumulated,accumulated+values,color=color,alpha=.5,label=label)
        accumulated+=values
    np.testing.assert_allclose(accumulated+model.data['noise_additive_bias'],curves[representative],atol=1e-12)
    ax.fill_between(model.data['r_kpc'],intervals[0]-model.data['noise_additive_bias'],intervals[2]-model.data['noise_additive_bias'],color='black',alpha=.1,label='total physical KTZ 16–84%')
    ax.plot(model.data['r_kpc'],accumulated,'k--',label='joint representative: exact component sum')
    ax.errorbar(model.data['r_kpc'],model.data['xi_centered'],yerr=model.data['total_error'],fmt='o',ms=2,color='black',ecolor='.6',lw=.7,label='PHANGS-MUSE')
    ax.set(xlim=(0,5),ylim=(-.2,1.04),xlabel='separation r [kpc]',ylabel=r'$\xi_{3,\mathrm{pp}}$',title='NGC 628: corrected complete KTZ components')
    ax.axhline(0,color='.6',lw=.6);ax.legend(fontsize=8,frameon=False,ncol=2)
    fig.savefig(output/'ngc628_window_projection_components.png',dpi=200)
    plt.close(fig)
    (output/'run_progress.json').write_text(json.dumps({'stage':'completed',**summary},indent=2)+'\n')
    return summary
