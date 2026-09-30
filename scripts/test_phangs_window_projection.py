"""Dense small-map checks independent of the compressed window implementation."""

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
from scipy.special import exp1, logit
from scripts.phangs_window_projection import build_window_responses, annular_project
from scripts.phangs_template_ktz import gaussian_source_cnp_kernel
from scripts.refit_ngc628_window_projection import ProjectedKTZ


def test_dense_projection():
    rng = np.random.default_rng(561)
    yy, xx = np.meshgrid(np.arange(7)*0.06, np.arange(8)*0.06, indexing="ij")
    xx -= 0.21
    yy -= 0.18
    valid = rng.random(xx.shape) > 0.2
    modulation = 0.3*np.sin(8*xx) + 0.2*np.cos(4*yy)
    radius = np.hypot(xx, yy)
    edges = np.arange(0.03, 0.451, 0.06)
    rate_log_slopes = [-0.2, 0.0, 0.17]
    result = build_window_responses(
        valid, xx, yy, modulation, 0.06, edges, 0.12, 0.25, rate_log_slopes, block_size=9,
    )
    iy, ix = np.nonzero(valid)
    x, y = xx[valid], yy[valid]
    r = radius[valid]
    labels = np.floor(r/0.12).astype(int)
    P = np.eye(len(r))
    for label in np.unique(labels):
        selected = np.flatnonzero(labels == label)
        P[np.ix_(selected, selected)] -= 1/len(selected)
    assert np.allclose(P@P, P)
    distance = np.hypot(x[:, None]-x, y[:, None]-y)
    midpoint_radius = np.hypot((x[:, None]+x)/2, (y[:, None]+y)/2)
    hy, hx = np.meshgrid(np.arange(13)/2, np.arange(15)/2, indexing="ij")
    ghalf = map_coordinates(modulation, [hy, hx], order=1, mode="nearest")
    gmid = ghalf[iy[:, None]+iy, ix[:, None]+ix]
    bins = np.searchsorted(edges, distance, side="right")-1
    kernel = np.exp(-result["kernel_radius"]**2/(2*0.15**2))
    Ckernel = np.interp(distance, result["kernel_radius"], kernel)
    for ei, rate_log_slope in enumerate(rate_log_slopes):
        rate = np.exp(rate_log_slope*(midpoint_radius-0.25))
        for ci, factor in enumerate([rate, rate*gmid, rate**2]):
            C = P @ (Ckernel*factor) @ P.T
            n, d = result["response"][ei, ci] @ kernel
            for b in range(len(edges)-1):
                selected = (bins == b) & (distance > 0)
                if not selected.any():
                    continue
                ndense = C[selected].mean()
                ddense = ((np.diag(C)[:, None]+np.diag(C)[None, :])/2)[selected].mean()
                np.testing.assert_allclose(n[b], ndense, rtol=2e-10, atol=2e-12)
                np.testing.assert_allclose(d[b], ddense, rtol=2e-10, atol=2e-12)
        field = np.exp(rate_log_slope*(radius-0.25))*(1+0.7*modulation)
        projected = annular_project(field, valid, radius, 0.12)[valid]
        np.testing.assert_allclose(projected, P@field[valid], atol=2e-14)
        N = sum(result["mean_response"][ei, k, 0]*0.7**k for k in range(3))
        D = sum(result["mean_response"][ei, k, 1]*0.7**k for k in range(3))
        for b in range(len(edges)-1):
            selected = (bins == b) & (distance > 0)
            if selected.any():
                np.testing.assert_allclose(N[b], np.outer(projected, projected)[selected].mean(), atol=1e-14)
                np.testing.assert_allclose(D[b], ((projected[:, None]**2+projected[None, :]**2)/2)[selected].mean(), atol=1e-14)
    print("Dense P C P^T, mean-field and arithmetic-denominator checks passed.")
    # Independently exercise the actual vectorised likelihood model, including
    # its Gaussian beam coordinates and the common four-term denominator.
    ell_grid=np.array([.1,.2,.4]);eps_grid=np.array([.2,.5,.8]);src_grid=np.array([.05,.15,.3])
    kr=result['kernel_radius'];kp=np.empty((3,3,len(kr)));kn=np.empty((3,3,3,len(kr)))
    for i,ell in enumerate(ell_grid):
        for j,eps in enumerate(eps_grid):
            kp[i,j]=np.r_[-np.log(eps),.5*(exp1(kr[1:]**2/(4*ell**2))-exp1(kr[1:]**2/(4*(ell*eps)**2)))]
            for k,length in enumerate(src_grid):
                kn[i,j,k]=gaussian_source_cnp_kernel(kr,ell,eps,length,quadrature_nodes=24)
    model=ProjectedKTZ.__new__(ProjectedKTZ)
    model.pinterp=RegularGridInterpolator((np.log(ell_grid),np.log(eps_grid),rate_log_slopes),np.einsum('ecabk,ijk->ijecab',result['response'][:,:2],kp))
    model.ninterp=RegularGridInterpolator((np.log(ell_grid),np.log(eps_grid),rate_log_slopes,np.log(src_grid)),np.einsum('eabk,ijlk->ijelab',result['response'][:,2],kn))
    model.minterp=RegularGridInterpolator((rate_log_slopes,),result['mean_response'])
    nb=len(edges)-1
    model.data=np.zeros(nb,dtype=[('noise_signal_fraction',float),('noise_additive_bias',float)])
    model.data['noise_signal_fraction']=.91;model.data['noise_additive_bias']=1e-4
    beam2=.5*(.12/2.354820045)**2
    ell=np.sqrt(.2**2-beam2);eps=np.sqrt((.5*.2)**2-beam2)/ell
    states=np.array([[np.log(ell),logit(eps),np.log(beta),np.log(scale),0,np.log(betaA),np.log(.15)] for beta,scale,betaA in [(0.9,.8,.2),(.4,.5,.6)]])
    predictions=model.predict(states)
    for si,(beta,scale,betaA) in enumerate([(0.9,.8,.2),(.4,.5,.6)]):
        K=np.interp(distance,kr,kp[1,1]);Knp=np.interp(distance,kr,kn[1,1,1])
        field=P@(1+scale*modulation[valid])
        C=P@((1+scale*gmid)*K+betaA*Knp)@P.T+beta*np.outer(field,field)
        for b in range(nb):
            selected=(bins==b)&(distance>0)
            if selected.any():
                expected=.91*C[selected].mean()/((np.diag(C)[:,None]+np.diag(C)[None,:])/2)[selected].mean()+1e-4
                np.testing.assert_allclose(predictions[si,b],expected,rtol=2e-10,atol=2e-12)
    np.testing.assert_allclose(model.components(states).sum(axis=1)+1e-4,predictions,atol=1e-14)
    print('Vectorised complete-KTZ predictions match dense four-term covariance.')


if __name__ == "__main__":
    test_dense_projection()
