"""Generate a reproducible corrected-window notebook, preserving old fits."""

import copy
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE=ROOT/'NGC628_PHANGS_halpha_map_systematic.ipynb'
TARGET=ROOT/'NGC628_PHANGS_window_projection_refit.ipynb'


def cell(kind,text,tag):
    result={'cell_type':kind,'id':tag,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':result.update(execution_count=None,outputs=[])
    return result


def main():
    original=json.loads(SOURCE.read_text())
    nb=copy.deepcopy(original)
    nb['cells']=nb['cells'][:14]
    nb['cells'][0]=cell('markdown',r'''# NGC 628: corrected finite-window KTZ refit

This refit addresses the sustained underprediction of the 0.6–2.5-kpc positive
shoulder. The metallicity map, full pair estimator, selection/noise pipeline,
1-kpc H-alpha template, seven physical coordinates, and physical prior are
unchanged. It does **not** choose a smoothing scale to improve the fit.

The previous approximation projected a stationary-equivalent *normalised*
curve. The new calculation instead builds the covariance with each measured
pair midpoint and endpoint annulus, applies $P C P^{\mathsf T}$, adds all four
projected raw terms, and only then divides by the common projected endpoint
second moment. All valid pairs across the full observed window enter the
annular averages, even when their separation exceeds the 5-kpc fit limit.
The deterministic mean field is projected as a map, not as an isotropic curve.

This fixes the order of operations; it does **not** eliminate the local-source
approximation or guarantee that the fitted posterior median will reproduce
the shoulder. The new responses are independently tested against dense
matrix projection in `scripts/test_phangs_window_projection.py`.

Old runs remain in `outputs/implications/NGC628_halpha_map_systematic/`.
All revised products are isolated in `outputs/implications/NGC628_window_projection_refit/`.
Only a converged, numerically validated refit should replace the manuscript
curves. The new median curves are not maximum-likelihood curves.
''','window-refit-introduction')
    nb['cells'][1]['source']=''.join(nb['cells'][1]['source']).replace(
        'OUTPUT_DIR = OUTPUT_ROOT / "NGC628_halpha_map_systematic"',
        'OUTPUT_DIR = OUTPUT_ROOT / "NGC628_window_projection_refit"',
    ).splitlines(keepends=True)
    s=''.join(nb['cells'][11]['source'])
    start=s.index('from scripts.phangs_annular_transfer import')
    end=s.index('# The paired dynamic-minus-fixed mean',start)
    s=s[:start]+'# Window projection is now applied to raw second moments in section 5.\n\n'+s[end:]
    nb['cells'][11]['source']=s.splitlines(keepends=True)
    for c in nb['cells']:
        if c['cell_type']=='code':
            c['execution_count']=None;c['outputs']=[];c['metadata'].pop('execution',None)
    nb['cells'] += [cell('markdown',r'''## 5. Corrected raw-moment window model and new MCMC

For the connected local model we retain
$C_{{\rm 2p},ij}=[\lambda_0(R_m)+\eta\lambda_0(R_m)g(\boldsymbol x_m)]K(r_{ij})$,
and $C_{{\rm NP},ij}\propto\lambda_0(R_m)^2 K_{\rm NP}(r_{ij})$.
The mean component is the outer product of
$P[\lambda_0(R)(1+\eta g)]$. Its axisymmetric, cross, and template-square pieces
are evaluated together; they are not additional fit parameters.

The two projected raw responses (pair numerator and arithmetic endpoint
second moment) are linear in the radial covariance kernel. A fine 10-pc
kernel grid below 0.6 kpc resolves injection-scale curvature; 60-pc nodes
continue beyond that point across the full window. The five original radial
rate nodes are retained. Projection is exact for these represented kernels
and adopted midpoint covariances, not for the full global source integral.

The complete KTZ posterior is newly sampled with 64 walkers, 10,000 burn-in
steps and at least 50,000 total steps, extending to the original 150,000-step
cap when max split Rhat exceeds 1.05. KT18 is also refitted using the same
full-window projection. Original posterior draws provide dispersed starting
positions only; they are never substituted for a new posterior.

An exploratory shoulder posterior-predictive check includes both parameter
uncertainty and a replicated correlated-observation draw. This is distinct
from the plotted 16–84% interval of the mean curve, and is not a confirmatory
test because the shoulder was identified after inspecting the data.
''','window-refit-method'),cell('code',r'''from scripts.refit_ngc628_window_projection import run_refit
INPUT_DIR = OUTPUT_ROOT / "NGC628_halpha_map_systematic"
previous_data = np.genfromtxt(INPUT_DIR / "ngc628_residual_covariance_diagnostics.csv", delimiter=",", names=True)
np.testing.assert_allclose(xi_observed, previous_data["xi_centered"], rtol=1e-10, atol=1e-12)
np.testing.assert_allclose(np.diag(total_covariance_raw), previous_data["total_error"]**2, rtol=1e-8, atol=1e-12)
assert np.all(fit_mask), "This refit expects all 83 original positive bins."
refit_summary = run_refit(INPUT_DIR, OUTPUT_DIR, fit_covariance, total_covariance_raw)
display(refit_summary)
''','window-refit-run'),cell('code',r'''display(Image(filename=str(OUTPUT_DIR / "ngc628_window_projection_comparison.png")))
display(Image(filename=str(OUTPUT_DIR / "ngc628_window_projection_components.png")))
display(Table.read(OUTPUT_DIR / "ngc628_transport_posterior.csv", format="ascii.csv"))
display(Markdown("**Convergence status:** " + refit_summary["status"] +
                 ". The previous curves are not replaced automatically."))
''','window-refit-visualisation')]
    TARGET.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    from scripts.embed_ngc628_window_refit_diagnostics import embed_diagnostics
    embed_diagnostics()
    note='\n\n> **NGC628 window-refit status:** the stored fit below underpredicts the positive shoulder. A corrected raw-second-moment projection is being refitted in [NGC628_PHANGS_window_projection_refit.ipynb](NGC628_PHANGS_window_projection_refit.ipynb). This old result is retained as a control, not a successful reproduction of that feature.\n'
    source=''.join(original['cells'][0]['source'])
    if '**NGC628 window-refit status:**' not in source:
        original['cells'][0]['source']=(source+note).splitlines(keepends=True)
        SOURCE.write_text(json.dumps(original,ensure_ascii=False,indent=1)+'\n')
    print(TARGET)


if __name__=='__main__':main()
