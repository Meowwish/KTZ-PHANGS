"""Preserve the executed refit while adding reproducible diagnostic cells."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'NGC628_PHANGS_window_projection_refit.ipynb'
OLD = ROOT/'NGC628_PHANGS_halpha_map_systematic.ipynb'
MASK_CONTROL = ROOT/'NGC628_PHANGS_implications.ipynb'


def embed_diagnostics():
    nb = nbformat.read(TARGET, as_version=4)
    stored = {c.get('id'): c for c in nb.cells}
    cells = [c for c in nb.cells if not c.get('id', '').startswith('completed-window-')]
    cells.append(nbformat.v4.new_markdown_cell(r'''## 6. Completed refit: improvements and remaining limitation

The MCMC has finished (complete KTZ max split Rhat = 1.021; KT18 = 1.003).
The projection fix lowers the exploratory 0.6–2.5-kpc shoulder RMSE from
0.0907 to 0.0725, a 20.1% reduction, but **the posterior median still
underpredicts the positive shoulder**. Its mean over those bins is 0.0383,
versus 0.1080 observed and 0.0200 for the old approximation. These are means
of the pointwise median curves, not medians of a parameter vector.

A separate, constrained, multistart likelihood diagnostic reaches
$\chi^2=4.495$, compared with 7.835 for the posterior-median curve. It largely
reproduces the shoulder (RMSE 0.0198), demonstrating that the adopted model
can produce a more suitable shape. However, its radial slope is at the
bound ($h_R^{-1}=-0.35\,{\rm kpc}^{-1}$, i.e. rate log-slope $+0.35$), about three H-alpha-prior standard
deviations above the independent estimate, and its source-pair length is
2.20 kpc (about the 93rd posterior percentile). Its arm amplitude $\eta$ is 0.584
(about the 3rd percentile). The fitted mean-field and broad distinct-event
contributions remain degenerate. This is not evidence that those particular
physical values have been measured.

The remaining median mismatch therefore involves prior weighting and
posterior volume, not simply a failure to find a suitable likelihood state.
A coordinate change with the correct Jacobian cannot change that posterior.
We do not remove the independent H-alpha constraint, select the 0.5-kpc
smoothing run, condition the posterior on long source lengths, or reduce the
covariance merely to obtain a more attractive curve.

The bins are strongly correlated. The shoulder discrepancy of the median
curve is 1.65 times its covariance-derived standard error; this is a
descriptive residual, not an independent-bin significance or a confirmatory
test. The exploratory replicated-observation upper-tail fraction is 0.108.
The plotted 16–84% region is uncertainty in the **mean model curve**, not a
prediction interval for a new noisy observation.

The separate grey likelihood curve below is labelled as a diagnostic only;
it does not replace the posterior median in the main figures. Dense small-map
tests verify the represented $P C P^{\mathsf T}$ and common denominator.
Eighty further kernel/quadrature checks at posterior draws have maximum
absolute correlation error 0.0047; these do not test rate_log_slope interpolation,
the radial covariance representation, or the retained local-source/WKBJ
approximation. No claim of a global diffusion solution is made.

This corrected calculation is currently isolated to NGC628. The four-galaxy
publication figures and comparison tables have **not** been partially
replaced: the same raw-moment projection must first be applied to the other
galaxies and geometry controls before a common scientific comparison.
''', id='completed-window-interpretation'))
    cells.append(nbformat.v4.new_code_cell('''from scripts.diagnose_ngc628_window_refit import audit_completed_refit
# Recompute diagnostics if running this notebook from top to bottom.
# This does not rerun MCMC or alter the posterior chains.
window_diagnostics = audit_completed_refit()
display(Table.read(OUTPUT_DIR / "ngc628_transport_posterior.csv", format="ascii.csv"))
display(Image(filename=str(OUTPUT_DIR / "ngc628_likelihood_shape_diagnostic.png")))
display(Markdown(
    "**Result:** the window correction improves the shoulder, but the posterior "
    "median remains low. The grey likelihood curve is a shape diagnostic, "
    "not the adopted posterior summary."
))
''', id='completed-window-audit'))
    for c in cells:
        previous = stored.get(c.get('id'))
        if (c.cell_type == 'code' and previous is not None
                and c.source == previous.source):
            c.outputs = previous.outputs
            c.execution_count = previous.execution_count
            c.metadata = previous.metadata
    nb.cells = cells
    nbformat.write(nb, TARGET)
    old = nbformat.read(OLD, as_version=4)
    old.cells[0].source = old.cells[0].source.replace(
        'A corrected raw-second-moment projection is being refitted in',
        'The completed, converged raw-second-moment refit is available in',
    )
    nbformat.write(old, OLD)
    control = nbformat.read(MASK_CONTROL, as_version=4)
    heading = '> **Finite-window method update:**'
    if heading not in control.cells[0].source:
        control.cells[0].source += '''

> **Finite-window method update:** this notebook preserves the historical
> PHANGS-mask control using the stationary-equivalent curve transfer. The
> completed raw-second-moment refit of the primary 1-kpc H-alpha template is
> displayed in [NGC628_PHANGS_window_projection_refit.ipynb](NGC628_PHANGS_window_projection_refit.ipynb).
> It improves, but does not fully reproduce, the positive shoulder. Comparing
> this stored mask fit against that corrected H-alpha fit is not a controlled
> geometry test: both must use the same corrected projection first.
'''
        nbformat.write(control, MASK_CONTROL)
    print(TARGET)


if __name__ == '__main__':
    embed_diagnostics()
