"""Build a visible, staged source-integral notebook without replacing old fits."""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'NGC628_PHANGS_halpha_map_systematic.ipynb'
TARGET = ROOT/'NGC628_PHANGS_source_integral_refit.ipynb'


def cell(kind, text, tag):
    result = {'cell_type': kind, 'id': tag, 'metadata': {}, 'source': text.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def cnp_notebook_cells():
    return [cell('markdown', r'''## 12. Analytic CNP regression control — not another physical component

For uniform source intensity, Gaussian source excess reduces exactly to a
triangular age-sum integral. With $z=s/t_*+s'/t_*$, its weight is $z$ on
$0<z<1$ and $2-z$ on $1<z<2$. The Gaussian endpoint covariance is
$$ [\ell_{\rm src}^2+2\sigma^2+2\kappa t_*z]I+2B. $$
The full elliptical beam is retained. Project the raw covariance moments on
the actual footprint before taking any ratio.

Multiplication by endpoint radial rates is **only a cheap regression control**,
not the physical variable-rate nonlocal CNP. The next step learns its
deterministic difference from $T D_{\lambda_0} E D_{\lambda_0}T^{\mathsf T}$ using
source integrals. Both pieces represent **one CNP**; neither is fitted
separately to galaxy data. No physical parameter is added to the seven coordinates.
''', 'source-integral-cnp-control-method'), cell('code', r'''import json
from pathlib import Path
from IPython.display import display
from scripts.phangs_cnp_baseline import test_gaussian_control
from scripts.build_ngc628_cnp_pilot import build_gaussian_control
diagnostic_output = Path.cwd() / "outputs/implications/NGC628_source_integral_refit"
cnp_gaussian_algebra = test_gaussian_control()
(diagnostic_output / "ngc628_cnp_control_algebra.json").write_text(json.dumps(cnp_gaussian_algebra, indent=2)+"\n")
display(cnp_gaussian_algebra)
cnp_control_summary = build_gaussian_control(Path.cwd(), diagnostic_output)
display(cnp_control_summary)
''', 'source-integral-cnp-control-run'), cell('markdown', r'''## 13. Nonlocal CNP source-response pilot and independent holdouts

Training covers the existing physical shape ranges in $\log(\kappa t_*)$,
$\log\sigma$, $h_R^{-1}$ and $\log\ell_{\rm src}$, with unchanged transport-grid
limits. Sixty-four space-filling points plus the stored exact reference point
train the numerical interpolation; 16 separately generated points are held out.
**Every observed pixel/pair** enters each source integral. These are theoretical
parameter-space samples, not a subset of the metallicity pixels.

For conditioning, the numerical residual is scaled by the uniform-source
zero-lag covariance. The observed metallicity correlation is never a training
target. Holdouts check positive second moments, the projected CNP-only shape and
complete-component mixtures over broad amplitude settings. The CNP-only shape
is a numerical diagnostic, not another fitted KTZ model.

Absolute correlation tolerance: 0.002. A failed pilot is **not ready for
inference**; no observational noise floor or narrower astrophysical prior is
introduced to pass it. Passing these points is not full-domain/source-completion
or native-line-selection validation. No posterior is automatically substituted.

Each exact point is checkpointed and code/input fingerprinted. Progress is in
`ngc628_cnp_pilot_progress.json`, with endpoint details in
`ngc628_cnp_pilot_integral_progress.json`. This stage may take hours.
''', 'source-integral-cnp-pilot-method'), cell('code', r'''from pathlib import Path
from IPython.display import Image, display
from scripts.build_ngc628_cnp_pilot import run_cnp_pilot
diagnostic_output = Path.cwd() / "outputs/implications/NGC628_source_integral_refit"
cnp_pilot_validation = run_cnp_pilot(Path.cwd(), diagnostic_output)
display(cnp_pilot_validation)
display(Image(filename=str(diagnostic_output / "ngc628_cnp_pilot_holdouts.png")))
''', 'source-integral-cnp-pilot-run')]


def main():
    original = json.loads(SOURCE.read_text())
    nb = copy.deepcopy(original)
    nb['cells'] = nb['cells'][:8]
    nb['cells'][0] = cell('markdown', r'''# NGC 628: global source-integral implementation and validation

**Stage:** numerical implementation and real-data reference calculation,
**not a completed new seven-parameter posterior**. Old fits are retained.
All new products go to `outputs/implications/NGC628_source_integral_refit/`.

This notebook implements the next correction after the raw-window projection
refit. The actual source field is transported before annular projection, the
same-event integral retains source smoothing instead of a midpoint replacement,
and the full elliptical face-on beam covariance is added to each endpoint
kernel. The distinct-event term remains one $C_{\rm NP}$ with axisymmetric
source weights only; its nonlocal radial response is a numerical extension of
the locally uniform leading term, **not** a new arm-coupled term.

The observed abundance construction is held fixed as a control. Metallicity
holes stay NaN and every valid pixel pair is used. Source preparation separates
geometric coverage, detected star-forming cells, low-EW cells assumed quiet in
the chosen tracer, and unclassified covered cells. Unknown cells are not
asserted to have zero source rate. Fixed H-alpha completion and radial
extrapolation are documented assumptions, not recovered observations.

The final real-data calculation fixes the same seven coordinates at one actual
joint draw from the previous posterior. Its curve is explicitly a **reference
point**, not a posterior median or a maximum-likelihood fit. No new posterior
16–84% interval is fabricated. Main scientific fit comparisons must continue
to use posterior medians once all corrections are validated and resampled.

Before a new MCMC: validate precision across the physical prior, construct and
test a fast response emulator, run native multi-line injection/recovery, obtain
independently restarted ensembles and target-statistic ESS, and rerun the same
algorithm for the four galaxies and three geometry controls. Current manuscript
figures/tables are not replaced by this NGC628-only diagnostic.
''', 'source-integral-introduction')
    nb['cells'][1]['source'] = ''.join(nb['cells'][1]['source']).replace(
        'OUTPUT_ROOT / "NGC628_halpha_map_systematic"',
        'OUTPUT_ROOT / "NGC628_source_integral_refit"').splitlines(keepends=True)
    nb['cells'][4] = cell('markdown', '## 2. Retain the catalogue-calibrated face-on coordinate transform\n\nThe catalogue sets the external spatial transform and reference radius only. No metallicity residual is used to fit the source geometry. The revised padded source is constructed in section 4.\n', 'source-integral-geometry')
    s = ''.join(nb['cells'][7]['source'])
    s = s[:s.index('# Build a deterministic source field')]
    s += r'''
# No legacy selected-pixel source smoothing is used in this notebook.
native_pixel_area_kpc2 = (native_pixel_scale_arcsec/206265.*DISTANCE_MPC*1e3)**2
np.savez_compressed(OUTPUT_DIR / "ngc628_observed_abundance_control.npz",
    x_grid_kpc=pixel_X, y_grid_kpc=pixel_Y, valid_mask=pixel_valid_grid,
    residual_dex=np.where(pixel_valid_grid, pixel_residual_centered_grid, np.nan),
    metallicity_12_log_oh=np.where(pixel_valid_grid, metallicity_grid, np.nan))
BASELINE_DIR = OUTPUT_ROOT / "NGC628_window_projection_refit"
with np.load(BASELINE_DIR / "ngc628_muse_scal_metallicity_residual_pixel_map.npz") as z:
    assert np.array_equal(pixel_valid_grid, z["valid_mask"])
    np.testing.assert_allclose(pixel_residual_centered_grid[pixel_valid_grid],
                              z["residual_dex"][pixel_valid_grid], rtol=1e-10, atol=1e-12)
print("Observed control map exactly matches the previous validated input;", pixel_valid_grid.sum(), "valid cells.")
'''
    nb['cells'][7]['source'] = s.splitlines(keepends=True)
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['execution_count'] = None
            c['outputs'] = []
            c['metadata'].pop('execution', None)
    nb['cells'] += [cell('markdown', r'''## 4. Coverage-aware padded H-alpha source

The source grid extends at least 1.62 kpc beyond the analysed outer radius.
The same PSF-matched released lines and calibrated continuum proxy are sampled
on that grid. A detected SF source requires the four BPT/Balmer lines above the
nominal S/N threshold, SF BPT classification, and EW at least 6 A. Low-EW cells
are assumed quiet only when EW plus twice the propagated/proxy-scatter error
is below 6 A with adequate continuum S/N. All other covered cells remain
unclassified. Thus the smoothing denominator includes supported quiet area,
not just selected bright cells. Mask-aware interpolation over unknown source
cells is recorded explicitly; no metallicity is filled.

The radial tracer sums qualified SF flux and divides by geometrically covered
disc area, **not** by selected SF area. Comparing with division by full ring
area diagnoses coverage-induced slope changes. Angular-block bootstrap keeps
flux and coverage together. Extrapolating covered surface brightness to missing
sectors still assumes representative coverage and is not a completeness proof.
An unknown-cell flux sensitivity scenario uses median detected attenuation;
it is not a confidence bound. The same fixed 1-kpc template smoothing remains.
''', 'source-integral-source-method'), cell('code', r'''from scripts.phangs_source_inputs import prepare_sources
source_input_summary = prepare_sources(globals(), OUTPUT_DIR)
display(source_input_summary)
''', 'source-integral-prepare'), cell('code', r'''from matplotlib.colors import ListedColormap, LogNorm
with np.load(OUTPUT_DIR / "ngc628_padded_source_inputs.npz") as z:
    source_axis = z["source_axis_kpc"]
    source_extent = [source_axis[0]-.03, source_axis[-1]+.03]*2
    classes = np.full(z["geometric_coverage"].shape, np.nan)
    classes[z["unclassified_covered"]] = 0
    classes[z["assumed_quiet"]] = 1
    classes[z["detected_SF"]] = 2
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    ha = z["selected_Halpha_flux"]
    positive = ha[np.isfinite(ha)&(ha > 0)]
    im = axes[0, 0].imshow(ha, origin="lower", extent=source_extent, cmap="magma",
        norm=LogNorm(*np.percentile(positive, [2, 99])))
    fig.colorbar(im, ax=axes[0, 0], label="qualified extinction-corrected H-alpha flux")
    axes[0, 0].set_title("Padded detected-SF tracer (no metal filling)")
    im = axes[0, 1].imshow(classes, origin="lower", extent=source_extent,
        cmap=ListedColormap(["#e6ab02", "#bdbdbd", "#1b9e77"]), vmin=-.5, vmax=2.5)
    cb = fig.colorbar(im, ax=axes[0, 1], ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["unknown covered", "assumed quiet", "detected SF"])
    axes[0, 1].set_title("White = no geometric coverage")
    im = axes[1, 0].imshow(np.where(z["modulation_support"], z["source_modulation"], np.nan),
        origin="lower", extent=source_extent, cmap="RdBu_r", vmin=-1, vmax=2)
    fig.colorbar(im, ax=axes[1, 0], label=r"fixed $g(\boldsymbol{x})$")
    axes[1, 0].set_title("1-kpc source template, including stated completion")
    extent = [pixel_centres[0]-.03, pixel_centres[-1]+.03]*2
    lim = np.percentile(np.abs(pixel_residual_centered_grid[pixel_valid_grid]), 99)
    im = axes[1, 1].imshow(np.where(pixel_valid_grid, pixel_residual_centered_grid, np.nan),
        origin="lower", extent=extent, cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=axes[1, 1], label="abundance residual [dex]")
    axes[1, 1].set_title("Actual correlation input: missing cells remain NaN")
    for ax in axes.ravel():
        for radius in [R_MIN_KPC, R_MAX_KPC]:
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, ec="0.2", lw=.8))
        ax.set(xlabel="face-on x [kpc]", ylabel="face-on y [kpc]")
    fig.savefig(OUTPUT_DIR / "ngc628_source_coverage_and_control_map.png", bbox_inches="tight")
    display_figure(fig)
profile = np.genfromtxt(OUTPUT_DIR / "ngc628_source_rate_coverage.csv", delimiter=",", names=True)
display(Table(profile))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
r = profile["radius_kpc"]
axes[0].plot(r, profile["qualified_Halpha_flux"]/profile["full_area_kpc2"], "o--", color="0.4", label="full ring area")
axes[0].plot(r, profile["covered_area_surface_density"], "o-", color="#d95f02", label="geometrically covered area")
axes[0].fill_between(r, profile["covered_area_surface_density"],
    profile["covered_area_surface_density"]+profile["unknown_flux_sensitivity_scenario"],
    color="#e6ab02", alpha=.2, label="unknown-flux sensitivity (not CI)")
axes[0].set(xlabel="radius [kpc]", ylabel="H-alpha flux per disc area", yscale="log", title="Source-rate coverage diagnostic")
axes[0].legend(frameon=False, fontsize=7)
axes[1].plot(r, profile["covered_area_kpc2"]/profile["full_area_kpc2"], "o-", color="#1b9e77")
axes[1].set(xlabel="radius [kpc]", ylabel="covered/full area", ylim=(0, 1.05), title="Coverage is not the SF-selected fraction")
fig.savefig(OUTPUT_DIR / "ngc628_source_rate_coverage.png", bbox_inches="tight")
display_figure(fig)
''', 'source-integral-source-visual'), cell('markdown', r'''## 5. Physical response and independent numerical checks

Let $G_u$ be the sampled Gaussian operator from all padded injection sites to
all valid observed positions, with endpoint covariance
$V_u=\ell_*^2u I+B_{\rm sky\to faceon}$ and $u\in[\epsilon^2,1]$.
Here $B=\sigma_{\rm sky}^2 A^{\mathsf T}A$ for the measured affine transform;
**the full beam covariance is added**, not half of it. Uniform physical ages
give $w(u)du=du/(1-\epsilon^2)$, evaluated by log-$u$ Gauss quadrature.
Define $T=\int w(u)G_u\,du$, $\lambda=\lambda_0(1+\eta g)$ and
$\lambda_0=\exp[-(R-R_{\rm ref})/h_R]$. In dimensionless covariance units,

$$ C_{\rm 2p}=\frac{4\pi\kappa t_*}{\Delta A}
       \int w(u)G_u D_\lambda G_u^{\mathsf T}\,du,\quad
   C_{\rm M}=\beta(T\lambda)(T\lambda)^{\mathsf T},\quad
   C_{\rm NP}=(\beta A_{\rm src})
       T D_{\lambda_0} E_{\ell_{\rm src}}D_{\lambda_0}T^{\mathsf T}. $$

$\Delta A$ is source-pixel area, $D$ denotes a diagonal source-rate matrix,
and $E$ has Gaussian isotropic source-pair spectrum. The latter is represented
spectrally so correlation lengths below source-grid size do not create a
spurious finite point variance. After the source integrals, apply the actual
annular projector $P$ to all terms. Sum the raw pair numerators and arithmetic
endpoint second moments before taking their ratio. No covariance at $r>5$
kpc is dropped from the annular averages. The observed $r=0$ anchor is fixed
to one and excluded from fitting.

Tests compare with dense $PCP^{\mathsf T}$, the uniform KT18/CNP limits,
transported mean harmonics and finite-$q$ same-event attenuation. A separate
small-map check compares 16/32 age nodes, 60/30-pc source grids and padding.
These verify the numerical response at test points, not across the full prior.
''', 'source-integral-response-method'), cell('code', r'''from scripts.test_phangs_source_integral import run_tests
numerical_validation = run_tests()
(OUTPUT_DIR / "ngc628_source_integral_validation.json").write_text(json.dumps(numerical_validation, indent=2)+"\n")
display(numerical_validation)
''', 'source-integral-unit-tests'), cell('markdown', r'''## 6. Finite-map normalisation recovery — limited scope

On a fixed irregular small map, draw Gaussian fields with known connected
covariance plus the transported deterministic mean. Every valid pixel pair is
used in every realisation. Compare $E[N/D]$ with $E[N]/E[D]$ and report Monte
Carlo error. The shaded interval here is the **single-map estimator** spread,
not a posterior interval. This experiment does not supply an empirical bias
correction for NGC628, and is not native multi-line/selection recovery or SBC.
''', 'source-integral-recovery-method'), cell('code', r'''from scripts.phangs_source_ratio_recovery import run_ratio_recovery
normalisation_recovery = run_ratio_recovery(OUTPUT_DIR)
display(normalisation_recovery)
display(Image(filename=str(OUTPUT_DIR / "ngc628_small_map_ratio_recovery.png")))
''', 'source-integral-ratio-recovery'), cell('markdown', r'''## 7. Full real-data reference calculation (may take a long time)

The coordinate vector is one actual prior-run joint draw nearest its internal
coordinate medians. Hold it fixed while replacing the source/beam/integration
implementation. This is not resampling the posterior. The same old covariance
and measured noise response are frozen controls, not newly validated errors.
The component panel uses the latent signal and noise-adjusted data, so its four
stacked terms sum exactly to the displayed latent total without displaying a
measurement-noise component. The comparison panel includes the measured noise
response. No posterior band is drawn for this fixed point.

Progress: `ngc628_source_integral_progress.json` and the runner log in the
isolated output folder. Completed notebook cells are saved immediately. The
real-data reference result must not be reported as a successful new fit.
''', 'source-integral-reference-method'), cell('code', r'''from scripts.reference_ngc628_source_integral import run_reference_point
reference_point_summary = run_reference_point(ROOT, OUTPUT_DIR)
display(reference_point_summary)
display(Image(filename=str(OUTPUT_DIR / "ngc628_global_source_reference_point.png")))
display(Markdown("**Next gate:** full-prior numerical convergence and emulator checks; native-flux recovery; independent MCMC restarts. No new posterior has been substituted for the existing fits."))
''', 'source-integral-reference-run'), cell('markdown', r'''## 8. Independent-ensemble inference utility — analytic test only

Four separately seeded/initialised ensemble runs are compared. Rhat uses the
same selected walker trajectory across those **independent restarts**, never
treating interacting walkers from one run as independent chains. Target
functions include squared radius in the closed-form Gaussian test; time-series
MCSE/ESS are computed from independent ensemble-average functions. An
artificially shifted restart must be rejected. This is a sampler test, not
NGC628 posterior convergence or full simulation-based calibration.

These appended diagnostic cells can run in a separate kernel while preserving
the executed source/reference cells above.
''', 'source-integral-independent-method'), cell('code', r'''import json
from pathlib import Path
from IPython.display import display
from scripts.phangs_independent_inference import test_gaussian_inference
independent_sampler_validation = test_gaussian_inference()
diagnostic_output = Path.cwd() / "outputs/implications/NGC628_source_integral_refit"
(diagnostic_output / "ngc628_independent_sampler_validation.json").write_text(json.dumps(independent_sampler_validation, indent=2)+"\n")
display(independent_sampler_validation)
''', 'source-integral-independent-test'), cell('markdown', r'''## 9. Real-map quadrature/source-grid refinement (long calculation)

Retain all 15,201 observed pixels and the identical joint reference coordinate.
Compare 32 versus 64 age nodes at 60-pc source spacing, then 60 versus 30-pc
source integration grids at 32 nodes. The latter refines the **same saved
source template** bilinearly; it adds no observations. The observer grid,
source completion, beam, noise response and annular projector stay fixed.
Thresholds concern absolute correlation changes, not percentages.

Passing these checks validates this point only, not the full physical prior or
the unknown-source completion. The full prior, source boundary/quiet assumptions
and native-flux propagation still require separate tests before new MCMC.
Progress is in `ngc628_source_precision_progress.json`; each completed variant
is cached separately, so long runs resume without repeating completed variants.
''', 'source-integral-precision-method'), cell('code', r'''from pathlib import Path
from IPython.display import Image, display
from scripts.audit_ngc628_source_integral import run_precision_audit
diagnostic_output = Path.cwd() / "outputs/implications/NGC628_source_integral_refit"
real_map_precision_audit = run_precision_audit(Path.cwd(), diagnostic_output)
display(real_map_precision_audit)
display(Image(filename=str(diagnostic_output / "ngc628_source_precision_audit.png")))
''', 'source-integral-precision-run'), cell('markdown', r'''## 10. Equivalent convolution integrator and real-map benchmark

For each age, multiplication by $G_u$ is a discrete Gaussian convolution, so
the source-matrix contractions can be evaluated without constructing a large
pixel-by-source sparse matrix. The distinct-event response is still exactly
$T D_{\lambda_0} E D_{\lambda_0}T^{\mathsf T}$ for the represented source grid:
for endpoint $j$, form $w_j=T_j\lambda_0$, convolve it with $E$, multiply by
$\lambda_0$, and convolve with $T$ before sampling **every** observed endpoint.
Row blocking limits memory; it does not subsample pixels or pairs.

The real-map benchmark must reproduce the original raw four-term moments and
the complete fixed-coordinate KTZ curve before a reusable response table is
built. Gaussian truncation, grid, source completion and physical amplitudes
remain unchanged. This is numerical equivalence, not a new fit.
''', 'source-integral-fft-method'), cell('code', r'''import json
from pathlib import Path
from IPython.display import display
from scripts.test_phangs_source_fft import run_fft_tests
from scripts.test_phangs_age_response import run_age_response_tests
from scripts.build_ngc628_age_library import run_fft_benchmark
diagnostic_output = Path.cwd() / "outputs/implications/NGC628_source_integral_refit"
convolution_checks = run_fft_tests()
single_age_checks = run_age_response_tests()
(diagnostic_output / "ngc628_fft_algebra_validation.json").write_text(json.dumps(
    {"convolution": convolution_checks, "single_age": single_age_checks}, indent=2)+"\n")
display(single_age_checks)
fft_real_map_benchmark = run_fft_benchmark(Path.cwd(), diagnostic_output)
display(fft_real_map_benchmark)
''', 'source-integral-fft-benchmark'), cell('markdown', r'''## 11. Reusable physical-age responses for CP and transported mean

This table contains the same-event covariance density and transported mean
fields at 65 intrinsic-variance nodes and nine external radial-rate slopes.
The variance range $0.015^2\le v\le3.2^2$ kpc$^2$ covers the existing injection
floor and transport-grid bound; $-0.35\le h_R^{-1}\le0.35$ kpc$^{-1}$ is unchanged.
For broad kernels the source grid expands to cover the required Gaussian
support. Newly included cells use the **same stated** axisymmetric exponential
extrapolation with $g=0$ outside measured support; this adds no observations or
new free parameters. Uncertain outer-disc source completion remains a separate
systematic, particularly for broad transport kernels.

Full pair-midpoint weights are accumulated once per variance node, not sampled,
then reused for all slopes. If $a=\sigma^2$ and $b=\ell_*^2$, uniform physical
ages imply
$$ C_{\rm 2p}=2\pi\int_a^b C_{\rm age}(v)\,dv,\qquad
   P T\lambda=\frac{1}{b-a}\int_a^b P G_v\lambda\,dv. $$
The prefactor follows from $\kappa t_*=(b-a)/2$. Splines are integrated in
**physical variance**, not log variance, even though nodes are log-spaced.
The output retains unnormalised covariance moments and projected mean fields.
Pair normalisation is applied only after their physical amplitudes are combined.

The table is checked against the known age-integrated reference point and
additional direct single-age points that are not knots. **It does not supply
a variable nonlocal CNP emulator and is not a complete KTZ fitting likelihood.**
For the reference check only, CNP is held at its independently integrated exact
value. No Poisson-only KTZ is fitted. Independent posterior resampling still
awaits the complete validated response and observing-recovery checks.

Progress: `ngc628_age_library_progress.json`. Each completed width node is
fingerprinted and saved in `single_age_response_nodes/`; interrupted table
generation resumes from those nodes without repeating completed work.
''', 'source-integral-age-library-method'), cell('code', r'''from pathlib import Path
from IPython.display import Image, display
from scripts.build_ngc628_age_library import build_age_library
diagnostic_output = Path.cwd() / "outputs/implications/NGC628_source_integral_refit"
age_response_library_validation = build_age_library(Path.cwd(), diagnostic_output)
display(age_response_library_validation)
display(Image(filename=str(diagnostic_output / "ngc628_age_library_validation.png")))
''', 'source-integral-age-library-run')] + cnp_notebook_cells()
    # Keep the bounded follow-on cells when regenerating this notebook.
    # Do not run the generator while a notebook execution writer is active.
    from scripts.refine_ngc628_cnp_response import notebook_cells as refinement_cells
    for refinement in refinement_cells():
        nb['cells'].append(cell(refinement['cell_type'], refinement['source'], refinement['id']))
    if TARGET.exists():
        old = json.loads(TARGET.read_text())
        by_id = {c['id']: c for c in old['cells']}
        for c in nb['cells']:
            previous = by_id.get(c['id'])
            if (previous and c['cell_type'] == 'code' and
                    ''.join(c['source']) == ''.join(previous['source'])):
                c['outputs'] = previous.get('outputs', [])
                c['execution_count'] = previous.get('execution_count')
    TARGET.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n')
    for old_path in [SOURCE, ROOT/'NGC628_PHANGS_window_projection_refit.ipynb']:
        old_nb = json.loads(old_path.read_text())
        intro = ''.join(old_nb['cells'][0]['source'])
        marker = '**NGC628 global-source correction:**'
        if marker not in intro:
            note = '\n\n> '+marker+' the next-stage implementation, source-coverage diagnostics and fixed-point global-integral checks are in [NGC628_PHANGS_source_integral_refit.ipynb](NGC628_PHANGS_source_integral_refit.ipynb). They are not a completed new posterior. This older fit remains a control; interacting-walker split Rhat alone is not an independent-restart convergence test.\n'
            old_nb['cells'][0]['source'] = (intro+note).splitlines(keepends=True)
            old_path.write_text(json.dumps(old_nb, ensure_ascii=False, indent=1)+'\n')
    print(TARGET)


if __name__ == '__main__':
    main()
