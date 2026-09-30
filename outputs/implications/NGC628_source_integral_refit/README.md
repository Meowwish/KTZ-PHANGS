# NGC 628: source-integral correction and validation

This folder is isolated from the previously fitted products. It does not
contain a newly fitted seven-parameter posterior. The companion notebook is
`NGC628_PHANGS_source_integral_refit.ipynb` in the workspace root.

## Implemented

- Global source-position integrals for the same-event covariance and the
  age-integrated mean field; no midpoint-rate replacement or untransported
  source-map mean.
- Full observing-beam covariance, transformed from sky to the measured face-on
  affine coordinates. Its variance is not halved. The linear-field beam model
  is still an approximation to nonlinear multi-line abundance propagation.
- One complete KTZ decomposition. The nonlocal distinct-event response uses
  the axisymmetric radial source rate only. It is a numerical extension of the
  locally uniform leading term, not a new arm-coupled CNP contribution.
- Explicit padded source grid, detected-SF/assumed-quiet/unknown/coverage
  classes, covered-area radial rate and angular-block uncertainty. Unknown
  source completion, source selection and radial extrapolation remain stated
  assumptions. No missing metallicity is filled.
- Exhaustive ordered pixel-pair statistics after actual annular projection.
  Projection contractions include all separations, not just fitted bins.
- Dense-matrix, uniform-source, finite-q harmonic and beam tests; small-map
  age/grid/padding convergence check. These are not full-prior precision tests.
- Covariance-level small-map normalisation recovery, explicitly distinguished
  from native-flux recovery or posterior SBC.
- Independent-ensemble sampler/diagnostics utility, tested on a closed-form
  Gaussian posterior. Interacting walkers are not independent chains for Rhat.

## Real-data reference calculation

`ngc628_source_integral_progress.json` records the live stage. On completion,
`ngc628_global_source_reference_point.json` and the PNG/CSV contain the new
response at one fixed actual joint draw from the previous posterior. The old
noise response and covariance are held fixed as controls. A chi-square or
shoulder discrepancy at this point is NOT a new fit-quality result.

The component panel is latent signal: its four stacked terms sum exactly to
its total. Noise-adjusted data are marked accordingly. The comparison panel
includes the frozen measurement response. No new posterior band is inferred
from one point. A cached point must match its source/coordinate fingerprint;
changed numerical assumptions require a new output directory/version.

Raster covered area divided by analytic annulus area can exceed one by a few
parts per thousand in fully covered rings because pixel-centre inclusion and
analytic areas differ. Do not interpret that as physical coverage above 100%.

## Required gates before publication refit

1. Grid/age/Gaussian-cutoff/padding convergence over the physical prior, source
   completion sensitivity, and a fast response emulator independently checked
   against this reference integrator.
2. Native multi-line injection through PSF, abundance conversion, flux noise,
   data-dependent selection and the same exhaustive pair estimator. The small
   Gaussian fixed-mask experiment does not replace this gate.
3. Independent MCMC ensemble restarts and target-function ESS/MCSE for transport
   coordinates, shoulder/tail predictions and component allocation. Gaussian
   utility tests are not an NGC628 convergence result or a physical validation.
4. Source-pair excess measured conditional on the same deterministic intensity,
   before interpreting broad CNP as association physics.
5. The same validated implementation for NGC628, NGC1566, NGC4254 and NGC4303,
   and all three externally fixed geometries; only then update common paper
   figures/tables using posterior medians and total-only 16–84% bands.

No source amplitude, physical prior, additional flow/anisotropy parameter or
maximum-likelihood substitute is chosen to force a better-looking result.

## Convolution and reusable response stage

The real-map 32/64-age and 60/30-pc source refinements completed successfully
at the reference point (maximum changes 4.27e-10 and 1.45e-4 respectively).
`phangs_source_fft.py` evaluates the same source covariance by convolution
instead of large sparse source matrices. It includes nonlocal CNP unchanged.
`ngc628_fft_real_map_benchmark.json` records its real-map equivalence and runtime.

`ngc628_age_library_progress.json` records generation of the reusable **CP/mean**
single-age response table. Nodes cover the existing source-width/rate bounds,
expanding the computational source grid instead of shrinking physical priors.
New source cells retain the explicit radial extrapolation and zero modulation
outside supported H-alpha. This completion is an assumption, not real data.
Every width node is checkpointed and code/input fingerprinted.

The physical-variance integral follows dv/(ell^2-sigma^2), not a uniform-log-age
measure. Tests use independent non-knot responses. The table is NOT a full KTZ
likelihood: a validated variable-CNP response is still required. No Poisson-only
KTZ fit is introduced. Do not describe table generation as completed MCMC.

The 65-width by 9-slope CP/mean table is now complete. The reference-point
integrated curve error is 3.70e-6 with CNP held exact solely for that numerical
test; the largest of four withheld single-age probe errors is 3.69e-5. These
tested-point results pass the 0.002 precision gate but do not establish
prior-wide accuracy, source-completion validity, or a new galaxy posterior.

The real-map convolution benchmark passed (maximum curve difference 6.66e-16;
about 63 s vs 138 s end-to-end at the known reference point). The bounded
follow-on runner `run_ngc628_cnp_pilot_notebook.py` waits for the age-library
runner to finish saving and pass its tested-point precision gate, then appends
and executes CNP control/pilot cells without overwriting previous code/outputs.
It does not automatically start MCMC.

The CNP pilot uses 64 source-response training points plus the stored reference
and 16 independently generated holdouts. Its analytic control and numerical
residual represent ONE CNP, never separately fitted to the observed curve.
No extra physical term/parameter is added. Observing pixels/pairs stay exhaustive;
parameter-space points are not subsampled metallicity pixels. Failed precision
or positivity checks prevent promotion to inference. Native-flux recovery is
still a separate required gate.

Follow-on progress: `ngc628_cnp_pipeline_progress.json`, then
`ngc628_cnp_control_progress.json` / `ngc628_cnp_pilot_progress.json`.
The CNP Gaussian control table and full nonlocal source-integral pilot are
complete. The pilot FAILED the predefined 0.002 numerical gate: its largest
pure-CNP shape error is 0.02550, and its largest mixture-probe error is 0.02540.
Its files and executed notebook outputs are preserved as failed validation,
not promoted to a likelihood. No new MCMC or manuscript replacement has started.

## Bounded CNP refinement (current stage)

The appended notebook section 14 and isolated `cnp_refinement_v1/` preserve
the failed pilot and reuse all 81 exact source responses as DEVELOPMENT data.
The pilot's old 16 holdouts are explicitly no longer independent validation.
The numerical representation reconstructs a single CNP from its projected
shape and positive second moment. Five-fold source-response cross-validation
chooses interpolation coordinates/kernel/slope factor, without observed
metallicity values or fresh validation outputs. Fresh zero-slope tests check
the optional homogeneous-continuum control condition; it is not assumed exact
for discrete source kernels.

Each of at most two stages adds 128 space-filling parameter points, then
freezes the representation before 16 NEW random and 8 NEW limit tests. These
are source-parameter nodes, not subsampled observing pixels. Every exact
response retains all 15201 observing pixels and 115366246 ordered pairs in
positive fitting bins, plus the full annular-projection contractions.

The gate includes second-moment allocation errors: if h=D_fast/D_true, the
maximum error for any positive allocation to the other valid pooled-pair
components is bounded by the CNP shape error plus 2*tanh(abs(log h)/4).
The same 0.002 threshold is retained; no astronomical prior, physical term,
source selection, noise floor or observed data are changed. Failure of stage 1
allows its tests to become stage-2 development, which then uses new tests.
Even passing these finite tests does not replace native-flux recovery or the
remaining prior/grid/source-boundary checks, and never starts MCMC automatically.

Progress is `cnp_refinement_v1/progress.json` (completed source nodes) and
`cnp_refinement_v1/integral_progress.json` (all-endpoint integral progress).
The notebook saves each completed refinement cell, including validation plots.
The runner rejects duplicate launches and resumes fingerprinted source nodes.

```sh
OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/private/tmp/ktz-ngc628-mpl /opt/anaconda3/bin/python -m scripts.run_ngc628_cnp_refinement_notebook
```

Do not run that command while the current refinement is active, or regenerate
the notebook while its execution writer is active.

To run only the new notebook sections while preserving previous outputs:

```sh
OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/private/tmp/ktz-ngc628-mpl /opt/anaconda3/bin/python -u scripts/run_ngc628_response_library_notebook.py
```

The bounded follow-on pilot can be reproduced separately after the preceding
runner has finished (do not start a duplicate while the current job is active):

```sh
OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/private/tmp/ktz-ngc628-mpl /opt/anaconda3/bin/python -u scripts/run_ngc628_cnp_pilot_notebook.py
```

## Reproduce

From the workspace root, with the existing PHANGS downloads:

```sh
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python -m scripts.test_phangs_source_integral
/opt/anaconda3/bin/python -m scripts.make_ngc628_source_integral_notebook
OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/private/tmp/ktz-ngc628-mpl /opt/anaconda3/bin/python -u scripts/run_ngc628_source_integral_notebook.py
```

The notebook runner needs permission to bind local Jupyter kernel ports. It
saves visual outputs after each completed cell. Existing four-galaxy manuscript
products have not been replaced by these diagnostics.
