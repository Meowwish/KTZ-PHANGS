# Numerical refinement after the failed CNP pilot

This is not a galaxy fit or a new physical CNP model. The original failed
65-training/16-holdout pilot is retained unchanged in the parent folder. All
81 original teacher responses are now development data. Former holdouts are
not described as independent tests of this refinement.

The notebook's section 14 documents positive-second-moment reconstruction,
five-fold numerical cross-validation, an optional homogeneous-limit factor,
and the any-positive-component-allocation error bound. Physical priors,
source completion, beam, exhaustive observing-pixel pairs, and the original
nonlocal radial-rate CNP definition are unchanged. No arm coupling, new physical
component or fitted amplitude is added. Negative projected CNP tails remain
allowed; positivity applies to its second moment, not its correlation.

At most two stages add 128 source-parameter training nodes each. Each stage
freezes its numerical representation before 16 fresh random holdouts and 8
fresh limit tests, including zero radial slope and source-length/rate boundaries.
If stage 1 fails, its tests may enter stage-2 development; stage 2 uses new
tests. The original absolute xi error threshold remains 0.002.

Files:

- `manifest.json`: code/input fingerprints and bounded schedule.
- `encoding_algebra.json`: synthetic reconstruction and allocation-bound tests.
- `legacy_development_cross_validation.json`: development diagnostics only.
- `progress.json`: completed source nodes and stage state.
- `integral_progress.json`: progress over all 15201 observing endpoints.
- `source_nodes/`: fingerprinted full-source CNP moment checkpoints.
- `stage_*_cross_validation.json`: settings frozen before fresh validation.
- `stage_*_validation.json`: fresh numerical precision results when complete.
- `stage_*_*.png`: source-response validation plots, not posterior/observed fits.
- `runner.log` / `execution_error.txt`: execution log / failure traceback if any.

Passing finite numerical tests alone is not publication validation. Source
completion/grid/boundary sensitivity, native multi-line recovery and independent
galaxy MCMC remain required. The runner never launches MCMC or replaces any
manuscript figure/table. It saves outputs to the existing companion notebook:
`NGC628_PHANGS_source_integral_refit.ipynb`.
