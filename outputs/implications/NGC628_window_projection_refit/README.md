# NGC628 full-window raw-second-moment refit

This run repairs the order of annular projection, pair compression, and
normalisation. It is not a smoothing-scale selection or an extra-component fit.
The same 1-kpc H-alpha template, seven parameter coordinates, physical prior,
observed correlation, and line-flux/selection covariance are retained.

The new midpoint local covariance is projected as `P C P^T` on the actual
valid mask before the four numerators and endpoint second moments are combined.
Every valid pair in the full observed window enters the annular averages;
the likelihood itself still uses only 0 < r < 5 kpc. The expected mean field is
projected directly, including the small within-annulus radial-envelope residual.
The covariance remains a local-source approximation, not a global diffusion
solution. Radial kernels and radial-rate dependence are represented numerically.

## Reproduction

Run `NGC628_PHANGS_window_projection_refit.ipynb` from top to bottom in the
project root, or execute `scripts/run_ngc628_window_refit_notebook.py`.
The notebook regenerates the original deterministic 200-draw observing pipeline
and verifies that the observed curve and marginal covariance match the old run.
It saves visual outputs after each completed cell.

Independent dense small-map tests:

```sh
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python -m scripts.test_phangs_window_projection
```

To recompute the completed-chain kernel/prior audits and constrained likelihood
shape diagnostic, without rerunning MCMC:

```sh
OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/private/tmp/ktz-ngc628-mpl /opt/anaconda3/bin/python -m scripts.diagnose_ngc628_window_refit
```

## Outputs and interpretation

- `run_progress.json`: precomputation or MCMC stage; this is not a result.
- `ngc628_full_window_second_moment_response_checkpoint.npz`: resumable eta-node cache.
- `ngc628_window_emulator_check.json`: kernel interpolation/quadrature point checks.
- `ngc628_transport_chain.npz` and `ngc628_kt18_transport_chain.npz`: newly sampled chains.
- `ngc628_window_refit_summary.json`: convergence, fitted medians and exploratory shoulder check.
- `ngc628_window_projection_comparison.png`: medians and total 16–84% band versus old results.
- `ngc628_window_projection_components.png`: additive joint-draw stack and only a total band.
- `ngc628_window_refit_diagnostics.json`: finite posterior-kernel checks, prior
  consistency, and posterior-versus-likelihood shape diagnostics.
- `ngc628_likelihood_shape_diagnostic.png`: explicitly diagnostic likelihood
  curve alongside the adopted posterior median; not a new main fit summary.

## Completed outcome

The chains converge (maximum split Rhat 1.021 for KTZ and 1.003 for KT18).
The shoulder RMSE decreases from 0.0907 to 0.0725, but the posterior median
still underpredicts the shoulder. Its bin-averaged value is 0.0383 versus
0.1080 observed. The constrained likelihood diagnostic reaches RMSE 0.0198,
but favours a radial slope at the prior-support boundary and a 2.20-kpc
distinct-event source-pair scale in the upper posterior tail. Its better
shape is not a measurement of those physical parameters.

Do not tune priors, the covariance, or H-alpha smoothing solely to move the
median towards that state. The current correction remains NGC628-only and
must be propagated to other galaxies and geometry controls before replacing
the common manuscript figures/tables. The unchanged local-source approximation
and numerical eta/radial representation remain limitations.

Do not replace manuscript results unless the run is converged and the numerical
checks pass. A shoulder statistic selected after inspection is exploratory,
and its replicated-observation distribution is not the same as the displayed
posterior interval of the mean curve. Old results are preserved in the separate
`NGC628_halpha_map_systematic` directory.
