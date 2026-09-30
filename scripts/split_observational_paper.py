"""Split the PHANGS application from the KTZ theory manuscript.

The transformation is intentionally marker-based so that all equations, tables,
figures, and prose in the existing observational analysis are preserved verbatim
apart from the small set of standalone-paper edits below.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THEORY_PATH = ROOT / "Non-uniform KT.tex"
OBS_PATH = ROOT / "KTZ_obs.tex"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence, found {count}: {old[:80]!r}")
    return text.replace(old, new, 1)


source = THEORY_PATH.read_text()
if "\\section{Implications}\n\\label{sec:implications}" not in source:
    if OBS_PATH.exists():
        print("The observational paper has already been split from the theory manuscript.")
        raise SystemExit(0)
    raise RuntimeError("Could not find the observational-section marker.")

obs_start = source.index("\\section{Implications}\n\\label{sec:implications}")
obs_end = source.index("\\section{Validity}\n\\label{sec:validity}")
obs_block = source[obs_start:obs_end]

discussion_start_marker = (
    "\\subsection{Application to observations}\n"
    "\\label{ssec:discussion_observations}"
)
discussion_start = source.index(discussion_start_marker)
discussion_end = source.index("\\section{Conclusions}", discussion_start)
obs_discussion = source[discussion_start:discussion_end].strip()

# ---------------------------------------------------------------------------
# Paper III: theory and controlled numerical validation.
# ---------------------------------------------------------------------------
theory = source[:obs_start] + source[obs_end:discussion_start] + source[discussion_end:]

theory = replace_once(
    theory,
    "\\title[Spiral metallicity correlations]{Metallicity fluctuation statistics in the interstellar medium and young stars - III. The effects of spiral structure}",
    "\\title[Non-uniform metal correlations]{Metallicity fluctuation statistics in the interstellar medium and young stars -- III. Non-uniform injection: theory and numerical validation}",
)

theory_abstract = r"""\begin{abstract}
We generalise stochastic metal-injection models from a spatially uniform source process to an inhomogeneous intensity with a deterministic spiral component.  After subtraction of only an axisymmetric abundance profile, the residual second moment separates into a connected same-event covariance, a product of the unremoved spiral mean fields, and, when distinct events remain associated, an additional connected source-pair covariance.  We derive the position- and orientation-dependent single-element statistic, its finite-window pair-product average, and the corresponding zero-lag multi-element correlation with channel-dependent yields, delays, injection widths, and stellar drift.  The dimensionless combinations expose how diffusion suppresses tightly spaced and high-order source harmonics and clarify which parameters remain identifiable from normalised correlations.  Controlled Monte Carlo experiments validate the local, finite-window, arm-sector, and multi-element expressions.  The full-window single-element statistic has an absolute RMSE of 0.009, while 120 off-diagonal elemental correlations are recovered with an RMSE of 0.033 and a Pearson coefficient of 0.993.  Clustered-source experiments separately validate the leading distinct-event term.  We state the limits imposed by local-WKBJ, frozen-pattern, isotropic-diffusion, and finite-boundary assumptions, providing a theoretical baseline for companion applications to resolved abundance maps.
\end{abstract}"""
theory = replace_once(theory, "\\begin{abstract}\n\\end{abstract}", theory_abstract)

theory = replace_once(
    theory,
    "\\autoref{sec:numerical_validation} tests both predictions with Monte Carlo realisations,\n"
    "\\autoref{sec:implications} presents the implications of our findings,\n"
    "\\autoref{sec:validity} discusses the model's domain of validity, and\n"
    "\\autoref{sec:discussion} considers how the theory can be tested with resolved observations.",
    "\\autoref{sec:numerical_validation} tests both predictions with Monte Carlo realisations, "
    "\\autoref{sec:validity} states the model's domain of validity, and "
    "\\autoref{sec:discussion} discusses the theoretical and numerical implications.",
)

theory = replace_once(
    theory,
    "This derivation is highly mathematical, so readers primarily interested in the results of the calculation may wish to skip to \\autoref{sec:implications}.",
    "This derivation is highly mathematical, so readers primarily interested in the interpretation and tested limits may wish to skip to \\autoref{sec:discussion}.",
)

theory = replace_once(
    theory,
    "The PHANGS application below is therefore a forward-modelling demonstration under explicit morphology and selection choices, not an extension of the controlled numerical validation to a claim of universal model adequacy.",
    "Applications to resolved abundance maps must therefore be treated as forward-modelling tests under explicit morphology and selection choices, not as automatic extensions of the controlled numerical validation to claims of universal model adequacy.",
)

theory = replace_once(
    theory,
    "In the four PHANGS--MUSE applications, the metal-inferred ordering of residual source-clustering scales matches the ordering measured independently from H~\\textsc{ii}-region excess pairs.  In NGC~4303, a KTZ model with independently fitted H$\\alpha$ geometry, constrained only at $r<3$~kpc, also predicts a held-out negative tail whose mean amplitude contains the observed value, whereas KT18 does not.  These are proof-of-concept physical cross-checks rather than a population detection: the four-object rank comparison is descriptive, and the H$\\alpha$ geometry was not pre-registered before inspection of the tail.\n\n",
    "",
)

THEORY_PATH.write_text(theory)

# ---------------------------------------------------------------------------
# Paper IV: standalone observational application.
# ---------------------------------------------------------------------------
preamble = source[: source.index("\\begin{document}")]
preamble = preamble.replace(
    "% mnras_template.tex \n"
    "%DIF LATEXDIFF DIFFERENCE FILE\n"
    "%DIF DEL RoSE_v0.tex   Thu Jun 11 20:40:19 2026\n"
    "%DIF ADD RoSE_v1.tex   Thu Jun 11 20:40:22 2026\n",
    "% mnras_template.tex\n",
    1,
)
preamble = replace_once(
    preamble,
    "\\title[Spiral metallicity correlations]{Metallicity fluctuation statistics in the interstellar medium and young stars - III. The effects of spiral structure}",
    "\\title[KTZ in PHANGS--MUSE galaxies]{Metallicity fluctuation statistics in the interstellar medium and young stars -- IV. A PHANGS--MUSE application of non-uniform injection}",
)

obs_abstract = r"""\begin{abstract}
We apply the non-uniform KTZ metal-correlation model developed in the companion theory paper to four nearby spiral galaxies observed by PHANGS--MUSE: NGC~628, NGC~1566, NGC~4254, and NGC~4303.  Emission-line maps are matched to a 120-pc point-spread function, sampled on a 60-pc face-on grid, and converted to oxygen-abundance residuals after line-quality, star-forming BPT, equivalent-width, and coverage selections.  Missing cells remain absent from the exhaustive finite-window pair statistic, and the same footprint, radial-profile subtraction, beam response, and selection covariance are propagated through KTZ and KT18.  Complete KTZ fits include same-event transport, a deterministic source-geometry mean field, and residual distinct-event clustering.  KTZ gives the lower covariance-calibrated discrepancy for NGC~628 and NGC~1566, but not for NGC~4254 or NGC~4303 under the primary PHANGS arm masks.  The two galaxies with the broadest metal-inferred source-pair terms also have the broadest independently measured H~\textsc{ii}-region excess-pair distributions.  An H$\alpha$ logarithmic-spiral geometry predicts most of the held-out negative tail in NGC~4303, although this geometry was examined after the feature was identified.  We therefore treat these four objects as a development sample and specify a phase-resolved, geometry-randomisation test for a future untouched sample.  The current results demonstrate observational feasibility and identify source geometry, selection, and covariance calibration as the limiting systematics.
\end{abstract}"""

obs_introduction = r"""\section{Introduction}
\label{sec:intro}

Spatially resolved metallicity fluctuations contain information that is absent from a radial abundance gradient.  Localised enrichment, turbulent transport, and the organisation of star formation determine both the amplitude and the coherence scale of residual abundance structure.  Integral-field spectroscopy now permits these residuals to be mapped across nearby galaxies, and two-point statistics show that chemically coherent structure commonly persists over hundreds of parsecs to kiloparsec scales \citep[e.g.,][]{2020MNRAS.499..193K,Li2021,2021MNRAS.508..489M,2023MNRAS.518..286L}.

The uniform stochastic-injection theory of \citet[hereafter \citetalias{KT18}]{KT18} connects the correlation function to injection width, diffusion, and the duration over which enrichment events accumulate.  Its multi-element extension, \citet[hereafter \citetalias{KT25}]{KT25}, adds channel-dependent yields, delays, and stellar drift.  These models provide physically interpretable baselines, but a uniform source process cannot retain the phase of a spiral pattern or distinguish a coherent arm-associated mean field from residual association-scale clustering.  Those limitations are especially relevant when observed correlations contain intermediate-scale shoulders or cross zero at large separation.

The companion theory paper in this series (Paper III) develops KTZ, an extension in which the deterministic injection intensity follows an externally specified two-dimensional source template.  It separates the same-event covariance, its local rate modulation, the product of any unremoved non-axisymmetric mean field, and an optional connected covariance between distinct events.  Paper III also derives finite-window and multi-element statistics and validates them against controlled Monte Carlo experiments.  Here we ask how that decomposition can be implemented on real maps once the beam, incomplete footprint, abundance selection, radial-profile subtraction, and correlated measurement errors are included.

We analyse NGC~628, NGC~1566, NGC~4254, and NGC~4303 using public PHANGS--MUSE emission-line products and source geometries measured independently of the metallicity residuals.  The aims are to determine which KTZ components are required by the measured radial correlations, test whether the inferred residual source-pair scale is reflected in the H~\textsc{ii}-region point process, and quantify how the predicted large-scale shape changes with the adopted arm tracer.  Because these galaxies and alternative geometries have already been inspected during method development, the resulting comparisons are exploratory rather than a pre-registered population test.  We therefore also define a phase-resolved statistic and geometry-randomisation protocol that can provide a falsifiable primary test on a future untouched sample.

\autoref{sec:framework} summarises the KTZ quantities used in the likelihood.  \autoref{sec:application} describes the public data, map construction, observing transfer, and inference.  \autoref{sec:discussion} interprets the current four-galaxy results and specifies the confirmatory phase-resolved experiment.
"""

framework = r"""\section{KTZ framework for resolved maps}
\label{sec:framework}

Paper III considers enrichment events with initial Gaussian width $\sigma$, diffusion coefficient $\kappa$, and maximum source age $t_\ast$.  The transport scale, relative injection width, and relative deterministic-mean strength used below are
\begin{align}
    \ell_\ast&=\sqrt{\sigma^2+2\kappa t_\ast},
    &\epsilon&=\frac{\sigma}{\ell_\ast},
    \label{eq:dimensionless_variables}
    \\
    \beta_{\rm src}
    &=\frac{4\pi\kappa\lambda_0t_\ast^2\langle M_\ast\rangle^2}
    {\langle M_\ast^2\rangle}.
    \label{eq:beta_src}
\end{align}
Here $\lambda_0$ is the local axisymmetric event rate and $M_\ast$ is the stellar mass represented by one parent event.  For a source harmonic of order $n$, the dimensionless smoothing coordinate is $q_n=nk_\perp\ell_\ast$.  The quantity $\beta_{\rm src}$ measures the deterministic mean-field power relative to the same-event second moment.

Let $\bs{x}_\pm=\bs{x}_m\pm\bs r/2$ be the two endpoints.  After subtraction of the axisymmetric abundance profile, complete KTZ writes their residual second moment as
\begin{equation}
    C_{\rm obs}(\bs{x}_-,\bs{x}_+)
    =C_{{\rm 2p},0}+C_{{\rm 2p},N}+C_{\rm M}+C_{\rm NP}.
    \label{eq:ktz_component_summary}
\end{equation}
The first two terms are the uniform and source-modulated parts of the connected same-event covariance.  The third is the product of the deterministic non-axisymmetric means left in the residual map and is therefore not a connected covariance about the full ensemble mean.  The fourth describes residual association between distinct events after the deterministic source intensity has been specified.  We adopt the isotropic Gaussian pair excess
\begin{equation}
    {\cal E}_{\rm src}(\Delta)
    ={\cal A}_{\rm src}
    \exp\left(-\frac{\Delta^2}{2\ell_{\rm src}^2}\right),
    \label{eq:Esrc_example}
\end{equation}
and propagate it through both age-integrated transport kernels and the same fixed source-rate template.  Thus $C_{\rm NP}$ enters the numerator and endpoint second moments before normalisation; it is not an additive correction to a completed KTZ curve.

Define $V_-=C_{\rm obs}(\bs{x}_-,\bs{x}_-)$ and $V_+=C_{\rm obs}(\bs{x}_+,\bs{x}_+)$.  If $\langle\cdot\rangle_3$ denotes an average over every allowed pair midpoint and orientation in a separation bin, the finite-window statistic is
\begin{equation}
    \xi_{3,{\rm pp}}(r)
    =\frac{\langle C_{\rm obs}\rangle_3(r)}
    {[\langle V_-\rangle_3(r)\langle V_+\rangle_3(r)]^{1/2}}.
    \label{eq:xi3_pp_def}
\end{equation}
The numerator and endpoint second moments must be averaged separately because averaging and normalisation do not commute.  The implementation below evaluates the Paper III kernels on the observed source template and applies the same finite footprint and map-processing operators used for the data.
"""

obs_block = obs_block.replace(
    "\\section{Implications}\n\\label{sec:implications}",
    "\\section{PHANGS--MUSE application}\n\\label{sec:application}",
    1,
)
obs_block = obs_block.replace(
    "Here $C_{\\rm NP}$ is not a post-fit correction.  It is the Gaussian residual source-pair term in equations~(\\ref{eq:Esrc_example}) and~(\\ref{eq:CNP_full}), with free ${\\cal A}_{\\rm src}$ and $\\ell_{\\rm src}$, and it enters both the numerator and the two endpoint second moments before normalisation.  In this application we evaluate the full real-space expression in the same local-source approximation as the Poisson term, so that its rate factor is locally proportional to $\\lambda^2$; this retains the uniform, linear, and quadratic powers of the fixed mask template.  This is more general than the compact zeroth-order approximation in equation~(\\ref{eq:CNP}), but it introduces no additional arm harmonic or morphology parameter: the residual pair excess remains isotropic and the deterministic arm pattern appears only through the independently specified $\\lambda$.  Figure~\\ref{fig:phangs_components} shows the four components using the same colours as the numerical validation; each component is summarised over the posterior rather than evaluated only at one coordinate vector.",
    "Here $C_{\\rm NP}$ is not a post-fit correction.  It is the Gaussian residual source-pair term in equation~(\\ref{eq:Esrc_example}), with free ${\\cal A}_{\\rm src}$ and $\\ell_{\\rm src}$, and it enters both the numerator and the two endpoint second moments before normalisation.  We evaluate the full real-space construction from Paper III in the same local-source approximation as the Poisson term, so that its rate factor is locally proportional to $\\lambda^2$; this retains the uniform, linear, and quadratic powers of the fixed mask template.  No additional arm harmonic or morphology parameter is introduced: the residual pair excess remains isotropic and the deterministic arm pattern appears only through the independently specified $\\lambda$.  Figure~\\ref{fig:phangs_components} shows the four components; each is summarised over the posterior rather than evaluated only at one coordinate vector.",
    1,
)
obs_block = obs_block.replace(
    "the same latent Gaussian form assumed in Appendix~\\ref{app:nonpoisson}",
    "the same latent Gaussian form as equation~(\\ref{eq:Esrc_example})",
    1,
)

obs_discussion = obs_discussion.replace(
    "\\subsection{Application to observations}\n\\label{ssec:discussion_observations}",
    "\\section{Discussion}\n\\label{sec:discussion}\n\n"
    "\\subsection{Interpretation of the development sample}\n"
    "\\label{ssec:discussion_development}",
    1,
)
obs_discussion = obs_discussion.replace(
    "the isotropic extension in Appendix~\\ref{app:nonpoisson}",
    "the isotropic distinct-event extension derived in Paper III",
)
obs_discussion = obs_discussion.replace(
    "Appendix~\\ref{app:identifiability} summarises the combinations accessible to the normalised statistics and the external information required to separate their dimensional factors.",
    "Paper III summarises the combinations accessible to normalised statistics and the external information required to separate their dimensional factors.",
)

confirmatory = r"""
\subsection{A confirmatory phase-resolved test}
\label{ssec:confirmatory_test}

The angle-averaged tail is not an efficient primary test of non-uniform injection in the present data: most galaxies approach zero at large separation, the tail bins are strongly covariant, and radial-profile subtraction can introduce a compensating negative lobe.  KTZ makes a more distinctive prediction before this phase information is averaged away.  Let $g(\bs{x})$ be a source template measured from H$\alpha$ and centred to have zero mean in each radial annulus.  For observed residuals $\delta Z_i$, pair midpoint $\bs{x}_{m,ij}$, and separation-bin indicator $I_b(i,j)$, define the geometry-matched statistic
\begin{equation}
    \widehat D_g(r_b)
    =
    \frac{
        \sum_{i<j}I_b(i,j)g(\bs{x}_{m,ij})\delta Z_i\delta Z_j
    }{
        \sum_{i<j}I_b(i,j)[\delta Z_i^2+\delta Z_j^2]/2
    }.
    \label{eq:phase_resolved_statistic}
\end{equation}
This is a continuous version of an arm-midpoint minus interarm-midpoint pair statistic.  It retains the linear $C_{{\rm 2p},N}$ response and the phase dependence of $C_{\rm M}$ at the higher-signal $0.2$--$2.5$~kpc separations.  KT18 has no physical source phase; its small non-zero response caused by the footprint and annular projector must nevertheless be propagated rather than assumed to vanish.

The four galaxies analysed here constitute the development sample used to freeze the abundance pipeline, priors, source-template algorithm, covariance estimator, and statistic in equation~(\ref{eq:phase_resolved_statistic}).  A confirmatory sample must be selected by data-quality criteria before its metallicity correlations are inspected.  The ordinary angle-averaged correlation will constrain the transport and residual-clustering coordinates, after which its phase-resolved counterpart will be predicted rather than used to choose the geometry.  The primary null distribution will be obtained by rotating or phase-randomising each H$\alpha$ template and refitting every randomised case with the same complete KTZ parameters and priors.  Success requires the joint posterior-predictive score of the true external geometry to exceed those of at least 95 per cent of the randomised geometries; comparison with KT18 is a pre-specified secondary baseline.  The large-separation tail and the H~\textsc{ii}-pair scale remain valuable secondary predictions, but neither is used to select galaxies or to redefine the primary statistic after unblinding.
"""

obs_conclusions = r"""\section{Conclusions}

We have constructed a reproducible application of complete KTZ to four PHANGS--MUSE spiral galaxies.  The emission-line data, abundance selection, finite footprint, radial-profile subtraction, beam, and correlated measurement response are carried through the same exhaustive pair estimator used for the model.  The inferred transport scales are short, while NGC~628 and NGC~1566 require the broadest residual source-pair contributions.  Their ordering agrees with an independent measurement from the H~\textsc{ii}-region point process, providing a physical cross-check without using the metal residual values themselves.

The present angle-averaged fits do not show that KTZ is uniformly preferred to KT18.  In NGC~4303 the PHANGS arm-region template underpredicts the negative tail, whereas an independently fitted H$\alpha$ logarithmic spiral predicts most of its mean amplitude even when the tail is excluded from the likelihood.  Because that alternative geometry was examined after the feature was noticed and because the best fit is sensitive to the radial-rate boundary, it remains a development-sample demonstration rather than a confirmatory detection.  The weak positive outer mean in NGC~1566 is statistically consistent with zero.

The strongest future test is therefore not whether a flexible model can pass through a near-zero radial tail.  It is whether the true external source phase predicts the geometry-weighted pair statistic in equation~(\ref{eq:phase_resolved_statistic}) better than identically fitted phase-randomised templates and the uniform KT18 model in an untouched sample.  This test preserves the two-dimensional information that distinguishes KTZ and supplies an explicit failure criterion.  Until it is completed, the four-galaxy results establish feasibility and motivate the test, but should not be interpreted as a population-level validation of non-uniform injection.

\section*{Acknowledgements}

\section*{Data Availability}

\bibliographystyle{mnras}
\bibliography{example}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Don't change these lines
\bsp
\label{lastpage}
\end{document}
"""

obs_document = (
    preamble
    + "\\begin{document}\n"
    + "\\label{firstpage}\n"
    + "\\pagerange{\\pageref{firstpage}--\\pageref{lastpage}}\n"
    + "\\maketitle\n\n"
    + obs_abstract
    + "\n\n"
    + "\\begin{keywords}\n"
    + "galaxies: abundances -- galaxies: ISM -- galaxies: spiral -- ISM: abundances -- methods: data analysis\n"
    + "\\end{keywords}\n\n"
    + obs_introduction
    + "\n"
    + framework
    + "\n"
    + obs_block
    + "\n"
    + obs_discussion
    + "\n"
    + confirmatory
    + "\n"
    + obs_conclusions
)

OBS_PATH.write_text(obs_document)
print(f"Updated {THEORY_PATH.name} and created {OBS_PATH.name}.")
