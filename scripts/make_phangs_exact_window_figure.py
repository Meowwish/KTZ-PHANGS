"""Manuscript figure and table for the exact-window joint fits of the four PHANGS--MUSE galaxies.

    python scripts/make_phangs_exact_window_figure.py

Reads outputs/implications/NGC<n>_exact_window_joint/ and writes
figs/phangs_four_galaxy_exact_window.pdf (+ .png in outputs/implications) and
outputs/implications/phangs_exact_window_posteriors.{csv,tex}.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
GALAXIES = ("NGC628", "NGC1566", "NGC4254", "NGC4303")
OUT = ROOT / "outputs" / "implications"; FIGS = ROOT / "figs"


def load(galaxy):
    d = OUT / f"{galaxy}_exact_window_joint"; g = galaxy.lower()
    summary = json.loads((d / f"{g}_exact_window_summary.json").read_text())
    curves = {t: dict(np.load(d / f"{g}_curves_{t}.npz")) for t in ("A", "R0", "KT18")}
    split = dict(np.load(d / f"{g}_split_data.npz"))
    stage1 = OUT / f"{galaxy}_halpha_map_systematic"
    diagnostics = np.genfromtxt(stage1 / f"{g}_residual_covariance_diagnostics.csv", delimiter=",", names=True)
    with np.load(stage1 / f"{g}_fit_covariance.npz") as c:
        err = np.sqrt(np.diag(c["covariance"]))
    return summary, curves, split, diagnostics, err


def fmt(p, digits=3):
    lo, med, hi = p
    return f"${med:.{digits}f}_{{-{med - lo:.{digits}f}}}^{{+{hi - med:.{digits}f}}}$"


def main():
    fig, axes = plt.subplots(2, len(GALAXIES), figsize=(3.6 * len(GALAXIES), 6.6), constrained_layout=True)
    rows = []; tex = []
    for col, galaxy in enumerate(GALAXIES):
        summary, curves, split, diagnostics, err = load(galaxy)
        r = diagnostics["r_kpc"]; xi = diagnostics["xi_centered"]
        ax = axes[0, col]
        ax.errorbar(r, xi, err, fmt="o", ms=2.2, color="k", lw=0.8, label="observed")
        ax.plot(r, curves["KT18"]["xi_median"], color="#1f77b4", lw=1.2, label="KT18")
        ax.plot(r, curves["R0"]["xi_median"], color="#ff7f0e", lw=1.2, ls="--", label=r"KTZ, $\sigma_q=0$")
        ax.fill_between(r, curves["A"]["xi_p16"], curves["A"]["xi_p84"], color="#7b3294", alpha=0.2)
        ax.plot(r, curves["A"]["xi_median"], color="#7b3294", lw=1.5, label=r"KTZ + pointing term")
        ax.axhline(0, color="0.5", lw=0.6); ax.set(title=galaxy.replace("NGC", "NGC "), xlabel="$r$ [kpc]", ylim=(-0.12, 0.42))
        if col == 0:
            ax.set_ylabel(r"$\xi_{3,{\rm pp}}(r)$"); ax.legend(frameon=False, fontsize=7.5)
        centres = 0.5 * (split["coarse_edges"][:-1] + split["coarse_edges"][1:])
        ax = axes[1, col]
        ax.errorbar(centres, split["D"], split["D_err"], fmt="o", color="k", ms=3, label="observed")
        ax.fill_between(centres, curves["A"]["D_p16"], curves["A"]["D_p84"], color="#7b3294", alpha=0.2)
        ax.plot(centres, curves["A"]["D_median"], color="#7b3294", lw=1.5, label="KTZ + pointing term")
        ax.plot(centres, curves["R0"]["D_median"], color="#ff7f0e", lw=1.2, ls="--", label="KTZ window difference")
        sq = summary["A"]["params"]["sigma_q"]
        ax.axhline(0, color="0.5", lw=0.6); ax.set(xlabel="$r$ [kpc]", title=rf"$\sigma_q={sq[1]:.2f}_{{-{sq[1]-sq[0]:.2f}}}^{{+{sq[2]-sq[1]:.2f}}}$", ylim=(-0.12, 0.2))
        if col == 0:
            ax.set_ylabel(r"$D(r)=\xi_{\rm same}-\xi_{\rm cross}$"); ax.legend(frameon=False, fontsize=7.5)
        pA = summary["A"]["params"]; vf = summary["A"]["pointing_variance_fraction"]
        rows.append(dict(galaxy=galaxy, n_pointings=summary["settings"]["n_pointings"], pointing_z=summary["settings"]["pointing_z"],
                         **{f"{k}_p16_50_84": pA[k] for k in ("ell_star", "epsilon", "beta_src", "eta", "inv_h_R", "A_src", "ell_src", "sigma_q")},
                         pointing_variance_fraction=vf, chi2_all_A=summary["A"]["chi2_all_median_curve"], chi2_joint_A=summary["A"]["chi2_joint_median"],
                         chi2_all_R0=summary["R0"]["chi2_all_median_curve"], chi2_joint_R0=summary["R0"]["chi2_joint_median"], chi2_all_KT18=summary["KT18"]["chi2_all_median_curve"],
                         ell_star_KT18=summary["KT18"]["params"]["ell_star"], ell_star_R0=summary["R0"]["params"]["ell_star"], n_data=summary["A"]["n_data"],
                         shoulder_obs=summary["A"]["shoulder_obs"], shoulder_ktz=summary["A"]["shoulder_ktz"], shoulder_pointing=summary["A"]["shoulder_pointing"]))
        tex.append(f"NGC~{galaxy[3:]} & {fmt(pA['ell_star'])} & {fmt(pA['epsilon'])} & {fmt(pA['beta_src'], 2)} & {fmt(pA['eta'], 2)} & {fmt(pA['inv_h_R'])} & "
                   f"{fmt(pA['ell_src'], 2)} & {fmt(pA['sigma_q'], 2)} & {100*vf[1]:.1f} & {summary['A']['chi2_joint_median']:.1f} & {summary['R0']['chi2_joint_median']:.1f} & {summary['KT18']['chi2_all_median_curve']:.1f}\\\\")
    fig.savefig(FIGS / "phangs_four_galaxy_exact_window.pdf", bbox_inches="tight"); fig.savefig(OUT / "phangs_four_galaxy_exact_window.png", dpi=150, bbox_inches="tight")
    json.dump(rows, (OUT / "phangs_exact_window_posteriors.json").open("w"), indent=1)
    header = ("Galaxy & $\\ell_\\ast$ [kpc] & $\\epsilon$ & $\\beta_{\\rm src}$ & $\\eta$ & $h_R^{-1}$ [kpc$^{-1}$] & $\\ell_{\\rm src}$ [kpc] & $\\sigma_q$ & "
              "$f_q$ [\\%] & $\\chi^2_{\\rm joint}$ & $\\chi^2_{\\rm joint}(\\sigma_q\\!=\\!0)$ & $\\chi^2_{\\rm KT18}$\\\\")
    (OUT / "phangs_exact_window_posteriors.tex").write_text(header + "\n\\midrule\n" + "\n".join(tex) + "\n")
    print("\n".join([header] + tex))


if __name__ == "__main__":
    main()
