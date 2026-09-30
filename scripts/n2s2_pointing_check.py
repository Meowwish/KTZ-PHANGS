"""H-beta-free calibration check of the pointing zero-point signature.

The stage-1 abundance maps use the Pilyugin & Grebel (2016) S calibration, whose inputs
(Balmer decrement, R3 = [O III]/H-beta, N2, S2 all normalised by H-beta) depend on the
per-pointing blue-end flux calibration and stellar-absorption correction of the MUSE
mosaic.  A per-pointing zero point should therefore appear in the S-calibration map and
be absent from a calibration that never uses H-beta.  Here the same selected pixels are
re-calibrated with the Dopita et al. (2016) N2S2-H-alpha relation,

    12 + log(O/H) = 8.77 + y + 0.45 (y + 0.3)^5,   y = log([N II]/[S II]) + 0.264 log([N II]/H-alpha),

which uses only red lines within 170 A of each other (no reddening correction, no H-beta),
and the same-/cross-pointing statistics are recomputed on the identical pixel set.

    python scripts/n2s2_pointing_check.py [NGC628 ...]
"""
from __future__ import annotations
import json, sys, io, contextlib
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts import ktz_exact_window as kew
from scripts.phangs_pointing_tiles import pointing_labels, shifted_tiling_labels

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "implications" / "hbeta_free_calibration_check"; OUT.mkdir(parents=True, exist_ok=True)
GALAXIES = sys.argv[1:] or ["NGC628", "NGC1566", "NGC4254", "NGC4303"]


def run_stage1_map(galaxy):
    """Execute the stage-1 notebook up to the pixel map and return its namespace."""
    nb = json.loads((ROOT / f"{galaxy}_PHANGS_halpha_map_systematic.ipynb").read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "__stage1__"}
    for cell in code:
        src = "".join(cell["source"])
        last = "np.savez_compressed(" in src and "metallicity_residual_pixel_map" in src
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, f"{galaxy}-stage1", "exec"), ns)
            ns["display_figure"] = lambda *a, **k: plt.close("all")   # after the definition cell: never show figures
            ns["display"] = lambda *a, **k: None
        if last:
            break
    return ns


def n2s2_metallicity(flux):
    with np.errstate(divide="ignore", invalid="ignore"):
        sii = flux["sii6716"] + flux["sii6730"]
        y = np.log10(flux["nii"]/sii) + 0.264*np.log10(flux["nii"]/flux["ha"])
        z = 8.77 + y + 0.45*(y + 0.3)**5
    ok = np.isfinite(z) & (flux["nii"] > 0) & (sii > 0) & (flux["ha"] > 0)
    return np.where(ok, z, np.nan)


def pointing_offsets(z, valid, radius, labels, width=0.2):
    """Joint least squares of per-pointing constants + annulus means (count-weighted zero-sum)."""
    zz = z[valid]; p = labels[valid]; ann = np.floor(radius[valid]/width).astype(int); ann -= ann.min()
    pu, p = np.unique(p, return_inverse=True); npt = len(pu); na = ann.max()+1; n = len(zz)
    Dm = np.zeros((n, npt+na)); Dm[np.arange(n), p] = 1; Dm[np.arange(n), npt+ann] = 1
    cnt = np.bincount(p, minlength=npt).astype(float)
    Dc = np.vstack([Dm, np.r_[cnt/cnt.sum()*np.sqrt(n), np.zeros(na)]]); coef = np.linalg.lstsq(Dc, np.r_[zz, 0.0], rcond=None)[0]
    return pu, coef[:npt]


def analyse(residual, valid, radius, labels, labels_info, rng, n_null=60):
    z = kew.detrend(residual, valid, radius)
    ps = kew.PairStatistic(valid)
    xa, xs, xc, cS, cC = ps.split(z, valid, labels)
    D, _ = kew.compress_difference(ps.r, xs, xc, cS, cC)
    nulls = np.array([kew.compress_difference(ps.r, *ps.split(z, valid, shifted_tiling_labels(labels_info, rng))[1:])[0] for _ in range(n_null)])
    sh = slice(2, 8); D0 = float(np.nanmean(D[sh])); nm = np.nanmean(nulls[:, sh], 1)
    _, off = pointing_offsets(z, valid, radius, labels)
    shoulder = (ps.r >= .6) & (ps.r <= 2.5)
    return dict(r=ps.r, xi_all=xa, xi_same=xs, xi_cross=xc, D=D, null=nulls, D_shoulder=D0, null_mean=float(nm.mean()), null_sd=float(nm.std()),
                z=float((D0-nm.mean())/nm.std()), p=float(np.mean(nm >= D0)), offsets=off, offset_rms=float(off.std()), field_rms=float(z[valid].std()),
                xi_shoulder_all=float(np.nanmean(xa[shoulder])), xi_shoulder_same=float(np.nanmean(xs[shoulder])), xi_shoulder_cross=float(np.nanmean(xc[shoulder])))




def tile_diagnostics(ns, labels, radius):
    """Per-tile means of individual line ratios and a bright/faint split (additive vs multiplicative offsets)."""
    valid = ns["pixel_valid_grid"].astype(bool); f = ns["line_flux"]; c = ns["corrected_line_flux"]
    with np.errstate(all="ignore"):
        q = {"S_cal": ns["metallicity_grid"], "N2S2": n2s2_metallicity(f), "logHaHb": np.log10(f["ha"]/f["hb"]), "logR3": np.log10(c["oiii"]/c["hb"]),
             "logN2Ha": np.log10(f["nii"]/f["ha"]), "logS2Ha": np.log10((f["sii6716"]+f["sii6730"])/f["ha"]), "logN2S2": np.log10(f["nii"]/(f["sii6716"]+f["sii6730"]))}
    v = valid.copy()
    for k in q:
        v &= np.isfinite(q[k])
    npt = int(labels.max())+1; ha = f["ha"]; med = np.nanmedian(ha[v]); out = {}
    for k, val in q.items():
        res, _ = ns["annular_residual_map"](val, v); vv = v & np.isfinite(res); z = kew.detrend(res, vv, radius)
        tmean = lambda mask: np.array([z[mask & (labels == p)].mean() if np.any(mask & (labels == p)) else np.nan for p in range(npt)])
        means = tmean(vv); bright = tmean(vv & (ha > med)); faint = tmean(vv & (ha <= med)); ok = np.isfinite(bright) & np.isfinite(faint) & np.isfinite(means)
        out[k] = dict(tile_means=means.tolist(), tile_rms=float(np.nanstd(means)), field_rms=float(z[vv].std()), bright_half_rms=float(bright[ok].std()), faint_half_rms=float(faint[ok].std()),
                      bright_faint_corr=float(np.corrcoef(bright[ok], faint[ok])[0, 1]))
    keys = list(q); M = np.array([out[k]["tile_means"] for k in keys]); ok = np.all(np.isfinite(M), axis=0)
    out["correlation"] = dict(keys=keys, matrix=np.corrcoef(M[:, ok]).tolist(), n_tiles=int(ok.sum()))
    return out


if __name__ == "__main__":
    results = {}
    fig, axes = plt.subplots(2, len(GALAXIES), figsize=(4.2*len(GALAXIES), 7.6), constrained_layout=True, squeeze=False)
    for col, galaxy in enumerate(GALAXIES):
        print(f"== {galaxy}: running stage-1 map construction", flush=True)
        ns = run_stage1_map(galaxy)
        valid = ns["pixel_valid_grid"].astype(bool); radius = ns["pixel_R"]
        pm = dict(x_grid_kpc=ns["pixel_X"], y_grid_kpc=ns["pixel_Y"], valid_mask=valid)
        info = pointing_labels(ROOT/"data"/"phangs", galaxy, pm); labels = info["pointing"]
        scal = ns["metallicity_grid"]; n2s2 = n2s2_metallicity(ns["line_flux"])
        valid_both = valid & np.isfinite(scal) & np.isfinite(n2s2)
        res_s, _ = ns["annular_residual_map"](scal, valid_both); res_n, _ = ns["annular_residual_map"](n2s2, valid_both)
        valid_both &= np.isfinite(res_s) & np.isfinite(res_n)
        rng = np.random.default_rng(7)
        a_s = analyse(res_s, valid_both, radius, labels, info, rng); a_n = analyse(res_n, valid_both, radius, labels, info, np.random.default_rng(7))
        corr = float(np.corrcoef(res_s[valid_both], res_n[valid_both])[0, 1])
        tiles = tile_diagnostics(ns, labels, radius)
        results[galaxy] = dict(n_pixels=int(valid_both.sum()), residual_correlation=corr, tile_diagnostics=tiles,
                               S_calibration={k: (v.tolist() if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) < 40 else v) for k, v in a_s.items() if k not in ("r", "xi_all", "xi_same", "xi_cross", "null")},
                               N2S2_calibration={k: (v.tolist() if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) < 40 else v) for k, v in a_n.items() if k not in ("r", "xi_all", "xi_same", "xi_cross", "null")})
        print(f"{galaxy}: {valid_both.sum():,} common pixels; residual correlation {corr:.3f}")
        keys = tiles["correlation"]["keys"]; C = np.array(tiles["correlation"]["matrix"])
        print("   tile-mean correlation with S_cal: " + ", ".join(f"{k} {C[0, i]:+.2f}" for i, k in enumerate(keys) if k != "S_cal"))
        print("   tile rms bright/faint half: " + ", ".join(f"{k} {tiles[k]['bright_half_rms']:.3f}/{tiles[k]['faint_half_rms']:.3f}" for k in ("S_cal", "N2S2", "logN2Ha", "logHaHb")))
        for name, a in (("S", a_s), ("N2S2", a_n)):
            print(f"   {name:5s}: shoulder D {a['D_shoulder']:+.4f} (null {a['null_mean']:+.4f} +- {a['null_sd']:.4f}) z = {a['z']:+.2f} p = {a['p']:.3f}; "
                  f"tile-offset rms {a['offset_rms']:.4f} dex (field rms {a['field_rms']:.4f}); xi shoulder all/same/cross {a['xi_shoulder_all']:.4f}/{a['xi_shoulder_same']:.4f}/{a['xi_shoulder_cross']:.4f}")
        np.savez_compressed(OUT/f"{galaxy.lower()}_hbeta_free_check.npz", r=a_s["r"], xi_all_S=a_s["xi_all"], xi_same_S=a_s["xi_same"], xi_cross_S=a_s["xi_cross"], D_S=a_s["D"], null_S=a_s["null"],
                            xi_all_N2S2=a_n["xi_all"], xi_same_N2S2=a_n["xi_same"], xi_cross_N2S2=a_n["xi_cross"], D_N2S2=a_n["D"], null_N2S2=a_n["null"],
                            offsets_S=a_s["offsets"], offsets_N2S2=a_n["offsets"], valid=valid_both, residual_S=np.where(valid_both, res_s, np.nan), residual_N2S2=np.where(valid_both, res_n, np.nan))
        centres = 0.5*(kew.COARSE_EDGES[:-1]+kew.COARSE_EDGES[1:])
        for row, (name, a, color) in enumerate((("S calibration (uses H$\\beta$)", a_s, "#7b3294"), ("N2S2-H$\\alpha$ calibration (no H$\\beta$)", a_n, "#e66101"))):
            ax = axes[row, col]
            ax.fill_between(centres, np.nanpercentile(a["null"], 16, 0), np.nanpercentile(a["null"], 84, 0), color="0.75", alpha=0.6, label="shifted-tiling null")
            ax.plot(centres, a["D"], "o-", color=color, label=f"observed D, z = {a['z']:+.1f}")
            ax.axhline(0, color="0.5", lw=0.6); ax.set(title=f"{galaxy}: {name}", xlabel="separation r [kpc]", ylabel=r"$D=\xi_{\rm same}-\xi_{\rm cross}$")
            ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT/"hbeta_free_calibration_check.png", dpi=140, bbox_inches="tight")
    json.dump(results, (OUT/"hbeta_free_calibration_check.json").open("w"), indent=1)
    print("wrote", OUT)
