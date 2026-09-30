"""Nested-model evidence for the exact-window joint fits (stand-alone driver for the stage-2 products).

    python -m scripts.run_exact_window_evidence NGC628 [NGC1566 ...]

Writes outputs/implications/NGC<n>_exact_window_joint/<n>_exact_window_evidence.json with the same
priors and data vector as the stage-2 notebook (flat eta prior, ell_src >= 0.12 kpc, sigma_q uniform on [0, 3]).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scripts import ktz_exact_window as kew
from scripts.ktz_exact_window_evidence import evidence_table

ROOT = Path(__file__).resolve().parents[1]
ELL_SRC_LOWER_KPC = 0.12


def load_model(galaxy):
    g = galaxy.lower(); s1 = ROOT/"outputs"/"implications"/f"{galaxy}_halpha_map_systematic"; s2 = ROOT/"outputs"/"implications"/f"{galaxy}_exact_window_joint"
    morphology = json.loads((s1/f"{g}_morphology_summary.json").read_text())
    diagnostics = np.genfromtxt(s1/f"{g}_residual_covariance_diagnostics.csv", delimiter=",", names=True)
    fit_covariance = np.load(s1/f"{g}_fit_covariance.npz")["covariance"]
    with np.load(s1/f"{g}_muse_scal_7p_ktz_cnp_full_pair_emulator.npz") as c:
        grids = {k: c[k] for k in ("ell_grid", "epsilon_grid", "source_length_grid")}
    resp = {k: np.load(s2/f"{g}_window_response_{k}.npz") for k in ("all", "same", "cross")}
    split = np.load(s2/f"{g}_split_data.npz"); C = np.load(s2/f"{g}_joint_covariance.npz")["covariance"]
    return kew.SplitKTZ(morphology, diagnostics, fit_covariance, resp["all"], resp["same"], resp["cross"], split, C, grids,
                        src_length_bounds=(ELL_SRC_LOWER_KPC, None), eta_prior="flat"), s2


if __name__ == "__main__":
    for galaxy in sys.argv[1:]:
        model, out_dir = load_model(galaxy); t0 = time.time()
        log_path = out_dir/f"{galaxy.lower()}_exact_window_evidence.log"
        def log(msg):
            print(f"[{galaxy}] {msg}", flush=True); log_path.open("a").write(msg+"\n")
        table = evidence_table(model, log=log)
        table["settings"] = dict(eta_prior="flat", ell_src_lower_kpc=ELL_SRC_LOWER_KPC, nlive=400, time_s=time.time()-t0)
        json.dump(table, (out_dir/f"{galaxy.lower()}_exact_window_evidence.json").open("w"), indent=1)
        log(f"done in {time.time()-t0:.0f} s")
