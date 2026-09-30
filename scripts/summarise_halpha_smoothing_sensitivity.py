"""Summarise the direct two-dimensional H-alpha template calculations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.make_halpha_map_systematic_notebooks import (
    refresh_markdown,
    variant_name,
)

OUTPUT_ROOT = ROOT / "outputs" / "implications"
GALAXIES = ("NGC628", "NGC1566", "NGC4254", "NGC4303")
FWHM_VALUES = (0.5, 1.0, 1.5)


def tag(fwhm_kpc: float) -> str:
    return str(fwhm_kpc).replace(".", "p")


def output_directory(galaxy: str, fwhm_kpc: float) -> Path:
    if fwhm_kpc == 1.0:
        suffix = "halpha_map_systematic"
    else:
        suffix = f"halpha_map_fwhm{tag(fwhm_kpc)}_systematic"
    return OUTPUT_ROOT / f"{galaxy}_{suffix}"


def load_fit_row(path: Path, model: str) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return next(row for row in rows if row["model"].startswith(model))


def main() -> None:
    rows: list[dict[str, object]] = []
    for galaxy in GALAXIES:
        stem = galaxy.lower()
        for fwhm_kpc in FWHM_VALUES:
            directory = output_directory(galaxy, fwhm_kpc)
            fit = load_fit_row(directory / f"{stem}_ktz_fit_summary.csv", "KTZ")
            with (directory / f"{stem}_morphology_summary.json").open() as handle:
                summary = json.load(handle)
            curve = np.genfromtxt(
                directory / f"{stem}_xi_observed_and_models.csv",
                delimiter=",",
                names=True,
                encoding="utf-8",
            )
            tail = (curve["r_kpc"] >= 3.3) & (curve["r_kpc"] <= 4.5)
            intervals = summary["posterior_intervals_p16_median_p84"]
            rows.append(
                {
                    "galaxy": galaxy,
                    "halpha_template_fwhm_kpc": fwhm_kpc,
                    "template_rms_contrast": summary["mask_contrast_amplitude"],
                    "covariance_calibrated_discrepancy": fit[
                        "covariance_calibrated_discrepancy"
                    ],
                    "mcmc_max_rhat": summary["mcmc_max_rhat"],
                    "converged_at_Rhat_1p05": summary["mcmc_max_rhat"] <= 1.05,
                    "ell_star_median_kpc": intervals["ell_star"][1],
                    "epsilon_median": intervals["epsilon"][1],
                    "eta_median": intervals["eta"][1],
                    "CNP_amplitude_median": intervals[
                        "source_excess_amplitude"
                    ][1],
                    "CNP_scale_median_kpc": intervals[
                        "source_clustering_length"
                    ][1],
                    "observed_tail_mean_3p3_4p5_kpc": float(
                        np.nanmean(curve["xi_observed"][tail])
                    ),
                    "KTZ_tail_mean_3p3_4p5_kpc": float(
                        np.nanmean(curve["xi_KTZ_median"][tail])
                    ),
                }
            )

    output_path = OUTPUT_ROOT / "phangs_halpha_smoothing_sensitivity.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)

    # Refresh only descriptive cells, preserving every executed calculation.
    for galaxy in GALAXIES:
        galaxy_rows = [row for row in rows if row["galaxy"] == galaxy]
        table_lines = [
            "| FWHM [kpc] | calibrated Q | ell_star [kpc] | ell_src [kpc] | tail mean | max Rhat | status |\n",
            "| --- | --- | --- | --- | --- | --- | --- |\n",
        ]
        for row in galaxy_rows:
            status = (
                "converged" if row["converged_at_Rhat_1p05"]
                else "not converged: diagnostic only"
            )
            table_lines.append(
                f"| {row['halpha_template_fwhm_kpc']:.1f} | "
                f"{float(row['covariance_calibrated_discrepancy']):.3f} | "
                f"{row['ell_star_median_kpc']:.3f} | "
                f"{row['CNP_scale_median_kpc']:.3f} | "
                f"{row['KTZ_tail_mean_3p3_4p5_kpc']:+.4f} | "
                f"{row['mcmc_max_rhat']:.3f} | {status} |\n"
            )
        for fwhm_kpc in FWHM_VALUES:
            notebook_path = ROOT / (
                f"{galaxy}_PHANGS_{variant_name(fwhm_kpc)}.ipynb"
            )
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            refresh_markdown(notebook, fwhm_kpc)
            notebook["cells"] = [
                cell for cell in notebook["cells"]
                if "halpha-smoothing-summary" not in cell.get("metadata", {}).get("tags", [])
            ]
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {"tags": ["halpha-smoothing-summary"]},
                "source": [
                    "## 8. Pre-specified H-alpha smoothing sensitivity\n\n",
                    "These are separate complete-KTZ fits at 0.5/1.0/1.5-kpc "
                    "template FWHM, with the same abundance map, observing transfer, "
                    "prior prescription, and seven likelihood coordinates. The "
                    "tail mean is over 3.3--4.5 kpc. Smaller Q is a lower "
                    "covariance-calibrated discrepancy, not model evidence. "
                    "We retain 1.0 kpc as the declared central template rather "
                    "than selecting the best curve. A failed convergence rule "
                    "invalidates parameter conclusions from that run.\n\n",
                    *table_lines,
                ],
            })
            notebook_path.write_text(
                json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
