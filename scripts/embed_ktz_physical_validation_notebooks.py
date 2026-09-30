"""Append the executed exploratory-diagnostic panel to the four PHANGS notebooks."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "implications"
GALAXIES = ("NGC628", "NGC1566", "NGC4254", "NGC4303")
CELL_TAG = "ktz-physical-validation"


def output_directory(galaxy: str) -> Path:
    if galaxy == "NGC628":
        return ROOT / "outputs" / "implications"
    return ROOT / "outputs" / "implications" / galaxy


def summary_text(galaxy: str) -> str:
    source_summary = np.genfromtxt(
        ROOT / "outputs/implications/phangs_hii_source_pair_summary.csv",
        delimiter=",", names=True, dtype=None, encoding="utf-8",
    )
    row = source_summary[source_summary["galaxy"] == galaxy]
    if np.ndim(row) > 0:
        row = row[0]
    lines = [
        f"Qualified H II regions: {int(row['n_hii'])}",
        (
            "H II excess-pair Gaussian: "
            f"A={row['hii_amplitude']:.3f} "
            f"[{row['hii_amplitude_lo']:.3f}, {row['hii_amplitude_hi']:.3f}], "
            f"ell={row['hii_length_kpc']:.3f} "
            f"[{row['hii_length_lo_kpc']:.3f}, "
            f"{row['hii_length_hi_kpc']:.3f}] kpc"
        ),
        (
            "PHANGS-mask KTZ source-pair posterior: "
            f"A_src={row['ktz_A_src_median']:.3f} "
            f"[{row['ktz_A_src_p16']:.3f}, {row['ktz_A_src_p84']:.3f}], "
            f"ell_src={row['ktz_ell_src_median_kpc']:.3f} "
            f"[{row['ktz_ell_src_p16_kpc']:.3f}, "
            f"{row['ktz_ell_src_p84_kpc']:.3f}] kpc"
        ),
    ]
    if galaxy == "NGC4303":
        tail = np.genfromtxt(
            ROOT / "outputs/implications/ngc4303_heldout_tail_summary.csv",
            delimiter=",", names=True, dtype=None, encoding="utf-8",
        )
        for tail_row in tail:
            lines.append(
                f"{tail_row['model_or_geometry']}: held-out tail mean="
                f"{tail_row['tail_mean_median']:+.4f} "
                f"[{tail_row['tail_mean_p16']:+.4f}, "
                f"{tail_row['tail_mean_p84']:+.4f}], "
                f"importance ESS={tail_row['importance_effective_sample_size']:.0f}"
            )
    if galaxy in {"NGC1566", "NGC4303"}:
        likelihood = np.genfromtxt(
            ROOT / (
                "outputs/implications/"
                "ngc1566_ngc4303_maximum_likelihood_summary.csv"
            ),
            delimiter=",", names=True, dtype=None, encoding="utf-8",
        )
        for likelihood_row in likelihood[likelihood["galaxy"] == galaxy]:
            lines.append(
                f"{likelihood_row['geometry']}: retained-chain best likelihood "
                f"Q={likelihood_row['best_likelihood_covariance_calibrated_Q']:.3f}, "
                f"tail mean={likelihood_row['best_likelihood_mean_3p3_4p5']:+.4f}, "
                f"KTZ/KT18 AIC={likelihood_row['ktz_aic']:.2f}/"
                f"{likelihood_row['kt18_aic']:.2f}"
            )
    return "\n".join(lines) + "\n"


def update_notebook(galaxy: str, direct_halpha: bool = False) -> None:
    suffix = "halpha_map_systematic" if direct_halpha else "implications"
    notebook_path = ROOT / f"{galaxy}_PHANGS_{suffix}.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = [
        cell for cell in notebook["cells"]
        if CELL_TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    if not direct_halpha:
        for cell in cells:
            if cell["cell_type"] != "markdown":
                continue
            text = "".join(cell["source"])
            text = text.replace("the optional isotropic appendix", "the isotropic appendix")
            text = text.replace("KTZ + isotropic C_NP", "complete KTZ")
            text = text.replace("The primary KTZ fit", "The PHANGS-mask KTZ control")
            cell["source"] = text.splitlines(keepends=True)
    cells.append({
        "cell_type": "markdown",
        "metadata": {"tags": [CELL_TAG]},
        "source": [
            "## Exploratory external diagnostics\n",
            "\n",
            "This diagnostic does not add another metallicity-correlation fit. "
            "Qualified PHANGS H II-region centroids are compared with an "
            "inhomogeneous random catalogue drawn from the independently fixed "
            "PHANGS-mask radial-plus-arm source intensity. The source-pair "
            "comparison uses the mask-control posterior, not the primary "
            "direct-H-alpha CNP posterior. The first 0--0.12 kpc bin is "
            "excluded because the nebular segmentation/deblending procedure "
            "suppresses distinct catalogue centroids at that separation. "
            "For NGC 4303, the second panel removes every bin at $r\\geq3$ kpc "
            "from the likelihood and predicts the 3.3--4.5 kpc negative tail "
            "by importance reweighting the converged chains. For NGC 1566 and "
            "NGC 4303, a separate diagnostic subtracts the declared prior from "
            "the saved posterior log density and identifies the highest-likelihood "
            "retained chain state. It tests whether posterior volume, rather than "
            "the KTZ curve family, causes a visible tail mismatch; it does not "
            "replace the posterior summaries.\n",
        ],
    })

    figure_path = output_directory(galaxy) / (
        f"{galaxy.lower()}_physical_validation.png"
    )
    encoded_figure = base64.b64encode(figure_path.read_bytes()).decode("ascii")
    code_source = [
        "from IPython.display import Image, display\n",
        "physical_validation_figure = ROOT / "
        + repr(str(figure_path.relative_to(ROOT))) + "\n",
        "display(Image(filename=str(physical_validation_figure)))\n",
        "display(Table.read(\n",
        "    OUTPUT_ROOT / \"phangs_hii_source_pair_summary.csv\",\n",
        "    format=\"ascii.csv\",\n",
        "))\n",
    ]
    if galaxy == "NGC4303":
        code_source.extend([
            "display(Table.read(\n",
            "    OUTPUT_ROOT / \"ngc4303_heldout_tail_summary.csv\",\n",
            "    format=\"ascii.csv\",\n",
            "))\n",
        ])
    if galaxy in {"NGC1566", "NGC4303"}:
        code_source.extend([
            "display(Image(filename=str(\n",
            "    OUTPUT_ROOT / \"ngc1566_ngc4303_maximum_likelihood_diagnostic.png\"\n",
            ")))\n",
            "display(Table.read(\n",
            "    OUTPUT_ROOT / \"ngc1566_ngc4303_maximum_likelihood_summary.csv\",\n",
            "    format=\"ascii.csv\",\n",
            "))\n",
        ])
        likelihood_figure_path = OUTPUT_ROOT / (
            "ngc1566_ngc4303_maximum_likelihood_diagnostic.png"
        )
        encoded_likelihood_figure = base64.b64encode(
            likelihood_figure_path.read_bytes()
        ).decode("ascii")
    else:
        encoded_likelihood_figure = None
    output_blocks = [
        {
            "data": {
                "image/png": encoded_figure,
                "text/plain": [
                    f"<Exploratory external diagnostics for {galaxy}>"
                ],
            },
            "metadata": {},
            "output_type": "display_data",
        },
    ]
    if encoded_likelihood_figure is not None:
        output_blocks.append({
            "data": {
                "image/png": encoded_likelihood_figure,
                "text/plain": [
                    f"<Maximum-likelihood tail diagnostic for {galaxy}>"
                ],
            },
            "metadata": {},
            "output_type": "display_data",
        })
    output_blocks.append({
        "name": "stdout",
        "output_type": "stream",
        "text": summary_text(galaxy).splitlines(keepends=True),
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [CELL_TAG]},
        "outputs": output_blocks,
        "source": code_source,
    })
    notebook["cells"] = cells
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for galaxy in GALAXIES:
        update_notebook(galaxy)
        update_notebook(galaxy, direct_halpha=True)


if __name__ == "__main__":
    main()
