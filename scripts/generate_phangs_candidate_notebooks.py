#!/usr/bin/env python3
"""Generate independent PHANGS/KTZ notebooks from the NGC 1566 template."""

from __future__ import annotations

import copy
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
TEMPLATE = WORKSPACE / "NGC1566_PHANGS_implications.ipynb"

CANDIDATES = {
    "NGC628": {
        "display": "NGC 628",
        "data_token": "NGC0628",
        "output_prefix": "ngc628",
        "flat_directories": True,
        "distance": 9.8,
        "r25": 4.9,
        "inclination": 8.9,
        "pa": 20.7,
        "psf": 0.92,
        "seed": 628,
        "catalog_id": "ADP.2025-05-14T09:12:30.213",
        "mask_id": "ADP.2025-05-14T09:12:30.214",
        "maps_id": "ADP.2021-07-16T10:20:56.338",
        "r_band_id": "ADP.2021-07-16T10:20:56.336",
        "maps_bytes": 854029440,
        "r_band_bytes": 18161280,
    },
    "NGC1300": {
        "display": "NGC 1300",
        "distance": 19.0,
        "r25": 3.0,
        "inclination": 31.8,
        "pa": 278.0,
        "psf": 0.89,
        "seed": 1300,
        "catalog_id": "ADP.2025-05-14T09:12:30.217",
        "mask_id": "ADP.2025-05-14T09:12:30.218",
        "maps_id": "ADP.2021-07-16T10:20:56.362",
        "r_band_id": "ADP.2021-07-16T10:20:56.360",
        "maps_bytes": 564168960,
        "r_band_bytes": 12012480,
    },
    "NGC4254": {
        "display": "NGC 4254",
        "distance": 13.1,
        "r25": 2.5,
        "inclination": 34.4,
        "pa": 68.1,
        "psf": 0.89,
        "seed": 4254,
        "catalog_id": "ADP.2025-05-14T09:12:30.237",
        "mask_id": "ADP.2025-05-14T09:12:30.238",
        "maps_id": "ADP.2021-07-16T10:20:56.482",
        "r_band_id": "ADP.2021-07-16T10:20:56.480",
        "maps_bytes": 510897600,
        "r_band_bytes": 10880640,
    },
    "NGC4303": {
        "display": "NGC 4303",
        "distance": 17.0,
        "r25": 3.4,
        "inclination": 23.5,
        "pa": 312.4,
        "psf": 0.78,
        "seed": 4303,
        "catalog_id": "ADP.2025-05-14T09:12:30.239",
        "mask_id": "ADP.2025-05-14T09:12:30.240",
        "maps_id": "ADP.2021-07-16T10:20:56.494",
        "r_band_id": "ADP.2021-07-16T10:20:56.492",
        "maps_bytes": 342838080,
        "r_band_bytes": 7312320,
    },
    "NGC4321": {
        "display": "NGC 4321",
        "distance": 15.2,
        "r25": 3.0,
        "inclination": 38.5,
        "pa": 156.2,
        "psf": 1.16,
        "seed": 4321,
        "catalog_id": "ADP.2025-05-14T09:12:30.241",
        "mask_id": "ADP.2025-05-14T09:12:30.242",
        "maps_id": "ADP.2021-07-16T10:20:56.506",
        "r_band_id": "ADP.2021-07-16T10:20:56.504",
        "maps_bytes": 521732160,
        "r_band_bytes": 11111040,
    },
    "NGC4535": {
        "display": "NGC 4535",
        "distance": 15.8,
        "r25": 4.1,
        "inclination": 44.7,
        "pa": 179.7,
        "psf": 0.56,
        "seed": 4535,
        "catalog_id": "ADP.2025-05-14T09:12:30.243",
        "mask_id": "ADP.2025-05-14T09:12:30.244",
        "maps_id": "ADP.2021-07-16T10:20:56.518",
        "r_band_id": "ADP.2021-07-16T10:20:56.516",
        "maps_bytes": 232479360,
        "r_band_bytes": 4965120,
    },
}


def replace_required(text: str, old: str, new: str, *, count: int | None = None) -> str:
    found = text.count(old)
    if found == 0:
        raise RuntimeError(f"Template token not found: {old!r}")
    if count is not None and found != count:
        raise RuntimeError(f"Expected {count} occurrence(s) of {old!r}; found {found}")
    return text.replace(old, new)


def render_notebook(template: dict, galaxy: str, cfg: dict) -> dict:
    nb = copy.deepcopy(template)
    data_token = cfg.get("data_token", galaxy)
    compact_lower = cfg.get("output_prefix", galaxy.lower())
    psf = f"{cfg['psf']:.2f}"

    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        source = source.replace("NGC 1566", cfg["display"])
        source = source.replace("NGC1566", data_token)
        source = source.replace("ngc1566", compact_lower)
        source = source.replace("0.80asec", f"{psf}asec")
        source = source.replace("0.80 arcsec", f"{psf} arcsec")
        source = source.replace("DISTANCE_MPC = 17.7", f"DISTANCE_MPC = {cfg['distance']:.1f}")
        source = source.replace("R25_ARCMIN = 3.6", f"R25_ARCMIN = {cfg['r25']:.1f}")
        source = source.replace(
            "INCLINATION_DEG = 29.5",
            f"INCLINATION_DEG = {cfg['inclination']:.1f}",
        )
        source = source.replace(
            "POSITION_ANGLE_DEG = 214.7",
            f"POSITION_ANGLE_DEG = {cfg['pa']:.1f}",
        )
        source = source.replace(
            "COPT_PSF_FWHM_ARCSEC = 0.80",
            f"COPT_PSF_FWHM_ARCSEC = {cfg['psf']:.2f}",
        )
        source = source.replace(
            "np.random.default_rng(156621)",
            f"np.random.default_rng({cfg['seed'] * 100 + 21})",
        )
        source = source.replace(
            "np.random.default_rng(1566)",
            f"np.random.default_rng({cfg['seed']})",
        )
        source = source.replace(
            "ADP.2025-05-14T09:12:30.227", cfg["catalog_id"]
        )
        source = source.replace(
            "ADP.2025-05-14T09:12:30.228", cfg["mask_id"]
        )
        source = source.replace(
            "ADP.2021-07-16T10:20:56.422", cfg["maps_id"]
        )
        source = source.replace(
            "ADP.2021-07-16T10:20:56.420", cfg["r_band_id"]
        )
        maps_mib = cfg["maps_bytes"] / 1024**2
        r_band_mib = cfg["r_band_bytes"] / 1024**2
        source = source.replace(
            "approximately 431 MiB and 9 MiB",
            f"approximately {maps_mib:.0f} MiB and {r_band_mib:.0f} MiB",
        )
        if cfg.get("flat_directories"):
            source = source.replace(
                f'DATA_DIR = DATA_ROOT / "{data_token}"',
                "DATA_DIR = DATA_ROOT",
            )
            source = source.replace(
                f'OUTPUT_DIR = OUTPUT_ROOT / "{data_token}"',
                "OUTPUT_DIR = OUTPUT_ROOT",
            )
        cell["source"] = source.splitlines(keepends=True)
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    serialized = json.dumps(nb, ensure_ascii=False)
    forbidden = ("NGC1566", "NGC 1566", "ngc1566", "0.80asec")
    leftovers = [token for token in forbidden if token in serialized]
    if leftovers:
        raise RuntimeError(f"Template tokens remain for {galaxy}: {leftovers}")
    return nb


def main() -> None:
    template = json.loads(TEMPLATE.read_text())
    for galaxy, cfg in CANDIDATES.items():
        output = WORKSPACE / f"{galaxy}_PHANGS_implications.ipynb"
        notebook = render_notebook(template, galaxy, cfg)
        output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        print(output)


if __name__ == "__main__":
    main()
