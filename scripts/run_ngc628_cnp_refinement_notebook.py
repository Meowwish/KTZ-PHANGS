"""Append and execute bounded numerical refinement, preserving earlier cells."""

import contextlib
import fcntl
import json
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from scripts.refine_ngc628_cnp_response import notebook_cells
from scripts.test_phangs_cnp_emulator import run_tests

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'NGC628_PHANGS_source_integral_refit.ipynb'
OUTPUT = ROOT/'outputs/implications/NGC628_source_integral_refit/cnp_refinement_v1'


def run():
    OUTPUT.mkdir(exist_ok=True)
    nb = None
    try:
        algebra = run_tests()
        (OUTPUT/'encoding_algebra.json').write_text(json.dumps(algebra, indent=2)+'\n')
        nb = nbformat.read(PATH, as_version=4)
        existing = {c.id for c in nb.cells}
        added = notebook_cells()
        for cell in added:
            if cell.id not in existing:
                nb.cells.append(cell)
        nbformat.write(nb, PATH)
        def save(**kwargs):
            nbformat.write(nb, PATH)
        client = NotebookClient(nb, timeout=None, kernel_name='python3',
            resources={'metadata': {'path': str(ROOT)}}, on_cell_executed=save)
        code_ids = {c.id for c in added if c.cell_type == 'code'}
        with client.setup_kernel():
            for i, cell in enumerate(nb.cells):
                if cell.id in code_ids:
                    print('Executing bounded CNP refinement cell:', cell.id, flush=True)
                    client.execute_cell(cell, i)
        print('Bounded CNP refinement complete; no galaxy MCMC started.', flush=True)
    except Exception:
        (OUTPUT/'execution_error.txt').write_text(traceback.format_exc())
        raise
    finally:
        if nb is not None:
            nbformat.write(nb, PATH)


if __name__ == '__main__':
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT/'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('The bounded refinement is already running; do not launch a duplicate.')
        with (OUTPUT/'runner.log').open('a', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                run()
