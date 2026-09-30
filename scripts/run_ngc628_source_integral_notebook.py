"""Execute staged validation and reference integration; save every finished cell."""

import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'NGC628_PHANGS_source_integral_refit.ipynb'
OUTPUT = ROOT/'outputs/implications/NGC628_source_integral_refit'
nb = nbformat.read(PATH, as_version=4)


def save(**kwargs):
    nbformat.write(nb, PATH)


def announce(cell, cell_index, **kwargs):
    print(f'Executing cell {cell_index}: '+cell.source[:90].replace('\n', ' '), flush=True)


client = NotebookClient(nb, timeout=None, kernel_name='python3',
    resources={'metadata': {'path': str(ROOT)}}, on_cell_executed=save, on_cell_start=announce)
if __name__ == '__main__':
    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        client.execute()
    except Exception:
        (OUTPUT/'ngc628_source_integral_execution_error.txt').write_text(traceback.format_exc())
        raise
    finally:
        save()
    print('NGC628 source-integral validation/reference notebook execution complete.', flush=True)
