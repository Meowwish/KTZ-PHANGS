"""Run only the new convolution/library cells; preserve previous visual outputs."""

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


if __name__ == '__main__':
    client = NotebookClient(nb, timeout=None, kernel_name='python3',
        resources={'metadata': {'path': str(ROOT)}}, on_cell_executed=save)
    try:
        with client.setup_kernel():
            for i, c in enumerate(nb.cells):
                if c.get('id') in ('source-integral-fft-benchmark', 'source-integral-age-library-run'):
                    print('Executing new response cell:', c.id, flush=True)
                    client.execute_cell(c, i)
    except Exception:
        (OUTPUT/'ngc628_response_library_execution_error.txt').write_text(traceback.format_exc())
        raise
    finally:
        save()
    print('Convolution benchmark and reusable response library completed.', flush=True)
