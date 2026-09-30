"""Continue after the running age job; avoid concurrent notebook overwrites."""

import json
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from make_ngc628_source_integral_notebook import cnp_notebook_cells

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'NGC628_PHANGS_source_integral_refit.ipynb'
OUTPUT = ROOT/'outputs/implications/NGC628_source_integral_refit'


def wait_for_upstream():
    state_path = OUTPUT/'ngc628_cnp_pipeline_progress.json'
    state_path.write_text(json.dumps({'stage': 'waiting for age-response validation and completed notebook save',
                                    'not_a_galaxy_fit': True}, indent=2)+'\n')
    start = time.monotonic()
    while time.monotonic()-start < 12*3600:
        error = OUTPUT/'ngc628_response_library_execution_error.txt'
        if error.exists():
            raise RuntimeError('Upstream response calculation failed; see '+str(error))
        log = OUTPUT/'ngc628_response_library_runner.log'
        gate = OUTPUT/'ngc628_age_library_validation.json'
        if gate.exists() and log.exists() and 'Convolution benchmark and reusable response library completed.' in log.read_text():
            validation = json.loads(gate.read_text())
            if not validation['tested_CP_mean_precision_gate_passed']:
                raise RuntimeError('CP/mean interpolation precision failed. Refine the numerical response before continuing.')
            return
        time.sleep(15)
    raise RuntimeError('Upstream response calculation exceeded the bounded dependency wait.')


if __name__ == '__main__':
    nb = None
    try:
        wait_for_upstream()
        nb = nbformat.read(PATH, as_version=4)
        existing = {c.get('id') for c in nb.cells}
        for c in cnp_notebook_cells():
            if c['id'] not in existing:
                c['source'] = ''.join(c['source'])
                nb.cells.append(nbformat.from_dict(c))
        nbformat.write(nb, PATH)
        def save(**kwargs):
            nbformat.write(nb, PATH)
        client = NotebookClient(nb, timeout=None, kernel_name='python3',
            resources={'metadata': {'path': str(ROOT)}}, on_cell_executed=save)
        with client.setup_kernel():
            for i, c in enumerate(nb.cells):
                if c.get('id') in ('source-integral-cnp-control-run', 'source-integral-cnp-pilot-run'):
                    (OUTPUT/'ngc628_cnp_pipeline_progress.json').write_text(json.dumps(
                        {'stage': c.id, 'not_a_galaxy_fit': True}, indent=2)+'\n')
                    print('Executing nonlocal CNP response cell:', c.id, flush=True)
                    client.execute_cell(c, i)
        validation = json.loads((OUTPUT/'ngc628_cnp_pilot_validation.json').read_text())
        (OUTPUT/'ngc628_cnp_pipeline_progress.json').write_text(json.dumps({'stage': 'CNP pilot completed',
            'tested_CNP_precision_gate_passed': validation['tested_CNP_precision_gate_passed'],
            'not_a_galaxy_fit': True}, indent=2)+'\n')
    except Exception:
        (OUTPUT/'ngc628_cnp_pilot_execution_error.txt').write_text(traceback.format_exc())
        raise
    finally:
        if nb is not None:
            nbformat.write(nb, PATH)
    print('Nonlocal CNP source-response pilot complete; no galaxy MCMC was started.', flush=True)
