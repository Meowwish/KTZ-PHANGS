"""Execute only the appended audit, retaining every expensive refit output."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'NGC628_PHANGS_window_projection_refit.ipynb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached', action='store_true',
                        help='Embed existing validated diagnostic products; do not reoptimise.')
    args = parser.parse_args()
    original = nbformat.read(TARGET, as_version=4)
    index = next(i for i, c in enumerate(original.cells)
                 if c.get('id') == 'completed-window-audit')
    setup = nbformat.v4.new_code_cell('''from pathlib import Path
from astropy.table import Table
from IPython.display import display, Image, Markdown
OUTPUT_DIR = Path("outputs/implications/NGC628_window_projection_refit")
''')
    audit = nbformat.v4.new_code_cell(original.cells[index].source)
    if args.cached:
        audit.source = audit.source.replace(
            'window_diagnostics = audit_completed_refit()',
            'import json\nwindow_diagnostics = json.loads((OUTPUT_DIR / "ngc628_window_refit_diagnostics.json").read_text())',
        )
    temporary = nbformat.v4.new_notebook(cells=[setup, audit])
    client = NotebookClient(
        temporary, timeout=None, kernel_name='python3',
        resources={'metadata': {'path': str(ROOT)}},
    )
    try:
        client.execute()
    finally:
        original.cells[index].outputs = temporary.cells[1].outputs
        previous = [c.execution_count for c in original.cells[:index]
                    if c.cell_type == 'code' and c.execution_count is not None]
        original.cells[index].execution_count = max(previous, default=0)+1
        original.cells[index].metadata = temporary.cells[1].metadata
        nbformat.write(original, TARGET)
    print('Completed audit; original refit outputs preserved.', flush=True)


if __name__ == '__main__':
    main()
