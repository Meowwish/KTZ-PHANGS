#!/usr/bin/env python3
"""Execute the data-preparation and map cells of PHANGS implication notebooks.

The KTZ inference is intentionally not run here.  This helper writes
the executed Li21-style selection, metallicity-map, and radial-diagnostic cells
back to the source notebook so they are immediately visible in Jupyter.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_map_stage(path: Path, last_cell: int = 9) -> None:
    notebook = nbformat.read(path, as_version=4)
    if len(notebook.cells) <= last_cell:
        raise ValueError(f"{path} contains only {len(notebook.cells)} cells")

    partial = deepcopy(notebook)
    partial.cells = partial.cells[: last_cell + 1]
    client = NotebookClient(
        partial,
        timeout=-1,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent.resolve())}},
    )
    client.execute()

    for index, executed_cell in enumerate(partial.cells):
        if executed_cell.cell_type != "code":
            continue
        notebook.cells[index].outputs = deepcopy(executed_cell.outputs)
        notebook.cells[index].execution_count = executed_cell.execution_count
    nbformat.write(notebook, path)
    print(f"Executed Li21 map stage through cell {last_cell}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--last-cell", type=int, default=9)
    args = parser.parse_args()
    for notebook_path in args.notebooks:
        execute_map_stage(notebook_path.resolve(), args.last_cell)


if __name__ == "__main__":
    main()
