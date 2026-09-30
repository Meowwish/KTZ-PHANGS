"""Execute the refit, saving notebook outputs after each completed cell."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'NGC628_PHANGS_window_projection_refit.ipynb'
nb=nbformat.read(PATH,as_version=4)


def save(**kwargs):
    nbformat.write(nb,PATH)


def announce(cell,cell_index,**kwargs):
    print(f'Executing cell {cell_index}: '+cell.source[:90].replace('\n',' '),flush=True)


client=NotebookClient(nb,timeout=None,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},on_cell_executed=save,on_cell_start=announce)
try:
    client.execute()
finally:
    save()
print('NGC628 corrected-window notebook execution complete.',flush=True)
