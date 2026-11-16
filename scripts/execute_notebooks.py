"""Execute notebooks in order and fail on any erroneous code cell."""
import asyncio
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
if sys.platform=='win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
for name in ['EDA.ipynb','SQL_analysis.ipynb','Model_results.ipynb']:
    notebook=nbformat.read(ROOT/name,as_version=4)
    NotebookClient(notebook,timeout=300,kernel_name='python3',
                   resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(notebook,ROOT/name)
    print('Executed:',name,flush=True)
