"""Выполнить подробный учебный ноутбук и сохранить таблицы и графики в outputs ячеек."""
import asyncio
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'Model_training_detailed.ipynb'
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
notebook = nbformat.read(path, as_version=4)


def announce(cell, cell_index):
    """Печатать текущую ячейку, чтобы видеть продвижение длительного обучения."""
    if cell.cell_type == 'code':
        print(f'Cell {cell_index}: {cell.source.splitlines()[0][:100]}', flush=True)


def checkpoint(cell, cell_index, **kwargs):
    """Сохранять выполненные ячейки и выводить текст обучения после завершения этапа."""
    nbformat.write(notebook, path)
    for output in cell.get('outputs', []):
        if output.output_type == 'stream' and output.get('text', '').strip():
            print(output.text[-2000:], flush=True)


client = NotebookClient(notebook, timeout=1800, kernel_name='python3',
                        resources={'metadata': {'path': str(ROOT)}},
                        on_cell_start=announce, on_cell_executed=checkpoint)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print('Executed:', path.name, flush=True)
