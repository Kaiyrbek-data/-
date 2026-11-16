"""Create a portable submission archive, excluding caches and local backups."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--destination',type=Path,default=ROOT/'release'/'Diabetes_Readmission_Project.zip')
args=parser.parse_args()
destination=args.destination.resolve()
destination.parent.mkdir(parents=True,exist_ok=True)
excluded={'.git','.venv','__pycache__','.ipynb_checkpoints','.pytest_cache','_backup','release','work'}
with ZipFile(destination,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.resolve()==destination: continue
        relative=path.relative_to(ROOT)
        if any(part in excluded for part in relative.parts): continue
        if path.suffix in ['.pyc','.pyo'] or path.name.endswith(('-journal','-wal','-shm')): continue
        archive.write(path,arcname=Path('Diabetes_Readmission_Project')/relative)
print(destination)
