"""Single-run I/O benchmark. OS caches can affect results."""
from pathlib import Path
from contextlib import closing
import sys
import time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config import DATA,TABLES
from src.data import load_data
from src.database import connect

rows=[]
for name,loader in [
    ('ZIP CSV / pandas',lambda:load_data()[0]),
    ('Parquet / pandas',lambda:pd.read_parquet(DATA/'raw.parquet')),
]:
    started=time.perf_counter(); frame=loader()
    rows.append(dict(method=name,rows=len(frame),read_time_s=time.perf_counter()-started,
                     memory_MiB=frame.memory_usage(deep=True).sum()/1024**2))
    del frame
with closing(connect()) as connection:
    started=time.perf_counter()
    frame=pd.read_sql_query('SELECT * FROM encounters',connection)
    rows.append(dict(method='SQLite / pandas (normalized encounters only)',rows=len(frame),
                     read_time_s=time.perf_counter()-started,memory_MiB=frame.memory_usage(deep=True).sum()/1024**2))
result=pd.DataFrame(rows)
result.to_csv(TABLES/'io_benchmark.csv',index=False)
print(result.to_string(index=False))
