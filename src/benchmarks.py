"""Сравнение одного и того же набора данных и одинакового расчёта двумя способами."""
import time
import tracemalloc
import pandas as pd
import numpy as np
from .config import DATA, TABLES, TARGET
from .pipeline import ensure_data_ready


def _measure(operation):
    """Вернуть результат, время и пик памяти, отслеживаемой tracemalloc."""
    tracemalloc.start()
    started = time.perf_counter()
    try:
        result = operation()
        return result, time.perf_counter()-started, tracemalloc.get_traced_memory()[1]/1024**2
    finally:
        tracemalloc.stop()


def run_all():
    """Сравнить ZIP CSV с Parquet и Pandas GROUP BY с SQLite GROUP BY."""
    from .data import load_data
    from .search import read_query
    ensure_data_ready()
    rows = []
    csv_frame = None
    for label, reader in [('ZIP CSV', lambda: load_data()[0]),
                          ('Parquet', lambda: pd.read_parquet(DATA/'raw.parquet'))]:
        frame, seconds, peak = _measure(reader)
        if csv_frame is None:
            csv_frame = frame
            same = True
        else:
            same = csv_frame.equals(frame)
        rows.append(dict(format=label, rows=len(frame), columns=len(frame.columns),
                         seconds=seconds, peak_python_mb=peak,
                         dataframe_mb=frame.memory_usage(deep=True).sum()/1024**2,
                         results_match=same))
    def pandas_groupby():
        frame = pd.read_parquet(DATA/'raw.parquet', columns=['age', 'readmitted'])
        frame[TARGET] = frame.readmitted.eq('<30').astype(int)
        return frame.groupby('age')[TARGET].mean().rename('rate').reset_index().sort_values('age')
    pandas_result, p_sec, p_mem = _measure(pandas_groupby)
    sql_result, s_sec, s_mem = _measure(lambda: read_query(
        f'SELECT age, AVG({TARGET}) AS rate FROM encounters GROUP BY age ORDER BY age'))
    match = (pandas_result.age.astype(str).tolist() == sql_result.age.astype(str).tolist()
             and np.allclose(pandas_result.rate, sql_result.rate))
    grouping = pd.DataFrame([
        dict(approach='Pandas groupby', seconds=p_sec, peak_python_mb=p_mem, results_match=match),
        dict(approach='SQLite GROUP BY', seconds=s_sec, peak_python_mb=s_mem, results_match=match)])
    results = {'csv_vs_parquet': pd.DataFrame(rows), 'pandas_vs_sql': grouping}
    for name, table in results.items():
        table.to_csv(TABLES/f'benchmark_{name}.csv', index=False)
    return results
