"""ZIP ingestion, validated cohort and patient-disjoint partitions."""

import csv
import hashlib
import json
from io import StringIO
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .config import *


def load_data(path=ZIP_PATH):
    """Прочитать CSV и справочники прямо из ZIP, проверить ключи и значения ответа.

    Возвращает: исходную таблицу, словарь справочников и сведения об архиве.
    Знак «?» и пустая ячейка считаются пропуском."""
    if not path.exists():
        raise FileNotFoundError(f'ZIP not found: {path}')
    with ZipFile(path) as archive:
        with archive.open('diabetic_data.csv') as source:
            raw = pd.read_csv(
                source, dtype='string', keep_default_na=False, na_values=['?', ''], low_memory=False
            )
        mapping_text = archive.read('IDS_mapping.csv').decode('utf-8-sig')
        size = archive.getinfo('diabetic_data.csv').file_size
    for column in ID_COLS + NUM_COLS:
        raw[column] = pd.to_numeric(raw[column], errors='raise').astype('int64')
    if raw[ID_COLS].isna().any().any() or not raw.encounter_id.is_unique:
        raise ValueError('Missing identifiers or duplicate encounter_id')
    if not raw.readmitted.isin(['NO', '>30', '<30']).all():
        raise ValueError('Unknown/missing target values')
    mappings, current = {}, None
    for row in csv.reader(StringIO(mapping_text)):
        if not row or not row[0].strip():
            continue
        key = row[0].strip()
        if key in CODE_COLS:
            current = key
            mappings[current] = {}
        elif current and key.isdigit():
            mappings[current][key] = row[1].strip()
    return (
        raw,
        mappings,
        {'csv_bytes': size, 'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
    )


def make_cohort(raw):
    """Population defined before fitting: alive and not discharged to hospice."""
    excluded = raw.discharge_disposition_id.isin(DEATH_CODES + HOSPICE_CODES)
    cohort = raw.loc[~excluded].copy()
    cohort[TARGET] = cohort.readmitted.eq('<30').astype('int8')
    flow = {
        'raw_encounters': len(raw),
        'raw_patients': raw.patient_nbr.nunique(),
        'death_encounters': int(raw.discharge_disposition_id.isin(DEATH_CODES).sum()),
        'hospice_encounters': int(raw.discharge_disposition_id.isin(HOSPICE_CODES).sum()),
        'cohort_encounters': len(cohort),
        'cohort_patients': cohort.patient_nbr.nunique(),
        'cohort_positive': int(cohort[TARGET].sum()),
    }
    return cohort, flow


def patient_split(cohort, seed=SEED):
    """60/10/10/20 percent of patients for fit/calibration/validation/test."""

    def split(frame, fraction, random_state):
        """Отделить долю fraction ПАЦИЕНТОВ: все визиты одного человека уходят в одну часть."""
        a, b = next(
            GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=random_state).split(
                frame, groups=frame.patient_nbr
            )
        )
        return frame.iloc[a].copy(), frame.iloc[b].copy()

    development, test = split(cohort, 0.20, seed)
    fit, holdout = split(development, 0.25, seed + 1)
    calibration, validation = split(holdout, 0.50, seed + 2)
    parts = dict(fit=fit, calibration=calibration, validation=validation, test=test)
    for name, frame in parts.items():
        if frame[TARGET].nunique() != 2:
            raise ValueError(f'Both classes are required in {name}')
        for other, another in parts.items():
            if name != other and not set(frame.patient_nbr).isdisjoint(another.patient_nbr):
                raise AssertionError(f'Patient leakage: {name}/{other}')
    return parts


def optimize_types(frame):
    """Сжать типы столбцов: числа — в меньший целый тип, текст — в category."""
    optimized = frame.copy()
    for column in optimized.select_dtypes(include='number'):
        optimized[column] = pd.to_numeric(optimized[column], downcast='integer')
    for column in optimized.select_dtypes(include=['string', 'object']):
        optimized[column] = optimized[column].astype('category')
    return optimized


def save_data_assets(raw, cohort, parts, source_info, flow):
    """Сохранить данные в Parquet, список разбиения и сводку (память до и после сжатия)."""
    ensure_directories()
    raw.to_parquet(DATA / 'raw.parquet', index=False)
    cohort.to_parquet(DATA / 'cohort.parquet', index=False)
    manifest = pd.concat(
        [f[['encounter_id', 'patient_nbr']].assign(split=n) for n, f in parts.items()],
        ignore_index=True,
    )
    manifest.to_parquet(DATA / 'split_manifest.parquet', index=False)
    optimized = optimize_types(raw)
    memory = {
        'before_MiB': raw.memory_usage(deep=True).sum() / 1024**2,
        'after_MiB': optimized.memory_usage(deep=True).sum() / 1024**2,
    }
    memory['reduction_pct'] = 100 * (1 - memory['after_MiB'] / memory['before_MiB'])
    meta = dict(
        source_info=source_info,
        cohort_flow=flow,
        memory=memory,
        seed=SEED,
        cohort_policy='Exclude death and hospice discharge codes; retain repeated encounters',
        prediction_time='At discharge',
        authors=AUTHORS,
    )
    (REPORTS / 'data_summary.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    split_info = pd.DataFrame(
        [
            {
                'split': n,
                'encounters': len(f),
                'patients': f.patient_nbr.nunique(),
                'positives': int(f[TARGET].sum()),
                'positive_rate': float(f[TARGET].mean()),
            }
            for n, f in parts.items()
        ]
    )
    split_info.to_csv(TABLES / 'split_summary.csv', index=False)
    return meta
