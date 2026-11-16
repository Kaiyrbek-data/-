"""Relational SQLite storage with repeatable project-table ingestion."""

import sqlite3
from contextlib import closing
import pandas as pd
from .config import *
from .features import icd9_group


def connect(path=DB_PATH):
    """Открыть соединение с SQLite и включить проверку внешних ключей."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute('PRAGMA foreign_keys=ON')
    return connection


def import_database(raw, mappings, parts):
    """Rebuild only the project's source tables, retaining experiment history."""
    encounter_cols = [c for c in raw if c not in DRUG_COLS + DIAG_COLS + ['patient_nbr']]
    other_cols = [c for c in encounter_cols if c != 'encounter_id']
    integer_cols = set(NUM_COLS)
    col_sql = ',\n'.join(f'"{c}" {"INTEGER" if c in integer_cols else "TEXT"}' for c in other_cols)
    schema = f'''
    CREATE TABLE patients(patient_nbr INTEGER PRIMARY KEY);
    CREATE TABLE admission_types(code TEXT PRIMARY KEY, description TEXT NOT NULL);
    CREATE TABLE discharge_dispositions(code TEXT PRIMARY KEY, description TEXT NOT NULL);
    CREATE TABLE admission_sources(code TEXT PRIMARY KEY, description TEXT NOT NULL);
    CREATE TABLE encounters(
      encounter_id INTEGER PRIMARY KEY,
      patient_nbr INTEGER NOT NULL REFERENCES patients(patient_nbr),
      {col_sql},
      readmitted_30d INTEGER NOT NULL CHECK(readmitted_30d IN (0,1)),
      FOREIGN KEY(admission_type_id) REFERENCES admission_types(code),
      FOREIGN KEY(discharge_disposition_id) REFERENCES discharge_dispositions(code),
      FOREIGN KEY(admission_source_id) REFERENCES admission_sources(code));
    CREATE TABLE diagnoses(
      encounter_id INTEGER REFERENCES encounters(encounter_id),
      position INTEGER CHECK(position BETWEEN 1 AND 3), icd9 TEXT, diagnosis_group TEXT,
      PRIMARY KEY(encounter_id,position));
    CREATE TABLE drugs(drug_id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
    CREATE TABLE encounter_medications(
      encounter_id INTEGER REFERENCES encounters(encounter_id),
      drug_id INTEGER REFERENCES drugs(drug_id), status TEXT NOT NULL,
      PRIMARY KEY(encounter_id,drug_id));
    CREATE TABLE split_assignments(
      encounter_id INTEGER PRIMARY KEY REFERENCES encounters(encounter_id),
      split TEXT NOT NULL CHECK(split IN ('fit','calibration','validation','test')));
    CREATE INDEX idx_encounters_patient ON encounters(patient_nbr);
    CREATE INDEX idx_diagnoses_group ON diagnoses(diagnosis_group);
    '''
    with closing(connect()) as connection:
        connection.execute('BEGIN')
        for table in [
            'split_assignments',
            'encounter_medications',
            'diagnoses',
            'drugs',
            'encounters',
            'patients',
            'admission_types',
            'discharge_dispositions',
            'admission_sources',
        ]:
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
        connection.executescript(schema)
        raw[['patient_nbr']].drop_duplicates().to_sql(
            'patients', connection, if_exists='append', index=False
        )
        for column, table in zip(
            CODE_COLS, ['admission_types', 'discharge_dispositions', 'admission_sources']
        ):
            connection.executemany(
                f'INSERT INTO {table}(code,description) VALUES (?,?)', mappings[column].items()
            )
        encounters = raw[["encounter_id", "patient_nbr"] + other_cols].copy()
        encounters[TARGET] = raw.readmitted.eq('<30').astype(int)
        encounters.to_sql('encounters', connection, if_exists='append', index=False, chunksize=3000)
        diagnoses = raw[['encounter_id'] + DIAG_COLS].melt(
            id_vars='encounter_id', var_name='position', value_name='icd9'
        )
        diagnoses['position'] = diagnoses.position.str[-1].astype(int)
        diagnoses['diagnosis_group'] = diagnoses.icd9.map(icd9_group)
        diagnoses.to_sql('diagnoses', connection, if_exists='append', index=False, chunksize=3000)
        connection.executemany(
            'INSERT INTO drugs(drug_id,name) VALUES (?,?)', enumerate(DRUG_COLS, start=1)
        )
        medication = raw[['encounter_id'] + DRUG_COLS].melt(
            id_vars='encounter_id', var_name='drug', value_name='status'
        )
        medication = medication[medication.status.ne('No') & medication.status.notna()].copy()
        medication['drug_id'] = medication.drug.map(
            {name: i for i, name in enumerate(DRUG_COLS, start=1)}
        )
        medication[['encounter_id', 'drug_id', 'status']].to_sql(
            'encounter_medications', connection, if_exists='append', index=False, chunksize=3000
        )
        manifest = pd.concat(
            [frame[['encounter_id']].assign(split=name) for name, frame in parts.items()]
        )
        manifest.to_sql('split_assignments', connection, if_exists='append', index=False)
        violations = connection.execute('PRAGMA foreign_key_check').fetchall()
        if violations:
            raise ValueError(f'Foreign key violations: {violations[:3]}')
        connection.commit()


def load_sql_queries():
    """Прочитать запросы из sql/analysis.sql в словарь {имя: текст SQL}."""
    text = (ROOT / 'sql' / 'analysis.sql').read_text(encoding='utf-8-sig')
    queries = {}
    for section in text.split('-- QUERY: ')[1:]:
        title, sql = section.split('\n', 1)
        queries[title.strip()] = sql.strip().rstrip(';')
    return queries


def run_sql_analysis():
    """Выполнить все запросы, вернуть результаты как DataFrame и сохранить в CSV."""
    results = {}
    with closing(connect()) as connection:
        for name, query in load_sql_queries().items():
            result = pd.read_sql_query(query, connection)
            result.to_csv(TABLES / f'sql_{name}.csv', index=False)
            results[name] = result
    return results


def save_experiments(run_id, metrics, predictions):
    """Дописать метрики и прогнозы запуска в таблицы experiments и predictions."""
    with closing(connect()) as connection:
        metrics.assign(run_id=run_id).to_sql(
            'experiments', connection, if_exists='append', index=False
        )
        predictions.assign(run_id=run_id).to_sql(
            'predictions', connection, if_exists='append', index=False
        )
        connection.commit()
