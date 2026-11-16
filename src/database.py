"""База SQLite, подготовка данных для поиска и готовые SQL-запросы."""
import csv
import re
import sqlite3

import pandas as pd

from src import config

NUMERIC = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
    'age_num', 'total_prior_visits',
]
TEXT = ['age', 'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult',
        'change', 'diabetesMed', 'readmitted']
LOOKUPS = {'admission_types': 'admission_type_id',
           'discharge_dispositions': 'discharge_disposition_id',
           'admission_sources': 'admission_source_id'}
MEDICATIONS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
]


def _mapping_from_csv() -> dict:
    """Прочитать три справочника, разделённые пустыми строками одного CSV."""
    blocks = {}
    current = None
    with config.MAPPING_CSV.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.reader(stream):
            if not row or not any(cell.strip() for cell in row):
                current = None
            elif row[0].strip() in LOOKUPS.values():
                current = row[0].strip()
                blocks[current] = []
            elif current:
                description = row[1].strip()
                if description in {'NULL', 'Not Mapped', 'Not Available', 'Unknown/Invalid'}:
                    description = 'Unknown'
                blocks[current].append((int(row[0]), description))
    return {key: pd.DataFrame(rows, columns=['id', 'description'])
            for key, rows in blocks.items()}


def _diagnosis_group(code) -> str:
    """Сгруппировать ICD-9 для автономного создания поисковой базы из CSV."""
    if pd.isna(code):
        return 'Missing'
    try:
        value = float(code)
    except (TypeError, ValueError):
        return 'Other'
    if 250 <= value < 251:
        return 'Diabetes'
    for low, high, extra, name in [
        (390, 460, 785, 'Circulatory'), (460, 520, 786, 'Respiratory'),
        (520, 580, 787, 'Digestive'), (800, 1000, None, 'Injury'),
        (710, 740, None, 'Musculoskeletal'), (580, 630, 788, 'Genitourinary'),
        (140, 240, None, 'Neoplasms'),
    ]:
        if low <= value < high or (extra is not None and extra <= value < extra + 1):
            return name
    return 'Other'


def prepare_database() -> dict:
    """Создать поисковую базу из исходных CSV, не затрагивая данные для модели."""
    for path in (config.RAW_CSV, config.MAPPING_CSV):
        if not path.is_file():
            raise FileNotFoundError(f'Файл данных не найден: {path}. Положите его в data/.')
    df = pd.read_csv(config.RAW_CSV, na_values='?', keep_default_na=False, low_memory=False)
    # Исключаем выписки «умер/хоспис»: они не относятся к группе риска возврата.
    df = df.loc[~df.discharge_disposition_id.isin(config.EXPIRED_OR_HOSPICE)
                & df.gender.ne('Unknown/Invalid')].copy()
    df['readmit_30'] = df.readmitted.eq('<30').astype(int)
    df['age_num'] = pd.to_numeric(df.age.str.extract(r'\[(\d+)-', expand=False)) + 5
    df['total_prior_visits'] = df[['number_outpatient', 'number_emergency', 'number_inpatient']].sum(axis=1)
    for position in (1, 2, 3):
        df[f'diag_{position}_group'] = df[f'diag_{position}'].map(_diagnosis_group)
    for column in ('race', 'payer_code', 'medical_specialty'):
        df[column] = df[column].fillna('Unknown')
    for column in ('A1Cresult', 'max_glu_serum'):
        df[column] = df[column].replace('None', 'NotMeasured').fillna('NotMeasured')
    return build_database(df, _mapping_from_csv())


def build_database(df, mapping: dict) -> dict:
    """Записать семь связанных таблиц; заменить готовую базу только после проверки."""
    config.ensure_dirs()
    frame = df.copy()
    for column in frame.select_dtypes(include='category').columns:
        frame[column] = frame[column].astype('string')
    temporary = config.DB_PATH.with_suffix('.building.db')
    if temporary.exists():
        temporary.unlink()
    con = sqlite3.connect(temporary)
    try:
        con.execute('PRAGMA foreign_keys = ON')
        con.execute('CREATE TABLE patients (patient_nbr INTEGER PRIMARY KEY, race TEXT, gender TEXT)')
        frame.groupby('patient_nbr', as_index=False)[['race', 'gender']].first().to_sql(
            'patients', con, if_exists='append', index=False)
        for table, key in LOOKUPS.items():
            con.execute(f'CREATE TABLE {table} ({key} INTEGER PRIMARY KEY, description TEXT)')
            values = mapping[key].rename(columns={'id': key})[[key, 'description']].copy()
            # В данных бывают коды, отсутствующие в официальном справочнике.
            absent = set(frame[key].dropna().astype(int)) - set(values[key].astype(int))
            if absent:
                values = pd.concat([values, pd.DataFrame({key: sorted(absent), 'description': 'Unknown'})])
            values.to_sql(table, con, if_exists='append', index=False)
        columns = ['encounter_id', 'patient_nbr', *LOOKUPS.values(), *TEXT, *NUMERIC, 'readmit_30']
        definitions = ['encounter_id INTEGER PRIMARY KEY',
                       'patient_nbr INTEGER REFERENCES patients(patient_nbr)']
        definitions += [f'{key} INTEGER REFERENCES {table}({key})' for table, key in LOOKUPS.items()]
        definitions += [f'"{key}" TEXT' for key in TEXT]
        definitions += [f'{key} INTEGER' for key in NUMERIC]
        definitions += ['readmit_30 INTEGER NOT NULL CHECK (readmit_30 IN (0, 1))']
        con.execute('CREATE TABLE encounters (' + ', '.join(definitions) + ')')
        frame[columns].to_sql('encounters', con, if_exists='append', index=False)
        con.executescript('''
            CREATE TABLE encounter_diagnoses (
                encounter_id INTEGER REFERENCES encounters(encounter_id),
                position INTEGER CHECK (position BETWEEN 1 AND 3),
                icd9_code TEXT, icd9_group TEXT,
                PRIMARY KEY (encounter_id, position));
            CREATE TABLE encounter_medications (
                encounter_id INTEGER REFERENCES encounters(encounter_id),
                medication TEXT, status TEXT,
                PRIMARY KEY (encounter_id, medication));
        ''')
        for position in (1, 2, 3):
            diagnoses = frame[['encounter_id', f'diag_{position}', f'diag_{position}_group']].dropna().copy()
            diagnoses.columns = ['encounter_id', 'icd9_code', 'icd9_group']
            diagnoses['position'] = position
            diagnoses.to_sql('encounter_diagnoses', con, if_exists='append', index=False)
        medications = frame.melt(id_vars='encounter_id',
                                 value_vars=[key for key in MEDICATIONS if key in frame],
                                 var_name='medication', value_name='status')
        medications = medications.loc[medications.status.notna() & medications.status.ne('No')]
        medications.to_sql('encounter_medications', con, if_exists='append', index=False, chunksize=10000)
        con.executescript('''
            CREATE INDEX idx_encounters_patient ON encounters(patient_nbr);
            CREATE INDEX idx_diagnosis_group ON encounter_diagnoses(icd9_group);
            CREATE INDEX idx_medication ON encounter_medications(medication);
        ''')
        if con.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Нарушены связи между таблицами базы.')
        tables = ['patients', 'encounters', 'encounter_diagnoses', 'encounter_medications', *LOOKUPS]
        counts = {table: con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in tables}
        con.commit()
    except BaseException:
        con.close()
        temporary.unlink(missing_ok=True)
        raise
    con.close()
    temporary.replace(config.DB_PATH)
    return counts


def load_queries() -> dict:
    """Прочитать именованные запросы из sql/queries.sql."""
    text = (config.SQL_DIR / 'queries.sql').read_text(encoding='utf-8-sig')
    return dict(block.split('\n', 1) for block in re.split(r'^-- name:\s*', text, flags=re.MULTILINE)[1:])


def run_query(sql: str, params=()) -> pd.DataFrame:
    """Выполнить запрос с параметрами через соединение только для чтения."""
    if not config.DB_PATH.is_file():
        raise FileNotFoundError('База ещё не создана. Выберите подготовку данных в меню.')
    with sqlite3.connect(config.DB_PATH.resolve().as_uri() + '?mode=ro', uri=True) as con:
        con.execute('PRAGMA query_only = ON')
        return pd.read_sql_query(sql, con, params=params)


def search_encounters(patient_nbr=None, encounter_id=None, age=None, gender=None,
                      diagnosis=None, readmit_30=None, limit=20) -> pd.DataFrame:
    """Найти визиты по необязательным фильтрам без подстановки ввода в SQL."""
    fields = {'e.patient_nbr': patient_nbr, 'e.encounter_id': encounter_id,
              'e.age': age, 'p.gender': gender, 'd.icd9_group': diagnosis,
              'e.readmit_30': readmit_30}
    conditions, params = [], []
    for field, value in fields.items():
        if value is not None:
            conditions.append(f'{field} = ?')
            params.append(value)
    where = ' WHERE ' + ' AND '.join(conditions) if conditions else ''
    sql = '''SELECT e.encounter_id, e.patient_nbr, e.age, p.gender,
                    d.icd9_group AS primary_diagnosis, e.time_in_hospital,
                    e.number_inpatient, e.readmitted, e.readmit_30
             FROM encounters e JOIN patients p USING(patient_nbr)
             LEFT JOIN encounter_diagnoses d
               ON d.encounter_id=e.encounter_id AND d.position=1'''
    return run_query(sql + where + ' ORDER BY e.encounter_id LIMIT ?',
                     (*params, max(1, min(int(limit), 100))))
