"""База SQLite: создание таблиц, запуск запросов и поиск госпитализаций."""
import re
import sqlite3

import pandas as pd

from src import config, data

# Справочники: таблица -> столбец с кодом
LOOKUPS = {
    "admission_types": "admission_type_id",
    "discharge_dispositions": "discharge_disposition_id",
    "admission_sources": "admission_source_id",
}
ENCOUNTER_COLUMNS = [
    "encounter_id", "patient_nbr", *LOOKUPS.values(), "age", "age_num",
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "total_prior_visits", "payer_code", "medical_specialty", "max_glu_serum", "A1Cresult",
    "change", "diabetesMed", "readmitted", "readmit_30",
]
MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "acarbose", "insulin", "glyburide-metformin",
]

SCHEMA = """
CREATE TABLE patients (patient_nbr INTEGER PRIMARY KEY, race TEXT, gender TEXT);
CREATE TABLE admission_types (admission_type_id INTEGER PRIMARY KEY, description TEXT);
CREATE TABLE discharge_dispositions (discharge_disposition_id INTEGER PRIMARY KEY, description TEXT);
CREATE TABLE admission_sources (admission_source_id INTEGER PRIMARY KEY, description TEXT);
CREATE TABLE encounters (
    encounter_id INTEGER PRIMARY KEY,
    patient_nbr INTEGER REFERENCES patients(patient_nbr),
    admission_type_id INTEGER REFERENCES admission_types(admission_type_id),
    discharge_disposition_id INTEGER REFERENCES discharge_dispositions(discharge_disposition_id),
    admission_source_id INTEGER REFERENCES admission_sources(admission_source_id),
    age TEXT, age_num INTEGER,
    time_in_hospital INTEGER, num_lab_procedures INTEGER, num_procedures INTEGER,
    num_medications INTEGER, number_outpatient INTEGER, number_emergency INTEGER,
    number_inpatient INTEGER, number_diagnoses INTEGER, total_prior_visits INTEGER,
    payer_code TEXT, medical_specialty TEXT, max_glu_serum TEXT, A1Cresult TEXT,
    change TEXT, diabetesMed TEXT, readmitted TEXT, readmit_30 INTEGER);
CREATE TABLE encounter_diagnoses (
    encounter_id INTEGER REFERENCES encounters(encounter_id),
    position INTEGER, icd9_code TEXT, icd9_group TEXT,
    PRIMARY KEY (encounter_id, position));
CREATE TABLE encounter_medications (
    encounter_id INTEGER REFERENCES encounters(encounter_id),
    medication TEXT, status TEXT,
    PRIMARY KEY (encounter_id, medication));
CREATE INDEX idx_encounters_patient ON encounters(patient_nbr);
"""


def diagnoses_table(clean) -> pd.DataFrame:
    """Три столбца диагнозов -> длинная таблица: одна строка на диагноз."""
    parts = []
    for position in [1, 2, 3]:
        part = clean[["encounter_id", f"diag_{position}", f"diag_{position}_group"]].dropna()
        part.columns = ["encounter_id", "icd9_code", "icd9_group"]
        parts.append(part.assign(position=position))
    return pd.concat(parts)


def medications_table(clean) -> pd.DataFrame:
    """Столбцы лекарств -> длинная таблица. Храним только назначенные (не «No»)."""
    long = clean.melt(id_vars="encounter_id", value_vars=MEDICATIONS,
                      var_name="medication", value_name="status")
    return long[long["status"] != "No"]


def build_database() -> dict:
    """Создать базу из исходных CSV. Вернуть {таблица: число строк}."""
    clean = data.clean(data.load_raw())      # та же очистка, что и для модели
    mapping = data.load_mapping()
    tables = {
        "patients": clean.groupby("patient_nbr", as_index=False)[["race", "gender"]].first(),
        "encounters": clean[ENCOUNTER_COLUMNS],
        "encounter_diagnoses": diagnoses_table(clean),
        "encounter_medications": medications_table(clean),
    }
    for table, key in LOOKUPS.items():
        tables[table] = mapping[key].rename(columns={"id": key})

    config.DB_PATH.unlink(missing_ok=True)   # пересоздаём базу с нуля
    connection = sqlite3.connect(config.DB_PATH)
    try:
        connection.executescript(SCHEMA)
        for name, table in tables.items():
            table.to_sql(name, connection, if_exists="append", index=False)
    finally:
        connection.close()
    return {name: len(table) for name, table in tables.items()}


def load_queries() -> dict:
    """Прочитать запросы из sql/queries.sql в словарь {имя: текст SQL}."""
    text = config.SQL_FILE.read_text(encoding="utf-8")
    blocks = re.split(r"^-- name:\s*", text, flags=re.MULTILINE)[1:]
    return dict(block.split("\n", 1) for block in blocks)


def run_query(sql: str, params=()) -> pd.DataFrame:
    """Выполнить SQL-запрос и вернуть результат как DataFrame. Базу создаёт при первом вызове."""
    if not config.DB_PATH.exists():
        build_database()
    connection = sqlite3.connect(config.DB_PATH)
    try:
        return pd.read_sql_query(sql, connection, params=params)
    finally:
        connection.close()


def search_encounters(patient_nbr=None, age=None, gender=None, readmit_30=None, limit=20) -> pd.DataFrame:
    """Найти госпитализации по фильтрам. Фильтр со значением None не применяется."""
    filters = {"e.patient_nbr": patient_nbr, "e.age": age, "p.gender": gender, "e.readmit_30": readmit_30}
    used = {field: value for field, value in filters.items() if value is not None}
    where = " WHERE " + " AND ".join(f"{field} = ?" for field in used) if used else ""
    sql = """SELECT e.encounter_id, e.patient_nbr, e.age, p.gender,
                    d.icd9_group AS primary_diagnosis, e.time_in_hospital,
                    e.number_inpatient, e.readmitted
             FROM encounters e
             JOIN patients p ON p.patient_nbr = e.patient_nbr
             LEFT JOIN encounter_diagnoses d ON d.encounter_id = e.encounter_id AND d.position = 1"""
    # значения передаются параметрами (?), а не вставляются в текст запроса
    return run_query(sql + where + " ORDER BY e.encounter_id LIMIT ?", (*used.values(), limit))
