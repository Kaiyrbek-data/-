"""Чтение SQLite и поиск по необязательным параметрам."""
import sqlite3
import pandas as pd
from .config import DB_PATH


def read_query(sql, params=()):
    """Выполнить SQL только для чтения; значения передаются параметрами."""
    if not DB_PATH.exists():
        raise FileNotFoundError('База не создана. Выполните python main.py prepare.')
    with sqlite3.connect(DB_PATH.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        connection.execute('PRAGMA query_only=ON')
        return pd.read_sql_query(sql, connection, params=params)


def search_encounters(patient_nbr=None, age=None, gender=None, diagnosis=None,
                      readmitted_30d=None, limit=20):
    """Найти госпитализации в исходных данных, возвращая не более 100 строк."""
    values = {'e.patient_nbr': patient_nbr, 'e.age': age, 'e.gender': gender,
              'd.diagnosis_group': diagnosis, 'e.readmitted_30d': readmitted_30d}
    conditions, params = [], []
    for field, value in values.items():
        if value is not None:
            conditions.append(f'{field} = ?')
            params.append(value)
    sql = '''SELECT e.encounter_id, e.patient_nbr, e.age, e.gender,
                    d.diagnosis_group, e.time_in_hospital, e.readmitted
             FROM encounters e LEFT JOIN diagnoses d
               ON d.encounter_id=e.encounter_id AND d.position=1'''
    if conditions:
        sql += ' WHERE ' + ' AND '.join(conditions)
    return read_query(sql + ' ORDER BY e.encounter_id LIMIT ?',
                      (*params, max(1, min(int(limit), 100))))
