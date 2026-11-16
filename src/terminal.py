"""Простое меню поиска по базе госпитализаций."""
import sqlite3
from datetime import datetime

import pandas as pd

from src import config, database

QUERY_LABELS = ['По основному диагнозу', 'По возрасту', 'По прошлым госпитализациям',
                'По типу выписки', 'По инсулину', 'Отделения выше среднего',
                '10 пациентов с наибольшим числом визитов', 'По номеру визита']
DIAGNOSES = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury',
             'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
DIAGNOSIS_LABELS = ['Болезни системы кровообращения', 'Болезни органов дыхания',
                    'Болезни органов пищеварения', 'Диабет', 'Травмы',
                    'Болезни костно-мышечной системы', 'Болезни мочеполовой системы',
                    'Новообразования', 'Прочие']


def _integer(prompt, minimum=0, maximum=None, default=None):
    """Запрашивать целое число до корректного ввода; Enter возвращает значение по умолчанию."""
    while True:
        raw = input(prompt).strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value < minimum or (maximum is not None and value > maximum):
                raise ValueError
            return value
        except ValueError:
            print('Введите целое число в указанном диапазоне или нажмите Enter.')


def _choose(prompt, options):
    """Выбрать значение по номеру, оставляя фильтр пустым при Enter."""
    for number, label in enumerate(options, 1):
        print(f' {number} — {label}')
    choice = _integer(prompt, 1, len(options))
    return options[choice - 1] if choice is not None else None


def _ready():
    """Создать базу при первом обращении и повторно использовать её при следующих запусках."""
    if not config.DB_PATH.exists():
        print('Создаю базу из data/*.csv. Это требуется только при первом запуске...')
        counts = database.prepare_database()
        print(f'Готово: {counts["encounters"]:,} госпитализаций, {counts["patients"]:,} пациентов.')


def _search():
    """Запросить простые фильтры поиска; пустой ответ означает любое значение."""
    print('Enter — пропустить фильтр. Номера пациентов можно взять из первых строк без фильтров.')
    patient = _integer('Номер пациента: ')
    encounter = _integer('Номер госпитализации: ')
    ages = [f'[{start}-{start + 10})' for start in range(0, 100, 10)]
    age = _choose('Номер возрастной группы: ', ages)
    gender = _choose('Номер пола (Female — женский, Male — мужской): ', ['Female', 'Male'])
    print('Группы основного диагноза:')
    label = _choose('Номер группы: ', DIAGNOSIS_LABELS)
    diagnosis = DIAGNOSES[DIAGNOSIS_LABELS.index(label)] if label is not None else None
    target = _integer('Возврат в течение 30 дней: 1 — да, 0 — нет, Enter — любой: ', 0, 1)
    limit = _integer('Сколько строк показать (1–100, Enter = 20): ', 1, 100, 20)
    return database.search_encounters(patient, encounter, age, gender, diagnosis, target, limit)


def _query():
    """Выбрать и выполнить один из восьми запросов проекта."""
    queries = database.load_queries()
    names = list(queries)
    for number, name in enumerate(names, 1):
        print(f' {number} — {QUERY_LABELS[number - 1]} ({name})')
    choice = _integer('Номер запроса (Enter — назад): ', 1, len(names))
    return database.run_query(queries[names[choice - 1]]) if choice else None


def _show(result):
    """Показать результат с понятным сообщением при отсутствии совпадений."""
    if result.empty:
        print('По этим фильтрам ничего не найдено.')
    else:
        print(result.to_string(index=False))
        print(f'Показано строк: {len(result)}')


def main():
    """Запустить меню и обработать ошибки ввода и отсутствующие файлы."""
    last_result = None
    while True:
        print('\n=== База госпитализаций пациентов с диабетом ===\n'
              '1 — Общая информация\n2 — Поиск госпитализаций\n'
              '3 — Готовые SQL-запросы\n4 — Сохранить последний результат в CSV\n'
              '5 — Пересоздать базу из исходных CSV\n0 — Выход')
        try:
            choice = input('Выберите пункт: ').strip()
            if choice == '0':
                return
            if choice not in {'1', '2', '3', '4', '5'}:
                print('Нет такого пункта, попробуйте ещё раз.')
                continue
            if choice == '4':
                if last_result is None:
                    print('Сначала выполните поиск или SQL-запрос.')
                    continue
                config.ensure_dirs()
                folder = config.REPORTS_DIR / 'sql_results'
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f'search_{datetime.now():%Y%m%d_%H%M%S_%f}.csv'
                last_result.to_csv(path, index=False, encoding='utf-8-sig')
                print(f'Сохранено: {path}')
                continue
            if choice == '5':
                print('Существующая локальная база будет заменена данными из CSV.')
                if input('Введите ДА для продолжения: ').strip().upper() == 'ДА':
                    print(database.prepare_database())
                continue
            _ready()
            if choice == '1':
                result = database.run_query('''SELECT COUNT(*) AS encounters,
                    COUNT(DISTINCT patient_nbr) AS patients,
                    ROUND(AVG(readmit_30)*100, 1) AS readmit_pct FROM encounters''')
                print('Госпитализации / пациенты / доля возвратов в течение 30 дней (%):')
                print('Исключены выписки «умер/хоспис» и неизвестный пол. Это учебные исторические данные.')
            else:
                result = _search() if choice == '2' else _query()
            if result is not None:
                last_result = result
                _show(result)
        except EOFError:
            return
        except KeyboardInterrupt:
            print('\nОтменено. Возвращаюсь в меню.')
        except (OSError, ValueError, KeyError, sqlite3.Error, pd.errors.DatabaseError) as error:
            print(f'Не удалось выполнить действие: {error}')
