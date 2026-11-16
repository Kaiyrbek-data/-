"""Консольное меню проекта: поиск по базе, SQL-запросы, обучение моделей и прогноз."""
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


def _train():
    """Обучить и сравнить модели: быстрый режим для показа или полный с сохранением."""
    # Импорт внутри функции: поиск по базе работает и без библиотек машинного обучения.
    from src import data_loader, models, preprocessing
    print(' 1 — Быстрое обучение (~30 секунд, модель не сохраняется)\n'
          ' 2 — Полное обучение с подбором параметров (~5 минут, модель сохраняется)')
    mode = _integer('Режим (Enter = 1): ', 1, 2, 1)
    print('Обучаю модели, подождите...')
    clean = preprocessing.clean(data_loader.load_raw())
    full = mode == 2
    table = models.train_and_compare(clean, tune_models=full, save=full)
    if full:
        print(f'Модель сохранена: {config.MODEL_PATH}')
    return table.drop(columns='best_params')


def _predict():
    """Спросить признаки пациента и показать вероятность возврата в течение 30 дней."""
    from src import models
    bundle = models.load_model()   # нет файла модели -> FileNotFoundError с подсказкой
    print('Enter — значение по умолчанию (типичный пациент).')
    age = _integer('Возраст, лет (0–99, Enter = 65): ', 0, 99, 65)
    patient = {
        'age_num': age // 10 * 10 + 5,   # середина интервала, как при обучении: 67 -> 65
        'time_in_hospital': _integer('Дней в больнице (1–14, Enter = 4): ', 1, 14, 4),
        'num_lab_procedures': _integer('Лабораторных анализов (1–132, Enter = 43): ', 1, 132, 43),
        'num_procedures': _integer('Других процедур (0–6, Enter = 1): ', 0, 6, 1),
        'num_medications': _integer('Количество лекарств (1–81, Enter = 16): ', 1, 81, 16),
        'number_outpatient': _integer('Амбулаторных визитов за год (Enter = 0): ', 0, 50, 0),
        'number_emergency': _integer('Визитов в скорую за год (Enter = 0): ', 0, 80, 0),
        'number_inpatient': _integer('Госпитализаций за год (Enter = 0): ', 0, 30, 0),
        'number_diagnoses': _integer('Количество диагнозов (1–16, Enter = 7): ', 1, 16, 7),
    }
    patient['total_prior_visits'] = (patient['number_outpatient'] + patient['number_emergency']
                                     + patient['number_inpatient'])
    print('Группа основного диагноза (Enter = болезни системы кровообращения):')
    label = _choose('Номер группы: ', DIAGNOSIS_LABELS)
    if label is not None:
        patient['diag_1_group'] = DIAGNOSES[DIAGNOSIS_LABELS.index(label)]
    print('Инсулин (Enter = не назначен): No — нет, Steady — доза без изменений, '
          'Up — доза повышена, Down — доза снижена')
    insulin = _choose('Номер: ', ['No', 'Steady', 'Up', 'Down'])
    if insulin is not None:
        patient['insulin'] = insulin

    result = models.predict_patient(patient, bundle)
    risk = 'ВЫСОКИЙ' if result['high_risk'] else 'низкий'
    print(f'\nВероятность повторной госпитализации в течение 30 дней: {result["probability"] * 100:.1f}%')
    print(f'Риск: {risk} (порог {result["threshold"] * 100:.1f}%)')
    print(f'Модель: {result["model"]}. Это учебный прогноз, а не медицинское заключение.')


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
              '5 — Пересоздать базу из исходных CSV\n'
              '6 — Обучить и сравнить модели\n7 — Прогноз для одного пациента\n0 — Выход')
        try:
            choice = input('Выберите пункт: ').strip()
            if choice == '0':
                return
            if choice not in {'1', '2', '3', '4', '5', '6', '7'}:
                print('Нет такого пункта, попробуйте ещё раз.')
                continue
            if choice == '6':
                last_result = _train()
                _show(last_result)
                continue
            if choice == '7':
                _predict()
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
