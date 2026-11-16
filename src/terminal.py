"""Русское меню, команды воспроизведения и поиск по SQLite."""
import argparse
import json
import sqlite3
from pathlib import Path
import pandas as pd
from .config import TABLES
from .pipeline import prepare_data, ensure_data_ready

MENU = '''
========== Госпитализации пациентов с диабетом ==========
1 — Подготовить данные и базу заново
2 — Выполнить SQL-запросы
3 — Анализ данных: построить графики
4 — Обучить и сравнить модели
5 — Прогноз для одного пациента
6 — Сравнить CSV / Parquet и Pandas / SQL
7 — Поиск госпитализаций в базе
8 — Показать результаты моделей
0 — Выход'''


def print_table(frame):
    """Показать таблицу без индекса или сообщение об отсутствии строк."""
    print(frame.to_string(index=False) if not frame.empty else 'Совпадений не найдено.')


def ask_integer(prompt, default=None, minimum=0, maximum=None):
    """Повторять ввод до корректного целого числа; Enter возвращает значение по умолчанию."""
    while True:
        value = input(prompt).strip()
        if not value:
            return default
        try:
            number = int(value)
            if number < minimum or (maximum is not None and number > maximum):
                raise ValueError
            return number
        except ValueError:
            print('Введите целое число в допустимом диапазоне или нажмите Enter.')


def sql_menu():
    """Выполнить выбранный запрос или все запросы с сохранением результатов."""
    from .database import load_sql_queries, run_sql_analysis
    from .search import read_query
    ensure_data_ready()
    queries = load_sql_queries()
    names = list(queries)
    for number, name in enumerate(names, 1):
        print(f'{number} — {name}')
    choice = input('Номер запроса; a — выполнить все; Enter — назад: ').strip().lower()
    if not choice:
        return
    if choice == 'a':
        print(f'Сохранено результатов: {len(run_sql_analysis())}. Папка reports/tables/.')
    else:
        index = int(choice)-1
        if not 0 <= index < len(names):
            raise ValueError('Такого номера запроса нет.')
        sql = queries[names[index]]
        print(sql)
        print_table(read_query(sql))


def search_menu():
    """Найти госпитализации по простым фильтрам, пропуская необязательные поля."""
    from .search import search_encounters
    ensure_data_ready()
    print('Поиск по исходным историческим данным. Enter пропускает любой фильтр.')
    patient = ask_integer('Номер пациента: ')
    age_start = ask_integer('Нижняя граница возраста (0, 10, ..., 90): ', maximum=90)
    if age_start is not None and age_start % 10:
        raise ValueError('Возрастная группа должна начинаться с 0, 10, ..., 90.')
    age = f'[{age_start}-{age_start+10})' if age_start is not None else None
    gender_choice = ask_integer('Пол: 1 — женский, 2 — мужской, Enter — любой: ', minimum=1, maximum=2)
    gender = {1: 'Female', 2: 'Male'}.get(gender_choice)
    target = ask_integer('Возврат <30 дней: 1 — да, 0 — нет, Enter — любой: ', maximum=1)
    limit = ask_integer('Показать строк (1–100, Enter = 20): ', default=20, minimum=1, maximum=100)
    print_table(search_encounters(patient_nbr=patient, age=age, gender=gender,
                                  readmitted_30d=target, limit=limit))


def train(quick=False, with_eda=False):
    """Обучить модели на едином разбиении и обновить производные результаты."""
    from .modeling import train_models
    from scripts.finalize_analysis import finalize_analysis
    train_models(prepare_data(with_eda), quick=quick)
    finalize_analysis()


def menu():
    """Обрабатывать действия до выхода, возвращаясь в меню при ошибке или отмене."""
    while True:
        print(MENU)
        try:
            choice = input('Выберите пункт: ').strip()
            if choice == '0':
                return
            if choice == '1':
                prepare_data()
            elif choice == '2':
                sql_menu()
            elif choice == '3':
                prepare_data(with_eda=True)
            elif choice == '4':
                mode = input('1 — быстрый поиск; 2 — полный. Оба режима обновляют модели [1]: ').strip() or '1'
                if mode not in {'1', '2'}:
                    raise ValueError('Выберите 1 или 2.')
                train(quick=mode == '1')
            elif choice == '5':
                from .inference import console_prediction
                console_prediction()
            elif choice == '6':
                from .benchmarks import run_all
                for name, frame in run_all().items():
                    print(name)
                    print_table(frame)
            elif choice == '7':
                search_menu()
            elif choice == '8':
                print_table(pd.read_csv(TABLES/'test_metrics.csv'))
            else:
                print('Нет такого пункта. Выберите номер от 0 до 8.')
        except EOFError:
            return
        except KeyboardInterrupt:
            print('\nОтменено.')
        except (OSError, ValueError, KeyError, sqlite3.Error, pd.errors.DatabaseError) as error:
            print(f'Ошибка: {error}')


def main():
    """Запустить меню либо отдельную команду подготовки, обучения или прогноза."""
    parser = argparse.ArgumentParser(description='Прогноз повторной госпитализации')
    parser.add_argument('command', nargs='?', default='menu',
                        choices=['menu','prepare','eda','train','all','predict','results','sql','search','benchmark'])
    parser.add_argument('--quick', action='store_true', help='Уменьшить бюджет поиска моделей')
    parser.add_argument('--input', type=Path, help='JSON с признаками пациента')
    args = parser.parse_args()
    try:
        if args.command == 'menu':
            menu()
        elif args.command in {'prepare', 'eda'}:
            prepare_data(with_eda=args.command == 'eda')
        elif args.command in {'train', 'all'}:
            train(args.quick, with_eda=args.command == 'all')
        elif args.command == 'results':
            print_table(pd.read_csv(TABLES/'test_metrics.csv'))
        elif args.command == 'sql':
            sql_menu()
        elif args.command == 'search':
            search_menu()
        elif args.command == 'benchmark':
            from .benchmarks import run_all
            for name, frame in run_all().items():
                print(name)
                print_table(frame)
        elif args.command == 'predict':
            from .inference import predict_record, console_prediction
            if args.input:
                print(json.dumps(predict_record(json.loads(args.input.read_text(encoding='utf-8-sig'))),
                                 ensure_ascii=False, indent=2))
            else:
                console_prediction()
    except (EOFError, KeyboardInterrupt):
        print('\nОтменено.')
    except (OSError, ValueError, KeyError, sqlite3.Error, pd.errors.DatabaseError) as error:
        parser.exit(1, f'Ошибка: {error}\n')
