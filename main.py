"""Консольное меню проекта. Запуск: python main.py"""
from datetime import datetime

import pandas as pd

from src import config, data, database, models

MENU = """
=== Повторная госпитализация пациентов с диабетом ===
1 — Общая информация
2 — Поиск госпитализаций
3 — Готовые SQL-запросы
4 — Сохранить последний результат в CSV
5 — Пересоздать базу из исходных CSV
6 — Обучить и сравнить модели
7 — Прогноз для одного пациента
0 — Выход"""

DIAGNOSES = ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
             "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"]


def ask_number(prompt, minimum, maximum, default=None):
    """Спрашивать целое число, пока ввод не станет верным. Enter — значение по умолчанию."""
    while True:
        text = input(prompt).strip()
        if not text:
            return default
        try:
            number = int(text)
            if minimum <= number <= maximum:
                return number
        except ValueError:
            pass
        print(f"Введите целое число от {minimum} до {maximum} или нажмите Enter.")


def ask_choice(prompt, options):
    """Выбрать вариант из списка по номеру. Enter — не выбирать (None)."""
    for number, option in enumerate(options, 1):
        print(f" {number} — {option}")
    number = ask_number(prompt, 1, len(options))
    return options[number - 1] if number else None


def show_info():
    """Пункт 1: сколько в базе госпитализаций и пациентов."""
    return database.run_query(
        "SELECT COUNT(*) AS encounters, COUNT(DISTINCT patient_nbr) AS patients, "
        "ROUND(AVG(readmit_30) * 100, 1) AS readmit_pct FROM encounters")


def search():
    """Пункт 2: поиск госпитализаций по фильтрам. Enter пропускает фильтр."""
    print("Enter — пропустить фильтр.")
    patient = ask_number("Номер пациента: ", 0, 10**10)
    ages = [f"[{start}-{start + 10})" for start in range(0, 100, 10)]
    age = ask_choice("Номер возрастной группы: ", ages)
    gender = ask_choice("Номер пола: ", ["Female", "Male"])
    target = ask_number("Возврат раньше 30 дней (1 — да, 0 — нет): ", 0, 1)
    limit = ask_number("Сколько строк показать (Enter = 20): ", 1, 100, 20)
    return database.search_encounters(patient, age, gender, target, limit)


def run_sql():
    """Пункт 3: выполнить один из готовых запросов."""
    queries = database.load_queries()
    name = ask_choice("Номер запроса: ", list(queries))
    return database.run_query(queries[name]) if name else None


def save_csv(result):
    """Пункт 4: сохранить последний показанный результат в CSV."""
    if result is None:
        print("Сначала выполните поиск или запрос.")
        return
    config.EXPORT_DIR.mkdir(exist_ok=True)
    path = config.EXPORT_DIR / f"result_{datetime.now():%Y%m%d_%H%M%S}.csv"
    result.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Сохранено: {path}")


def train():
    """Пункт 6: обучить модели. Режим 2 подбирает параметры и сохраняет лучшую модель."""
    print(" 1 — Быстро (~30 секунд, модель не сохраняется)\n"
          " 2 — Полное обучение (~5 минут, модель сохраняется)")
    full = ask_number("Режим (Enter = 1): ", 1, 2, 1) == 2
    print("Обучаю модели...")
    clean = data.clean(data.load_raw())
    return models.train_all(clean, tune=full, save=full).drop(columns="best_params")


def predict():
    """Пункт 7: спросить признаки пациента и показать вероятность возврата."""
    print("Enter — значение по умолчанию.")
    age = ask_number("Возраст (Enter = 65): ", 0, 99, 65)
    patient = {
        "age_num": age // 10 * 10 + 5,      # середина интервала: 67 -> 65
        "time_in_hospital": ask_number("Дней в больнице (Enter = 4): ", 1, 14, 4),
        "num_lab_procedures": ask_number("Лабораторных анализов (Enter = 43): ", 1, 132, 43),
        "num_procedures": ask_number("Других процедур (Enter = 1): ", 0, 6, 1),
        "num_medications": ask_number("Количество лекарств (Enter = 16): ", 1, 81, 16),
        "number_outpatient": ask_number("Амбулаторных визитов за год (Enter = 0): ", 0, 50, 0),
        "number_emergency": ask_number("Визитов в скорую за год (Enter = 0): ", 0, 80, 0),
        "number_inpatient": ask_number("Госпитализаций за год (Enter = 0): ", 0, 30, 0),
        "number_diagnoses": ask_number("Количество диагнозов (Enter = 7): ", 1, 16, 7),
    }
    patient["total_prior_visits"] = (patient["number_outpatient"] + patient["number_emergency"]
                                     + patient["number_inpatient"])
    diagnosis = ask_choice("Номер группы основного диагноза: ", DIAGNOSES)
    if diagnosis:
        patient["diag_1_group"] = diagnosis
    insulin = ask_choice("Номер статуса инсулина: ", ["No", "Steady", "Up", "Down"])
    if insulin:
        patient["insulin"] = insulin

    result = models.predict_patient(patient)
    risk = "ВЫСОКИЙ" if result["high_risk"] else "низкий"
    print(f"\nВероятность возврата в течение 30 дней: {result['probability'] * 100:.1f}%")
    print(f"Риск: {risk} (порог {result['threshold'] * 100:.1f}%)")


def main():
    """Показывать меню, пока пользователь не выберет выход. Ошибки не роняют программу."""
    actions = {"1": show_info, "2": search, "3": run_sql, "5": database.build_database, "6": train}
    last_result = None
    while True:
        print(MENU)
        try:
            choice = input("Выберите пункт: ").strip()
            if choice == "0":
                break
            elif choice == "4":
                save_csv(last_result)
            elif choice == "7":
                predict()
            elif choice in actions:
                result = actions[choice]()
                if isinstance(result, pd.DataFrame):
                    last_result = result
                    print(result.to_string(index=False))
                elif result is not None:
                    print(result)
            else:
                print("Нет такого пункта, попробуйте ещё раз.")
        except FileNotFoundError as error:
            print(f"Нет файла: {error}")
        except (ValueError, KeyError) as error:
            print(f"Ошибка: {error}")
        except (KeyboardInterrupt, EOFError):      # Ctrl+C или конец ввода
            print("\nВыход.")
            break


if __name__ == "__main__":
    main()
