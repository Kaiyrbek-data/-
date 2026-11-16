# Соответствие требованиям

| Требование | Реализация и проверяемое свидетельство |
|---|---|
| Python 3.x | Модули src/*.py, main.py, app.py; версии в artifacts/environment.json |
| Не менее 5 функций кроме main | load_data, make_cohort, patient_split, optimize_types, run_eda, build_pipeline, train_models, predict_record и другие |
| Списки и словари | src/config.py, model_candidates, справочники SQLite, схема ввода |
| if/elif/else | ICD-9 группировка, главное меню, валидация ввода |
| for/while | Анализ признаков, подбор моделей, повторный ввод и меню |
| Чтение и запись файлов | ZIP, Parquet, CSV, JSON, SQLite, joblib |
| try/except | main.py и src/inference.py: отсутствующий файл, неверное число, неверная категория популяции |
| Пользовательский интерфейс | Консольное меню и Streamlit с вводом признаков, результатами и фильтром графиков/SQL |
| Модульность | src/data.py, database.py, features.py, eda.py, modeling.py, inference.py |
| Baseline + минимум 2 модели | Dummy + LogisticRegression + RandomForest + LightGBM |
| Кросс-валидация | 3-fold StratifiedGroupKFold по patient_nbr; параметрический поиск внутри Pipeline |
| Качество, время, память, размер | reports/tables/test_metrics.csv и validation_metrics.csv |
| Базы данных | data/project.sqlite3; пациенты, госпитализации, справочники, диагнозы, препараты, сплиты и эксперименты |
| ER-диаграмма | reports/figures/er_diagram.png и docs/schema.md |
| Не менее 8 SQL-запросов | 12 запросов в sql/analysis.sql; JOIN, GROUP BY/HAVING, CTE, подзапросы, оконные функции |
| SQL → Pandas | src/database.py: pd.read_sql_query; сохранённые reports/tables/sql_*.csv |
| EDA | Все 50 столбцов, пропуски и кодированные неизвестные, выбросы, баланс классов, корреляции и повторные пациенты |
| Нет утечки | Четыре непересекающиеся группы пациентов; preprocessing обучается отдельно в каждом CV-фолде |
| Калибровка и дисбаланс | Sigmoid на отдельной calibration; варианты весов классов; порог F1 выбирается только на validation |
| Метрики | ROC-AUC, AP (явно определённая версия PR-AUC), F1 положительного класса, precision, recall, Brier, log-loss, ECE |
| Интерпретация и ошибки | Permutation importance на validation, confusion matrix, FN/FP, метрики по подгруппам на test |
| Сохранение результатов | joblib + Parquet + SQLite experiments/predictions |
| Требование большого объёма | Исходный набор: 101 766 строк, CSV около 19 МБ. Исключение требуется согласовать с преподавателем. Таблицы нормализации не выдаются за новые независимые наблюдения |
| Оптимизация памяти | Измерение до/после, уменьшение разрядности, category, конвертация ZIP→Parquet |
| GitHub | Общая версия опубликована в Kaiyrbek-data/my-; инструкция запуска в README |
| Вклад участников | project_metadata.json; фактическое распределение требуется подтвердить участникам |
| Отчёт и презентация | reports/Project_report.pdf, reports/Defense_slides.pdf; редактируемая презентация также сохранена в Spaces |
| Moodle и защита | Загрузка и согласование даты выполняются участниками; проект не отправлялся третьим лицам |

Повторный подбор решений после просмотра test превращает test в validation. Для новых решений нужен новый независимый test.
