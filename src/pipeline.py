"""Подготовка данных из исходного ZIP и повторное использование готовой базы."""
from .config import DATA, DB_PATH, REPORTS, ensure_directories


def prepare_data(with_eda=False, log=print):
    """Создать Parquet, разбиение по пациентам и SQLite по единой схеме проекта."""
    from .data import load_data, make_cohort, patient_split, save_data_assets
    from .database import import_database, run_sql_analysis
    ensure_directories()
    log('Загружаю данные и разделяю пациентов...')
    raw, mappings, info = load_data()
    cohort, flow = make_cohort(raw)
    parts = patient_split(cohort)
    save_data_assets(raw, cohort, parts, info, flow)
    log(f'Госпитализаций: {len(cohort):,}; пациентов: {cohort.patient_nbr.nunique():,}.')
    import_database(raw, mappings, parts)
    results = run_sql_analysis()
    if not results['09_split_leakage_check'].empty:
        raise ValueError('Найдено пересечение пациентов между выборками.')
    log(f'База готова. Выполнено SQL-запросов: {len(results)}.')
    if with_eda:
        from .eda import run_eda
        run_eda(raw, mappings, parts)
        log('Графики и таблицы анализа сохранены в reports/.')
    return parts


def data_is_ready():
    """Проверить наличие базы и файлов, нужных для анализа и поиска."""
    return all(path.is_file() for path in [DB_PATH, DATA/'raw.parquet',
               DATA/'cohort.parquet', DATA/'split_manifest.parquet', REPORTS/'data_summary.json'])


def ensure_data_ready(log=print):
    """Автоматически подготовить отсутствующие рабочие данные без обучения модели."""
    if not data_is_ready():
        log('Первый запуск: создаю локальную базу и рабочие данные из ZIP...')
        prepare_data(log=log)


def load_analysis_data():
    """Загрузить только fit для исследования связей признаков с целевой переменной."""
    import pandas as pd
    from .features import icd9_group
    ensure_data_ready()
    cohort = pd.read_parquet(DATA/'cohort.parquet')
    manifest = pd.read_parquet(DATA/'split_manifest.parquet')
    fit_ids = manifest.loc[manifest['split'].eq('fit'), 'encounter_id']
    frame = cohort.loc[cohort.encounter_id.isin(fit_ids)].copy()
    frame['diag_1_group'] = frame.diag_1.map(icd9_group)
    return frame
