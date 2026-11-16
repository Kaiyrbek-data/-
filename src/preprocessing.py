"""Аудит качества, очистка данных и создание признаков (Часть 1)."""
import pandas as pd

from src import config

TARGET = "readmit_30"       # целевая переменная: вернулся ли пациент за 30 дней
GROUP = "patient_nbr"       # по этому столбцу делим на train/test (один пациент — одна сторона)

NUMERIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "total_prior_visits", "age_num",
]

CATEGORICAL_FEATURES = [
    "race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "medical_specialty", "payer_code", "A1Cresult", "max_glu_serum",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "acarbose", "insulin", "glyburide-metformin",
    "change", "diabetesMed",
]

DIAG_GROUPS = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing",
]

# Диапазоны кодов ICD-9 по схеме Strack et al., 2014 (границы включительно)
ICD9_RANGES = {
    "Circulatory": [(390, 459), (785, 785)],
    "Respiratory": [(460, 519), (786, 786)],
    "Digestive": [(520, 579), (787, 787)],
    "Injury": [(800, 999)],
    "Musculoskeletal": [(710, 739)],
    "Genitourinary": [(580, 629), (788, 788)],
    "Neoplasms": [(140, 239)],
}


def audit(df) -> dict:
    """Собрать факты о качестве исходных данных ДО очистки (ничего не меняет)."""
    visits_per_patient = df["patient_nbr"].value_counts()
    repeat_patients = visits_per_patient[visits_per_patient > 1]

    missing = (df.isna().mean() * 100).round(2)
    missing = missing[missing > 0].sort_values(ascending=False)

    expired = df["discharge_disposition_id"].isin(config.EXPIRED_OR_HOSPICE)

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "unique_patients": int(df["patient_nbr"].nunique()),
        "patients_with_repeat_visits": int(len(repeat_patients)),
        "rows_from_repeat_patients": int(repeat_patients.sum()),
        "duplicate_encounter_ids": int(df["encounter_id"].duplicated().sum()),
        "missing_percent": {col: float(value) for col, value in missing.items()},
        "gender_unknown_rows": int((df["gender"] == "Unknown/Invalid").sum()),
        "expired_or_hospice_rows": int(expired.sum()),
        # Ошибка в данных: пациент умер, но записан как «вернувшийся»
        "expired_but_marked_readmitted": int((expired & (df["readmitted"] != "NO")).sum()),
        "target_distribution": {str(k): int(v) for k, v in df["readmitted"].value_counts().items()},
    }


def map_icd9(code) -> str:
    """Перевести код диагноза ICD-9 в одну из ~9 крупных групп болезней."""
    # Почему группируем: у диагноза ~700 разных кодов, многие встречаются
    # несколько раз — модель на таких редких значениях ничему не научится.
    if pd.isna(code):
        return "Missing"
    code = str(code).strip()
    if code.startswith(("V", "E")):
        return "Other"          # V и E — дополнительные коды (обследования, внешние причины)
    try:
        number = float(code)
    except ValueError:
        return "Other"          # нечисловой код

    if 250 <= number < 251:     # 250.00–250.99 — все виды диабета
        return "Diabetes"
    whole = int(number)         # 428.0 и 428.9 относятся к одному диапазону
    for group, ranges in ICD9_RANGES.items():
        for low, high in ranges:
            if low <= whole <= high:
                return group
    return "Other"


def find_near_constant(df) -> list:
    """Найти столбцы, где одно значение занимает больше 99.95% строк."""
    result = []
    for col in df.columns:
        # dropna=False: пропуск тоже считается «значением»
        top_share = df[col].value_counts(dropna=False, normalize=True).iloc[0]
        if top_share > config.NEAR_CONSTANT_THRESHOLD:
            result.append(col)
    return result


def clean(df) -> pd.DataFrame:
    """Очистить исходные данные и добавить новые признаки. Возвращает новую таблицу."""
    df = df.copy()
    # Если на вход пришла таблица со сжатыми типами (category), возвращаем обычный
    # текст: в category нельзя просто так записать новое значение вроде "Unknown".
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(object)

    # 1. Умершие и выписанные в хоспис физически не могут вернуться в больницу.
    #    Если их оставить, модель выучит «умер = не вернётся» — бесполезное правило.
    df = df[~df["discharge_disposition_id"].isin(config.EXPIRED_OR_HOSPICE)]

    # 2. Пол не указан всего у 3 строк — это мусорная категория, проще удалить.
    df = df[df["gender"] != "Unknown/Invalid"]

    # 3. weight пропущен на ~97% — восстановить нельзя. Почти постоянные столбцы
    #    (лекарства, которые почти никому не назначали) не несут информации.
    to_drop = ["weight"] + [col for col in find_near_constant(df) if col != "weight"]
    df = df.drop(columns=to_drop)

    # 4. Пропуск здесь значит «анализ не делали». Это тоже информация о пациенте,
    #    поэтому делаем из пропуска отдельную категорию, а не удаляем строки.
    for col in ["A1Cresult", "max_glu_serum"]:
        df[col] = df[col].fillna("NotMeasured")

    # 5. Пропусков слишком много, чтобы удалять строки (medical_specialty ~49%).
    for col in ["medical_specialty", "payer_code", "race"]:
        df[col] = df[col].fillna("Unknown")

    # 6. Сотни кодов ICD-9 -> 9 групп + Missing (см. map_icd9).
    for position in [1, 2, 3]:
        df[f"diag_{position}_group"] = df[f"diag_{position}"].map(map_icd9, na_action=None)

    # 7. Целевая переменная: 1 только если вернулся раньше 30 дней.
    #    «>30» считаем нулём — задача именно про ранний возврат.
    df[TARGET] = (df["readmitted"] == "<30").astype("int8")

    # 8. Новые признаки: общее число обращений за прошлый год и возраст числом
    #    (середина интервала: "[70-80)" -> 70 + 5 = 75).
    df["total_prior_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    df["age_num"] = df["age"].astype(str).str.extract(r"\[(\d+)-")[0].astype(int) + 5

    # 9. После удаления строк индекс идёт с дырами — нумеруем заново.
    return df.reset_index(drop=True)


def get_feature_lists(df) -> tuple[list, list]:
    """Вернуть (числовые, категориальные) признаки, которые реально есть в таблице."""
    numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
    categorical = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    return numeric, categorical


def prepare_for_model(df) -> pd.DataFrame:
    """Привести категориальные признаки к строкам, чтобы кодировщик видел единый тип."""
    # Почему: admission_type_id и подобные — это номера категорий, а не количества.
    # Число 3 там не «больше» числа 1, поэтому превращаем их в текст "3" и "1".
    df = df.copy()
    _, categorical = get_feature_lists(df)
    for col in categorical:
        df[col] = df[col].astype(str).astype(object)
    return df
