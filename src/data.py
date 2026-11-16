"""Загрузка данных, экономия памяти, аудит качества и очистка."""
import csv

import pandas as pd

from src import config, features

UNKNOWN_DESCRIPTIONS = {"NULL", "Not Mapped", "Not Available", "Unknown/Invalid", ""}


def load_raw() -> pd.DataFrame:
    """Прочитать исходный CSV. Знак «?» в этих данных означает пропуск."""
    try:
        return pd.read_csv(config.RAW_CSV, na_values="?", low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл данных не найден: {config.RAW_CSV}") from None


def load_in_chunks(chunksize=20000) -> pd.DataFrame:
    """Прочитать CSV кусками: в памяти одновременно только один «сырой» кусок."""
    chunks = []
    for chunk in pd.read_csv(config.RAW_CSV, na_values="?", low_memory=False, chunksize=chunksize):
        chunks.append(optimize_dtypes(chunk))
    return optimize_dtypes(pd.concat(chunks, ignore_index=True))


def optimize_dtypes(df) -> pd.DataFrame:
    """Сжать типы: текст -> category, числа -> самый маленький подходящий тип."""
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            continue
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("category")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def memory_mb(df) -> float:
    """Сколько мегабайт таблица занимает в памяти."""
    return round(df.memory_usage(deep=True).sum() / 1024**2, 2)


def load_mapping() -> dict:
    """Прочитать три справочника из одного файла в словарь {имя: таблица id, description}."""
    with open(config.MAPPING_CSV, newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    blocks = {}
    name = None                       # None — ждём заголовок следующего справочника
    for row in rows:
        if not row or all(cell.strip() == "" for cell in row):
            name = None               # пустая строка разделяет справочники
        elif name is None:
            name = row[0].strip()
            blocks[name] = []
        else:
            description = row[1].strip()
            if description in UNKNOWN_DESCRIPTIONS:
                description = "Unknown"
            blocks[name].append({"id": int(row[0]), "description": description})
    return {name: pd.DataFrame(items) for name, items in blocks.items()}


def audit(df) -> dict:
    """Собрать факты о качестве исходных данных. Ничего не меняет."""
    visits = df["patient_nbr"].value_counts()
    missing = (df.isna().mean() * 100).round(2)
    expired = df["discharge_disposition_id"].isin(config.EXPIRED_OR_HOSPICE)
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "unique_patients": len(visits),
        "patients_with_repeat_visits": int((visits > 1).sum()),
        "rows_from_repeat_patients": int(visits[visits > 1].sum()),
        "duplicate_encounter_ids": int(df["encounter_id"].duplicated().sum()),
        "gender_unknown_rows": int((df["gender"] == "Unknown/Invalid").sum()),
        "expired_or_hospice_rows": int(expired.sum()),
        "expired_but_marked_readmitted": int((expired & (df["readmitted"] != "NO")).sum()),
        "missing_percent": missing[missing > 0].sort_values(ascending=False).to_dict(),
        "target_distribution": df["readmitted"].value_counts().to_dict(),
    }


def find_near_constant(df) -> list:
    """Найти столбцы, где одно значение занимает больше 99.95% строк."""
    result = []
    for col in df.columns:
        top_share = df[col].value_counts(dropna=False, normalize=True).iloc[0]
        if top_share > config.NEAR_CONSTANT_THRESHOLD:
            result.append(col)
    return result


def clean(df) -> pd.DataFrame:
    """Очистить исходные данные и добавить новые признаки."""
    df = df.copy()
    # 1. Умер или переведён в хоспис — вернуться в больницу не может
    df = df[~df["discharge_disposition_id"].isin(config.EXPIRED_OR_HOSPICE)]
    # 2. Пол не указан всего у 3 строк
    df = df[df["gender"] != "Unknown/Invalid"]
    # 3. Вес пропущен на 97%; почти постоянные столбцы не несут информации
    to_drop = ["weight"] + [col for col in find_near_constant(df) if col != "weight"]
    df = df.drop(columns=to_drop)
    # 4. Пропуск = анализ не делали. Это информация, а не ошибка
    for col in ["A1Cresult", "max_glu_serum"]:
        df[col] = df[col].fillna("NotMeasured")
    # 5. Пропусков слишком много, чтобы удалять строки
    for col in ["medical_specialty", "payer_code", "race"]:
        df[col] = df[col].fillna("Unknown")
    # 6. Новые признаки и ответ
    df = features.add_features(df)
    return df.reset_index(drop=True)