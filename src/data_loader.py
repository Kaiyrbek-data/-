"""Загрузка исходных данных и экономия памяти (Часть 1)."""
import csv

import pandas as pd

from src import config

# Так в справочнике записано «значение неизвестно» — приводим к одному слову
UNKNOWN_DESCRIPTIONS = {"NULL", "Not Mapped", "Not Available", "Unknown/Invalid", ""}


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    """Прочитать исходный CSV целиком; знак «?» считается пропуском."""
    try:
        # В этом датасете пропуски записаны знаком «?», а не пустой ячейкой.
        # Без na_values pandas считал бы «?» обычным значением.
        return pd.read_csv(path, na_values="?", low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Файл данных не найден: {path}. Скачайте с UCI и положите в папку data/"
        ) from None


def load_in_chunks(path=config.RAW_CSV, chunksize=20000) -> pd.DataFrame:
    """Прочитать CSV кусками по chunksize строк, сразу сжимая типы каждого куска."""
    # Почему кусками: в памяти одновременно находится только один «сырой» кусок,
    # поэтому пик памяти = один кусок. Такой подход работает и для файлов,
    # которые целиком в оперативную память не помещаются.
    chunks = []
    try:
        reader = pd.read_csv(path, na_values="?", low_memory=False, chunksize=chunksize)
        for chunk in reader:
            chunks.append(optimize_dtypes(chunk))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Файл данных не найден: {path}. Скачайте с UCI и положите в папку data/"
        ) from None
    df = pd.concat(chunks, ignore_index=True)
    # После склейки категории разных кусков могут не совпасть, и столбец снова
    # станет текстовым — поэтому сжимаем типы ещё раз.
    return optimize_dtypes(df)


def optimize_dtypes(df) -> pd.DataFrame:
    """Вернуть копию таблицы с более экономными типами столбцов."""
    df = df.copy()
    for col in df.columns:
        series = df[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            continue  # уже сжат
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Текст с малым числом разных значений выгодно хранить как category:
            # в памяти лежит словарь значений и короткие номера вместо строк.
            df[col] = series.astype("category")
        elif pd.api.types.is_integer_dtype(series):
            # int64 -> самый маленький целый тип, в который помещаются значения
            df[col] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            df[col] = pd.to_numeric(series, downcast="float")
    return df


def memory_mb(df) -> float:
    """Сколько мегабайт занимает таблица в памяти (с учётом строк)."""
    return round(df.memory_usage(deep=True).sum() / 1024**2, 2)


def load_mapping(path=config.MAPPING_CSV) -> dict:
    """Прочитать три справочника из IDS_mapping.csv в словарь {имя: DataFrame(id, description)}."""
    # В одном файле лежат три таблицы подряд, разделённые строкой «,».
    # pd.read_csv такое не разберёт, поэтому читаем построчно модулем csv.
    try:
        with open(path, newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл справочников не найден: {path}") from None

    blocks = {}
    current_name = None  # None = ждём строку с именем нового блока
    for row in rows:
        is_separator = len(row) == 0 or all(cell.strip() == "" for cell in row)
        if is_separator:
            current_name = None
        elif current_name is None:
            current_name = row[0].strip()  # первая строка блока — его заголовок
            blocks[current_name] = []
        else:
            description = row[1].strip() if len(row) > 1 else ""
            if description in UNKNOWN_DESCRIPTIONS:
                description = "Unknown"
            blocks[current_name].append({"id": int(row[0]), "description": description})

    return {name: pd.DataFrame(items, columns=["id", "description"]) for name, items in blocks.items()}
