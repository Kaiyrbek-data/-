"""Пути и константы проекта. Остальные файлы берут их только отсюда."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
EXPORT_DIR = ROOT / "exports"

RAW_CSV = DATA_DIR / "diabetic_data.csv"          # исходные данные
MAPPING_CSV = DATA_DIR / "IDS_mapping.csv"        # расшифровки кодов
CLEAN_PARQUET = DATA_DIR / "clean.parquet"        # очищенные данные (создаёт программа)
DB_PATH = DATA_DIR / "readmission.db"             # база SQLite (создаёт программа)
SQL_FILE = ROOT / "sql" / "queries.sql"
MODEL_PATH = MODELS_DIR / "best_model.joblib"     # лучшая модель
METRICS_PATH = MODELS_DIR / "metrics.parquet"     # таблица сравнения моделей

RANDOM_STATE = 42
TEST_SIZE = 0.2                                   # доля ПАЦИЕНТОВ в тесте
EXPIRED_OR_HOSPICE = [11, 13, 14, 19, 20, 21]     # коды выписки: умер или хоспис
NEAR_CONSTANT_THRESHOLD = 0.9995                  # одно значение занимает больше 99.95% строк
