"""Пути и константы проекта. Все остальные модули берут пути только отсюда."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = ROOT / "logs"
SQL_DIR = ROOT / "sql"

RAW_CSV = DATA_DIR / "diabetic_data.csv"
MAPPING_CSV = DATA_DIR / "IDS_mapping.csv"
CLEAN_PARQUET = PROCESSED_DIR / "clean.parquet"
DB_PATH = PROCESSED_DIR / "readmission.db"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
AUDIT_JSON = REPORTS_DIR / "data_audit.json"
METRICS_CSV = REPORTS_DIR / "metrics.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2                                   # доля ПАЦИЕНТОВ в тесте
EXPIRED_OR_HOSPICE = [11, 13, 14, 19, 20, 21]     # discharge_disposition_id: умер / хоспис
NEAR_CONSTANT_THRESHOLD = 0.9995                  # столбец бесполезен, если одно значение > 99.95%


def ensure_dirs() -> None:
    """Создать выходные папки, если их ещё нет."""
    for folder in [PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
