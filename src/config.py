"""Paths and domain definitions. No dependency on the current directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / 'diabetes+130-us+hospitals+for+years+1999-2008.zip'
DATA = ROOT / 'data'
REPORTS = ROOT / 'reports'
FIGURES = REPORTS / 'figures'
TABLES = REPORTS / 'tables'
ARTIFACTS = ROOT / 'artifacts'
DB_PATH = DATA / 'project.sqlite3'
TARGET = 'readmitted_30d'
SEED = 42
AUTHORS = ['Zhenissov Kaiyrbek', 'Bekarys']
ID_COLS = ['encounter_id', 'patient_nbr']
NUM_COLS = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses']
CODE_COLS = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
DIAG_COLS = ['diag_1', 'diag_2', 'diag_3']
DRUG_COLS = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
             'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
             'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
             'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
             'glipizide-metformin', 'glimepiride-pioglitazone',
             'metformin-rosiglitazone', 'metformin-pioglitazone']
DEATH_CODES = ['11', '19', '20', '21']
HOSPICE_CODES = ['13', '14']
SPECIAL_MISSING = {
    'gender': ['Unknown/Invalid'], 'admission_type_id': ['5', '6', '8'],
    'discharge_disposition_id': ['18', '25', '26'],
    'admission_source_id': ['9', '15', '17', '20', '21'],
}
# Fixed domain exclusions, not selected by looking at test results.
DROP_FEATURES = ID_COLS + ['readmitted', TARGET, 'weight', 'examide', 'citoglipton']
AGE_ORDER = [f'[{i}-{i+10})' for i in range(0, 100, 10)]

def ensure_directories():
    for path in [DATA, REPORTS, FIGURES, TABLES, ARTIFACTS]:
        path.mkdir(parents=True, exist_ok=True)
