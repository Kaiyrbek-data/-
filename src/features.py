"""Признаки модели: группы диагнозов, возраст числом, прошлые обращения, ответ."""
import pandas as pd

TARGET = "readmit_30"      # ответ: 1 — вернулся раньше 30 дней
GROUP = "patient_nbr"      # по пациенту делим данные на train и test

NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "total_prior_visits", "age_num",
]
CATEGORICAL = [
    "race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "medical_specialty", "payer_code", "A1Cresult", "max_glu_serum",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "acarbose", "insulin", "glyburide-metformin",
    "change", "diabetesMed",
]
FEATURES = NUMERIC + CATEGORICAL   # 36 признаков

# Диапазоны кодов ICD-9 (Strack et al., 2014), границы включительно
ICD9_RANGES = {
    "Circulatory": [(390, 459), (785, 785)],
    "Respiratory": [(460, 519), (786, 786)],
    "Digestive": [(520, 579), (787, 787)],
    "Injury": [(800, 999)],
    "Musculoskeletal": [(710, 739)],
    "Genitourinary": [(580, 629), (788, 788)],
    "Neoplasms": [(140, 239)],
}


def map_icd9(code) -> str:
    """Перевести код диагноза ICD-9 в одну из 9 групп болезней."""
    if pd.isna(code):
        return "Missing"
    code = str(code).strip()
    if code.startswith(("V", "E")):      # дополнительные коды: обследования, внешние причины
        return "Other"
    try:
        number = float(code)
    except ValueError:                   # нечисловой код
        return "Other"
    if 250 <= number < 251:              # 250.xx — диабет
        return "Diabetes"
    for group, ranges in ICD9_RANGES.items():
        for low, high in ranges:
            if low <= int(number) <= high:
                return group
    return "Other"


def add_features(df) -> pd.DataFrame:
    """Добавить группы диагнозов, ответ, сумму прошлых обращений и возраст числом."""
    for position in [1, 2, 3]:
        df[f"diag_{position}_group"] = df[f"diag_{position}"].map(map_icd9, na_action=None)
    df[TARGET] = (df["readmitted"] == "<30").astype("int8")
    df["total_prior_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    # "[70-80)" -> 70 + 5 = 75: середина возрастного интервала
    df["age_num"] = df["age"].astype(str).str.extract(r"\[(\d+)-")[0].astype(int) + 5
    return df


def prepare_for_model(df) -> pd.DataFrame:
    """Сделать категориальные признаки текстом: код 3 не «больше» кода 1."""
    df = df.copy()
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).astype(object)
    return df
