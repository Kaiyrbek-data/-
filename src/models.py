"""Модели: разбиение по пациентам, обучение, метрики, калибровка, прогноз."""
import io
import json
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config, features

DUMMY, LOGREG, LGBM = "Dummy (baseline)", "Logistic Regression", "LightGBM"

# Значения по умолчанию для прогноза — самые частые в данных
DEFAULTS = {
    "race": "Caucasian", "gender": "Female", "admission_type_id": "1",
    "discharge_disposition_id": "1", "admission_source_id": "7",
    "medical_specialty": "Unknown", "payer_code": "Unknown",
    "A1Cresult": "NotMeasured", "max_glu_serum": "NotMeasured",
    "diag_1_group": "Circulatory", "diag_2_group": "Circulatory", "diag_3_group": "Diabetes",
    "change": "No", "diabetesMed": "Yes",
}


# ------------------------------------------------------------ разбиение

def split_by_patient(df):
    """Разделить на train и test так, чтобы каждый пациент был только в одной части."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df[features.GROUP]))
    return df.iloc[train_idx], df.iloc[test_idx]


def make_cv():
    """Кросс-валидация на 4 части без общих пациентов и с равной долей возвратов."""
    return StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=config.RANDOM_STATE)


# ------------------------------------------------------------ модели

def make_pipeline(classifier, scale=True) -> Pipeline:
    """Собрать Pipeline: подготовка признаков + модель."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]   # пропуск -> медиана
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))            # единый масштаб чисел
    prep = ColumnTransformer([
        ("num", Pipeline(numeric_steps), features.NUMERIC),
        # категория -> столбцы из 0 и 1; редкие категории (< 50 строк) объединяются
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), features.CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("clf", classifier)])


def get_models() -> dict:
    """Вернуть словарь {имя: (pipeline, сетка параметров, число вариантов поиска)}."""
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    lgbm = LGBMClassifier(objective="binary", is_unbalance=True, verbose=-1, n_jobs=-1,
                          random_state=config.RANDOM_STATE)
    lgbm_grid = {
        "clf__n_estimators": [200, 400, 600],
        "clf__learning_rate": [0.02, 0.05, 0.1],
        "clf__num_leaves": [15, 31, 63],
        "clf__min_child_samples": [20, 50, 100],
        "clf__subsample": [0.8, 1.0],
        "clf__subsample_freq": [1],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_lambda": [0, 1, 5],
    }
    return {
        DUMMY: (make_pipeline(DummyClassifier(strategy="prior")), None, 0),
        LOGREG: (make_pipeline(logreg), {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0]}, 5),
        LGBM: (make_pipeline(lgbm, scale=False), lgbm_grid, 12),   # деревьям масштаб не нужен
    }


def fit_model(pipeline, grid, n_iter, X, y, groups, tune=True):
    """Обучить модель. Вернуть (модель, лучшие параметры, секунды обучения)."""
    start = time.perf_counter()
    if grid is None or not tune:
        model, params = pipeline.fit(X, y), {}
    else:
        search = RandomizedSearchCV(pipeline, grid, n_iter=n_iter, scoring="average_precision",
                                    cv=make_cv(), random_state=config.RANDOM_STATE, n_jobs=1)
        search.fit(X, y, groups=groups)
        model, params = search.best_estimator_, search.best_params_
    return model, params, round(time.perf_counter() - start, 1)


# ------------------------------------------------------------ порог и метрики

def best_f1_threshold(y_true, proba) -> float:
    """Найти порог вероятности, при котором F1 максимален."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def choose_threshold(model, X, y, groups) -> float:
    """Выбрать порог только по train: по прогнозам из кросс-валидации."""
    proba = cross_val_predict(clone(model), X, y, groups=groups, cv=make_cv(),
                              method="predict_proba")[:, 1]
    return best_f1_threshold(y, proba)


def model_size_mb(model) -> float:
    """Размер сохранённой модели в мегабайтах."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return round(buffer.getbuffer().nbytes / 1024**2, 3)


def evaluate(name, model, X_test, y_test, threshold) -> dict:
    """Посчитать метрики модели на тестовой выборке."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "brier": round(brier_score_loss(y_test, proba), 4),
        "threshold": round(threshold, 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "size_mb": model_size_mb(model),
    }


# ------------------------------------------------------------ калибровка

def calibrate(model, X, y, groups):
    """Исправить завышенные вероятности. Вернуть (калиброванную модель, её порог)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.RANDOM_STATE)
    fit_idx, cal_idx = next(splitter.split(X, groups=groups))
    base = clone(model).fit(X.iloc[fit_idx], y.iloc[fit_idx])           # 75% пациентов
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calibrated.fit(X.iloc[cal_idx], y.iloc[cal_idx])                    # поправка на остальных 25%
    proba = calibrated.predict_proba(X.iloc[cal_idx])[:, 1]
    return calibrated, best_f1_threshold(y.iloc[cal_idx], proba)


# ------------------------------------------------------------ весь сценарий

def train_all(clean, tune=True, save=True) -> pd.DataFrame:
    """Обучить все модели, откалибровать лучшую и вернуть таблицу сравнения."""
    df = features.prepare_for_model(clean)
    train, test = split_by_patient(df)
    X_train, y_train, groups = train[features.FEATURES], train[features.TARGET], train[features.GROUP]
    X_test, y_test = test[features.FEATURES], test[features.TARGET]

    rows, fitted = [], {}
    for name, (pipeline, grid, n_iter) in get_models().items():
        model, params, seconds = fit_model(pipeline, grid, n_iter, X_train, y_train, groups, tune)
        threshold = 0.5 if name == DUMMY else choose_threshold(model, X_train, y_train, groups)
        row = evaluate(name, model, X_test, y_test, threshold)
        row.update(train_sec=seconds, best_params=json.dumps(params))
        rows.append(row)
        fitted[name] = model

    best = max(rows[1:], key=lambda row: row["pr_auc"])     # Dummy не участвует
    start = time.perf_counter()
    calibrated, threshold = calibrate(fitted[best["model"]], X_train, y_train, groups)
    name = best["model"] + " + isotonic calibration"
    row = evaluate(name, calibrated, X_test, y_test, threshold)
    row.update(train_sec=round(time.perf_counter() - start, 1), best_params=best["best_params"])
    rows.append(row)

    table = pd.DataFrame(rows)
    if save:
        save_model(calibrated, threshold, name, table)
    return table


# ------------------------------------------------------------ сохранение и прогноз

def save_model(model, threshold, name, table) -> None:
    """Сохранить лучшую модель (joblib) и таблицу сравнения моделей (Parquet)."""
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "name": name}, config.MODEL_PATH)
    table.to_parquet(config.METRICS_PATH, index=False)


def load_model() -> dict:
    """Загрузить сохранённую модель вместе с порогом и названием."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("Модель не найдена. Обучите её: пункт 6 меню, режим 2.")
    return joblib.load(config.MODEL_PATH)


def predict_patient(patient: dict) -> dict:
    """Посчитать вероятность возврата для одного пациента."""
    bundle = load_model()
    row = {col: patient.get(col, np.nan) for col in features.NUMERIC}       # нет числа -> медиана
    for col in features.CATEGORICAL:                                        # нет категории -> типичная
        row[col] = str(patient.get(col, DEFAULTS.get(col, "No")))
    probability = float(bundle["model"].predict_proba(pd.DataFrame([row]))[0, 1])
    return {
        "probability": probability,
        "high_risk": probability >= bundle["threshold"],
        "threshold": bundle["threshold"],
        "model": bundle["name"],
    }
