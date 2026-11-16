"""Обучение, сравнение и сохранение моделей; прогноз для одного пациента (Часть 1)."""
import io
import json
import sqlite3
import time
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")  # рисуем сразу в файл, окно с графиком не нужно
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config, preprocessing

DUMMY_NAME = "Dummy (baseline)"
LR_NAME = "Logistic Regression"
LGBM_NAME = "LightGBM"

METRIC_COLUMNS = [
    "model", "roc_auc", "pr_auc", "brier", "threshold", "precision", "recall", "f1",
    "train_sec", "predict_ms_per_1000", "size_mb", "best_params",
]

ERROR_ANALYSIS_COLUMNS = [
    "number_inpatient", "number_emergency", "time_in_hospital",
    "num_medications", "number_diagnoses", "age_num",
]

# Значения по умолчанию для прогноза, если пользователь не указал признак
# (самые частые значения в данных)
DEFAULT_CATEGORICAL = {
    "race": "Caucasian", "gender": "Female",
    "admission_type_id": "1", "discharge_disposition_id": "1", "admission_source_id": "7",
    "medical_specialty": "Unknown", "payer_code": "Unknown",
    "A1Cresult": "NotMeasured", "max_glu_serum": "NotMeasured",
    "diag_1_group": "Circulatory", "diag_2_group": "Circulatory", "diag_3_group": "Diabetes",
    "change": "No", "diabetesMed": "Yes",
}


# ---------------------------------------------------------------- разбиение

def split_by_patient(df):
    """Разделить строки на train/test так, чтобы каждый пациент был только в одной части."""
    # Почему не обычный train_test_split: у 16 773 пациентов по несколько визитов.
    # Если один визит попадёт в train, а другой в test, модель «узнает» пациента
    # и качество на test окажется завышенным (утечка данных).
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df[preprocessing.GROUP]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    common = set(train[preprocessing.GROUP]) & set(test[preprocessing.GROUP])
    assert len(common) == 0, "Утечка: один пациент попал и в train, и в test"
    return train, test


def make_cv():
    """Кросс-валидация: 4 части, пациенты не пересекаются, доля возвратов одинаковая."""
    return StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=config.RANDOM_STATE)


# ---------------------------------------------------------------- модели

def make_preprocessor(num_cols, cat_cols, scale=True) -> ColumnTransformer:
    """Собрать преобразование признаков: числа — пропуски (+ масштаб), категории — one-hot."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        # Логистической регрессии нужен единый масштаб признаков.
        # Деревьям (LightGBM) он не нужен: они сравнивают значение с порогом.
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), num_cols),
        # handle_unknown="ignore": новая категория при прогнозе не вызовет ошибку.
        # min_frequency=50: редкие категории объединяются в одну, меньше шума.
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat_cols),
    ])


def make_models(num_cols, cat_cols) -> dict:
    """Вернуть словарь {имя: (pipeline, сетка параметров или None)}."""
    # Всё внутри Pipeline: медиана, масштаб и список категорий считаются только
    # по обучающей части — в том числе внутри каждой части кросс-валидации.
    dummy = Pipeline([
        ("prep", make_preprocessor(num_cols, cat_cols)),
        ("clf", DummyClassifier(strategy="prior")),   # всем выдаёт среднюю долю возвратов
    ])
    logreg = Pipeline([
        ("prep", make_preprocessor(num_cols, cat_cols)),
        # class_weight="balanced": ошибки на редком классе (11%) весят больше
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    lgbm = Pipeline([
        ("prep", make_preprocessor(num_cols, cat_cols, scale=False)),
        ("clf", LGBMClassifier(objective="binary", is_unbalance=True, verbose=-1, n_jobs=-1,
                               random_state=config.RANDOM_STATE)),
    ])
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
        DUMMY_NAME: (dummy, None),
        LR_NAME: (logreg, {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0]}),
        LGBM_NAME: (lgbm, lgbm_grid),
    }


def tune(pipeline, grid, n_iter, X, y, groups):
    """Подобрать параметры случайным поиском по сетке; вернуть (лучшую модель, параметры)."""
    search = RandomizedSearchCV(
        pipeline, grid, n_iter=n_iter,
        scoring="average_precision",       # PR-AUC: честная метрика при редком классе
        cv=make_cv(), random_state=config.RANDOM_STATE, n_jobs=1,
    )
    search.fit(X, y, groups=groups)        # groups: пациенты не пересекаются между частями CV
    return search.best_estimator_, search.best_params_


# ---------------------------------------------------------------- оценка

def best_f1_threshold(y_true, proba) -> float:
    """Найти порог вероятности, при котором F1 максимален."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # последняя точка кривой не имеет своего порога, поэтому [:-1]
    return float(thresholds[np.argmax(f1[:-1])])


def choose_threshold(model, X_train, y_train, groups) -> float:
    """Выбрать порог только по train: прогнозы «из кросс-валидации», test не участвует."""
    # Почему не 0.5: положительных всего 11%, и модель почти никогда не даёт
    # вероятность выше 0.5 после калибровки — при пороге 0.5 recall был бы около нуля.
    proba = cross_val_predict(clone(model), X_train, y_train, groups=groups,
                              cv=make_cv(), method="predict_proba")[:, 1]
    return best_f1_threshold(y_train, proba)


def model_size_mb(model) -> float:
    """Размер модели в МБ (сохраняем в память, а не на диск)."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return round(buffer.getbuffer().nbytes / 1024**2, 3)


def evaluate(name, model, X_test, y_test, threshold) -> dict:
    """Посчитать метрики качества модели на тестовой выборке."""
    start = time.perf_counter()
    proba = model.predict_proba(X_test)[:, 1]
    predict_sec = time.perf_counter() - start
    pred = (proba >= threshold).astype(int)
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "brier": round(brier_score_loss(y_test, proba), 4),   # ошибка самих вероятностей
        "threshold": round(threshold, 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "predict_ms_per_1000": round(predict_sec / len(X_test) * 1000 * 1000, 2),
        "size_mb": model_size_mb(model),
    }


def calibrate(best_model, X_train, y_train, groups):
    """Откалибровать вероятности лучшей модели; вернуть (модель, порог)."""
    # Из-за class_weight / is_unbalance модель завышает вероятности: говорит «40%»,
    # а реально возвращается ~15%. Изотоническая регрессия учит поправку
    # «что модель сказала -> что было на самом деле» на данных, которых модель не видела.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.RANDOM_STATE)
    fit_idx, cal_idx = next(splitter.split(X_train, groups=groups))
    base = clone(best_model).fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
    # FrozenEstimator: базовую модель не переобучать, учить только поправку
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calibrated.fit(X_train.iloc[cal_idx], y_train.iloc[cal_idx])
    proba = calibrated.predict_proba(X_train.iloc[cal_idx])[:, 1]
    return calibrated, best_f1_threshold(y_train.iloc[cal_idx], proba)


# ---------------------------------------------------------------- сохранение

def save_experiment(table) -> None:
    """Сохранить таблицу метрик в CSV, Parquet и (если база есть) в таблицу experiments."""
    config.ensure_dirs()
    table.to_csv(config.METRICS_CSV, index=False)
    table.to_parquet(config.REPORTS_DIR / "metrics.parquet", index=False)
    if config.DB_PATH.exists():
        with_time = table.copy()
        with_time["run_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = sqlite3.connect(config.DB_PATH)
        try:
            # append: история всех запусков копится, старые результаты не теряются
            with_time.to_sql("experiments", connection, if_exists="append", index=False)
        finally:
            connection.close()


def plot_calibration(fitted, X_test, y_test) -> str:
    """График калибровки: предсказанная вероятность против реальной доли возвратов."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Идеальная калибровка")
    for name, model in fitted.items():
        if name == DUMMY_NAME:
            continue  # у Dummy одна вероятность на всех — кривой нет
        proba = model.predict_proba(X_test)[:, 1]
        true_share, predicted = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        ax.plot(predicted, true_share, marker="o", label=name)
    ax.set_xlabel("Предсказанная вероятность")
    ax.set_ylabel("Реальная доля возвратов")
    ax.set_title("Калибровка вероятностей (тестовая выборка)")
    ax.legend(fontsize=8)
    path = config.FIGURES_DIR / "08_calibration.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def plot_confusion(model, name, threshold, X_test, y_test) -> str:
    """Матрица ошибок сохранённой модели при её пороге."""
    pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    matrix = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["Не вернулся", "Вернулся < 30 дн."]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Прогноз модели")
    ax.set_ylabel("На самом деле")
    ax.set_title(f"Матрица ошибок: {name}\nпорог {threshold:.3f}", fontsize=10)
    path = config.FIGURES_DIR / "09_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def save_feature_importance(model, X_test, y_test) -> pd.DataFrame:
    """Permutation importance: насколько падает PR-AUC, если перемешать один признак."""
    # Берём 6000 строк test: метод обучает ничего, но делает много прогнозов.
    sample = X_test.sample(n=min(6000, len(X_test)), random_state=config.RANDOM_STATE)
    result = permutation_importance(
        model, sample, y_test.loc[sample.index], scoring="average_precision",
        n_repeats=5, random_state=config.RANDOM_STATE,
    )
    table = pd.DataFrame({
        "feature": sample.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    table.to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)

    top = table.head(15).iloc[::-1]   # переворачиваем, чтобы самый важный был сверху
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#2a78d6")
    ax.set_xlabel("Падение PR-AUC при перемешивании признака")
    ax.set_ylabel("Признак")
    ax.set_title("Важность признаков (permutation importance), топ-15")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "10_feature_importance.png", dpi=130)
    plt.close(fig)
    return table


def save_error_analysis(model, threshold, test) -> pd.DataFrame:
    """Сравнить пропущенных (FN) и пойманных (TP) пациентов по средним значениям признаков."""
    features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    X = test[features] if features else test
    pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
    actual = test[preprocessing.TARGET].to_numpy()
    groups = {
        "FN - пропущенные (вернулись, модель не предупредила)": (actual == 1) & (pred == 0),
        "TP - пойманные (вернулись, модель предупредила)": (actual == 1) & (pred == 1),
    }
    rows = []
    for label, mask in groups.items():
        row = {"group": label, "count": int(mask.sum())}
        for col in ERROR_ANALYSIS_COLUMNS:
            row[col] = round(float(test.loc[mask, col].mean()), 2) if mask.any() else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(config.REPORTS_DIR / "error_analysis.csv", index=False)
    return table


# ---------------------------------------------------------------- главный сценарий

def train_and_compare(df, tune_models=True, save=True) -> pd.DataFrame:
    """Обучить Dummy, LR и LightGBM, откалибровать лучшую, вернуть таблицу сравнения."""
    df = preprocessing.prepare_for_model(df)
    num_cols, cat_cols = preprocessing.get_feature_lists(df)
    features = num_cols + cat_cols

    train, test = split_by_patient(df)
    X_train, y_train, groups = train[features], train[preprocessing.TARGET], train[preprocessing.GROUP]
    X_test, y_test = test[features], test[preprocessing.TARGET]

    rows, fitted = [], {}
    for name, (pipeline, grid) in make_models(num_cols, cat_cols).items():
        start = time.perf_counter()
        best_params = {}
        if grid is not None and tune_models:
            n_iter = 12 if name == LGBM_NAME else len(grid["clf__C"])
            model, best_params = tune(pipeline, grid, n_iter, X_train, y_train, groups)
        else:
            model = pipeline.fit(X_train, y_train)
        train_sec = time.perf_counter() - start

        if name == DUMMY_NAME:
            threshold = 0.5   # у Dummy одна вероятность на всех, подбирать нечего
        else:
            threshold = choose_threshold(model, X_train, y_train, groups)

        row = evaluate(name, model, X_test, y_test, threshold)
        row["train_sec"] = round(train_sec, 1)
        row["best_params"] = json.dumps(best_params)
        rows.append(row)
        fitted[name] = model

    # Лучшая по PR-AUC (Dummy не считаем) -> калибруем
    candidates = [row for row in rows if row["model"] != DUMMY_NAME]
    best_row = max(candidates, key=lambda row: row["pr_auc"])
    best_name = best_row["model"]

    start = time.perf_counter()
    calibrated, cal_threshold = calibrate(fitted[best_name], X_train, y_train, groups)
    cal_name = f"{best_name} + isotonic calibration"
    row = evaluate(cal_name, calibrated, X_test, y_test, cal_threshold)
    row["train_sec"] = round(time.perf_counter() - start, 1)
    row["best_params"] = best_row["best_params"]
    rows.append(row)
    fitted[cal_name] = calibrated

    table = pd.DataFrame(rows)[METRIC_COLUMNS]

    if save:
        config.ensure_dirs()
        joblib.dump({
            "model": calibrated, "threshold": cal_threshold, "features": features,
            "num_cols": num_cols, "cat_cols": cat_cols, "name": cal_name,
        }, config.MODEL_PATH)
        save_experiment(table)
        plot_calibration(fitted, X_test, y_test)
        plot_confusion(calibrated, cal_name, cal_threshold, X_test, y_test)
        save_feature_importance(calibrated, X_test, y_test)
        save_error_analysis(calibrated, cal_threshold, test[features + [preprocessing.TARGET]])
    return table


def leakage_demo(df) -> pd.DataFrame:
    """Сравнить разбиение по пациентам со случайным по строкам (где пациенты пересекаются)."""
    # При случайном разбиении тысячи пациентов оказываются и в train, и в test.
    # Такая оценка нечестная, даже если цифры получаются близкими: модель в жизни
    # будет работать с новыми пациентами, которых она раньше не видела.
    df = preprocessing.prepare_for_model(df)
    num_cols, cat_cols = preprocessing.get_feature_lists(df)
    features = num_cols + cat_cols
    pipeline = make_models(num_cols, cat_cols)[LGBM_NAME][0]

    by_patient = split_by_patient(df)
    by_row = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
                              stratify=df[preprocessing.TARGET])
    rows = []
    for label, (train, test) in {"по пациентам (правильно)": by_patient,
                                 "случайно по строкам (утечка)": by_row}.items():
        model = clone(pipeline).fit(train[features], train[preprocessing.TARGET])
        proba = model.predict_proba(test[features])[:, 1]
        common = set(train[preprocessing.GROUP]) & set(test[preprocessing.GROUP])
        rows.append({
            "split": label,
            "patients_in_both_train_and_test": len(common),
            "roc_auc": round(roc_auc_score(test[preprocessing.TARGET], proba), 4),
            "pr_auc": round(average_precision_score(test[preprocessing.TARGET], proba), 4),
        })
    table = pd.DataFrame(rows)
    config.ensure_dirs()
    table.to_csv(config.REPORTS_DIR / "leakage_demo.csv", index=False)
    return table


# ---------------------------------------------------------------- прогноз

def load_model() -> dict:
    """Загрузить сохранённую модель вместе с порогом и списком признаков."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {config.MODEL_PATH}. Сначала обучите её: пункт 6 меню, режим 2."
        )
    return joblib.load(config.MODEL_PATH)


def predict_patient(patient: dict, bundle: dict | None = None) -> dict:
    """Посчитать вероятность возврата для одного пациента, заданного словарём признаков."""
    if bundle is None:
        bundle = load_model()

    row = {}
    for col in bundle["num_cols"]:
        # Неуказанное число -> NaN; внутри модели его заменит медиана (SimpleImputer)
        row[col] = patient.get(col, np.nan)
    for col in bundle["cat_cols"]:
        # Неуказанная категория -> типичное значение; для лекарств — "No" (не назначено)
        default = DEFAULT_CATEGORICAL.get(col, "No")
        row[col] = str(patient.get(col, default))

    X = pd.DataFrame([row])[bundle["features"]]
    probability = float(bundle["model"].predict_proba(X)[0, 1])
    return {
        "probability": probability,
        "high_risk": bool(probability >= bundle["threshold"]),
        "threshold": float(bundle["threshold"]),
        "model": bundle["name"],
    }
