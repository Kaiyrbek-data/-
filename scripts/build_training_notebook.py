"""Создать учебный ноутбук с реальным обучением, подбором параметров и оценкой."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


md(r'''
# Обучение модели: от исходных категорий до метрик и разбора ошибок

**Проект:** повторная госпитализация пациентов с диабетом. **Авторы:** Кайырбек Женисов и Бекарыс Ережеп.

Это исполняемый учебный эксперимент: ячейки **обучают модели заново**, а не только читают старые метрики.
Читать и запускать сверху вниз: Jupyter / VS Code → **Restart Kernel and Run All**.
Зависимости: `python -m pip install -r requirements.txt` из корня репозитория.

Путь работы:
1. Загрузить источник, определить целевую переменную и популяцию.
2. Разделить **пациентов**, а не отдельные визиты, на четыре части.
3. Понять признаки, пропуски, категории, one-hot и стандартизацию.
4. Обучить Dummy, Logistic Regression и LightGBM.
5. Найти гиперпараметры на групповой CV и проверить, стало ли лучше.
6. Откалибровать вероятности и выбрать порог на отдельных данных.
7. Зафиксировать модель, рассчитать метрики на test и разобрать ошибки.

Новый эксперимент сохраняется в `reports/model_training_detailed/` и
`artifacts/notebook_training/`; действующая модель сайта не перезаписывается.
Обучение занимает несколько минут или дольше в зависимости от компьютера.
Готовые outputs позволяют изучить этот файл без повторного обучения.
''')
md('''
## 1. Настройки и воспроизводимость

По умолчанию используются **все данные**, три групповых CV-фолда, три набора параметров
логистической регрессии и четыре — LightGBM. `EXTENDED_SEARCH=True` увеличивает
бюджет до 8/12 кандидатов, но не гарантирует улучшения. Случайность фиксируем.

Это отдельный учебный запуск. Его показатели могут отличаться от прежних отчётов проекта.
Test из этого набора уже просматривался в предыдущих экспериментах: внутри текущего
запуска он не используется для выбора решений, но полностью новое независимое
подтверждение потребовало бы других пациентов / нового источника данных.
''')
code('''
from pathlib import Path
import sys, json, time, platform, importlib.metadata
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'src' / 'features.py').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve, brier_score_loss)
from threadpoolctl import threadpool_limits
import joblib

from src.config import SEED, TARGET, NUM_COLS, CODE_COLS, DIAG_COLS, ARTIFACTS, REPORTS
from src.data import load_data, make_cohort, patient_split
from src.features import (FeatureBuilder, INPUT_COLS, MODEL_NUM_COLS,
                          MODEL_CAT_COLS, build_pipeline, icd9_group)
from src.modeling import model_candidates, choose_threshold, evaluate, group_bootstrap
get_ipython().run_line_magic('matplotlib', 'inline')

EXTENDED_SEARCH = False
N_ITER = {'LogisticRegression': 8 if EXTENDED_SEARCH else 3,
          'LightGBM': 12 if EXTENDED_SEARCH else 4}
CV_SPLITS = 3
BOOTSTRAP_REPEATS = 200
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = REPORTS / 'model_training_detailed'
MODEL_OUT = ARTIFACTS / 'notebook_training'
OUT.mkdir(parents=True, exist_ok=True)
MODEL_OUT.mkdir(parents=True, exist_ok=True)
pd.set_option('display.max_columns', 30)
sns.set_theme(style='whitegrid')

def show_plot(name):
    """Сохранить текущий график и показать его непосредственно в ноутбуке."""
    plt.tight_layout()
    plt.savefig(OUT / (name + '.png'), dpi=130, bbox_inches='tight')
    plt.show()
    plt.close()

versions = {name: importlib.metadata.version(name) for name in
            ['pandas', 'numpy', 'scikit-learn', 'lightgbm', 'joblib']}
display(pd.Series({'python': platform.python_version(), **versions}, name='version'))
print('Бюджет поиска:', N_ITER, '; исходные данные не уменьшаются.')
''')
md('''
## 2. Данные, цель и момент предсказания

Одна строка — госпитализация. `readmitted`: `<30` → 1, `NO` и `>30` → 0.
Это **зарегистрированный** возврат в системе, а не все возможные обращения пациента.
Прогноз делается при выписке: число процедур и исход выписки в этот момент уже известны.
Для прогноза при поступлении набор признаков пришлось бы изменить.

Знак `?` означает пропуск. Категория `None` в лабораторных результатах означает,
что исследование не выполнялось, поэтому её сохраняем. До разбиения исключаем
смерть и хоспис. Неизвестный пол сохраняется и затем обрабатывается как пропуск.
Повторные госпитализации одного пациента не удаляем как дубликаты.
''')
code('''
raw, mappings, source_info = load_data()
cohort, flow = make_cohort(raw)
display(pd.Series(flow, name='число записей'))
display(raw.head(3))
assert raw.encounter_id.is_unique
assert cohort[TARGET].isin([0, 1]).all()
target_table = cohort[TARGET].value_counts().sort_index().rename('rows').to_frame()
target_table['percent'] = target_table.rows / len(cohort) * 100
display(target_table)
print('SHA256 исходного архива:', source_info['archive_sha256'])
''')
md('''
## 3. Защита от утечки: четыре независимые группы пациентов

| Часть | Доля пациентов примерно | Для чего используется |
|---|---:|---|
| fit | 60% | обучение, импутация, стандартизация, словарь категорий, CV и гиперпараметры |
| calibration | 10% | обучение поправки к вероятностям уже обученной модели |
| validation | 10% | сравнение вариантов, выбор модели и порога |
| test | 20% | итоговая проверка зафиксированных решений |

Доли строк могут отличаться от долей пациентов: у людей разное число визитов.
Один `patient_nbr` не может находиться в двух частях. Перемешивание отдельных
строк этого не обеспечивает. Пока не доходим до итоговой оценки, признаки и
ответы test не используем для обучения или выбора решений.
''')
code('''
parts = patient_split(cohort)
fit, calibration, validation, test = [parts[name] for name in
                                    ['fit', 'calibration', 'validation', 'test']]
split_summary = pd.DataFrame([
    {'split': name, 'rows': len(frame), 'patients': frame.patient_nbr.nunique(),
     'positive_rate': frame[TARGET].mean()}
    for name, frame in parts.items()])
display(split_summary)
overlap = pd.DataFrame({a: {b: len(set(fa.patient_nbr) & set(fb.patient_nbr))
                           for b, fb in parts.items()} for a, fa in parts.items()})
display(overlap)
for a, fa in parts.items():
    for b, fb in parts.items():
        if a != b:
            assert set(fa.patient_nbr).isdisjoint(fb.patient_nbr)
X_fit, y_fit = fit[INPUT_COLS], fit[TARGET]
X_cal, y_cal = calibration[INPUT_COLS], calibration[TARGET]
X_val, y_val = validation[INPUT_COLS], validation[TARGET]
assert not {'encounter_id', 'patient_nbr', 'readmitted', TARGET} & set(INPUT_COLS)
split_summary.to_csv(OUT / 'splits.csv', index=False)
''')
md('''
## 4. Какие столбцы числовые, а какие — категории

Числовые признаки описывают количество: дни, процедуры, предыдущие обращения.
Коды `admission_type_id`, `discharge_disposition_id`, `admission_source_id` —
**названия категорий**, хотя выглядят как числа: код 4 не означает «вдвое больше» кода 2.
Возраст в источнике — интервал, его оставляем категорией. Номера пациентов и
госпитализаций, исходный ответ и бинарная цель запрещены как входы модели.

Категориальный dtype Pandas помогает хранить строки компактнее, но сам по себе
не превращает их в пригодные для логистической регрессии признаки.
Для модели ниже применяется OneHotEncoder, а не произвольные номера категорий.
''')
code('''
column_roles = pd.DataFrame([
    {'column': col, 'role': 'число' if col in NUM_COLS else 'категория / исходный код',
     'dtype': str(X_fit[col].dtype), 'unique_fit': X_fit[col].nunique(),
     'missing_fit_pct': X_fit[col].isna().mean()*100}
    for col in INPUT_COLS])
display(column_roles)
category_demo = X_fit['medical_specialty'].astype('category')
display(pd.Series({'строки, МБ': X_fit.medical_specialty.memory_usage(deep=True)/1024**2,
                   'category, МБ': category_demo.memory_usage(deep=True)/1024**2}))
''')
md('''
## 5. Feature engineering: укрупнение диагнозов и дополнительные признаки

Сотни ICD-9 кодов переводим в укрупнённые группы: например, `250.83` → Diabetes,
`428` → Circulatory. Коды V/E → Other; отсутствующий диагноз → Missing.
Число назначенных препаратов и изменений дозы, наличие анализов/записи веса,
сумма предыдущих обращений — детерминированные признаки одной госпитализации.

Не добавляем общее число визитов пациента за весь источник: так можно использовать
его будущие посещения. `FeatureBuilder` одинаково применяется в обучении и прогнозе.
Два фиксированно исключённых лекарственных столбца постоянны в источнике;
остальные редкие категории объединяются **по fit конкретного CV-фолда**.
''')
code('''
examples = ['250.83', '428', '486', '820', 'V57', None]
display(pd.DataFrame({'ICD-9': examples, 'group': [icd9_group(value) for value in examples]}))
feature_builder = FeatureBuilder()
engineered = feature_builder.fit_transform(X_fit)
display(pd.concat([X_fit[['diag_1', 'number_outpatient', 'number_emergency', 'number_inpatient']].head(),
                   engineered[['diag_1_group', 'prior_visits_total', 'n_diabetes_drugs',
                               'n_dose_changes', 'A1C_measured']].head()], axis=1))
display(pd.Series({'числовых признаков после FeatureBuilder': len(MODEL_NUM_COLS),
                   'категориальных': len(MODEL_CAT_COLS)}))
''')
md(r'''
## 6. Пропуски и стандартизация чисел — показываем реальные преобразования

Для чисел: пропуск заменяется медианой, затем `z = (x - mean_fit) / std_fit`.
Среднее преобразованной обучающей части близко к 0, стандартное отклонение — к 1
для непостоянных столбцов (`ddof=0`). При прогнозе используется **та же** медиана,
среднее и масштаб. На validation/test ничего заново не вычисляем.

Логистической регрессии стандартизация помогает сопоставлять масштабы признаков.
Деревьям она обычно не нужна; здесь для совместимости с проектом используется
единый Pipeline. Демо ниже обучено только на fit. В реальной CV эти шаги заново
обучаются **внутри каждого фолда**, а не до разбиения на CV.
[Pipeline и ColumnTransformer — документация](https://scikit-learn.org/stable/modules/compose.html).
''')
code('''
numeric_demo = Pipeline([
    ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
    ('scale', StandardScaler())])
scaled_fit = numeric_demo.fit_transform(engineered[MODEL_NUM_COLS])
numeric_audit = pd.DataFrame({
    'feature': MODEL_NUM_COLS,
    'median_fit': numeric_demo['impute'].statistics_,
    'mean_before': engineered[MODEL_NUM_COLS].mean().to_numpy(),
    'mean_after': scaled_fit.mean(axis=0),
    'std_after': scaled_fit.std(axis=0)})
display(numeric_audit.round(6))
display(pd.DataFrame(scaled_fit[:5, :8], columns=MODEL_NUM_COLS[:8]))
assert np.isfinite(scaled_fit).all()
''')
md('''
## 7. Категории: заполнение пропусков, редкие значения и one-hot

Для категории пропуск становится `Missing`, а OneHotEncoder создаёт бинарные
индикаторы. Например, одно поле «инсулин» превращается в отдельные признаки
для No / Steady / Up / Down. В одном поле действует только один индикатор.

`min_frequency=30` объединяет редкие значения в группу. Неизвестная категория при
прогнозе попадает в эту группу, если она есть; иначе кодируется нулями. Это не
ошибка и не обучение на новых данных. Не кодируем классы как 0/1/2/3 с выдуманным
порядком. Словарь one-hot определяется только обучающими строками.
''')
code('''
categorical_demo = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='Missing', keep_empty_features=True)),
    ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=30,
                             sparse_output=True, dtype=np.float32))])
encoded_fit = categorical_demo.fit_transform(engineered[MODEL_CAT_COLS])
encoder = categorical_demo['encode']
encoded_names = encoder.get_feature_names_out(MODEL_CAT_COLS)
display(pd.DataFrame({'column': MODEL_CAT_COLS,
                     'categories_seen_fit': [len(values) for values in encoder.categories_],
                     'rare_categories': [0 if values is None else len(values)
                                        for values in encoder.infrequent_categories_]}))
print('Форма one-hot матрицы:', encoded_fit.shape, '; ненулевых элементов:', encoded_fit.nnz)
print('Не превращаем всю разреженную матрицу в плотную: показываем только маленький фрагмент.')
display(pd.DataFrame(encoded_fit[:5, :12].toarray(), columns=encoded_names[:12]))
unseen_example = engineered[MODEL_CAT_COLS].head(1).copy()
unseen_example['medical_specialty'] = 'PreviouslyUnseenSpecialty'
assert categorical_demo.transform(unseen_example).shape[1] == encoded_fit.shape[1]
''')
md('''
## 8. Итоговый Pipeline и групповая кросс-валидация

Единая цепочка: **FeatureBuilder → ColumnTransformer → модель**.
ColumnTransformer параллельно обрабатывает числовые и категориальные столбцы.
Каждый кандидат гиперпараметров проверяется на одинаковых трёх CV-фолдах внутри fit.
Внутри фолда медианы, стандартизация и словарь one-hot обучаются на train-фрагменте,
а затем применяются к отложенным пациентам этого фолда.

StratifiedGroupKFold сохраняет группы пациентов и старается сбалансировать классы;
идеально одинаковые доли редкого класса не гарантируются.
[Описание метода](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
''')
code('''
cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
folds = list(cv.split(X_fit, y_fit, groups=fit.patient_nbr))
fold_audit = []
for number, (train_idx, holdout_idx) in enumerate(folds, 1):
    common = set(fit.iloc[train_idx].patient_nbr) & set(fit.iloc[holdout_idx].patient_nbr)
    assert not common
    fold_audit.append({'fold': number, 'train_rows': len(train_idx), 'holdout_rows': len(holdout_idx),
                       'train_positive_rate': y_fit.iloc[train_idx].mean(),
                       'holdout_positive_rate': y_fit.iloc[holdout_idx].mean(),
                       'patient_overlap': len(common)})
display(pd.DataFrame(fold_audit))
candidate_definitions = model_candidates(quick=False)
display(build_pipeline(candidate_definitions['LogisticRegression'][0]))
''')
md('''
## 9. Что будем измерять и зачем нужен Dummy

| Метрика | Смысл | Направление |
|---|---|---|
| ROC-AUC | насколько хорошо модель ранжирует положительные и отрицательные случаи | больше лучше |
| Average precision (AP) | качество precision–recall при редком положительном классе | больше лучше |
| Recall | какую долю фактических возвратов обнаружили | больше, но вместе с precision |
| Precision | какая доля предупреждений оказалась правильной | больше, но вместе с recall |
| F1 | гармоническое среднее precision и recall | больше лучше при выбранном пороге |
| Brier / log-loss | ошибка вероятностей | меньше лучше |
| Accuracy | доля правильных классов | опасно использовать отдельно при дисбалансе |

Здесь TP — обнаруженный возврат, FP — ложное предупреждение, FN — пропущенный
возврат, TN — правильно определённое отсутствие возврата. Формулы:
**precision = TP / (TP + FP)**; **recall = TP / (TP + FN)**;
**F1 = 2 × precision × recall / (precision + recall)**.
Brier — среднее `(вероятность − фактический класс)²` по всем записям.

В этом ноутбуке PR-AUC означает **AP**, а не трапецеидальную площадь под PR-кривой.
AP оценивает ранжирование, а F1/precision/recall зависят от порога. Dummy выдаёт всем
долю положительного класса на fit. При пороге 0.5 он обычно никого не предупреждает:
accuracy высокая, recall равен нулю. «Предупреждать всех» даёт recall=1, но низкий precision.
Оба примера показывают, почему одной метрики недостаточно.
[Определение AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
''')
code('''
dummy = build_pipeline(DummyClassifier(strategy='prior')).fit(X_fit, y_fit)
dummy_val = dummy.predict_proba(X_val)[:, 1]
display(pd.DataFrame([
    {'policy': 'Dummy, порог 0.5', **evaluate(y_val, dummy_val, 0.5)},
    {'policy': 'Dummy, предупреждать всех', **evaluate(y_val, dummy_val, 0.0)}
]).round(4))
print('AP случайного постоянного ранжирования на validation ≈ доле единиц:', y_val.mean())
''')
md('''
## 10. Начальные модели — точка отсчёта перед улучшениями

Logistic Regression — понятная линейная модель; LightGBM — ансамбль деревьев,
который может учитывать нелинейности и взаимодействия признаков. Здесь обе
модели сначала обучаются с начальными параметрами. Сохраняем validation-метрики,
чтобы потом сравнить улучшения с этой точкой, а не только между разными алгоритмами.

В проекте есть также Random Forest. В этом разборе оставляем две разные обучаемые
модели и baseline: это сокращает время Run All без исключения данных из обучения.
''')
code('''
fitted = {'Dummy': dummy}
initial_rows, fit_times = [], {}
for family in ['LogisticRegression', 'LightGBM']:
    print('Обучение начальной модели:', family, flush=True)
    estimator = clone(candidate_definitions[family][0])
    pipeline = build_pipeline(estimator)
    started = time.perf_counter()
    with threadpool_limits(limits=1, user_api='blas'):
        pipeline.fit(X_fit, y_fit)
    label = family + ' / initial'
    fit_times[label] = time.perf_counter()-started
    fitted[label] = pipeline
    p = pipeline.predict_proba(X_val)[:, 1]
    initial_rows.append({'model': label, **evaluate(y_val, p, 0.5), 'fit_seconds': fit_times[label]})
initial_metrics = pd.DataFrame(initial_rows)
display(initial_metrics.round(4))
''')
md('''
## 11. Что такое гиперпараметры и что ищет RandomizedSearchCV

Параметры модели (например, коэффициенты регрессии) находятся при `fit`.
**Гиперпараметры** задают способ обучения и выбираются снаружи:

- LR: `C` — обратная сила регуляризации; `class_weight` — веса ошибок классов.
- LightGBM: число деревьев, learning rate, сложность дерева (`num_leaves`),
  минимум наблюдений в листе, регуляризация, доли строк/признаков, вес единиц.

Сравниваем кандидатов по **среднему AP на групповой CV**, не по test и не по accuracy.
`model__` в имени адресует шаг `model` внутри Pipeline. Для каждого кандидата
показываем среднее, разброс по фолдам и время. Малый бюджет — это выбор нескольких
вариантов из пространства, а не доказательство найденного глобального оптимума.
Повышенный вес редкого класса может увеличить recall и ухудшить вероятности.
SMOTE и другие способы oversampling здесь не применяются.
''')
code('''
search_spaces = []
for family in ['LogisticRegression', 'LightGBM']:
    _, grid, _ = candidate_definitions[family]
    for name, values in grid.items():
        search_spaces.append({'family': family, 'hyperparameter': name, 'values': str(values)})
display(pd.DataFrame(search_spaces))
searches, search_tables = {}, []
for family in ['LogisticRegression', 'LightGBM']:
    estimator, grid, _ = candidate_definitions[family]
    search = RandomizedSearchCV(
        build_pipeline(clone(estimator)), param_distributions=grid,
        n_iter=N_ITER[family], scoring={'ap': 'average_precision', 'roc_auc': 'roc_auc'},
        refit='ap', cv=folds, random_state=SEED, n_jobs=1, error_score='raise',
        return_train_score=True, verbose=1)
    print(f'Подбор {family}: {N_ITER[family]} кандидатов × {CV_SPLITS} фолда', flush=True)
    started = time.perf_counter()
    with threadpool_limits(limits=1, user_api='blas'):
        search.fit(X_fit, y_fit)
    label = family + ' / tuned'
    fit_times[label] = time.perf_counter()-started
    searches[family] = search
    fitted[label] = search.best_estimator_
    table = pd.DataFrame(search.cv_results_)
    table['family'] = family
    table['ap_train_minus_cv'] = table.mean_train_ap - table.mean_test_ap
    search_tables.append(table)
    display(table[['params', 'mean_train_ap', 'mean_test_ap', 'std_test_ap',
                   'mean_test_roc_auc', 'ap_train_minus_cv', 'mean_fit_time',
                   'rank_test_ap']].sort_values('rank_test_ap'))
    print('Лучшие гиперпараметры:', search.best_params_)
    print('Лучший CV AP:', round(search.best_score_, 5))
    table.to_csv(OUT / ('cv_' + family + '.csv'), index=False)
''')
md('''
## 12. Действительно ли подбор улучшил модель?

Сравниваем начальную и подобранную модели на одной validation с одинаковым
порогом 0.5. Метрики ранжирования от этого порога не зависят. Уменьшение разрыва
train–CV помогает замечать переобучение, но само по себе не гарантирует лучшего
результата на других данных. Изменение AP показываем со знаком: отрицательное
изменение нельзя называть улучшением. Test всё ещё не используется.
''')
code('''
raw_validation_rows = []
for label, model in fitted.items():
    p = model.predict_proba(X_val)[:, 1]
    raw_validation_rows.append({'model': label, **evaluate(y_val, p, 0.5)})
raw_validation = pd.DataFrame(raw_validation_rows)
display(raw_validation.round(4))
improvements = []
for family in searches:
    base = raw_validation.set_index('model').loc[family + ' / initial']
    tuned = raw_validation.set_index('model').loc[family + ' / tuned']
    improvements.append({'family': family, 'AP_before': base.pr_auc_ap,
                         'AP_after': tuned.pr_auc_ap, 'delta_AP': tuned.pr_auc_ap-base.pr_auc_ap,
                         'ROC_before': base.roc_auc, 'ROC_after': tuned.roc_auc})
improvements = pd.DataFrame(improvements)
display(improvements.round(5))
for row in improvements.itertuples():
    print(row.family, ':', 'AP вырос' if row.delta_AP > 0 else 'AP не вырос',
          f'({row.delta_AP:+.5f}); это результат данного бюджета и validation.')
''')
md('''
## 13. Калибровка: вероятность и правильный порядок случаев — разные задачи

AP/ROC-AUC оценивают ранжирование: у кого риск выше. Калибровка отвечает другому
вопросу: среди случаев с прогнозом около 20% действительно ли возвращается около 20%?
Обучаем sigmoid-поправку на **calibration**, замораживая уже обученный Pipeline.
Validation остаётся отдельной и используется только для сравнения результатов.

Монотонная sigmoid-поправка обычно сохраняет порядок и поэтому не обязана менять
AP/ROC-AUC. Brier и log-loss могут улучшиться или ухудшиться — это проверяем таблицей.
Точка на диагонали reliability diagram соответствует совпадению средней вероятности
с частотой исхода в группе; малые группы дают нестабильные точки.
[Калибровка вероятностей](https://scikit-learn.org/stable/modules/calibration.html).
''')
code('''
calibrated_models = {'Dummy': dummy}
calibration_rows, val_probabilities = [], {}
for label, model in fitted.items():
    if label == 'Dummy':
        calibrated = model
    else:
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid')
        calibrated.fit(X_cal, y_cal)
    calibrated_models[label] = calibrated
    before = model.predict_proba(X_val)[:, 1]
    after = calibrated.predict_proba(X_val)[:, 1]
    val_probabilities[label] = after
    calibration_rows.append({'model': label,
        'Brier_before': brier_score_loss(y_val, before),
        'Brier_after': brier_score_loss(y_val, after),
        'AP_before': average_precision_score(y_val, before),
        'AP_after': average_precision_score(y_val, after)})
calibration_table = pd.DataFrame(calibration_rows)
display(calibration_table.round(5))
plt.figure(figsize=(7, 5))
for label in searches:
    key = label + ' / tuned'
    for stage, probabilities in [('raw', fitted[key].predict_proba(X_val)[:, 1]),
                                  ('calibrated', val_probabilities[key])]:
        observed, predicted = calibration_curve(y_val, probabilities, n_bins=8, strategy='quantile')
        plt.plot(predicted, observed, marker='o', label=label + ' ' + stage)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Средняя предсказанная вероятность'); plt.ylabel('Доля возвратов')
plt.title('Калибровка на validation'); plt.legend(fontsize=8)
show_plot('calibration_validation')
''')
md('''
## 14. Выбор порога и лучшей модели — только на validation

Порог 0.5 не является универсальным. При редком классе и откалиброванных
вероятностях он может дать почти нулевой recall. Для каждой обучаемой модели
выбираем порог максимального F1 на validation. Dummy оставляем с порогом 0.5.
Лучшую модель выбираем по validation AP, при равенстве — по меньшему Brier.
Начальный вариант тоже может победить: подбор не должен принудительно вытеснять его.

Дополнительно показываем цену порога: число предупреждений на 100 случаев,
precision и recall. Можно выбрать иную политику, но **до итогового test**.
Здесь выбор F1 учебный: клинические стоимости ошибок и допустимая нагрузка не заданы.
''')
code('''
thresholds, validation_rows = {}, []
for label, probabilities in val_probabilities.items():
    threshold = 0.5 if label == 'Dummy' else choose_threshold(y_val, probabilities)
    thresholds[label] = threshold
    validation_rows.append({'model': label, **evaluate(y_val, probabilities, threshold)})
validation_metrics = pd.DataFrame(validation_rows)
display(validation_metrics.sort_values('pr_auc_ap', ascending=False).round(5))
winner = (validation_metrics.query("model != 'Dummy'")
          .sort_values(['pr_auc_ap', 'brier'], ascending=[False, True]).iloc[0]['model'])
chosen_threshold = thresholds[winner]
selected_model = calibrated_models[winner]
selection = {'run_id': RUN_ID, 'winner': winner, 'threshold': chosen_threshold,
             'criterion': 'validation AP, tie-break lower Brier',
             'test_not_used_for_selection': True, 'source_sha256': source_info['archive_sha256'],
             'search_budget': N_ITER, 'versions': versions}
(OUT / 'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
print('Решение зафиксировано ДО test:', winner, '; порог:', chosen_threshold)

p_val = val_probabilities[winner]
threshold_table = pd.DataFrame([
    {'threshold': value, **evaluate(y_val, p_val, value),
     'alerts_per_100': 100*np.mean(p_val >= value)}
    for value in sorted(set([0.05, 0.10, 0.15, 0.20, 0.50, chosen_threshold]))])
display(threshold_table[['threshold', 'precision', 'recall', 'f1', 'alerts_per_100']].round(4))
precision, recall, cutoffs = precision_recall_curve(y_val, p_val)
f1_values = 2*precision[:-1]*recall[:-1] / np.maximum(precision[:-1]+recall[:-1], 1e-15)
plt.figure(figsize=(8, 4))
for values, label in [(precision[:-1], 'Precision'), (recall[:-1], 'Recall'), (f1_values, 'F1')]:
    plt.plot(cutoffs, values, label=label)
plt.axvline(chosen_threshold, color='black', linestyle='--', label='Выбранный порог')
plt.xlabel('Порог вероятности'); plt.ylabel('Метрика'); plt.legend()
plt.xlim(0, min(0.8, float(np.quantile(p_val, .995)) + .05))
show_plot('threshold_validation')
''')
md('''
## 15. Итоговая оценка на test

Теперь все Pipeline, калибраторы, модель-победитель и пороги уже зафиксированы.
Впервые в этом запуске получаем прогнозы test для сравнения фиксированных вариантов.
Не меняем победителя по этой таблице и не подбираем порог повторно.
Validation-метрики были использованы для выбора и потому оптимистичнее как основание
для заявления о качестве; test показывает перенос на отложенных пациентов этого набора.
Оценка относится к госпитализациям: пациенты с несколькими визитами имеют больший вес.
''')
code('''
X_test, y_test = test[INPUT_COLS], test[TARGET]
test_rows, test_probabilities = [], {}
for label, model in calibrated_models.items():
    started = time.perf_counter()
    probabilities = model.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter()-started
    test_probabilities[label] = probabilities
    test_rows.append({'model': label, **evaluate(y_test, probabilities, thresholds[label]),
                      'selected_on_validation': label == winner,
                      'predict_ms_per_1000': elapsed/len(test)*1e6})
test_metrics = pd.DataFrame(test_rows)
display(test_metrics.round(5))
for filename, table in [('initial_validation', initial_metrics), ('tuning_changes', improvements),
                         ('calibration_validation', calibration_table), ('validation_metrics', validation_metrics),
                         ('thresholds_validation', threshold_table), ('test_metrics', test_metrics)]:
    table.to_csv(OUT / (filename + '.csv'), index=False)
''')
md('''
## 16. ROC, precision–recall и матрица ошибок

ROC показывает компромисс между обнаруженными возвратами и ложноположительной
долей. PR-кривая при редком классе нагляднее показывает качество предупреждений;
горизонтальная линия — доля единиц test. Матрица ошибок использует **выбранный
ранее** порог: FN — пропущенный возврат, FP — лишнее предупреждение.

Полезно считать не только проценты: например, recall 50% означает, что половина
фактических возвратов осталась не обнаружена. Это не клинически валидированный инструмент.
''')
code('''
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for label, probabilities in test_probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    axes[0].plot(fpr, tpr, label=label)
    axes[1].step(recall, precision, where='post', label=label)
axes[0].plot([0,1], [0,1], 'k--')
axes[1].axhline(y_test.mean(), color='gray', linestyle='--', label='Доля единиц test')
axes[0].set(xlabel='False positive rate', ylabel='Recall', title='ROC на test')
axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall на test')
for ax in axes:
    ax.legend(fontsize=7)
show_plot('test_curves')

p_test = test_probabilities[winner]
predicted = (p_test >= chosen_threshold).astype(int)
matrix = confusion_matrix(y_test, predicted, labels=[0,1])
plt.figure(figsize=(6,4))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Прогноз 0', 'Прогноз 1'], yticklabels=['Факт 0', 'Факт 1'])
plt.title(winner + ': матрица ошибок test')
show_plot('confusion_test')
display(pd.DataFrame(classification_report(y_test, predicted, output_dict=True, zero_division=0)).T)
tn, fp, fn, tp = matrix.ravel()
display(pd.Series({'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
                   'пропущено возвратов': fn, 'лишних предупреждений': fp}))
''')
md('''
## 17. Неопределённость: bootstrap по пациентам

Визиты одного пациента зависимы. Поэтому bootstrap перевыбирает **пациентов**,
а все их визиты получают общий вес. Показываем 95% percentile-интервалы ROC-AUC,
AP и F1 при фиксированной модели и пороге. Они не включают всю неопределённость
подбора гиперпараметров, не являются интервалами индивидуального риска и не
доказывают качество на другой больнице. 200 повторов — учебный компромисс;
для более стабильных границ увеличьте число повторов до нового итогового запуска.
''')
code('''
intervals = group_bootstrap(y_test, p_test, test.patient_nbr,
                            chosen_threshold, n_bootstrap=BOOTSTRAP_REPEATS)
display(pd.Series(intervals, name='95% bootstrap').round(5))
(OUT / 'bootstrap_intervals.json').write_text(json.dumps(intervals, indent=2), encoding='utf-8')
''')
md('''
## 18. Почему модель предсказывает: permutation importance и коэффициенты

Перемешиваем один исходный признак на validation и смотрим, насколько падает AP.
Большое падение означает, что модель использовала информацию этого признака.
Используем исходные поля и весь Pipeline: зависимые производные признаки
пересчитываются после перемешивания. Часть validation берётся для экономии времени,
а не для обучения. Важности описательные, не причинные; коррелирующие поля могут
заменять друг друга. Отрицательная важность может возникнуть из-за шума.

Для LR дополнительно выводим коэффициенты уже преобразованных признаков.
Коэффициент числа относится к изменению на одно обучающее стандартное отклонение;
интерпретация категорий зависит от кодирования и регуляризации. Это не эффект лечения.
''')
code('''
sample = validation.sample(n=min(1000, len(validation)), random_state=SEED)
importance = permutation_importance(selected_model, sample[INPUT_COLS], sample[TARGET],
                                    scoring='average_precision', n_repeats=3,
                                    random_state=SEED, n_jobs=1)
importance_table = pd.DataFrame({'feature': INPUT_COLS,
    'importance_mean': importance.importances_mean,
    'importance_std': importance.importances_std}).sort_values('importance_mean', ascending=False)
display(importance_table.head(20))
view = importance_table.head(15).sort_values('importance_mean')
plt.figure(figsize=(9,6))
plt.barh(view.feature, view.importance_mean, xerr=view.importance_std)
plt.xlabel('Падение validation AP'); plt.title('Permutation importance выбранной модели')
show_plot('importance_validation')
importance_table.to_csv(OUT / 'permutation_importance.csv', index=False)

lr = fitted['LogisticRegression / tuned']
names = lr.named_steps['preprocess'].get_feature_names_out()
coefficient_table = pd.DataFrame({'feature': names, 'coefficient': lr.named_steps['model'].coef_[0]})
display(coefficient_table.assign(abs_value=lambda frame: frame.coefficient.abs())
        .sort_values('abs_value', ascending=False).head(20))
''')
md('''
## 19. Разбор ошибок и возрастных / половых групп

Сравниваем FN, FP, TP и TN по прошлым госпитализациям и длительности пребывания.
Затем описательно проверяем группы: число записей/пациентов, частоту исхода,
precision и recall. Не объявляем малые группы надёжными выводами о различиях,
не делаем причинных или дискриминационных заключений. Этот анализ test не служит
новым подбором модели. Если после него изменить решение, понадобится другая
независимая проверка качества.
''')
code('''
errors = test[['patient_nbr', 'age', 'gender', 'number_inpatient', 'time_in_hospital', TARGET]].copy()
errors['prediction'] = predicted
errors['probability'] = p_test
errors['error_type'] = np.select([
    errors[TARGET].eq(1) & errors.prediction.eq(0),
    errors[TARGET].eq(0) & errors.prediction.eq(1),
    errors[TARGET].eq(1) & errors.prediction.eq(1)], ['FN','FP','TP'], default='TN')
error_profiles = errors.groupby('error_type').agg(
    encounters=(TARGET, 'size'), patients=('patient_nbr','nunique'),
    mean_prior_inpatient=('number_inpatient','mean'),
    mean_stay=('time_in_hospital','mean'), mean_probability=('probability','mean'))
display(error_profiles.round(3))
subgroups = []
for field in ['age', 'gender']:
    for group, frame in errors.groupby(field, dropna=False):
        positives = int(frame[TARGET].sum())
        metrics = evaluate(frame[TARGET], frame.probability, chosen_threshold) if frame[TARGET].nunique()==2 else {}
        subgroups.append({'field': field, 'group': str(group), 'n': len(frame),
                          'patients': frame.patient_nbr.nunique(), 'positives': positives,
                          'precision': metrics.get('precision', np.nan),
                          'recall': metrics.get('recall', np.nan),
                          'note': 'малая группа: осторожная интерпретация' if len(frame)<200 or positives<20 else 'описательная оценка'})
subgroup_table = pd.DataFrame(subgroups)
display(subgroup_table)
error_profiles.to_csv(OUT / 'error_profiles.csv')
subgroup_table.to_csv(OUT / 'subgroups.csv', index=False)
''')
md('''
## 20. Сохранение нового Pipeline и воспроизводимый прогноз

Сохраняется вся цепочка, включая обработку категорий, стандартизацию и калибратор.
При загрузке не нужно вручную повторять one-hot и подгонять новый scaler.
Проверяем совпадение вероятностей до и после сохранения на небольшом примере.
Файл отдельный: действующая `artifacts/best_model.joblib` не заменяется.
Версии Python/библиотек и SHA256 источника записаны в selection.json.
Не загружайте pickle/joblib из неизвестных источников.
''')
code('''
model_path = MODEL_OUT / 'detailed_model.joblib'
bundle = {'model': selected_model, 'threshold': chosen_threshold, 'name': winner,
          'run_id': RUN_ID, 'input_columns': INPUT_COLS, 'selection': selection}
joblib.dump(bundle, model_path, compress=3)
loaded = joblib.load(model_path)
example = X_fit.head(3)
before = selected_model.predict_proba(example)[:,1]
after = loaded['model'].predict_proba(example)[:,1]
np.testing.assert_allclose(before, after, rtol=0, atol=1e-12)
display(pd.DataFrame({'probability': after, 'prediction': (after >= loaded['threshold']).astype(int)}))
print('Сохранён:', model_path.relative_to(ROOT))
print('Размер, МБ:', model_path.stat().st_size/1024**2)
''')
md('''
## 21. Итог: что получилось в этом запуске

Следующая ячейка собирает выводы из реально рассчитанных значений. Подбор,
калибровка и порог — разные этапы: улучшение одной метрики не означает улучшения
всех. Нет обещания ROC-AUC 0.9 или клинической пригодности на историческом наборе.
''')
code(r'''
selected = test_metrics.set_index('model').loc[winner]
cal_row = calibration_table.set_index('model').loc[winner]
lines = [f'**Выбрана на validation:** {winner}. Порог: {chosen_threshold:.4f}.',
         f'**Test:** ROC-AUC {selected.roc_auc:.4f}; AP {selected.pr_auc_ap:.4f}; '
         f'precision {selected.precision:.4f}; recall {selected.recall:.4f}; F1 {selected.f1:.4f}.',
         f'**Вероятности на validation:** Brier до {cal_row.Brier_before:.5f}, после {cal_row.Brier_after:.5f}.',
         f'**Ошибки test:** пропущено {fn} возвратов; ложных предупреждений {fp}; обнаружено {tp}.']
for row in improvements.itertuples():
    lines.append(f'**{row.family}:** изменение validation AP после подбора {row.delta_AP:+.5f}.')
lines += ['**Ограничения:** историческая выборка, нет временной/межбольничной проверки; '
          'стоимость ошибок не задана; test ранее изучался в проекте. Учебный эксперимент, не медицинское решение.',
          '**Результаты:** reports/model_training_detailed/. Новая модель: artifacts/notebook_training/detailed_model.joblib.']
display(Markdown('\n\n'.join(lines)))
(OUT / 'summary.md').write_text('\n\n'.join(lines), encoding='utf-8')
assert all(set(a.patient_nbr).isdisjoint(b.patient_nbr)
           for name_a, a in parts.items() for name_b, b in parts.items() if name_a != name_b)
assert all(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() for p in test_probabilities.values())
print('Контроль пройден: группы независимы, вероятности корректны, сохранение воспроизводится.')
''')
md('''
## Источники и вопросы для защиты

- [Pipeline / ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html).
- [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- [Калибровка вероятностей](https://scikit-learn.org/stable/modules/calibration.html).
- [Average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- [Данные UCI](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

Попробуйте объяснить: почему patient_nbr не признак, но нужен для разбиения;
чем category отличается от one-hot; почему scaler нельзя обучать на test;
чем параметр отличается от гиперпараметра; почему высокая accuracy недостаточна;
почему калибровка не обязана повышать AP; почему выбор другого порога меняет
precision/recall; почему после анализа test нельзя подбирать модель по его результатам.
''')

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
notebook.metadata.language_info = {'name': 'python', 'version': '3.11'}
target = ROOT / 'Model_training_detailed.ipynb'
nbf.write(notebook, target)
print(f'Created {target.name}: {len(cells)} cells')
