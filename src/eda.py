"""Full audit of 50 source columns, train-only associations and saved figures."""

import json
import textwrap
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from .config import *
from .features import icd9_group

sns.set_theme(style='whitegrid', palette=['#147D92', '#F4A340', '#6C7A89', '#B6D8CE'])
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 160, 'font.family': 'DejaVu Sans'})


def save_figure(name):
    """Сохранить текущий рисунок в reports/figures и закрыть его."""
    plt.tight_layout()
    plt.savefig(FIGURES / f'{name}.png', bbox_inches='tight')
    plt.close()


def category_report(frame, column):
    """Таблица по категориям признака: число визитов, пациентов и доля возвратов."""
    temp = frame.assign(category=frame[column].astype('string').fillna('Missing'))
    report = temp.groupby('category', observed=True).agg(
        encounters=(TARGET, 'size'), patients=('patient_nbr', 'nunique'), positives=(TARGET, 'sum')
    )
    report['share_pct'] = 100 * report.encounters / len(frame)
    report['readmission_pct'] = 100 * report.positives / report.encounters
    return report.sort_values('encounters', ascending=False)


def plot_category(frame, column, order=None):
    """Столбчатые графики: размер категорий и доля возвратов в каждой."""
    report = category_report(frame, column)
    report.to_csv(TABLES / f'feature_{column}.csv')
    view = report.reindex(order).dropna(subset=['encounters']) if order else report.head(12)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    labels = ['\n'.join(textwrap.wrap(str(x), 22)) for x in view.index]
    positions = np.arange(len(view))
    axes[0].bar(positions, view.encounters, color='#147D92')
    axes[1].bar(positions, view.readmission_pct.where(view.encounters >= 100), color='#F4A340')
    axes[1].axhline(
        frame[TARGET].mean() * 100, linestyle='--', color='#283E51', label='Fit-set average'
    )
    for ax in axes:
        ax.set_xticks(positions, labels, rotation=55, ha='right', fontsize=8)
    axes[0].set(title=f'{column}: encounters', ylabel='Encounters')
    axes[1].set(title='Readmission <30 days (groups n >= 100)', ylabel='Readmission, %')
    axes[1].legend(fontsize=8)
    fig.suptitle('Fit partition only; top 12 categories where applicable', fontsize=10)
    save_figure(f'category_{column}')
    return report


def draw_er_diagram():
    """Нарисовать ER-диаграмму базы данных."""
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.set(xlim=(0, 14), ylim=(-0.2, 8.5))
    ax.axis('off')
    positions = {
        'patients': (0.3, 5.6, 2.6, 1.25, 'PK patient_nbr'),
        'encounters': (
            4.3,
            4.2,
            3.4,
            2.35,
            'PK encounter_id\nFK patient_nbr\nFK admission_type_id\nFK discharge_disposition_id\nFK admission_source_id',
        ),
        'admission_types': (0.3, 3.55, 2.6, 1.2, 'PK code\ndescription'),
        'discharge_dispositions': (0.3, 1.65, 2.9, 1.2, 'PK code\ndescription'),
        'admission_sources': (4.3, 0.3, 3.4, 1.2, 'PK code\ndescription'),
        'diagnoses': (9.6, 6.55, 3.7, 1.5, 'PK FK encounter_id + position\nicd9, diagnosis_group'),
        'encounter_medications': (
            9.6,
            4.1,
            3.7,
            1.6,
            'PK FK encounter_id + drug_id\nstatus (non-No only)',
        ),
        'drugs': (9.6, 1.9, 3.7, 1.2, 'PK drug_id\nname'),
        'split_assignments': (4.3, 6.85, 3.4, 1.1, 'PK FK encounter_id\nsplit'),
    }
    for title, (x, y, w, h, body) in positions.items():
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle='round,pad=0.07', facecolor='#EFF6F7', edgecolor='#147D92'
            )
        )
        ax.text(x + 0.12, y + h - 0.25, title, fontsize=10, fontweight='bold', va='top')
        ax.text(x + 0.12, y + h - 0.63, body, fontsize=9, va='top', linespacing=1.5)
    for start, end, label in [
        ((2.9, 6.1), (4.3, 5.7), '1 : N'),
        ((2.9, 4.1), (4.3, 4.8), '1 : N'),
        ((3.2, 2.2), (4.5, 4.2), '1 : N'),
        ((6, 1.5), (6, 4.2), '1 : N'),
        ((7.7, 6), (9.6, 7.1), '1 : N'),
        ((7.7, 5), (9.6, 4.9), '1 : N'),
        ((11.5, 3.1), (11.5, 4.1), '1 : N'),
        ((6, 6.55), (6, 6.85), '1 : 1'),
    ]:
        ax.annotate(
            '', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='#64748B', lw=1.5)
        )
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.08,
            label,
            fontsize=9,
            color='#334155',
        )
    ax.text(0.3, 8.1, 'SQLite relational schema', fontsize=22, fontweight='bold', color='#183547')
    ax.text(
        0.3,
        0.03,
        'patients contains identifiers only; age and other encounter-varying attributes stay in encounters',
        fontsize=9,
    )
    save_figure('er_diagram')


def run_eda(raw, mappings, parts):
    """Выполнить весь исследовательский анализ: таблицы и графики в reports/."""
    ensure_directories()
    fit = parts['fit'].copy()
    profile = pd.DataFrame(
        {
            'dtype': raw.dtypes.astype(str),
            'missing_count': raw.isna().sum(),
            'missing_pct': 100 * raw.isna().mean(),
            'unique': raw.nunique(),
        }
    )
    profile['dominant_pct'] = [
        raw[c].value_counts(dropna=False, normalize=True).iloc[0] * 100 for c in raw
    ]
    profile.to_csv(TABLES / 'source_column_audit.csv')
    raw.readmitted.value_counts().rename_axis('readmitted').to_frame('encounters').to_csv(
        TABLES / 'raw_target.csv'
    )
    raw.readmitted.value_counts().reindex(['NO', '>30', '<30']).plot.bar(
        figsize=(7, 4), color=['#147D92', '#6C7A89', '#F4A340']
    )
    plt.ylabel('Encounters')
    plt.title('Source target distribution')
    plt.xticks(rotation=0)
    save_figure('01_target_balance')
    patient_counts = raw.groupby('patient_nbr').size()
    patient_counts.value_counts().sort_index().rename_axis('encounters_per_patient').to_csv(
        TABLES / 'patient_recurrence.csv'
    )
    patient_counts.value_counts().sort_index().plot.bar(figsize=(11, 4), color='#147D92')
    plt.xlabel('Encounters per patient')
    plt.ylabel('Patients')
    plt.title('Repeated patients in source data')
    save_figure('02_patient_recurrence')
    missing = fit.drop(columns=ID_COLS + ['readmitted', TARGET]).isna()
    for column, codes in SPECIAL_MISSING.items():
        missing[column] |= fit[column].isin(codes)
    mt = pd.DataFrame({'count': missing.sum(), 'percent': missing.mean() * 100}).sort_values(
        'percent', ascending=False
    )
    mt.to_csv(TABLES / 'fit_missingness.csv')
    mt.query('count>0')['percent'].sort_values().plot.barh(figsize=(9, 5), color='#147D92')
    plt.xlabel('Unknown values, %')
    plt.title('Formal and coded missing values on fit')
    save_figure('03_missing_values')
    by_target = missing.groupby(fit[TARGET]).mean().T * 100
    by_target.to_csv(TABLES / 'missingness_by_target.csv')
    varying = missing.columns[(missing.mean() > 0) & (missing.mean() < 1)]
    plt.figure(figsize=(10, 7))
    sns.heatmap(missing[varying].corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Co-occurrence of unknown values on fit')
    save_figure('04_missingness_correlation')
    num = fit[NUM_COLS]
    desc = num.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc['zero_pct'] = num.eq(0).mean() * 100
    desc.to_csv(TABLES / 'numeric_descriptive.csv')
    fit.groupby(TARGET)[NUM_COLS].agg(['mean', 'median', 'std']).T.to_csv(
        TABLES / 'numeric_by_target.csv'
    )
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    outside = num.lt(q1 - 1.5 * iqr) | num.gt(q3 + 1.5 * iqr)
    outliers = pd.DataFrame(
        {
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'outside_pct': outside.mean() * 100,
            'maximum': num.max(),
            'note': np.where(iqr.eq(0), 'IQR=0; do not auto-delete', 'Diagnostic only'),
        }
    )
    outliers.to_csv(TABLES / 'outlier_diagnostics.csv')
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    for c, ax in zip(NUM_COLS, axes.ravel()):
        sns.histplot(data=fit, x=c, bins=30, ax=ax, color='#147D92')
        ax.set_title(c)
    save_figure('05_numeric_distributions')
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    for c, ax in zip(NUM_COLS, axes.ravel()):
        sns.boxplot(data=fit, x=TARGET, y=c, showfliers=False, ax=ax)
        ax.set_title(c)
    save_figure('06_numeric_by_target')
    corr = fit[NUM_COLS].corr(method='spearman')
    corr.to_csv(TABLES / 'spearman.csv')
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Spearman correlations on fit')
    save_figure('07_spearman')
    fit[NUM_COLS + [TARGET]].corr(method='spearman')[TARGET].drop(TARGET).sort_values().to_csv(
        TABLES / 'target_spearman.csv'
    )
    for c in [
        'race',
        'gender',
        'age',
        'weight',
        'payer_code',
        'medical_specialty',
        'A1Cresult',
        'max_glu_serum',
        'change',
        'diabetesMed',
    ]:
        plot_category(fit, c, AGE_ORDER if c == 'age' else None)
    for c in CODE_COLS:
        report = category_report(fit, c)
        report['description'] = [mappings[c].get(str(x), 'Unmapped') for x in report.index]
        report.to_csv(TABLES / f'feature_{c}.csv')
        fit[c + '_label'] = fit[c].map(lambda x: f'{x}: {mappings[c].get(str(x),"Unknown")}')
        plot_category(fit, c + '_label')
    for c in DIAG_COLS:
        fit[c].fillna('Missing').value_counts().to_csv(TABLES / f'raw_{c}_counts.csv')
        fit[c + '_group'] = fit[c].map(icd9_group)
        plot_category(fit, c + '_group')
    medication = (
        pd.DataFrame({c: fit[c].fillna('Missing').value_counts() for c in DRUG_COLS})
        .T.fillna(0)
        .astype(float)
    )
    medication = medication.reindex(columns=['No', 'Steady', 'Up', 'Down', 'Missing'], fill_value=0)
    medication.to_csv(TABLES / 'medication_counts.csv')
    pct = medication / len(fit) * 100
    pct.to_csv(TABLES / 'medication_percent.csv')
    plt.figure(figsize=(9, 11))
    sns.heatmap(pct, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Medication states on fit, %')
    save_figure('08_medication_states')
    pd.concat({c: category_report(fit, c) for c in DRUG_COLS}, names=['drug', 'status']).to_csv(
        TABLES / 'medication_target_rates.csv'
    )
    for c in ['number_outpatient', 'number_emergency', 'number_inpatient']:
        fit[c + '_band'] = pd.cut(
            fit[c], [-np.inf, 0, 1, 2, np.inf], labels=['0', '1', '2', '3+']
        ).astype('string')
        plot_category(fit, c + '_band', ['0', '1', '2', '3+'])
    constants = profile.index[profile['unique'].eq(1)].tolist()
    facts = {
        'raw_rows': len(raw),
        'columns': len(raw.columns),
        'patients': raw.patient_nbr.nunique(),
        'duplicates': int(raw.duplicated().sum()),
        'duplicate_encounter_ids': int(raw.encounter_id.duplicated().sum()),
        'repeated_patients': int(patient_counts.gt(1).sum()),
        'max_encounters_per_patient': int(patient_counts.max()),
        'raw_positive_count': int(raw.readmitted.eq('<30').sum()),
        'raw_positive_rate': float(raw.readmitted.eq('<30').mean()),
        'source_constants': constants,
        'fit_rows': len(fit),
        'fit_positive_rate': float(fit[TARGET].mean()),
        'fit_out_of_range_length_of_stay': int((~fit.time_in_hospital.between(1, 14)).sum()),
    }
    (REPORTS / 'eda_facts.json').write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    draw_er_diagram()
    write_feature_conclusions(raw, profile)
    return facts


def write_feature_conclusions(raw, profile):
    """Записать текстовые выводы по каждому исходному столбцу."""
    numeric = pd.read_csv(TABLES / 'numeric_descriptive.csv', index_col=0)
    outliers = pd.read_csv(TABLES / 'outlier_diagnostics.csv', index_col=0)
    drug_rates = pd.read_csv(TABLES / 'medication_target_rates.csv')
    lines = [
        '# Подробные выводы EDA по всем исходным столбцам',
        '',
        'Единица наблюдения — госпитализация. Общий аудит относится к исходным данным; связи с целью рассчитаны только на fit.',
        'Частоты не являются доказательством причинности. Исходные файлы и исходные записи не исправлялись автоматически.',
        '',
    ]
    for c in raw.columns:
        p = profile.loc[c]
        lines += [
            f'## {c}',
            f'- Тип: {p["dtype"]}; уникальных значений: {int(p["unique"])}.',
            f'- Пропуски: {int(p["missing_count"]):,} ({p["missing_pct"]:.2f}%).',
            f'- Доля наиболее частого значения: {p["dominant_pct"]:.2f}%.',
        ]
        if c in ID_COLS:
            decision = 'Технический идентификатор, исключён из входов модели.'
            if c == 'patient_nbr':
                decision += ' Используется для всех разбиений и групповой кросс-валидации; повторы не удаляются.'
        elif c == 'readmitted':
            decision = 'Цель: <30 → 1, >30 и NO → 0. Исходный столбец запрещён во входах модели.'
        elif c == 'weight':
            decision = '96,86% пропусков. Исходные интервалы исключены из первой модели, используется индикатор наличия записи. Строки не удаляются.'
        elif c in ['examide', 'citoglipton']:
            decision = 'Постоянный в источнике столбец, не несёт различий между наблюдениями; исключён по фиксированной спецификации.'
        elif c in DIAG_COLS:
            decision = 'Коды ICD-9 сохраняются строками и внутри Pipeline объединяются в укрупнённые группы. V/E относятся к Other; пропуски отдельны.'
        elif c in CODE_COLS:
            decision = 'Номинальный код со справочником IDS_mapping.csv. Не интерпретировать как число. Неизвестные коды нормализуются в Pipeline.'
        elif c in ['A1Cresult', 'max_glu_serum']:
            decision = 'None — отдельное состояние отсутствия проведённого анализа, не обычный NaN. Категория сохраняется, добавляется индикатор измерения.'
        elif c in NUM_COLS:
            decision = 'Количественный признак. Проверены квартили, нули, хвосты, IQR и связь с целью на fit. Экстремальные значения не удаляются автоматически. Медианная импутация и масштабирование обучаются внутри CV.'
        elif c in DRUG_COLS:
            decision = 'Категории No/Steady/Up/Down. Частоты и доли <30 представлены в medication_target_rates.csv. Редкость не доказывает бесполезность; различия не являются эффектом лекарства.'
        else:
            decision = 'Категориальный признак. Импутация отдельной категорией и One-Hot Encoding с объединением редких уровней обучаются только на тренировочных фолдах.'
        if c in NUM_COLS:
            n = numeric.loc[c]
            o = outliers.loc[c]
            lines += [
                f'- На fit: среднее {n["mean"]:.2f}, медиана {n["50%"]:.2f}, P99 {n["99%"]:.2f}, максимум {n["max"]:.0f}.',
                f'- Нулей {n["zero_pct"]:.2f}%; IQR {o["iqr"]:.2f}; за диагностическими границами IQR {o["outside_pct"]:.2f}%.',
            ]
        else:
            path = TABLES / f'feature_{c+"_group" if c in DIAG_COLS else c}.csv'
            if c in DRUG_COLS:
                rate_table = drug_rates[drug_rates['drug'].eq(c)].rename(
                    columns={'status': 'category'}
                )
            elif path.exists():
                rate_table = pd.read_csv(path)
            else:
                rate_table = pd.DataFrame()
            if len(rate_table):
                first = rate_table.sort_values('encounters', ascending=False).iloc[0]
                lines += [
                    f'- На fit наиболее многочисленная категория {first["category"]}: {int(first["encounters"]):,} госпитализаций; доля <30 внутри неё {first["readmission_pct"]:.2f}%.'
                ]
                stable = rate_table[rate_table['encounters'].ge(100)]
                if len(stable) > 1:
                    low = stable.sort_values('readmission_pct').iloc[0]
                    high = stable.sort_values('readmission_pct').iloc[-1]
                    lines += [
                        f'- Среди категорий с n ≥ 100 наблюдаемые доли <30 на fit: от {low["readmission_pct"]:.2f}% ({low["category"]}) до {high["readmission_pct"]:.2f}% ({high["category"]}). Это описательное сравнение без утверждения значимости или причинности.'
                    ]
        lines += [f'**Решение:** {decision}', '']
    (REPORTS / 'EDA_detailed_conclusions.md').write_text('\n'.join(lines), encoding='utf-8')
