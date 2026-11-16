"""Исследовательский анализ данных (EDA): статистика, выбросы и графики (Часть 3)."""
import matplotlib

matplotlib.use("Agg")  # рисуем сразу в файл, окно с графиком не нужно
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from src import config, preprocessing

sns.set_theme(style="whitegrid")

MAIN_COLOR = "#2a78d6"
ACCENT_COLOR = "#d6492a"     # выделяем то, что важно для задачи (возврат < 30 дней)
BOXPLOT_COLUMNS = ["time_in_hospital", "num_lab_procedures", "num_medications", "number_diagnoses"]


def _save(fig, filename) -> str:
    """Сохранить рисунок в папку графиков и закрыть его (иначе копится память)."""
    config.ensure_dirs()
    path = config.FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def _numeric_columns(df) -> list:
    """Числовые признаки, которые есть в таблице."""
    return preprocessing.get_feature_lists(df)[0]


def describe_numeric(df) -> pd.DataFrame:
    """Описательная статистика числовых признаков (среднее, медиана, квартили...)."""
    summary = df[_numeric_columns(df)].describe().T.round(2)
    summary.to_csv(config.REPORTS_DIR / "numeric_summary.csv", index_label="feature")
    return summary


def iqr_outliers(df) -> pd.DataFrame:
    """Посчитать выбросы по правилу IQR: всё за пределами Q1 - 1.5*IQR и Q3 + 1.5*IQR."""
    # Выбросы считаем, но НЕ удаляем: 10 госпитализаций за год — не ошибка ввода,
    # а редкий, но реальный пациент. Именно такие чаще всего и возвращаются.
    rows = []
    for col in _numeric_columns(df):
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({
            "feature": col, "q1": q1, "q3": q3, "lower_bound": low, "upper_bound": high,
            "outliers": count, "outliers_pct": round(count / len(df) * 100, 2),
        })
    table = pd.DataFrame(rows)
    table.to_csv(config.REPORTS_DIR / "iqr_outliers.csv", index=False)
    return table


def plot_target(df) -> str:
    """График 1: доли исходов NO / >30 / <30."""
    share = df["readmitted"].astype(str).value_counts(normalize=True).reindex(["NO", ">30", "<30"]) * 100
    labels = ["Не вернулся", "Вернулся позже 30 дней", "Вернулся раньше 30 дней"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, share.values, color=[MAIN_COLOR, MAIN_COLOR, ACCENT_COLOR])
    ax.bar_label(bars, fmt="%.1f%%")
    ax.set_xlabel("Исход после выписки")
    ax.set_ylabel("Доля визитов, %")
    ax.set_title("Распределение целевой переменной: классы несбалансированы")
    return _save(fig, "01_target_distribution.png")


def plot_rate_by_inpatient(df) -> str:
    """График 2: процент возвратов по числу госпитализаций за прошлый год."""
    visits = df["number_inpatient"].clip(upper=6)          # 6 и больше — одна группа «6+»
    rate = df.groupby(visits)[preprocessing.TARGET].mean() * 100
    labels = [str(value) if value < 6 else "6+" for value in rate.index]
    average = df[preprocessing.TARGET].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, rate.values, color=MAIN_COLOR)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
    ax.axhline(average, color=ACCENT_COLOR, linestyle="--", label=f"Среднее: {average:.1f}%")
    ax.set_xlabel("Госпитализаций за прошлый год")
    ax.set_ylabel("Вернулись раньше 30 дней, %")
    ax.set_title("Чем больше прошлых госпитализаций, тем выше риск возврата")
    ax.legend()
    return _save(fig, "02_rate_by_prior_inpatient.png")


def plot_rate_by_diagnosis(df) -> str:
    """График 3: процент возвратов по группе основного диагноза."""
    stats = df.groupby("diag_1_group", observed=True)[preprocessing.TARGET].agg(["mean", "count"])
    stats = stats.sort_values("mean")
    average = df[preprocessing.TARGET].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(stats.index.astype(str), stats["mean"] * 100, color=MAIN_COLOR)
    ax.bar_label(bars, labels=[f"{m * 100:.1f}%  (n={n})" for m, n in zip(stats["mean"], stats["count"])],
                 padding=3, fontsize=9)
    ax.axvline(average, color=ACCENT_COLOR, linestyle="--", label=f"Среднее: {average:.1f}%")
    ax.set_xlim(0, stats["mean"].max() * 100 * 1.35)
    ax.set_xlabel("Вернулись раньше 30 дней, %")
    ax.set_ylabel("Группа основного диагноза (ICD-9)")
    ax.set_title("Доля возвратов по группе основного диагноза")
    ax.legend(loc="lower right")
    return _save(fig, "03_rate_by_primary_diagnosis.png")


def plot_boxplots(df) -> str:
    """График 4: boxplot четырёх числовых признаков отдельно для вернувшихся и нет."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    for ax, col in zip(axes, BOXPLOT_COLUMNS):
        sns.boxplot(data=df, x=preprocessing.TARGET, y=col, hue=preprocessing.TARGET,
                    palette=[MAIN_COLOR, ACCENT_COLOR], legend=False, ax=ax)
        ax.set_xlabel("Вернулся < 30 дней (0 = нет, 1 = да)")
        ax.set_ylabel(col)
    fig.suptitle("Распределение числовых признаков у вернувшихся и не вернувшихся")
    return _save(fig, "04_boxplots_by_target.png")


def plot_correlation(df) -> str:
    """График 5: корреляция Спирмена между числовыми признаками и ответом."""
    # Спирмен, а не Пирсон: счётчики визитов сильно скошены, а Спирмен работает
    # с рангами и не чувствителен к выбросам.
    corr = df[_numeric_columns(df) + [preprocessing.TARGET]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot_kws={"size": 8}, ax=ax)
    ax.set_xlabel("Признак")
    ax.set_ylabel("Признак")
    ax.set_title("Корреляция Спирмена: числовые признаки и readmit_30")
    return _save(fig, "05_correlation_heatmap.png")


def plot_visits_per_patient(df) -> str:
    """График 6: сколько визитов приходится на одного пациента."""
    visits = df[preprocessing.GROUP].value_counts().clip(upper=10)   # 10 = «10 и больше»
    counts = visits.value_counts().sort_index()
    labels = [str(value) if value < 10 else "10+" for value in counts.index]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts.values, color=MAIN_COLOR)
    ax.bar_label(bars, fontsize=8)
    ax.set_yscale("log")   # иначе столбец «1 визит» делает остальные невидимыми
    ax.set_xlabel("Число визитов одного пациента")
    ax.set_ylabel("Число пациентов (логарифмическая шкала)")
    ax.set_title("У многих пациентов несколько визитов:\nделить на train/test нужно по пациентам")
    return _save(fig, "06_visits_per_patient.png")


def plot_interactive(df) -> str:
    """График 7 (интерактивный, Plotly): процент возвратов по возрасту для каждой группы диагноза."""
    stats = (df.groupby(["age", "diag_1_group"], observed=True)[preprocessing.TARGET]
               .agg(readmit_pct="mean", visits="count").reset_index())
    stats = stats[stats["visits"] >= 100]          # на малых группах процент слишком шумный
    stats["readmit_pct"] = (stats["readmit_pct"] * 100).round(1)
    stats["age"] = stats["age"].astype(str)
    fig = px.line(
        stats.sort_values("age"), x="age", y="readmit_pct", color="diag_1_group", markers=True,
        hover_data=["visits"],
        labels={"age": "Возраст", "readmit_pct": "Вернулись раньше 30 дней, %",
                "diag_1_group": "Группа диагноза", "visits": "Визитов"},
        title="Доля возвратов по возрасту и группе основного диагноза",
    )
    config.ensure_dirs()
    path = config.FIGURES_DIR / "07_interactive_age_diagnosis.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return str(path)


def run_all(df) -> list[str]:
    """Сохранить обе таблицы статистики и все 7 графиков; вернуть пути к графикам."""
    config.ensure_dirs()
    describe_numeric(df)
    iqr_outliers(df)
    plots = [plot_target, plot_rate_by_inpatient, plot_rate_by_diagnosis, plot_boxplots,
             plot_correlation, plot_visits_per_patient, plot_interactive]
    return [plot(df) for plot in plots]
