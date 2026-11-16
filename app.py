"""Сайт проекта. Запуск: streamlit run app.py"""
import pandas as pd
import streamlit as st

from src import config, data, database, models

st.set_page_config(page_title="Повторная госпитализация", layout="wide")
st.title("Повторная госпитализация пациентов с диабетом")


@st.cache_data
def load_clean() -> pd.DataFrame:
    """Загрузить и очистить данные один раз, дальше брать из кэша."""
    return data.clean(data.load_raw())


def show_rate(df, column, title):
    """Столбчатая диаграмма: процент возвратов по значениям столбца."""
    rate = (df.groupby(column)["readmit_30"].mean() * 100).round(1)
    st.subheader(title)
    st.bar_chart(rate)


clean = load_clean()
tab_data, tab_sql, tab_models, tab_predict = st.tabs(["Анализ", "SQL", "Модели", "Прогноз"])

# ------------------------------------------------------------ Анализ с фильтрами
with tab_data:
    ages = sorted(clean["age"].unique())
    groups = sorted(clean["diag_1_group"].unique())
    age_filter = st.multiselect("Возраст", ages, default=ages)
    gender_filter = st.multiselect("Пол", ["Female", "Male"], default=["Female", "Male"])
    group_filter = st.multiselect("Группа основного диагноза", groups, default=groups)

    selected = clean[clean["age"].isin(age_filter) & clean["gender"].isin(gender_filter)
                     & clean["diag_1_group"].isin(group_filter)]
    if selected.empty:
        st.warning("По этим фильтрам нет данных.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Госпитализаций", f"{len(selected):,}")
        col2.metric("Пациентов", f"{selected['patient_nbr'].nunique():,}")
        col3.metric("Вернулись раньше 30 дней", f"{selected['readmit_30'].mean() * 100:.1f}%")
        show_rate(selected, selected["number_inpatient"].clip(upper=6), "Возвраты по числу прошлых госпитализаций, %")
        show_rate(selected, "diag_1_group", "Возвраты по группе диагноза, %")

# ------------------------------------------------------------ SQL-запросы
with tab_sql:
    queries = database.load_queries()
    name = st.selectbox("Запрос", list(queries))
    st.code(queries[name], language="sql")
    st.dataframe(database.run_query(queries[name]), width="stretch")
    st.image(str(config.ROOT / "docs" / "er_diagram.png"), caption="Схема базы данных")

# ------------------------------------------------------------ Сравнение моделей
with tab_models:
    if config.METRICS_PATH.exists():
        st.dataframe(pd.read_parquet(config.METRICS_PATH).drop(columns="best_params"), width="stretch")
    else:
        st.info("Таблицы пока нет. Обучите модели: python main.py, пункт 6, режим 2.")

# ------------------------------------------------------------ Прогноз
with tab_predict:
    with st.form("patient"):
        age = st.slider("Возраст", 5, 95, 65, step=10)
        days = st.slider("Дней в больнице", 1, 14, 4)
        labs = st.slider("Лабораторных анализов", 1, 132, 43)
        medications = st.slider("Количество лекарств", 1, 81, 16)
        diagnoses = st.slider("Количество диагнозов", 1, 16, 7)
        outpatient = st.number_input("Амбулаторных визитов за год", 0, 50, 0)
        emergency = st.number_input("Визитов в скорую за год", 0, 80, 0)
        inpatient = st.number_input("Госпитализаций за год", 0, 30, 0)
        group = st.selectbox("Группа основного диагноза", groups)
        insulin = st.selectbox("Инсулин", ["No", "Steady", "Up", "Down"])
        submitted = st.form_submit_button("Рассчитать риск")

    if submitted:
        patient = {
            "age_num": age, "time_in_hospital": days, "num_lab_procedures": labs,
            "num_medications": medications, "number_diagnoses": diagnoses,
            "number_outpatient": outpatient, "number_emergency": emergency,
            "number_inpatient": inpatient,
            "total_prior_visits": outpatient + emergency + inpatient,
            "diag_1_group": group, "insulin": insulin,
        }
        try:
            result = models.predict_patient(patient)
            text = f"вероятность возврата в течение 30 дней {result['probability'] * 100:.1f}%"
            if result["high_risk"]:
                st.error(f"Высокий риск: {text}")
            else:
                st.success(f"Низкий риск: {text}")
        except FileNotFoundError as error:
            st.info(str(error))
