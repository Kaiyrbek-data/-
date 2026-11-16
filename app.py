"""Optional dashboard: python -m streamlit run app.py"""

import json
import pandas as pd
import streamlit as st
from src.config import *
from src.inference import load_schema, predict_record
from src.labels import LABELS, STATE_LABELS
from contextlib import closing
from src.pipeline import ensure_data_ready, load_analysis_data
from src.database import load_sql_queries
from src.search import read_query


@st.cache_data
def code_descriptions():
    from src.database import connect

    result = {}
    with closing(connect()) as connection:
        for column, table in zip(
            CODE_COLS, ['admission_types', 'discharge_dispositions', 'admission_sources']
        ):
            result[column] = dict(
                connection.execute(f'SELECT code,description FROM {table}').fetchall()
            )
    return result


st.set_page_config(page_title='Diabetes readmission', page_icon='🏥', layout='wide')
st.title('Повторная госпитализация пациентов с диабетом')
st.caption('Учебный проект • Zhenissov Kaiyrbek и Bekarys • прогноз при выписке')
try:
    with st.spinner('Подготовка локальной базы данных...'):
        ensure_data_ready(log=lambda message: None)
except (OSError, ValueError) as error:
    st.error(f'Не удалось подготовить данные: {error}')
    st.stop()

tabs = st.tabs(['Предсказание', 'Модели', 'Анализ', 'SQL', 'Данные'])

with tabs[0]:
    st.write(
        'Оценка вероятности зарегистрированной повторной госпитализации менее чем через 30 дней.'
    )
    st.info(
        'Модель разработана для выписанных живыми пациентов, не направленных в хоспис. Это учебная оценка, не медицинская рекомендация.'
    )
    if not (ARTIFACTS / 'best_model.joblib').exists():
        st.warning('Сначала выполните python main.py all в терминале.')
    else:
        schema = load_schema()
        descriptions = code_descriptions()
        st.write(
            'Все поля редактируются. Начальные значения взяты из обучающей выборки и не описывают конкретного пациента.'
        )
        values = {}
        with st.form('patient'):
            groups = {
                'Обращения и процедуры': NUM_COLS,
                'Пациент и госпитализация': [
                    'race',
                    'gender',
                    'age',
                    'weight',
                    *CODE_COLS,
                    'payer_code',
                    'medical_specialty',
                ],
                'Диагнозы и анализы': [
                    *DIAG_COLS,
                    'max_glu_serum',
                    'A1Cresult',
                    'change',
                    'diabetesMed',
                ],
                'Препараты': DRUG_COLS,
            }
            for title, columns in groups.items():
                with st.expander(title, expanded=title == 'Обращения и процедуры'):
                    layout = st.columns(3)
                    for i, column in enumerate(columns):
                        spec = schema['fields'][column]
                        default = schema['defaults'][column]
                        label = LABELS.get(column, column)
                        with layout[i % 3]:
                            if spec['type'] == 'integer':
                                values[column] = st.number_input(
                                    label,
                                    min_value=spec['min'],
                                    max_value=spec['max'],
                                    value=int(default),
                                    step=1,
                                    key=column,
                                )
                            elif column in DIAG_COLS:
                                values[column] = st.text_input(
                                    label, value=str(default or ''), key=column
                                )
                            else:
                                options = ['Missing'] + spec['values']

                                def display_value(value, column=column):
                                    if column in descriptions and value in descriptions[column]:
                                        return f'{value} — {descriptions[column][value]}'
                                    return STATE_LABELS.get(value, value)

                                chosen = st.selectbox(
                                    label,
                                    options,
                                    index=(
                                        options.index(str(default))
                                        if str(default) in options
                                        else 0
                                    ),
                                    format_func=display_value,
                                    key=column,
                                )
                                values[column] = None if chosen == 'Missing' else chosen
            submitted = st.form_submit_button('Получить предсказание', type='primary')
        if submitted:
            try:
                result = predict_record(values)
                st.metric('Оценённая вероятность', f'{result["probability"]:.1%}')
                st.write(
                    'Класс:',
                    (
                        '1 — повышенный риск по выбранному порогу'
                        if result['prediction']
                        else '0 — ниже выбранного порога'
                    ),
                )
                st.caption(
                    f'Модель: {result["model"]}. Порог выбран на validation: {result["threshold"]:.3f}.'
                )
                if result['unseen_categories']:
                    st.warning('Новые категории: ' + ', '.join(result['unseen_categories']))
            except (ValueError, FileNotFoundError) as error:
                st.error(str(error))
with tabs[1]:
    path = TABLES / 'test_metrics.csv'
    if path.exists():
        st.dataframe(pd.read_csv(path), width='stretch')
        for filename in ['model_curves.png', 'model_comparison.png']:
            if (FIGURES / filename).exists():
                st.image(str(FIGURES / filename))
with tabs[2]:
    st.caption(
        'Фильтры показывают обучающую часть fit. Остальные выборки не используются для исследования связей с целью.'
    )

    @st.cache_data
    def analysis_frame(cohort_stamp, manifest_stamp):
        return load_analysis_data()

    frame = analysis_frame(
        (DATA / 'cohort.parquet').stat().st_mtime_ns,
        (DATA / 'split_manifest.parquet').stat().st_mtime_ns,
    )
    columns = st.columns(3)
    ages = sorted(frame.age.dropna().unique(), key=lambda value: int(value[1:].split('-')[0]))
    genders = sorted(frame.gender.dropna().unique())
    diagnoses = sorted(frame.diag_1_group.dropna().unique())
    chosen_ages = columns[0].multiselect('Возраст', ages, default=ages)
    chosen_genders = columns[1].multiselect('Пол', genders, default=genders)
    chosen_diagnoses = columns[2].multiselect('Основной диагноз', diagnoses, default=diagnoses)
    selected = frame.loc[
        frame.age.isin(chosen_ages)
        & frame.gender.isin(chosen_genders)
        & frame.diag_1_group.isin(chosen_diagnoses)
    ]
    if selected.empty:
        st.warning('По выбранным фильтрам ничего не найдено.')
    else:
        metrics = st.columns(3)
        metrics[0].metric('Госпитализаций в выборке', f'{len(selected):,}')
        metrics[1].metric('Пациентов в выборке', f'{selected.patient_nbr.nunique():,}')
        metrics[2].metric('Возврат <30 дней', f'{selected[TARGET].mean():.1%}')
        rate = selected.groupby('diag_1_group')[TARGET].mean().mul(100).rename('Возврат, %')
        st.bar_chart(rate)
        age_rate = (
            selected.groupby('age')[TARGET].mean().mul(100).reindex(ages).rename('Возврат, %')
        )
        st.line_chart(age_rate)
        prior_rate = (
            selected.assign(prior=selected.number_inpatient.clip(upper=6))
            .groupby('prior')[TARGET]
            .mean()
            .mul(100)
        )
        prior_rate.index = prior_rate.index.map(lambda number: '6+' if number == 6 else str(number))
        st.bar_chart(prior_rate.rename('Возврат, %'))
    st.subheader('Подробные графики сохранённого анализа')
    choices = sorted(p.name for p in FIGURES.glob('*.png'))
    if choices:
        selected = st.selectbox('График', choices)
        st.image(str(FIGURES / selected))
with tabs[3]:
    queries = load_sql_queries()
    selected = st.selectbox('SQL-запрос', list(queries))
    st.code(queries[selected], language='sql')
    result = read_query(queries[selected])
    st.dataframe(result, width='stretch', hide_index=True)
    st.download_button(
        'Скачать результат CSV',
        result.to_csv(index=False).encode('utf-8-sig'),
        file_name=selected + '.csv',
        mime='text/csv',
    )
    if (FIGURES / 'er_diagram.png').exists():
        st.image(str(FIGURES / 'er_diagram.png'))
with tabs[4]:
    summary = json.loads((REPORTS / 'data_summary.json').read_text(encoding='utf-8'))
    flow = summary['cohort_flow']
    metrics = st.columns(3)
    metrics[0].metric('Исходных госпитализаций', f'{flow["raw_encounters"]:,}')
    metrics[1].metric('Для моделирования', f'{flow["cohort_encounters"]:,}')
    metrics[2].metric('Пациентов для моделирования', f'{flow["cohort_patients"]:,}')
    st.write(
        'Из моделирования исключены смерть и хоспис. Неизвестный пол сохраняется и обрабатывается как пропуск в Pipeline.'
    )
    st.write(
        'Обучение / калибровка / выбор модели / тест: примерно 60 / 10 / 10 / 20% пациентов. Пересечений пациентов нет.'
    )
    st.write('Исходный ZIP включён в проект. SQLite и Parquet создаются локально автоматически.')
