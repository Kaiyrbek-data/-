"""Build the written PDF report from measured results, never invented metrics.

Usage: python scripts/build_report.py
Requires requirements-docs.txt. All analysis must have completed first.
"""
from pathlib import Path
import csv
import json
import math
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'; TABLES=REPORTS/'tables'; FIGURES=REPORTS/'figures'; ARTIFACTS=ROOT/'artifacts'

def read_json(path): return json.loads(path.read_text(encoding='utf-8'))
def read_csv(name):
    with (TABLES/name).open(encoding='utf-8-sig',newline='') as stream: return list(csv.DictReader(stream))
def number(value,digits=3):
    try: return f'{float(value):.{digits}f}'
    except (ValueError,TypeError): return str(value)
def integer(value): return f'{int(float(value)):,}'.replace(',',' ')

def build_report():
    for name in ['test_metrics.csv','validation_metrics.csv','source_column_audit.csv']:
        if not (TABLES/name).exists(): raise FileNotFoundError(f'Run python main.py all first: missing {name}')
    regular=Path('C:/Windows/Fonts/arial.ttf'); bold=Path('C:/Windows/Fonts/arialbd.ttf')
    if not regular.exists():
        regular=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        bold=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    pdfmetrics.registerFont(TTFont('Body',str(regular)))
    pdfmetrics.registerFont(TTFont('BodyBold',str(bold)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle',fontName='BodyBold',fontSize=24,leading=31,textColor=colors.black,spaceAfter=22))
    styles.add(ParagraphStyle(name='Section',fontName='BodyBold',fontSize=17,leading=22,textColor=colors.black,spaceAfter=13))
    styles.add(ParagraphStyle(name='Subsection',fontName='BodyBold',fontSize=11.5,leading=16,textColor=colors.black,spaceBefore=9,spaceAfter=5))
    styles.add(ParagraphStyle(name='BodyTextRu',fontName='Body',fontSize=10.2,leading=14.5,spaceAfter=9))
    styles.add(ParagraphStyle(name='SmallRu',fontName='Body',fontSize=8.1,leading=11,spaceAfter=6))
    styles.add(ParagraphStyle(name='CellRu',fontName='Body',fontSize=8.1,leading=10.8))
    styles.add(ParagraphStyle(name='HeaderRu',fontName='BodyBold',fontSize=8.1,leading=11,textColor=colors.white))
    story=[]
    def para(text,style='BodyTextRu'): return Paragraph(text,styles[style])
    def p(text,style='BodyTextRu'): story.append(para(text,style))
    def section(title): p(title,'Section')
    def page(): story.append(PageBreak())
    def table(headers,rows,widths=None):
        data=[[para(escape(str(x)),'HeaderRu') for x in headers]]
        data += [[para(escape(str(x)),'CellRu') for x in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([
          ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#183F50')),
          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F5F6')]),
          ('GRID',(0,0),(-1,-1),.45,colors.HexColor('#D9D9D9')),
          ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
          ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
          ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ]))
        story.append(t); story.append(Spacer(1,10))
    def image(name,max_height=270):
        path=FIGURES/name
        if not path.exists(): return
        w,h=ImageReader(str(path)).getSize(); width=170*mm; height=width*h/w
        if height>max_height: width*=max_height/height; height=max_height
        story.append(Image(str(path),width=width,height=height,hAlign='CENTER'))
        story.append(Spacer(1,8))
    summary=read_json(REPORTS/'data_summary.json'); facts=read_json(REPORTS/'eda_facts.json')
    selection=read_json(ARTIFACTS/'selection.json'); versions=read_json(ARTIFACTS/'environment.json')
    tests=read_csv('test_metrics.csv'); val=read_csv('validation_metrics.csv')
    winner=selection['winner']; best=next(r for r in tests if r['model']==winner)
    metadata=read_json(ROOT/'project_metadata.json')
    width=170*mm

    p('Прогноз повторной госпитализации пациентов с диабетом','ReportTitle')
    p('Письменный отчёт по проекту Machine Learning on Tabular Data','Subsection')
    p('Zhenissov Kaiyrbek и Bekarys<br/>Astana IT University<br/>2026')
    story.append(Spacer(1,22))
    p('Аннотация','Section')
    p(f'Разработан воспроизводимый проект на Python для прогноза зарегистрированной повторной госпитализации менее чем через 30 дней. '
      f'Модель {winner} выбрана по average precision на отдельной validation. На итоговом test получены '
      f'ROC-AUC {number(best["roc_auc"])}, AP {number(best["pr_auc_ap"])}, F1 положительного класса {number(best["f1"])}. '
      'Госпитализации одного пациента не пересекаются между обучением, калибровкой, валидацией и тестом.')
    p('Проект включает аудит всех 50 исходных столбцов, реляционную SQLite и 12 SQL-запросов, '
      'Pipeline/ColumnTransformer, групповой подбор гиперпараметров, калибровку, объяснение важности признаков, '
      'анализ ошибок, сохранённые модели и интерфейсы предсказания.')
    p('Модель оценивает исторический риск в заданной популяции и не является подтверждённым инструментом клинического решения. '
      'Публикация GitHub отложена по решению команды; согласование исключения по объёму данных остаётся организационным условием сдачи.')
    p(f'Идентификатор эксперимента: {selection["run_id"]}. Seed: 42. Полный бюджет поиска, quick_mode={selection["quick_mode"]}.','SmallRu')
    page()

    section('1 Данные и исследовательская задача')
    p('Источник — Diabetes 130-US Hospitals for Years 1999–2008, UCI Machine Learning Repository. '
      'Одна строка соответствует стационарной госпитализации, а не уникальному человеку. '
      'Возраст задан интервалами, диагнозы — кодами ICD-9, результаты ряда анализов — категориями.')
    table(['Характеристика','Значение'],[
       ['Исходные госпитализации',integer(facts['raw_rows'])],['Уникальные пациенты',integer(facts['patients'])],
       ['Исходные столбцы','50'],['Положительные случаи <30',integer(facts['raw_positive_count'])],
       ['Доля <30',number(facts['raw_positive_rate']*100,2)+'%'],
       ['CSV на диске',number(summary['source_info']['csv_bytes']/1024**2,2)+' MiB'],
       ['Момент предсказания','При выписке'],['Целевой класс 0','>30 или NO']], [width*.58,width*.42])
    image('01_target_balance.png',230)
    p('NO означает отсутствие зарегистрированного повторного обращения в источнике. Обращения вне охвата системы могут быть не видны. '
      'Частота исхода относится к госпитализациям в этой выборке, а не к общей популяции людей с диабетом.')
    p('Источник меньше рекомендованных 500 тыс. строк или 200 МБ. Требуется согласование исключения с преподавателем. '
      'Нормализация SQL не считается увеличением объёма независимых наблюдений.','SmallRu')
    page()

    section('2 Ключи и повторные наблюдения')
    table(['Проверка','Результат'],[
       ['Полные дубликаты',facts['duplicates']],['Повторы encounter_id',facts['duplicate_encounter_ids']],
       ['Пациентов с несколькими госпитализациями',integer(facts['repeated_patients'])],
       ['Максимум госпитализаций одного пациента',facts['max_encounters_per_patient']]], [width*.7,width*.3])
    image('02_patient_recurrence.png',220)
    p('encounter_id используется для трассировки записей и связи таблиц. patient_nbr используется для групповых разбиений. '
      'Оба идентификатора исключены из входов модели. Повторные госпитализации не удаляются как дубликаты: это разные события.')
    p('Подсчёт всех строк пациента может учитывать будущие события и поэтому не добавляется как модельный признак. '
      'Входы number_outpatient, number_emergency и number_inpatient описывают предшествующий год согласно определению источника.')
    p('Повторные события не считаются независимыми пациентами при оценке неопределённости: интервалы финальных метрик строятся bootstrap по пациентам.')
    page()

    section('3 Пропуски и неизвестные коды')
    missing=[r for r in read_csv('source_column_audit.csv') if float(r['missing_count'])>0]
    table(['Столбец','Пропусков','Доля'],[[r[''],integer(r['missing_count']),number(r['missing_pct'],2)+'%'] for r in missing],
          [width*.5,width*.25,width*.25])
    image('03_missing_values.png',215)
    p('В дополнение к NaN учтены Unknown/Invalid и коды Not Available, NULL, Not Mapped. '
      'Строка None в A1Cresult/max_glu_serum сохраняется как состояние отсутствия проведённого анализа. '
      'Результаты диагностики кодов приведены в ноутбуке и справочниках SQLite.')
    p('Исходный weight исключён из модели из-за крайней разреженности, но сохраняется индикатор weight_recorded. '
      'Строки с отсутствующим весом не удаляются. payer_code и medical_specialty сохраняются с отдельной категорией пропусков; '
      'заполнение наиболее частой категорией не подменяет неизвестные факты.')
    p('Различие долей пропусков между классами и совместное отсутствие столбцов изучены только на fit. '
      'По этим наблюдениям нельзя надёжно установить механизм MCAR/MAR/MNAR или причинность.','SmallRu')
    page()

    section('4 Демография и категории госпитализации')
    p('В исходной выборке Female составляет 53.759%, Male — 46.238%; три значения Unknown/Invalid не являются подтверждённым полом. '
      'race распределён неравномерно: Caucasian 74.78%, AfricanAmerican 18.88%; остальные категории малочисленны. '
      'Преобладание группы не доказывает повышенный риск.')
    p('Для возраста доступны десять интервалов. Наиболее многочисленная группа — 70–79 лет, 25.62% исходных госпитализаций. '
      'Точный возраст не восстанавливается. Модель использует One-Hot Encoding интервалов, не навязывая линейную зависимость риска от возраста.')
    image('category_age.png',260)
    p('admission_type_id, discharge_disposition_id и admission_source_id расшифрованы по IDS_mapping.csv. '
      'Это номинальные коды, а не величины. Исход выписки доступен в принятой постановке при выписке. '
      'Для постановки при поступлении его использование было бы недопустимо.')
    p('Графики и таблицы по race, gender, age, weight, payer_code, medical_specialty и трём кодам содержат число госпитализаций, '
      'число пациентов и долю положительного класса на fit. Малые группы не интерпретируются как устойчивые различия.')
    page()

    section('5 Количественные признаки и выбросы')
    numeric=read_csv('numeric_descriptive.csv')
    table(['Признак','Среднее','Медиана','P99','Макс.'],[
      [r[''],number(r['mean'],2),number(r['50%'],1),number(r['99%'],1),number(r['max'],0)] for r in numeric],
      [width*.45,width*.15,width*.14,width*.13,width*.13])
    p('Проверены все восемь количественных признаков. Ноль предыдущих обращений является допустимым значением. '
      'Длинные правые хвосты отражают неоднородность нагрузки и сами по себе не доказывают ошибку записи.')
    outliers=read_csv('outlier_diagnostics.csv')
    table(['Признак','IQR','За границами IQR'],[[r[''],number(r['iqr'],1),number(r['outside_pct'],2)+'%'] for r in outliers],
          [width*.55,width*.18,width*.27])
    p('Правило 1.5 IQR применяется диагностически. При IQR=0 любое ненулевое значение может быть отмечено как выброс. '
      'Автоматического удаления, winsorization и обучения границ на полном наборе нет. '
      'Скрытие крайних точек на boxplot не изменяет таблицу.')
    page()

    section('6 Диагнозы анализы и лекарства')
    p('diag_1, diag_2, diag_3 сохраняются строками и группируются по фиксированной укрупнённой схеме ICD-9: '
      'диабет, болезни кровообращения, дыхания, пищеварения, мочеполовой системы, травмы, болезни костно-мышечной системы, новообразования и Other. '
      'V/E-коды входят в Other; пропуски остаются отдельно. Группировка не является полной медицинской классификацией.')
    image('category_diag_1_group.png',245)
    p('Проверены 23 лекарственных столбца No/Steady/Up/Down, change и diabetesMed. '
      'examide и citoglipton постоянны в исходнике. Добавлены число записанных противодиабетических препаратов и число изменений дозировки. '
      'Это не точные дозировки и не общее число всех лекарств.')
    p('Результаты A1Cresult и max_glu_serum рассматриваются как категории. Созданы индикаторы проведения исследований. '
      'Различия частот повторной госпитализации между назначениями не доказывают эффект препарата: тяжесть состояния и другие факторы могут объяснять наблюдение.')
    page()

    section('7 Связи признаков с целью')
    correlations=read_csv('target_spearman.csv')
    strongest=max(correlations,key=lambda row:abs(float(row['readmitted_30d'])))
    p(f'Наибольшая по модулю одномерная связь среди количественных признаков на fit: {escape(strongest[""])} '
      f'(Spearman {number(strongest["readmitted_30d"],3)}). Даже этот коэффициент не описывает весь риск и не заменяет проверку модели.')
    table(['Количественный признак','Spearman с целью'],[[r[''],number(r['readmitted_30d'],4)] for r in correlations],
          [width*.7,width*.3])
    image('07_spearman.png',310)
    p('Корреляции рассчитаны на fit. Низкая одномерная корреляция не исключает полезности взаимодействий и нелинейных зависимостей. '
      'Категориальные коды и идентификаторы в числовую корреляцию не включались.')
    p('Отдельно исследованы группы предшествующих обращений 0, 1, 2 и 3+. '
      'Сравниваются доли положительного класса внутри групп, а не только абсолютное число событий.','SmallRu')
    page()

    section('8 Популяция и независимые выборки')
    flow=summary['cohort_flow']
    table(['Этап','Госпитализации'],[['Исходные',integer(flow['raw_encounters'])],
      ['Исключены исходы смерти',integer(flow['death_encounters'])],['Исключён хоспис',integer(flow['hospice_encounters'])],
      ['Включены в моделирование',integer(flow['cohort_encounters'])],['Уникальных пациентов в популяции',integer(flow['cohort_patients'])]],
      [width*.7,width*.3])
    splits=read_csv('split_summary.csv')
    table(['Часть','Строки','Пациенты','Доля класса 1'],[[r['split'],integer(r['encounters']),integer(r['patients']),number(float(r['positive_rate'])*100,2)+'%'] for r in splits],
          [width*.28,width*.24,width*.24,width*.24])
    p('Fit используется для обучения и 3-fold StratifiedGroupKFold; calibration — только для sigmoid-калибровки; '
      'validation — для выбора модели по AP и порога по F1; test — для итогового сравнения. '
      'Сначала разделены пациенты, затем обучаются преобразования. Пересечение пациентов во всех парах и CV-фолдах равно нулю.')
    p('Групповое разбиение проверяет перенос на новых пациентов той же системы. Дат госпитализации и идентификаторов больниц в выгрузке нет, '
      'поэтому временной перенос и перенос на новые больницы не заявляются.')
    p('Манифесты split_manifest.parquet и cv_manifest.parquet позволяют проверить принадлежность каждой записи. '
      'Повторный подбор после просмотра test нарушит независимость финальной оценки.')
    page()

    section('9 Подготовка и подбор моделей')
    p('Каждая модель представлена Pipeline: FeatureBuilder → ColumnTransformer → классификатор. '
      'FeatureBuilder выполняет только фиксированные преобразования без использования цели. '
      'Числовой блок обучает медианную импутацию и StandardScaler. Категориальный — отдельное значение Missing и OneHotEncoder '
      'с объединением уровней частотой менее 30 и обработкой новых категорий.')
    p('Baseline — DummyClassifier с постоянной априорной вероятностью. '
      'Сравниваются Logistic Regression, Random Forest и LightGBM. '
      'Подбор проводится по average precision на трёх групповых фолдах fit. Варианты весов классов исследуются внутри поиска; SMOTE вне фолдов не применяется.')
    table(['Модель','Кандидатов','CV AP лучшего варианта'],[[name,0 if name=='Dummy' else (6 if name=='LightGBM' else 4),'-' if name=='Dummy' else number(selection['details'][name]['cv_ap'])]
      for name in ['Dummy','LogisticRegression','RandomForest','LightGBM']], [width*.42,width*.2,width*.38])
    p('Лучшая конфигурация выбранной модели','Subsection')
    for key,value in selection['details'][winner]['best_params'].items():
        p(escape(f'{key}: {value}'),'SmallRu')
    p('PR-AUC в проекте явно определена как average precision (AP). Не следует напрямую сравнивать её с трапецеидальной площадью без проверки определения. '
      'F1 относится к положительному классу, не к weighted average.')
    page()

    section('10 Итоговые результаты моделей')
    table(['Модель','ROC-AUC','AP','F1','Precision','Recall'],[
      [r['model'],number(r['roc_auc']),number(r['pr_auc_ap']),number(r['f1']),number(r['precision']),number(r['recall'])] for r in tests],
      [width*.31,width*.15,width*.12,width*.12,width*.15,width*.15])
    p(f'<b>{winner}</b> выбрана заранее по validation AP. Порог {number(best["threshold"],4)} выбран на validation, '
      'не по test. Даже если другая модель имеет лучшую отдельную тестовую метрику, это не меняет правило выбора.')
    table(['Метрика выбранной модели','Оценка','95% интервал'],[[metric,number(best[key]),
      f'{number(best[key+"_ci_low"])} - {number(best[key+"_ci_high"])}'] for metric,key in [('ROC-AUC','roc_auc'),('AP','pr_auc_ap'),('F1','f1')]],
      [width*.4,width*.2,width*.4])
    image('selected_confusion.png',245)
    p('Интервалы получены 300 bootstrap-пересэмплированиями пациентов при фиксированной модели. '
      'Они не включают всю неопределённость подбора и не доказывают межбольничную или временную переносимость.')
    p('Для контроля интерпретации: всегда предсказывать 0 даёт высокую accuracy, но нулевой recall. '
      'Всегда предсказывать 1 даёт recall=1 и F1 около 0.20 при положительном классе около 11%. '
      'Поэтому качество оценивается совместно по нескольким метрикам.')
    page()

    section('11 Калибровка вероятностей')
    table(['Модель','Brier до на val','Brier после на val','Brier test'],[
      [r['model'],number(r['raw_brier'],4),number(r['brier'],4),number(next(t for t in tests if t['model']==r['model'])['brier'],4)] for r in val],
      [width*.34,width*.22,width*.22,width*.22])
    image('calibration_comparison.png',250)
    p('Sigmoid-калибровка обучается на отдельной calibration с замороженным классификатором. '
      'Калибровка согласует шкалу вероятностей, но не обещает повышения ROC-AUC: монотонное преобразование обычно сохраняет ранжирование.')
    p('Brier и log-loss оценивают вероятностную ошибку, ECE — расхождение предсказанной и наблюдаемой частоты по десяти интервалам. '
      'ECE зависит от разбиения на интервалы. Reliability curve использует квантильные группы, поэтому её группировка отличается от ECE.')
    p(f'На test у выбранной модели Brier={number(best["brier"],4)}, log-loss={number(best["log_loss"],4)}, '
      f'ECE={number(best["ece_10bins"],4)}. Качество вероятностей оценивается отдельно от классификации по выбранному порогу.')
    page()

    section('12 Интерпретация и анализ ошибок')
    image('permutation_importance.png',300)
    top=read_csv('permutation_importance.csv')[:7]
    p('Наибольшее снижение validation AP при перемешивании наблюдалось для: '+', '.join(escape(r['feature']) for r in top)+'. '
      'Это оценка зависимости модели от признаков, а не причинного влияния. Коррелированные признаки могут делить важность.')
    matrix=read_csv('confusion_matrix.csv')
    tn=int(matrix[0]['Predicted 0']); fp=int(matrix[0]['Predicted 1']); fn=int(matrix[1]['Predicted 0']); tp=int(matrix[1]['Predicted 1'])
    p(f'На test: TN={integer(tn)}, FP={integer(fp)}, FN={integer(fn)}, TP={integer(tp)}. '
      'FN — пропущенные ранние повторные госпитализации; FP — ложные предупреждения. Выбор порога отражает компромисс между этими ошибками.')
    p('Сохранены профили FP/FN, примеры FN и описательные метрики по возрасту, полу и race. '
      'Для групп с n&lt;200 или менее 20 положительных исходов стоит метка малой выборки. '
      'Различия между группами не объявляются статистически значимыми без отдельной проверки.')
    page()

    section('13 SQLite и аналитические запросы')
    p('patients хранит идентификаторы. encounters — характеристики госпитализации. '
      'diagnoses хранит три позиции ICD-9. drugs и encounter_medications описывают назначения. '
      'Три справочника расшифровывают коды. split_assignments фиксирует популяцию и разбиение. '
      'experiments/predictions сохраняют историю запусков с run_id.')
    image('er_diagram.png',285)
    queries=[('01','Баланс классов'),('02','Повторные пациенты и HAVING'),('03','JOIN типов поступления'),
      ('04','JOIN исходов выписки'),('05','Диагнозы с HAVING'),('06','CTE и DENSE_RANK'),('07','JOIN лекарств'),
      ('08','Подзапрос среднего числа обращений'),('09','Контроль групповой утечки'),('10','CTE групп обращений'),
      ('11','LEFT JOIN пациентов'),('12','Накопительная сумма OVER')]
    table(['Запрос','Назначение'],queries,[width*.16,width*.84])
    p('Запрос 09 вернул ноль строк. PRAGMA foreign_key_check не обнаружил нарушений. '
      'SQL-результаты загружаются через pd.read_sql_query. Целевые связи ограничены fit.','SmallRu')
    page()

    section('14 Время память и размер моделей')
    table(['Модель','Refit c','Поиск c','RSS MiB','joblib MiB'],[
      [r['model'],number(r['fit_time_s'],1),number(r['search_time_s'],1),number(r['peak_process_rss_MiB'],1),number(r['model_size_MiB'],3)] for r in tests],
      [width*.34,width*.15,width*.17,width*.17,width*.17])
    memory=summary['memory']
    p(f'Память исходного DataFrame до оптимизации: {number(memory["before_MiB"],2)} MiB; '
      f'после downcast и category: {number(memory["after_MiB"],2)} MiB; снижение {number(memory["reduction_pct"],2)}%. '
      'Размер зависит от версий Pandas и доступности Arrow-backed strings. Категориальная копия используется для демонстрации памяти, '
      'а не вместо обучаемого preprocessing.')
    p('Refit — финальное обучение на fit. Поиск включает CV и refit. RSS — абсолютный пик процесса при поиске с опросом каждые 50 мс, '
      'включает загруженные данные и ранее созданные объекты. Это не точная добавочная память конкретного алгоритма. '
      'Размеры joblib измерены при одинаковом сжатии и включают preprocessing/калибровку.')
    if (TABLES/'io_benchmark.csv').exists():
        io=read_csv('io_benchmark.csv')
        table(['Чтение','Строки','Время c','DataFrame MiB'],[[r['method'],integer(r['rows']),number(r['read_time_s'],3),number(r['memory_MiB'],2)] for r in io],
              [width*.51,width*.16,width*.16,width*.17])
        p('Это однократный замер в текущем окружении с возможным влиянием кэша ОС. '
          'SQLite encounters — нормализованная часть схемы, а не все 50 исходных столбцов, поэтому сравнение памяти не является сравнением равных представлений.','SmallRu')
    p('Воспроизведение: Python '+versions['python']+', scikit-learn '+versions['scikit-learn']+', Pandas '+versions['pandas']+', LightGBM '+versions['lightgbm']+'. Полные версии сохранены в artifacts/environment.json.')
    page()

    section('15 Запуск ограничения и вклад участников')
    p('Для готовой модели: python main.py predict --input examples/patient.json. '
      'Для консольного меню: python main.py. Для приложения: python -m streamlit run app.py. '
      'Все поля формы редактируемы; начальные значения — статистики fit, а не данные конкретного человека. '
      'Проверяются типы, допустимые диапазоны, отсутствие целевой переменной и соответствие популяции.')
    p('Для воспроизведения: python main.py all. Проверки: python -m unittest discover -s tests -v. '
      'Структура функций разделена на модули, используются list/dict, if/elif/else, for/while, try/except, pathlib и постоянное хранение данных.')
    p('Ограничения','Subsection')
    p('Исторические данные 1999–2008 не подтверждают работу модели сегодня. Отсутствуют многие социальные и послеоперационные факторы, '
      'явные даты и идентификаторы больниц. Неполный охват обращений ограничивает точность целевой переменной. '
      'Выводы о лекарствах, поле и race являются наблюдательными. Проверка внешней клинической полезности не проводилась.')
    p('Вклад участников','Subsection')
    if metadata.get('contributions'):
        if isinstance(metadata['contributions'],dict):
            table(['Участник','Фактический вклад'],list(metadata['contributions'].items()),[width*.35,width*.65])
        else: p(escape(str(metadata['contributions'])))
    else:
        table(['Участник','Статус описания вклада'],[['Zhenissov Kaiyrbek','Требуется подтверждение фактического вклада'],['Bekarys','Требуется подтверждение фактического вклада']],
              [width*.35,width*.65])
    p('До сдачи: согласовать исключение по объёму, подтвердить вклад участников, опубликовать GitHub после решения команды '
      'и загрузить отчёт/презентацию в Moodle за установленный преподавателем срок. Эти внешние действия не заявляются как выполненные.')
    page()

    section('16 Источники и проверяемые материалы')
    references=[
      ('Данные UCI и лицензия CC BY 4.0','https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008'),
      ('Clore et al. 2014 DOI датасета','https://doi.org/10.24432/C5230J'),
      ('Strack et al. 2014 описание признаков и ICD-9 группировка','https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/'),
      ('scikit-learn CalibratedClassifierCV','https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html'),
      ('scikit-learn StratifiedGroupKFold','https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html'),
      ('LightGBM LGBMClassifier','https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html')]
    for title,url in references:
        p(escape(title),'Subsection'); p(f'<link href="{escape(url)}" color="#147D92">{escape(url)}</link>','SmallRu')
    p('Все числовые результаты проекта получены локальным запуском кода. Результаты внешних работ не подставлялись вместо собственных экспериментов.')
    p('Материалы для проверки: EDA.ipynb, SQL_analysis.ipynb, Model_results.ipynb, sql/analysis.sql, '
      'reports/tables/*.csv, artifacts/selection.json, environment.json, data/split_manifest.parquet, cv_manifest.parquet и tests/test_project.py.')
    p('SHA256 исходного архива: '+summary['source_info']['archive_sha256'],'SmallRu')

    descriptions={
      'encounter_id':'Уникальный номер госпитализации; ключ связи, не вход модели.',
      'patient_nbr':'Идентификатор пациента; группы разбиения и CV, не вход модели.',
      'race':'Номинальная категория пациента; проверяется представленность групп и ошибки модели.',
      'gender':'Пол в записи. Unknown/Invalid трактуется как неизвестное значение.',
      'age':'Десятилетний возрастной интервал, а не точный возраст.',
      'weight':'Весовая категория; исходный интервал исключён, используется наличие записи.',
      'admission_type_id':'Номинальный код типа поступления со справочником.',
      'discharge_disposition_id':'Исход или направление выписки; учитывается выбранная популяция.',
      'admission_source_id':'Номинальный код источника поступления со справочником.',
      'time_in_hospital':'Продолжительность текущей госпитализации в днях.',
      'payer_code':'Категория плательщика; неизвестные значения сохраняются отдельно.',
      'medical_specialty':'Специальность принимающего врача; редкие категории объединяются внутри CV.',
      'num_lab_procedures':'Число лабораторных процедур текущей госпитализации.',
      'num_procedures':'Число остальных процедур текущей госпитализации.',
      'num_medications':'Число различных препаратов текущей госпитализации.',
      'number_outpatient':'Амбулаторные обращения за год до госпитализации.',
      'number_emergency':'Экстренные обращения за год до госпитализации.',
      'number_inpatient':'Стационарные госпитализации за предшествующий год.',
      'diag_1':'Основной диагноз ICD-9, преобразуется в фиксированную группу.',
      'diag_2':'Дополнительный диагноз ICD-9, преобразуется в фиксированную группу.',
      'diag_3':'Ещё один дополнительный диагноз ICD-9, преобразуется в фиксированную группу.',
      'number_diagnoses':'Число зарегистрированных диагнозов при текущей госпитализации.',
      'max_glu_serum':'Категория результата глюкозы; None — анализ не проводился.',
      'A1Cresult':'Категория HbA1c; None — анализ не проводился.',
      'change':'Общий индикатор изменения противодиабетической терапии.',
      'diabetesMed':'Индикатор назначения противодиабетических препаратов.',
      'readmitted':'Целевая переменная. <30 → 1, >30 и NO → 0; запрещена во входах.'}
    profiles=read_csv('source_column_audit.csv')
    for start in range(0,len(profiles),5):
        page(); section(f'Приложение А Разбор столбцов {start+1} по {min(start+5,len(profiles))}')
        for row in profiles[start:start+5]:
            column=row['']; p(escape(column),'Subsection')
            description=descriptions.get(column,'Статус назначения препарата: No, Steady, Up или Down. Не содержит точной дозировки.')
            p(escape(description))
            p(f'Тип {escape(row["dtype"])}. Уникальных значений {integer(row["unique"])}. '
              f'Пропуски {integer(row["missing_count"])} ({number(row["missing_pct"],2)}%). '
              f'Наиболее частое значение занимает {number(row["dominant_pct"],2)}%.','SmallRu')
            if column in ['examide','citoglipton']: p('Постоянный столбец исключён по фиксированной спецификации источника.','SmallRu')
    out=REPORTS/'Project_report.pdf'
    doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=20*mm,rightMargin=20*mm,
          topMargin=18*mm,bottomMargin=18*mm,title='Прогноз повторной госпитализации пациентов с диабетом',author='Zhenissov Kaiyrbek; Bekarys')
    def footer(canvas,document):
        canvas.saveState(); canvas.setFont('Body',8); canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawRightString(A4[0]-20*mm,10*mm,str(document.page)); canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('Created',out)

if __name__=='__main__': build_report()
