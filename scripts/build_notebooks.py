"""Build explanatory notebooks; execute with nbclient for checked outputs."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]

def build(name,cells):
    notebook=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(text) if kind=='md' else nbf.v4.new_code_cell(text)
                                     for kind,text in cells])
    notebook.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    notebook.metadata.language_info={'name':'python','version':'3.11'}
    nbf.write(notebook,ROOT/name)

base="""from pathlib import Path
import sys, json
ROOT = Path.cwd()
if not (ROOT / 'src').exists(): ROOT = ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
from IPython.display import display, Image, Markdown
from src.config import *
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 60)
def table(name): return pd.read_csv(TABLES / name)
def figure(name, width=1000): display(Image(filename=str(FIGURES / name), width=width))
"""

eda=[('md',"""# Подробный EDA госпитализаций пациентов с диабетом

**Авторы:** Zhenissov Kaiyrbek и Bekarys.

Одна строка — одна госпитализация. Цель — зарегистрированная повторная госпитализация **менее чем через 30 дней**. Прогноз делается при выписке. Источник: [UCI](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), CC BY 4.0.

Сначала проверяем весь источник. Затем исключаем исходы смерти/хосписа по заранее заданной популяции и делим пациентов на fit/calibration/validation/test. Все связи признаков с целью ниже относятся **только к fit**. Код расчёта графиков находится в `src/eda.py`; ноутбук объединяет исходные проверки, результаты и объяснения.

Перед первым запуском: `python main.py all`. Для воспроизведения только EDA: `python main.py eda`. Весь источник меньше рекомендуемых 500 тыс. строк / 200 МБ; исключение следует согласовать с преподавателем."""),('code',base),
('md','## 1 Загрузка и структура\nЧитаем ZIP без распаковки. Сохраняем категорию None, а ? и пустые поля считаем пропусками.'),
('code',"from src.data import load_data, make_cohort, patient_split, optimize_types\nraw, mappings, source_info = load_data()\ncohort, flow = make_cohort(raw)\nparts = patient_split(cohort)\nfit = parts['fit'].copy()\nprint(raw.shape)\ndisplay(raw.head())\nraw.info(memory_usage='deep')"),
('md','## 2 Паспорт всех 50 столбцов\nКардинальность и доминирующее значение — диагностические признаки, а не автоматические критерии удаления.'),
('code',"display(pd.read_csv(TABLES/'source_column_audit.csv', index_col=0).round(3))"),
('md','## 3 Ключи и повторные пациенты\nencounter_id должен быть уникальным. Повторение patient_nbr означает разные госпитализации и не является дубликатом. Общее число госпитализаций пациента не используем как вход модели: оно может учитывать будущее.'),
('code',"counts = raw.groupby('patient_nbr').size()\ndisplay(pd.Series({'duplicate_rows':raw.duplicated().sum(),'duplicate_encounter_id':raw.encounter_id.duplicated().sum(),'patients':len(counts),'repeat_patients':counts.gt(1).sum(),'max_encounters':counts.max()}).to_frame('value'))\nfigure('02_patient_recurrence.png')"),
('md','## 4 Цель и популяция\n<30 → 1; >30 и NO → 0. NO означает отсутствие зарегистрированного повторного обращения в источнике. Смерть и хоспис исключены до разбиения. Повторные госпитализации остальных пациентов сохраняются.'),
('code',"display(table('raw_target.csv'))\nfigure('01_target_balance.png',700)\ndisplay(pd.Series(flow).to_frame('value'))\ndisplay(table('split_summary.csv'))\nassert all(set(a.patient_nbr).isdisjoint(b.patient_nbr) for na,a in parts.items() for nb,b in parts.items() if na!=nb)"),
('md','## 5 Пропуски\nНеизвестные коды тоже учитываются. None в лабораторных результатах — осмысленная категория отсутствия исследования. Тип пропусков MCAR/MAR/MNAR по одним частотам не устанавливаем.'),
('code',"display(table('fit_missingness.csv').round(3))\ndisplay(table('missingness_by_target.csv').round(3))\nfigure('03_missing_values.png')\nfigure('04_missingness_correlation.png')"),
('md','## 6 Количественные признаки\nРазбираем все восемь показателей: время в стационаре, лабораторные и другие процедуры, препараты, амбулаторные/экстренные/стационарные обращения за предшествующий год и число диагнозов. Коды и идентификаторы в числовую статистику не включены.'),
('code',"display(table('numeric_descriptive.csv').round(3))\nfigure('05_numeric_distributions.png')\ndisplay(table('numeric_by_target.csv'))\nfigure('06_numeric_by_target.png')"),
('md','## 7 Выбросы\nГраница 1.5 IQR — диагностический флаг, а не доказательство ошибки. При IQR=0 ненулевые обращения могут формально стать выбросами. Наблюдения не удаляются. В boxplot крайние точки скрыты только для читаемости.'),
('code',"display(table('outlier_diagnostics.csv').round(3))"),
('md','## 8 Демография и административные признаки\nПоказываем число госпитализаций, число пациентов и долю <30 внутри каждой категории. Малые группы не интерпретируем как надёжные различия риска. weight почти пустой; в первой модели используем только индикатор наличия записи.'),
('code',"for column in ['race','gender','age','weight','payer_code','medical_specialty']:\n    display(Markdown('### '+column))\n    display(table('feature_'+column+'.csv').round(3))\n    figure('category_'+column+'.png')"),
('md','## 9 Расшифровки типов поступления и исходов\nВсе три столбца _id — номинальные категории. Отсутствие NaN не означает полноту информации: существуют коды NULL/Not Available/Unknown. Исход выписки доступен для прогноза при выписке, но недоступен для прогноза при поступлении.'),
('code',"for column in CODE_COLS:\n    display(Markdown('### '+column))\n    display(table('feature_'+column+'.csv').round(3))\n    figure('category_'+column+'_label.png')"),
('md','## 10 Диагнозы ICD 9\nИсходные коды остаются строками. В Pipeline применяется фиксированная укрупнённая схема из статьи Strack et al. 2014. Коды V/E относятся к Other. Группировка сокращает кардинальность, но теряет детализацию.'),
('code',"from src.features import icd9_group\nfor column in DIAG_COLS:\n    display(Markdown('### '+column))\n    display(table('raw_'+column+'_counts.csv').head(15))\n    display(table('feature_'+column+'_group.csv').round(3))\n    figure('category_'+column+'_group.png')"),
('md','## 11 Анализы и общие признаки терапии\nЛабораторные результаты представлены категориями, не точными концентрациями. Различие частот <30 между вариантами лечения не доказывает эффект препарата: тяжесть состояния влияет и на назначение, и на исход.'),
('code',"for column in ['A1Cresult','max_glu_serum','change','diabetesMed']:\n    display(table('feature_'+column+'.csv').round(3))\n    figure('category_'+column+'.png')"),
('md','## 12 Все 23 лекарственных столбца\nNo — не назначался; Steady — дозировка без изменения; Up/Down — увеличение/уменьшение. Проверяем константы и редкие назначения. Столбцы examide и citoglipton постоянны в данном источнике.'),
('code',"display(table('medication_counts.csv'))\nfigure('08_medication_states.png',850)\ndisplay(table('medication_target_rates.csv').round(3))"),
('md','## 13 Корреляции и предыдущие обращения\nSpearman исследует монотонные связи, но не описывает всю нелинейную предсказательную информацию. Большая группа не обязательно имеет больший риск: сравниваем доли внутри групп.'),
('code',"figure('07_spearman.png')\ndisplay(table('target_spearman.csv').round(4))\nfor column in ['number_outpatient','number_emergency','number_inpatient']:\n    display(table('feature_'+column+'_band.csv').round(3))\n    figure('category_'+column+'_band.png')"),
('md','## 14 Память и хранение\nРазмер CSV на диске и память DataFrame различаются. Оптимизация типов показана отдельно, не подменяет обучаемые преобразования Pipeline. Исходные данные сохранены также в Parquet.'),
('code',"summary=json.loads((REPORTS/'data_summary.json').read_text(encoding='utf-8'))\ndisplay(pd.Series(summary['memory']).to_frame('value'))\nprint('CSV MiB:',source_info['csv_bytes']/1024**2)\nfigure('er_diagram.png')"),
('md','## 15 Выводы по каждому исходному столбцу'),
('code',"display(Markdown((REPORTS/'EDA_detailed_conclusions.md').read_text(encoding='utf-8')))"),
('md','## Ограничения\nВыборка историческая, зависит от критериев включения и охвата медицинской системы. Нет дат для честной временной валидации и идентификаторов больниц для проверки переноса между больницами. Нельзя считать повторные госпитализации независимыми пациентами. EDA не устанавливает причинность и не подтверждает клиническую применимость модели.')]
build('EDA.ipynb',eda)

sql=[('md','# SQLite и 12 аналитических SQL запросов\nZhenissov Kaiyrbek и Bekarys. Нормализация, JOIN, CTE, HAVING, оконные функции и чтение в Pandas. Целевые связи ограничены fit.'),('code',base),('code',"figure('er_diagram.png')"),
('code',"from contextlib import closing\nfrom src.database import connect, load_sql_queries\nwith closing(connect()) as connection:\n    for name,query in load_sql_queries().items():\n        display(Markdown('## '+name+'\\n```sql\\n'+query+'\\n```'))\n        result=pd.read_sql_query(query,connection)\n        print('Rows:',len(result))\n        display(result.head(25))\n    print('Foreign key violations:',connection.execute('PRAGMA foreign_key_check').fetchall())"),
('md','Запрос 09 должен вернуть ноль строк. В encounter_medications хранятся только состояния не-No; отсутствие назначения означает No. Нормализация не создаёт дополнительных независимых госпитализаций.')]
build('SQL_analysis.ipynb',sql)

models=[('md','# Модели и оценка результатов\nМодель выбирается по validation AP. Test не участвует в подборе гиперпараметров, калибровке и выборе порога. PR-AUC в проекте означает average precision (AP), а не трапецеидальную площадь.'),('code',base),
('code',"selection=json.loads((ARTIFACTS/'selection.json').read_text(encoding='utf-8'))\nprint('Selected on validation:',selection['winner'])\ndisplay(table('cv_split_audit.csv'))\ndisplay(table('validation_metrics.csv').round(4))\ndisplay(table('test_metrics.csv').round(4))"),
('md','## Калибровка и пороги\nSigmoid обучен на отдельной группе пациентов calibration. Порог максимального F1 выбирается только на validation. Монотонная калибровка не обещает улучшения ранжирования ROC-AUC.'),
('code',"figure('calibration_comparison.png')\nfigure('model_curves.png')\nfigure('model_comparison.png')"),
('md','## Интерпретация и ошибки\nPermutation importance измерена на validation. Низкая важность отдельного признака при наличии коррелированных признаков не доказывает его бесполезность. Ошибки и подгруппы test исследуются описательно, без повторного выбора модели.'),
('code',"figure('permutation_importance.png')\ndisplay(table('permutation_importance.csv').round(4))\ndisplay(table('error_profiles.csv'))\ndisplay(table('subgroup_metrics.csv').round(4))"),
('md','## Предсказание из сохранённого joblib\nПример содержит статистические значения по умолчанию с fit, а не сведения конкретного человека. Все поля можно изменить в JSON или Streamlit.'),
('code',"from src.inference import predict_record\nrecord=json.loads((ROOT/'examples'/'patient.json').read_text(encoding='utf-8'))\ndisplay(pd.Series(predict_record(record)).to_frame('value'))"),
('md','## Что означают интервалы\n95% интервалы получены bootstrap по пациентам. Они отражают неопределённость на данном test при фиксированной модели, не учитывают весь процесс подбора и не доказывают переносимость на современные больницы.')]
build('Model_results.ipynb',models)
print('Created EDA.ipynb, SQL_analysis.ipynb, Model_results.ipynb')
