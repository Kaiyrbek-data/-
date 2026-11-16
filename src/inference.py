"""Validated inference for JSON, console and Streamlit."""
import json
import math
import joblib
import pandas as pd
from .config import *
from .features import INPUT_COLS
from .labels import LABELS

def load_schema():
    return json.loads((ARTIFACTS/'input_schema.json').read_text(encoding='utf-8'))

def predict_record(record, model_path=None):
    schema=load_schema()
    unknown=set(record)-set(INPUT_COLS)
    if unknown:
        raise ValueError(f'Unknown input fields (IDs/target are forbidden): {sorted(unknown)}')
    missing=[c for c in INPUT_COLS if c not in record]
    values={**schema['defaults'],**record}
    unseen=[]
    for column,spec in schema['fields'].items():
        value=values[column]
        if value is None or value=='':
            values[column]=None
            continue
        if spec['type']=='integer':
            number=float(value)
            if not math.isfinite(number) or not number.is_integer() or not spec['min']<=number<=spec['max']:
                raise ValueError(f'{column}: expected integer in [{spec["min"]}, {spec["max"]}]')
            values[column]=int(number)
        else:
            values[column]=str(value)
            if str(value) not in spec['values']:
                unseen.append(column)
    if str(values['discharge_disposition_id']) in DEATH_CODES+HOSPICE_CODES:
        raise ValueError('This model excludes death and hospice discharges from its population.')
    if not any(values[c] is not None for c in NUM_COLS):
        raise ValueError('Provide at least one quantitative input.')
    path=model_path or ARTIFACTS/'best_model.joblib'
    if not path.exists():
        raise FileNotFoundError('Train a model first: python main.py all')
    bundle=joblib.load(path)
    probability=float(bundle['model'].predict_proba(pd.DataFrame([values]))[0,1])
    return {'model':bundle['name'],'probability':probability,'threshold':bundle['threshold'],
            'prediction':int(probability>=bundle['threshold']),
            'defaults_used_for':missing,'unseen_categories':unseen,
            'scope':'Educational estimate at discharge, not a medical recommendation.'}

def console_prediction():
    schema=load_schema(); record={}
    print('Введите значения. Enter оставляет показанное значение; ? означает пропуск.')
    print('Значения по умолчанию — статистики fit, а не данные конкретного пациента.')
    for column,spec in schema['fields'].items():
        default=schema['defaults'][column]
        options=f' ({", ".join(spec["values"][:8])})' if spec['type']=='category' else ''
        while True:
            value=input(f'{LABELS.get(column,column)} ({column}){options} [{default}]: ').strip()
            if not value: value=default
            if value=='?': value=None
            if spec['type']=='integer' and value is not None:
                try:
                    number=float(value)
                    if not number.is_integer() or not spec['min']<=number<=spec['max']: raise ValueError()
                    value=int(number)
                except (ValueError,TypeError):
                    print(f'Нужно целое число от {spec["min"]} до {spec["max"]}.'); continue
            record[column]=value
            break
    result=predict_record(record)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
