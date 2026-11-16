"""Stateless clinical grouping inside a leakage-safe sklearn Pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import *

def icd9_group(value):
    if pd.isna(value) or str(value).strip() in ['', '?']:
        return 'Missing'
    code = str(value).strip().upper()
    if code.startswith(('V', 'E')):
        return 'Other'
    try:
        number = int(code.split('.')[0])
    except ValueError:
        return 'Unmapped'
    if number == 250: return 'Diabetes'
    elif 390 <= number <= 459 or number == 785: return 'Circulatory'
    elif 460 <= number <= 519 or number == 786: return 'Respiratory'
    elif 520 <= number <= 579 or number == 787: return 'Digestive'
    elif 580 <= number <= 629 or number == 788: return 'Genitourinary'
    elif 800 <= number <= 999: return 'Injury'
    elif 710 <= number <= 739: return 'Musculoskeletal'
    elif 140 <= number <= 239: return 'Neoplasms'
    else: return 'Other'

DERIVED_NUM = ['n_diabetes_drugs', 'n_dose_changes', 'A1C_measured',
               'glucose_measured', 'weight_recorded', 'prior_visits_total']
MODEL_NUM_COLS = NUM_COLS + DERIVED_NUM
MODEL_CAT_COLS = ['race', 'gender', 'age', *CODE_COLS, 'payer_code', 'medical_specialty',
                  'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
                  *[c for c in DRUG_COLS if c not in ['examide', 'citoglipton']],
                  *[c + '_group' for c in DIAG_COLS]]
INPUT_COLS = list(dict.fromkeys(NUM_COLS + ['race', 'gender', 'age', 'weight', *CODE_COLS,
                 'payer_code', 'medical_specialty', *DIAG_COLS, 'max_glu_serum', 'A1Cresult',
                 'change', 'diabetesMed', *DRUG_COLS]))

class FeatureBuilder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        frame = X.copy()
        # Both inference and training use exactly the same deterministic rules.
        for column in INPUT_COLS:
            if column not in frame:
                frame[column] = np.nan
        for column in INPUT_COLS:
            if column not in NUM_COLS:
                frame[column] = frame[column].astype('string').replace({'?': pd.NA, '': pd.NA})
        for column, codes in SPECIAL_MISSING.items():
            frame[column] = frame[column].mask(frame[column].isin(codes))
        for column in NUM_COLS:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
        for column in DIAG_COLS:
            frame[column + '_group'] = frame[column].map(icd9_group)
        frame['n_diabetes_drugs'] = (frame[DRUG_COLS].notna() & frame[DRUG_COLS].ne('No')).sum(axis=1)
        frame['n_dose_changes'] = frame[DRUG_COLS].isin(['Up', 'Down']).sum(axis=1)
        frame['A1C_measured'] = (frame.A1Cresult.notna() & frame.A1Cresult.ne('None')).astype(int)
        frame['glucose_measured'] = (frame.max_glu_serum.notna() & frame.max_glu_serum.ne('None')).astype(int)
        frame['weight_recorded'] = frame.weight.notna().astype(int)
        frame['prior_visits_total'] = frame[['number_outpatient','number_emergency','number_inpatient']].sum(axis=1, min_count=1)
        result = frame[MODEL_NUM_COLS + MODEL_CAT_COLS].copy()
        # sklearn imputers use np.nan rather than pd.NA in object arrays.
        for column in MODEL_CAT_COLS:
            result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
        for column in MODEL_NUM_COLS:
            result[column] = result[column].astype(float)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(MODEL_NUM_COLS + MODEL_CAT_COLS, dtype=object)

def build_pipeline(estimator):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                        ('scale', StandardScaler())])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='Missing', keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=30,
                                  sparse_output=True, dtype=np.float32)),
    ])
    preprocess = ColumnTransformer([('numeric', numeric, MODEL_NUM_COLS),
                                    ('categorical', categorical, MODEL_CAT_COLS)],
                                   sparse_threshold=1.0)
    return Pipeline([('features', FeatureBuilder()), ('preprocess', preprocess), ('model', estimator)])
