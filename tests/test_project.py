"""Meaningful checks for patient leakage, preprocessing and persisted inference."""
import json
import sqlite3
import unittest
from contextlib import closing
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from src.config import *
from src.data import load_data,make_cohort,patient_split
from src.features import FeatureBuilder,INPUT_COLS,build_pipeline,icd9_group
from src.database import connect,load_sql_queries

class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw,cls.mappings,_=load_data()
        cls.cohort,cls.flow=make_cohort(cls.raw)
        cls.parts=patient_split(cls.cohort)

    def test_source_contract(self):
        self.assertEqual(self.raw.shape,(101766,50))
        self.assertTrue(self.raw.encounter_id.is_unique)
        self.assertEqual(int(self.raw.A1Cresult.eq('None').sum()),84748)
        self.assertEqual(int(self.raw.weight.isna().sum()),98569)

    def test_group_disjointness_and_cohort(self):
        groups={n:set(f.patient_nbr) for n,f in self.parts.items()}
        for a in groups:
            for b in groups:
                if a!=b: self.assertFalse(groups[a]&groups[b])
        self.assertEqual(sum(len(x) for x in self.parts.values()),len(self.cohort))
        self.assertFalse(self.cohort.discharge_disposition_id.isin(DEATH_CODES+HOSPICE_CODES).any())

    def test_icd_groups(self):
        expected={'250.83':'Diabetes','785':'Circulatory','786.5':'Respiratory',
                  'V45':'Other','E888':'Other','140':'Neoplasms','abc':'Unmapped'}
        for code,group in expected.items(): self.assertEqual(icd9_group(code),group)
        self.assertEqual(icd9_group(pd.NA),'Missing')

    def test_ids_and_target_do_not_affect_features(self):
        a=self.raw.head(8).copy(); b=a.copy()
        b['encounter_id']=999; b['patient_nbr']=777; b['readmitted']='<30'; b[TARGET]=1
        pd.testing.assert_frame_equal(FeatureBuilder().fit_transform(a),FeatureBuilder().fit_transform(b))

    def test_preprocessing_fits_train_only_and_handles_unseen(self):
        fit=self.raw.head(6)[INPUT_COLS].copy()
        fit['num_medications']=10
        future=fit.iloc[:1].copy(); future['num_medications']=10000
        future['race']='CompletelyUnseen'; future['diag_1']='V123'
        pipeline=build_pipeline(DummyClassifier()).fit(fit,[0,1,0,1,0,0])
        imputer=pipeline.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
        from src.features import MODEL_NUM_COLS
        self.assertEqual(imputer.statistics_[MODEL_NUM_COLS.index('num_medications')],10)
        prediction=pipeline.predict_proba(future)
        self.assertEqual(prediction.shape,(1,2))
        self.assertTrue(np.isfinite(prediction).all())

class IntegrationTests(unittest.TestCase):
    @unittest.skipUnless(DB_PATH.exists(),'Run prepare first')
    def test_relations_and_all_sql_queries(self):
        with closing(connect()) as con:
            self.assertFalse(con.execute('PRAGMA foreign_key_check').fetchall())
            self.assertEqual(con.execute('SELECT COUNT(*) FROM encounters').fetchone()[0],101766)
            queries=load_sql_queries(); self.assertGreaterEqual(len(queries),8)
            for name,sql in queries.items():
                result=pd.read_sql_query(sql,con)
                if name=='09_split_leakage_check': self.assertTrue(result.empty)

    @unittest.skipUnless((ARTIFACTS/'best_model.joblib').exists(),'Train first')
    def test_persisted_prediction_and_validation(self):
        from src.inference import load_schema,predict_record
        values=load_schema()['defaults']
        a=predict_record(values); b=predict_record(values)
        self.assertEqual(a['probability'],b['probability'])
        self.assertGreaterEqual(a['probability'],0); self.assertLessEqual(a['probability'],1)
        for invalid in [{'time_in_hospital':15},{'number_inpatient':-1},{'num_medications':1.5},
                        {'readmitted':'<30'},{'discharge_disposition_id':'11'}]:
            with self.assertRaises(ValueError): predict_record({**values,**invalid})

if __name__=='__main__': unittest.main()
