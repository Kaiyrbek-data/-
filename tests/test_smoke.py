"""Проверка запуска из клона: python -m unittest discover -s tests -v."""
from pathlib import Path
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from src import config, data, database, features, models


class StartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.old_db, cls.old_export = config.DB_PATH, config.EXPORT_DIR
        config.DB_PATH = Path(cls.tmp.name) / "readmission.db"
        config.EXPORT_DIR = Path(cls.tmp.name) / "exports"
        cls.clean = data.clean(data.load_raw())

    @classmethod
    def tearDownClass(cls):
        config.DB_PATH, config.EXPORT_DIR = cls.old_db, cls.old_export
        cls.tmp.cleanup()

    def test_data_and_patient_split(self):
        self.assertEqual(len(self.clean), 99340)
        prepared = features.prepare_for_model(self.clean)
        self.assertTrue(set(features.FEATURES).issubset(prepared.columns))
        train, test = models.split_by_patient(prepared)
        self.assertFalse(set(train.patient_nbr) & set(test.patient_nbr))
        self.assertEqual(len(train) + len(test), len(prepared))

    def test_database_queries_and_filters(self):
        # База отсутствует: первый SQL должен построить её сам.
        totals = database.run_query("SELECT COUNT(*) AS n FROM encounters")
        self.assertEqual(int(totals.iloc[0, 0]), 99340)
        with sqlite3.connect(config.DB_PATH) as connection:
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
        queries = database.load_queries()
        self.assertEqual(len(queries), 8)
        for name, query in queries.items():
            with self.subTest(query=name):
                self.assertFalse(database.run_query(query).empty)
        found = database.search_encounters(age="[70-80)", gender="Male", readmit_30=1, limit=7)
        self.assertEqual(len(found), 7)
        self.assertTrue(found.age.eq("[70-80)").all())
        self.assertTrue(found.gender.eq("Male").all())
        self.assertTrue(found.readmitted.eq("<30").all())
        self.assertTrue(database.search_encounters(patient_nbr=-1).empty)
        self.assertTrue(database.search_encounters(gender="Female' OR 1=1 --").empty)

    def test_saved_model_prediction(self):
        result = models.predict_patient({
            "age_num": 65, "time_in_hospital": 4, "num_lab_procedures": 43,
            "num_procedures": 1, "num_medications": 16, "number_diagnoses": 7,
            "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0,
            "total_prior_visits": 0, "diag_1_group": "Circulatory", "insulin": "No",
        })
        self.assertAlmostEqual(result["probability"], 0.09245283018867924, places=6)
        self.assertEqual(result["high_risk"], result["probability"] >= result["threshold"])
        self.assertTrue(0 <= models.predict_patient({})["probability"] <= 1)

    def test_terminal_and_csv(self):
        import main
        main.save_csv(database.search_encounters(limit=3))
        exported = list(config.EXPORT_DIR.glob("*.csv"))
        self.assertEqual(len(exported), 1)
        self.assertEqual(len(pd.read_csv(exported[0])), 3)
        # Запускаем настоящий main.py из другой рабочей папки.
        script = config.ROOT / "main.py"
        answers = "bad\n1\n3\n1\n2\n" + "\n" * 5 + "7\n" + "\n" * 11 + "0\n"
        run = subprocess.run([sys.executable, str(script)], cwd=self.tmp.name,
                             input=answers, text=True, encoding="utf-8", capture_output=True,
                             env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=120)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn("Traceback", run.stderr)
        self.assertNotIn("Ошибка:", run.stdout)
        self.assertIn("Вероятность возврата", run.stdout)

    def test_streamlit_startup_filters_sql_prediction(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=120).run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        self.assertEqual([t.label for t in app.tabs], ["Анализ", "SQL", "Модели", "Прогноз"])
        for query_name in database.load_queries():
            app.selectbox[0].select(query_name).run()
            self.assertEqual(len(app.exception), 0)
        app.multiselect[0].set_value([]).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("нет данных" in w.value for w in app.warning))
        app.multiselect[0].set_value(["[70-80)"]).run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        self.assertTrue(any("вероятность" in e.value.lower() for e in [*app.success, *app.error]))


if __name__ == "__main__":
    unittest.main()
