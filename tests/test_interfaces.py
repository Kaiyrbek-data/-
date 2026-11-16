"""Проверки объединённого меню, фильтров и безопасного поиска."""
import os
import subprocess
import sys
import unittest
from unittest.mock import patch
import pandas as pd
from src.config import ROOT, DATA
from src.pipeline import ensure_data_ready, load_analysis_data
from src.search import search_encounters, read_query


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_data_ready(log=lambda message: None)

    def test_search_filters_and_injection(self):
        result = search_encounters(age='[70-80)', gender='Female',
                                   diagnosis='Diabetes', readmitted_30d=1)
        self.assertFalse(result.empty)
        self.assertTrue(result.age.eq('[70-80)').all())
        self.assertTrue(result.gender.eq('Female').all())
        self.assertTrue(result.diagnosis_group.eq('Diabetes').all())
        self.assertTrue(result.readmitted.eq('<30').all())
        self.assertTrue(search_encounters(patient_nbr=-1).empty)
        self.assertTrue(search_encounters(gender="Female' OR 1=1 --").empty)

    def test_read_only_and_no_unnecessary_rebuild(self):
        with self.assertRaises(pd.errors.DatabaseError):
            read_query('DELETE FROM encounters')
        self.assertEqual(read_query('SELECT COUNT(*) AS n FROM encounters').iloc[0]['n'], 101766)
        with patch('src.pipeline.prepare_data') as prepare:
            ensure_data_ready()
            prepare.assert_not_called()

    def test_analysis_uses_fit_only(self):
        frame = load_analysis_data()
        manifest = pd.read_parquet(DATA/'split_manifest.parquet')
        fit_ids = set(manifest.loc[manifest['split'].eq('fit'), 'encounter_id'])
        self.assertEqual(set(frame.encounter_id), fit_ids)

    def test_menu_bad_input_search_and_sql(self):
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        answers = 'bad\n7\n' + '\n'*5 + '2\na\n8\n0\n'
        result = subprocess.run([sys.executable, 'main.py'], cwd=ROOT, input=answers,
                                capture_output=True, text=True, encoding='utf-8', env=env, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        self.assertNotIn('Ошибка:', result.stdout)
        self.assertIn('Нет такого пункта', result.stdout)
        self.assertIn('encounter_id', result.stdout)
        self.assertIn('Сохранено результатов: 12', result.stdout)

    def test_streamlit_prediction_filters_and_sql(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 5)
        app.multiselect[0].set_value([]).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('ничего не найдено' in warning.value for warning in app.warning))
        app.multiselect[0].set_value(['[70-80)']).run()
        self.assertEqual(len(app.exception), 0)
        for index, button in enumerate(app.button):
            if button.label == 'Получить предсказание':
                app.button[index].click().run()
                break
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(metric.label == 'Оценённая вероятность' for metric in app.metric))
        self.assertEqual(len(app.error), 0)
        query_box = next(box for box in app.selectbox if box.label == 'SQL-запрос')
        query_box.set_value('09_split_leakage_check').run()
        self.assertEqual(len(app.exception), 0)


if __name__ == '__main__':
    unittest.main()
