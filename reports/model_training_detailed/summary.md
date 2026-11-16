**Выбрана на validation:** LightGBM / initial. Порог: 0.1309.

**Test:** ROC-AUC 0.6701; AP 0.2316; precision 0.1938; recall 0.5253; F1 0.2832.

**Вероятности на validation:** Brier до 0.09155, после 0.09161.

**Ошибки test:** пропущено 1052 возвратов; ложных предупреждений 4841; обнаружено 1164.

**LogisticRegression:** изменение validation AP после подбора +0.00073.

**LightGBM:** изменение validation AP после подбора -0.00621.

**Ограничения:** историческая выборка, нет временной/межбольничной проверки; стоимость ошибок не задана; test ранее изучался в проекте. Учебный эксперимент, не медицинское решение.

**Результаты:** reports/model_training_detailed/. Новая модель: artifacts/notebook_training/detailed_model.joblib.