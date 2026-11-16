"""Human-readable field labels for the inference interfaces."""
LABELS={
 'time_in_hospital':'Дней в стационаре','num_lab_procedures':'Лабораторных процедур',
 'num_procedures':'Других процедур','num_medications':'Различных препаратов',
 'number_outpatient':'Амбулаторных обращений за год','number_emergency':'Экстренных обращений за год',
 'number_inpatient':'Госпитализаций за предыдущий год','number_diagnoses':'Зарегистрированных диагнозов',
 'race':'Категория race в записи','gender':'Пол в записи','age':'Возрастная группа','weight':'Весовая категория',
 'admission_type_id':'Тип поступления','discharge_disposition_id':'Исход выписки',
 'admission_source_id':'Источник поступления','payer_code':'Плательщик / страховка',
 'medical_specialty':'Специальность врача','diag_1':'Основной диагноз ICD-9',
 'diag_2':'Дополнительный диагноз ICD-9','diag_3':'Третий диагноз ICD-9',
 'max_glu_serum':'Категория результата глюкозы','A1Cresult':'Категория HbA1c',
 'change':'Изменение терапии диабета','diabetesMed':'Препараты для лечения диабета',
}
STATE_LABELS={'No':'No — не назначался / нет','Steady':'Steady — без изменения дозировки',
              'Up':'Up — дозировка увеличена','Down':'Down — дозировка уменьшена',
              'None':'None — анализ не проводился','Missing':'Неизвестно / пропуск',
              'Female':'Female — женский','Male':'Male — мужской',
              'Unknown/Invalid':'Неизвестное или некорректное значение',
              'Ch':'Ch — терапия изменена','Yes':'Yes — да'}
