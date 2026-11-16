-- name: q1_by_diagnosis
-- Основной диагноз учитываем один раз на визит, чтобы не завышать число госпитализаций.
SELECT d.icd9_group, COUNT(*) AS encounters,
       ROUND(AVG(e.readmit_30) * 100, 1) AS readmit_pct
FROM encounters AS e
JOIN encounter_diagnoses AS d ON d.encounter_id = e.encounter_id
WHERE d.position = 1
GROUP BY d.icd9_group
ORDER BY readmit_pct DESC, d.icd9_group;

-- name: q2_by_age
-- Возраст задан интервалами; сортируем по их нижней границе.
SELECT age, COUNT(*) AS encounters,
       ROUND(AVG(readmit_30) * 100, 1) AS readmit_pct
FROM encounters
GROUP BY age
ORDER BY CAST(SUBSTR(age, 2) AS INTEGER), age;

-- name: q3_by_prior_visits
-- Сравниваем возвраты по числу прошлых госпитализаций, оставляя значения до 10 включительно.
SELECT number_inpatient, COUNT(*) AS encounters,
       ROUND(AVG(readmit_30) * 100, 1) AS readmit_pct
FROM encounters
WHERE number_inpatient <= 10
GROUP BY number_inpatient
ORDER BY number_inpatient;

-- name: q4_by_discharge
-- Небольшие группы выписок исключаем, чтобы проценты не опирались на редкие случаи.
SELECT e.discharge_disposition_id, d.description, COUNT(*) AS encounters,
       ROUND(AVG(e.readmit_30) * 100, 1) AS readmit_pct
FROM encounters AS e
JOIN discharge_dispositions AS d
  ON d.discharge_disposition_id = e.discharge_disposition_id
GROUP BY e.discharge_disposition_id, d.description
HAVING COUNT(*) >= 500
ORDER BY readmit_pct DESC, e.discharge_disposition_id;

-- name: q5_by_insulin
-- Условие по инсулину в JOIN сохраняет визиты без назначения: их нет в таблице лекарств.
SELECT COALESCE(m.status, 'Not prescribed') AS insulin_status,
       COUNT(*) AS encounters,
       ROUND(AVG(e.readmit_30) * 100, 1) AS readmit_pct
FROM encounters AS e
LEFT JOIN encounter_medications AS m
  ON m.encounter_id = e.encounter_id AND m.medication = 'insulin'
GROUP BY COALESCE(m.status, 'Not prescribed')
ORDER BY readmit_pct DESC, insulin_status;

-- name: q6_specialties_above_average
-- Сравниваем неокруглённые доли со средним по всем визитам; Unknown не считаем отделением.
SELECT medical_specialty, COUNT(*) AS encounters,
       ROUND(AVG(readmit_30) * 100, 1) AS readmit_pct
FROM encounters
WHERE medical_specialty <> 'Unknown'
GROUP BY medical_specialty
HAVING COUNT(*) >= 500
   AND AVG(readmit_30) > (SELECT AVG(readmit_30) FROM encounters)
ORDER BY readmit_pct DESC, medical_specialty;

-- name: q7_top_patients
-- При одинаковом числе визитов номер пациента задаёт воспроизводимый порядок.
SELECT patient_nbr, COUNT(*) AS encounters
FROM encounters
GROUP BY patient_nbr
ORDER BY encounters DESC, patient_nbr
LIMIT 10;

-- name: q8_visit_number
-- Нумеруем доступные визиты пациента по encounter_id; пятый и последующие объединяем в 5+.
WITH numbered_visits AS (
    SELECT patient_nbr, encounter_id, readmit_30,
           ROW_NUMBER() OVER (
               PARTITION BY patient_nbr ORDER BY encounter_id
           ) AS visit_number
    FROM encounters
), visit_bands AS (
    SELECT CASE WHEN visit_number >= 5 THEN 5 ELSE visit_number END AS visit_number,
           readmit_30
    FROM numbered_visits
)
SELECT visit_number,
       CASE WHEN visit_number = 5 THEN '5+' ELSE CAST(visit_number AS TEXT) END AS visit_label,
       COUNT(*) AS encounters,
       ROUND(AVG(readmit_30) * 100, 1) AS readmit_pct
FROM visit_bands
GROUP BY visit_number
ORDER BY visit_number;
