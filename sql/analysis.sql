-- QUERY: 01_class_balance
SELECT readmitted, COUNT(*) AS encounters,
       ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (),2) AS share_pct
FROM encounters GROUP BY readmitted;

-- QUERY: 02_patient_recurrence
SELECT patient_nbr, COUNT(*) AS encounters, SUM(readmitted_30d) AS early_readmissions
FROM encounters GROUP BY patient_nbr HAVING COUNT(*) > 1
ORDER BY encounters DESC, patient_nbr LIMIT 20;

-- QUERY: 03_admission_types
SELECT a.code, a.description, COUNT(*) AS encounters,
       ROUND(100.0*AVG(e.readmitted_30d),2) AS readmission_pct
FROM encounters e JOIN admission_types a ON e.admission_type_id=a.code
JOIN split_assignments s ON s.encounter_id=e.encounter_id AND s.split='fit'
GROUP BY a.code,a.description ORDER BY encounters DESC;

-- QUERY: 04_discharge_outcomes
SELECT d.code,d.description,COUNT(*) AS encounters
FROM encounters e JOIN discharge_dispositions d ON e.discharge_disposition_id=d.code
GROUP BY d.code,d.description ORDER BY encounters DESC;

-- QUERY: 05_diagnosis_groups
SELECT d.diagnosis_group,COUNT(*) AS encounters,
       ROUND(100.0*AVG(e.readmitted_30d),2) AS readmission_pct
FROM diagnoses d JOIN encounters e USING(encounter_id)
JOIN split_assignments s USING(encounter_id)
WHERE d.position=1 AND s.split='fit'
GROUP BY d.diagnosis_group HAVING COUNT(*)>=100 ORDER BY readmission_pct DESC;

-- QUERY: 06_age_ranking
WITH age_rates AS (
 SELECT age,COUNT(*) AS encounters,AVG(readmitted_30d) AS rate
 FROM encounters e JOIN split_assignments s USING(encounter_id)
 WHERE s.split='fit' GROUP BY age
)
SELECT *, DENSE_RANK() OVER(ORDER BY rate DESC) AS rate_rank
FROM age_rates ORDER BY rate_rank;

-- QUERY: 07_medication_status
SELECT d.name,m.status,COUNT(*) AS encounters,
 ROUND(100.0*AVG(e.readmitted_30d),2) AS readmission_pct
FROM encounter_medications m JOIN drugs d USING(drug_id)
JOIN encounters e USING(encounter_id)
JOIN split_assignments s USING(encounter_id)
WHERE s.split='fit' GROUP BY d.name,m.status HAVING COUNT(*)>=100
ORDER BY d.name,encounters DESC;

-- QUERY: 08_high_prior_utilization
SELECT e.encounter_id,e.patient_nbr,e.number_inpatient,e.readmitted_30d
FROM encounters e JOIN split_assignments s USING(encounter_id)
WHERE s.split='fit' AND e.number_inpatient > (
 SELECT AVG(x.number_inpatient) FROM encounters x
 JOIN split_assignments z USING(encounter_id) WHERE z.split='fit'
) ORDER BY e.number_inpatient DESC,e.encounter_id LIMIT 30;

-- QUERY: 09_split_leakage_check
SELECT e.patient_nbr,COUNT(DISTINCT s.split) AS split_count
FROM encounters e JOIN split_assignments s USING(encounter_id)
GROUP BY e.patient_nbr HAVING COUNT(DISTINCT s.split)>1;

-- QUERY: 10_prior_visit_bands
WITH bands AS (
 SELECT CASE WHEN number_inpatient=0 THEN '0'
             WHEN number_inpatient=1 THEN '1'
             WHEN number_inpatient=2 THEN '2' ELSE '3+' END AS prior_band,
        readmitted_30d
 FROM encounters e JOIN split_assignments s USING(encounter_id) WHERE s.split='fit'
)
SELECT prior_band,COUNT(*) AS encounters,ROUND(100.0*AVG(readmitted_30d),2) AS readmission_pct
FROM bands GROUP BY prior_band ORDER BY prior_band;

-- QUERY: 11_patient_join
SELECT p.patient_nbr,COUNT(e.encounter_id) AS encounters
FROM patients p LEFT JOIN encounters e USING(patient_nbr)
GROUP BY p.patient_nbr ORDER BY encounters DESC,p.patient_nbr LIMIT 20;

-- QUERY: 12_specialty_window
WITH counts AS (
 SELECT COALESCE(medical_specialty,'Missing') AS specialty,COUNT(*) AS encounters
 FROM encounters GROUP BY COALESCE(medical_specialty,'Missing')
)
SELECT *,ROUND(100.0*SUM(encounters) OVER(ORDER BY encounters DESC,specialty
 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)/SUM(encounters) OVER(),2) AS cumulative_pct
FROM counts ORDER BY encounters DESC,specialty;
