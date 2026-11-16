# Реляционная схема SQLite

```mermaid
erDiagram
    patients ||--o{ encounters : has
    admission_types ||--o{ encounters : type
    discharge_dispositions ||--o{ encounters : outcome
    admission_sources ||--o{ encounters : source
    encounters ||--|{ diagnoses : diagnoses
    encounters ||--o{ encounter_medications : receives
    drugs ||--o{ encounter_medications : drug
    encounters ||--o| split_assignments : assigned
    patients { INTEGER patient_nbr PK }
    encounters {
      INTEGER encounter_id PK
      INTEGER patient_nbr FK
      TEXT admission_type_id FK
      TEXT discharge_disposition_id FK
      TEXT admission_source_id FK
      INTEGER readmitted_30d
    }
    diagnoses {
      INTEGER encounter_id PK,FK
      INTEGER position PK
      TEXT icd9
      TEXT diagnosis_group
    }
    encounter_medications {
      INTEGER encounter_id PK,FK
      INTEGER drug_id PK,FK
      TEXT status
    }
    drugs { INTEGER drug_id PK
            TEXT name }
    split_assignments { INTEGER encounter_id PK,FK
                        TEXT split }
```

В patients хранится только идентификатор. Возраст и прочие характеристики могут отличаться между госпитализациями, поэтому находятся в encounters.

diagnoses содержит три позиции на госпитализацию, включая отсутствующие значения. В encounter_medications хранятся только назначения со статусом, отличным от No; отсутствие пары encounter_id/drug_id означает No. Исходная широкая таблица без потери значений сохранена в data/raw.parquet и в ZIP.

split_assignments содержит только включённую в моделирование популяцию. Исходная таблица encounters содержит и исключённые исходы смерти/хосписа, что позволяет проверить поток исключений.

experiments и predictions — накопительный журнал запусков (run_id, model и результаты), а не исходные сущности пациентов. Для них намеренно нет каскадного удаления при повторном импорте источника.

Нормализация не увеличивает число независимых госпитализаций. Номера encounter_id не используются как даты.
