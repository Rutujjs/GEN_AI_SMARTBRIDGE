-- database/schema.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- Run this file once to create all required tables in healthcare_db.
-- Usage: psql -U postgres -d healthcare_db -f database/schema.sql
-- ─────────────────────────────────────────────────────────────────────────────

-- ── Table 1: vitals ──────────────────────────────────────────────────────────
-- Stores every incoming vital sign record received from the Kafka consumer.
CREATE TABLE IF NOT EXISTS vitals (
    id                SERIAL PRIMARY KEY,
    patient_id        INTEGER,
    heart_rate        FLOAT NOT NULL,
    spo2              FLOAT NOT NULL,            -- Oxygen saturation (%)
    body_temperature  FLOAT NOT NULL,            -- Celsius
    systolic_bp       FLOAT NOT NULL,            -- Systolic blood pressure (mmHg)
    diastolic_bp      FLOAT NOT NULL,            -- Diastolic blood pressure (mmHg)
    timestamp         TIMESTAMP NOT NULL,
    received_at       TIMESTAMP DEFAULT NOW()    -- When our system received it
);

-- ── Table 2: anomaly_results ──────────────────────────────────────────────────
-- Stores the ML-generated anomaly detection results for each vital record.
CREATE TABLE IF NOT EXISTS anomaly_results (
    id                      SERIAL PRIMARY KEY,
    vital_id                INTEGER REFERENCES vitals(id) ON DELETE CASCADE,
    patient_id              INTEGER,

    -- ML scores from each model (0 = normal, 1 = most anomalous)
    isolation_forest_score  FLOAT,
    autoencoder_score       FLOAT,
    combined_score          FLOAT,               -- Weighted average of both models

    -- Classification result
    severity                VARCHAR(10) NOT NULL, -- 'LOW', 'MEDIUM', or 'HIGH'
    is_anomaly              BOOLEAN NOT NULL,

    -- Snapshot of vitals at time of detection (for easy querying)
    heart_rate              FLOAT,
    spo2                    FLOAT,
    body_temperature        FLOAT,
    systolic_bp             FLOAT,
    diastolic_bp            FLOAT,

    detected_at             TIMESTAMP DEFAULT NOW()
);

-- ── Table 3: email_alerts ─────────────────────────────────────────────────────
-- Tracks HIGH severity alerts that triggered an email notification.
CREATE TABLE IF NOT EXISTS email_alerts (
    id              SERIAL PRIMARY KEY,
    anomaly_id      INTEGER REFERENCES anomaly_results(id) ON DELETE CASCADE,
    patient_id      INTEGER,
    severity        VARCHAR(10),
    email_sent_to   VARCHAR(255),
    sent_at         TIMESTAMP DEFAULT NOW(),
    success         BOOLEAN DEFAULT TRUE
);

-- ── Indexes for performance ───────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_vitals_patient    ON vitals(patient_id);
CREATE INDEX IF NOT EXISTS idx_vitals_timestamp  ON vitals(timestamp);
CREATE INDEX IF NOT EXISTS idx_anomaly_severity  ON anomaly_results(severity);
CREATE INDEX IF NOT EXISTS idx_anomaly_detected  ON anomaly_results(detected_at);

-- ── Confirm ───────────────────────────────────────────────────────────────────
DO $$ BEGIN
    RAISE NOTICE 'Schema created successfully: vitals, anomaly_results, email_alerts';
END $$;
