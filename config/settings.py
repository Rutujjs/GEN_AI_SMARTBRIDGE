# config/settings.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for the entire Healthcare Anomaly Detection System.
# Update the values below to match your local environment before running.
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Kafka ─────────────────────────────────────────────────────────────────────
KAFKA_BROKER        = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC         = "vitals"           # Topic name for vital sign messages
KAFKA_GROUP_ID      = "healthcare-group" # Consumer group ID
KAFKA_DELAY_SECONDS = 0.5               # Delay between producer messages (simulate real-time)

# ── PostgreSQL ────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = os.getenv("DB_PORT",     "5432")
DB_NAME     = os.getenv("DB_NAME",     "healthcare_db")
DB_USER     = os.getenv("DB_USER",     "postgres")       # ← Change this
DB_PASSWORD = os.getenv("DB_PASSWORD", "Rutuja_1107")  # ← Change this

# ── SMTP Email Alerts ─────────────────────────────────────────────────────────
SMTP_HOST       = "smtp.gmail.com"
SMTP_PORT       = 587
EMAIL_SENDER    = os.getenv("EMAIL_SENDER",   "your_email@gmail.com")   # ← Change
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD", "your_app_password")      # ← Use Gmail App Password
EMAIL_RECEIVER  = os.getenv("EMAIL_RECEIVER", "doctor@hospital.com")    # ← Change

# ── Machine Learning ──────────────────────────────────────────────────────────
MODEL_DIR               = "models/saved"  # Directory to save trained models
ISOLATION_FOREST_CONTAMINATION = 0.05    # Expected fraction of anomalies (5%)
AUTOENCODER_EPOCHS      = 30
AUTOENCODER_BATCH_SIZE  = 32
AUTOENCODER_THRESHOLD   = 0.05           # Reconstruction error threshold for anomaly

# ── Severity Thresholds ───────────────────────────────────────────────────────
# Combined anomaly score (0–1) determines severity level
SEVERITY_LOW_MAX    = 0.4   # Score < 0.4  → LOW
SEVERITY_MEDIUM_MAX = 0.7   # Score 0.4–0.7 → MEDIUM
                             # Score > 0.7  → HIGH (triggers email)

# Hard clinical thresholds — automatically HIGH regardless of ML score
CLINICAL_THRESHOLDS = {
    "heart_rate_min":       30,    # BPM too low
    "heart_rate_max":       150,   # BPM too high
    "spo2_min":             85,    # % oxygen saturation critically low
    "body_temp_min":        35.0,  # °C hypothermia
    "body_temp_max":        40.0,  # °C high fever
    "systolic_bp_min":      70,    # mmHg critically low
    "systolic_bp_max":      180,   # mmHg hypertensive crisis
    "diastolic_bp_min":     40,
    "diastolic_bp_max":     120,
}

# ── Flask API ─────────────────────────────────────────────────────────────────
FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
FLASK_DEBUG = False   # Set True for development

# ── Dataset Path ──────────────────────────────────────────────────────────────
DATASET_PATH = "data/human_vital_signs_dataset_2024.csv"

# Feature columns used by ML models
FEATURE_COLUMNS = [
    "heart_rate",
    "spo2",
    "body_temperature",
    "systolic_bp",
    "diastolic_bp",
]
