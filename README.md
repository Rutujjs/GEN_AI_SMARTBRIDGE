# 🏥 AI-Driven Healthcare Anomaly Detection System

A real-time healthcare monitoring system that detects abnormal patient vital signs using machine learning and generates automated alerts.

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SYSTEM OVERVIEW                             │
│                                                                     │
│  CSV Dataset ──► Kafka Producer ──► Kafka Topic (vitals)           │
│                                           │                         │
│                                    Kafka Consumer                   │
│                                           │                         │
│                          ┌────────────────┼────────────────┐        │
│                          ▼                ▼                ▼        │
│                   Isolation Forest   Autoencoder     Severity       │
│                   (Scikit-learn)   (TensorFlow)   Classifier        │
│                          └────────────────┼────────────────┘        │
│                                           │                         │
│                                    PostgreSQL DB                    │
│                                           │                         │
│                                    Flask REST API                   │
│                                           │                         │
│                          ┌────────────────┴────────────────┐        │
│                          ▼                                  ▼       │
│                  Frontend Dashboard                  SMTP Email     │
│               (HTML + Tailwind + Chart.js)       (HIGH alerts only) │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
healthcare-anomaly-detection/
│
├── config/
│   └── settings.py            # All configuration (Kafka, DB, SMTP, thresholds)
│
├── kafka/
│   ├── producer.py            # Reads CSV, streams records to Kafka
│   └── consumer.py            # Receives messages, runs ML, stores results
│
├── models/
│   ├── isolation_forest.py    # Isolation Forest anomaly detector
│   ├── autoencoder.py         # TensorFlow Autoencoder model
│   └── severity_classifier.py # Classifies anomaly scores → LOW/MEDIUM/HIGH
│
├── database/
│   ├── db.py                  # PostgreSQL connection helper
│   └── schema.sql             # Table definitions
│
├── api/
│   └── app.py                 # Flask REST API
│
├── notifications/
│   └── email_alert.py         # SMTP email sender for HIGH severity
│
├── frontend/
│   ├── templates/
│   │   └── index.html         # Dashboard (Tailwind + Chart.js)
│   └── static/
│       └── js/
│           └── dashboard.js   # Frontend JS logic
│
├── scripts/
│   └── train_models.py        # One-time model training script
│
├── data/
│   └── (place dataset CSV here)
│
├── requirements.txt
└── README.md
```

---

## 🔧 Prerequisites

Install these before running:

1. **Python 3.9** — https://www.python.org/downloads/release/python-390/
2. **Apache Kafka + ZooKeeper** — https://kafka.apache.org/downloads
3. **PostgreSQL** — https://www.postgresql.org/download/

---

## 🚀 Step-by-Step Setup & Run Instructions

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Set up PostgreSQL
```bash
# Login to PostgreSQL
psql -U postgres

# Create the database
CREATE DATABASE healthcare_db;

# Connect and create tables
\c healthcare_db
\i database/schema.sql
```

### Step 3 — Configure settings
Edit `config/settings.py` and update:
- PostgreSQL credentials (DB_USER, DB_PASSWORD)
- SMTP credentials (EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER)
- Kafka broker address if different

### Step 4 — Start ZooKeeper (Terminal 1)
```bash
# From your Kafka installation directory:
bin/zookeeper-server-start.sh config/zookeeper.properties
# On Windows:
bin\windows\zookeeper-server-start.bat config\zookeeper.properties
```

### Step 5 — Start Kafka Broker (Terminal 2)
```bash
bin/kafka-server-start.sh config/server.properties
# On Windows:
bin\windows\kafka-server-start.bat config\server.properties
```

### Step 6 — Create Kafka Topic (Terminal 3)
```bash
bin/kafka-topics.sh --create --topic vitals --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

### Step 7 — Train ML Models (Terminal 3)
```bash
python scripts/train_models.py
```
This trains and saves both Isolation Forest and Autoencoder models.

### Step 8 — Start Kafka Consumer (Terminal 3)
```bash
python kafka/consumer.py
```

### Step 9 — Start Kafka Producer (Terminal 4)
```bash
python kafka/producer.py
```

### Step 10 — Start Flask API + Dashboard (Terminal 5)
```bash
python api/app.py
```

### Step 11 — Open Dashboard
Navigate to: **http://localhost:5000**

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/vitals` | Latest 50 vital sign records |
| GET | `/api/anomalies` | Latest 50 anomaly detections |
| GET | `/api/stats` | Summary stats (counts per severity) |
| GET | `/api/health` | Health check |

---

## ⚠️ Severity Levels

| Level | Trigger Condition |
|-------|------------------|
| LOW | Anomaly score < 0.4 |
| MEDIUM | Anomaly score 0.4 – 0.7 |
| HIGH | Anomaly score > 0.7 → triggers email |

HIGH severity also checks hard clinical thresholds (e.g., HR > 150, SpO2 < 85%).
