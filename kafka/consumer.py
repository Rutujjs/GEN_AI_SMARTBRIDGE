# kafka/consumer.py — Uses confluent-kafka (compatible with Kafka 4.x)
import json
import sys
import os
import numpy as np
from confluent_kafka import Consumer, KafkaError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import KAFKA_BROKER, KAFKA_TOPIC, KAFKA_GROUP_ID, FEATURE_COLUMNS
from database.db import get_connection, insert_vital, insert_anomaly_result, log_email_alert
from models.isolation_forest import IsolationForestDetector
from models.autoencoder import AutoencoderDetector
from models.severity_classifier import classify_severity, is_clinically_critical
from notifications.email_alert import send_high_alert_email


def create_consumer():
    print(f"[Consumer] Connecting to Kafka at {KAFKA_BROKER}...")
    consumer = Consumer({
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': KAFKA_GROUP_ID,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,
    })
    consumer.subscribe([KAFKA_TOPIC])
    print(f"[Consumer] Listening to topic: '{KAFKA_TOPIC}'")
    return consumer


def load_models():
    print("[Consumer] Loading trained ML models...")
    iso_detector = IsolationForestDetector()
    iso_detector.load()
    ae_detector = AutoencoderDetector()
    ae_detector.load()
    print("[Consumer] Models loaded successfully.")
    return iso_detector, ae_detector


def extract_features(record):
    features = [record[col] for col in FEATURE_COLUMNS]
    return np.array(features).reshape(1, -1)


def process_message(record, iso, ae, conn):
    vital_id = insert_vital(conn, record)
    features = extract_features(record)
    iso_score = iso.predict_score(features)
    ae_score = ae.predict_score(features)
    combined_score = round(0.4 * iso_score + 0.6 * ae_score, 4)
    forced_high = is_clinically_critical(record)
    severity = "HIGH" if forced_high else classify_severity(combined_score)
    is_anomaly = severity in ("MEDIUM", "HIGH")

    anomaly_record = {
        "vital_id": vital_id, "patient_id": record["patient_id"],
        "isolation_forest_score": round(iso_score, 4),
        "autoencoder_score": round(ae_score, 4),
        "combined_score": combined_score, "severity": severity,
        "is_anomaly": is_anomaly, "heart_rate": record["heart_rate"],
        "spo2": record["spo2"], "body_temperature": record["body_temperature"],
        "systolic_bp": record["systolic_bp"], "diastolic_bp": record["diastolic_bp"],
    }
    anomaly_id = insert_anomaly_result(conn, anomaly_record)

    if severity == "HIGH":
        email_success = send_high_alert_email(record, anomaly_record)
        log_email_alert(conn, anomaly_id=anomaly_id, patient_id=record["patient_id"],
                       severity=severity, email_sent_to=os.getenv("EMAIL_RECEIVER", ""), success=email_success)

    return {"patient_id": record["patient_id"], "severity": severity,
            "combined_score": combined_score, "is_anomaly": is_anomaly}


def run_consumer():
    iso, ae = load_models()
    conn = get_connection()
    print("[Consumer] Database connected.")
    consumer = create_consumer()
    processed = 0
    print("[Consumer] Waiting for messages... (Press Ctrl+C to stop)\n")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"[Consumer ERROR] {msg.error()}")
                    break

            record = json.loads(msg.value().decode("utf-8"))
            try:
                result = process_message(record, iso, ae, conn)
                processed += 1
                severity_icon = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}.get(result["severity"], "?")
                print(f"{severity_icon} [{processed}] Patient {result['patient_id']} | "
                      f"Severity: {result['severity']} | Score: {result['combined_score']}")
            except Exception as e:
                print(f"[Consumer WARNING] Failed to process record: {e}")
                try:
                    conn = get_connection()
                except Exception:
                    pass

    except KeyboardInterrupt:
        print(f"\n[Consumer] Stopped. Total records processed: {processed}")
    finally:
        conn.close()
        consumer.close()


if __name__ == "__main__":
    run_consumer()
