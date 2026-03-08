# kafka/producer.py — Uses confluent-kafka (compatible with Kafka 4.x)
import json
import time
import sys
import os
import pandas as pd
from confluent_kafka import Producer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import KAFKA_BROKER, KAFKA_TOPIC, KAFKA_DELAY_SECONDS, DATASET_PATH


def load_dataset(filepath):
    print(f"[Producer] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    column_map = {
        "Patient ID": "patient_id", "Heart Rate": "heart_rate",
        "Oxygen Saturation": "spo2", "Body Temperature": "body_temperature",
        "Systolic Blood Pressure": "systolic_bp",
        "Diastolic Blood Pressure": "diastolic_bp", "Timestamp": "timestamp",
    }
    df = df.rename(columns=column_map)
    df = df[list(column_map.values())].dropna()
    print(f"[Producer] Dataset loaded: {len(df)} records found.")
    return df


def delivery_report(err, msg):
    if err is not None:
        print(f"[Producer ERROR] Delivery failed: {err}")


def create_producer():
    print(f"[Producer] Connecting to Kafka broker at {KAFKA_BROKER}...")
    producer = Producer({'bootstrap.servers': KAFKA_BROKER, 'acks': 'all', 'retries': 3})
    print(f"[Producer] Connected. Streaming to topic: '{KAFKA_TOPIC}'")
    return producer


def stream_dataset(producer, df):
    total = len(df)
    for idx, row in df.iterrows():
        message = {
            "patient_id": int(row["patient_id"]),
            "heart_rate": float(row["heart_rate"]),
            "spo2": float(row["spo2"]),
            "body_temperature": float(row["body_temperature"]),
            "systolic_bp": float(row["systolic_bp"]),
            "diastolic_bp": float(row["diastolic_bp"]),
            "timestamp": str(row["timestamp"]),
        }
        producer.produce(KAFKA_TOPIC, key=str(message["patient_id"]),
                        value=json.dumps(message).encode("utf-8"), callback=delivery_report)
        producer.poll(0)
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"[Producer] Sent {idx+1}/{total} | Patient {message['patient_id']} | HR={message['heart_rate']} | SpO2={message['spo2']}")
        time.sleep(KAFKA_DELAY_SECONDS)
    producer.flush()
    print("[Producer] All records sent.")


if __name__ == "__main__":
    try:
        df = load_dataset(DATASET_PATH)
        producer = create_producer()
        stream_dataset(producer, df)
    except KeyboardInterrupt:
        print("\n[Producer] Stopped by user.")
    except Exception as e:
        print(f"[Producer ERROR] {e}")
        sys.exit(1)
    finally:
        print("[Producer] Shutting down.")
