# scripts/train_models.py
# ─────────────────────────────────────────────────────────────────────────────
# One-time model training script.
#
# Loads the vital signs dataset, filters to "normal" records,
# and trains both the Isolation Forest and Autoencoder models.
#
# Run this BEFORE starting the Kafka consumer:
#   python scripts/train_models.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DATASET_PATH, FEATURE_COLUMNS, MODEL_DIR
from models.isolation_forest import IsolationForestDetector
from models.autoencoder import AutoencoderDetector


def load_training_data(filepath: str) -> np.ndarray:
    """
    Load and preprocess the dataset for model training.

    Strategy:
      - We train on ALL data (both normal and anomalous) for Isolation Forest
        since it's an unsupervised model that handles mixed data.
      - For Autoencoder, we ideally train on "Low Risk" records only, but
        if the Risk Category column isn't available, we use all data.

    Column name mapping from the Kaggle dataset:
        "Patient ID"               → patient_id
        "Heart Rate"               → heart_rate
        "Oxygen Saturation"        → spo2
        "Body Temperature"         → body_temperature
        "Systolic Blood Pressure"  → systolic_bp
        "Diastolic Blood Pressure" → diastolic_bp
    """
    print(f"[Train] Loading data from: {filepath}")

    df = pd.read_csv(filepath)

    # Rename columns to our standard names
    df = df.rename(columns={
        "Patient ID":               "patient_id",
        "Heart Rate":               "heart_rate",
        "Oxygen Saturation":        "spo2",
        "Body Temperature":         "body_temperature",
        "Systolic Blood Pressure":  "systolic_bp",
        "Diastolic Blood Pressure": "diastolic_bp",
        "Risk Category":            "risk_category",
    })

    print(f"[Train] Full dataset: {len(df)} rows")

    # ── Subset for Autoencoder training: use only "Low Risk" records ──────────
    if "risk_category" in df.columns:
        df_normal = df[df["risk_category"].str.lower() == "low risk"]
        print(f"[Train] Low Risk records for Autoencoder training: {len(df_normal)}")
    else:
        # Fallback: use all data if risk category isn't available
        df_normal = df
        print("[Train] Warning: No 'risk_category' column. Training Autoencoder on all data.")

    # Extract only the feature columns, drop rows with missing values
    X_all    = df[FEATURE_COLUMNS].dropna().values
    X_normal = df_normal[FEATURE_COLUMNS].dropna().values

    print(f"[Train] Feature matrix shapes:")
    print(f"        All data (for IsolationForest): {X_all.shape}")
    print(f"        Normal data (for Autoencoder):  {X_normal.shape}")

    return X_all, X_normal


def train_all_models():
    """
    Train and save both ML models.
    """
    print("=" * 60)
    print("  Healthcare Anomaly Detection — Model Training")
    print("=" * 60)

    # Ensure model save directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_all, X_normal = load_training_data(DATASET_PATH)

    # ── Train Isolation Forest ────────────────────────────────────────────────
    print("\n── Training Isolation Forest ──────────────────────────────")
    iso = IsolationForestDetector()
    iso.train(X_all)
    print("[Train] Isolation Forest: DONE ✅")

    # ── Train Autoencoder ─────────────────────────────────────────────────────
    print("\n── Training Autoencoder ────────────────────────────────────")
    ae = AutoencoderDetector()
    ae.train(X_normal)
    print("[Train] Autoencoder: DONE ✅")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("\n── Sanity Check: Predicting on a sample ────────────────────")

    # Test with a normal-looking sample
    normal_sample = np.array([[72, 98.0, 36.8, 118, 76]])  # Healthy vitals
    iso_score = iso.predict_score(normal_sample)
    ae_score  = ae.predict_score(normal_sample)
    print(f"Normal vitals [HR=72, SpO2=98, Temp=36.8, SBP=118, DBP=76]:")
    print(f"  Isolation Forest score: {iso_score}  (expect low)")
    print(f"  Autoencoder score:      {ae_score}   (expect low)")

    # Test with an anomalous sample
    anomalous_sample = np.array([[155, 82.0, 40.5, 185, 125]])  # Critical vitals
    iso_score2 = iso.predict_score(anomalous_sample)
    ae_score2  = ae.predict_score(anomalous_sample)
    print(f"\nAnomalous vitals [HR=155, SpO2=82, Temp=40.5, SBP=185, DBP=125]:")
    print(f"  Isolation Forest score: {iso_score2}  (expect high)")
    print(f"  Autoencoder score:      {ae_score2}   (expect high)")

    print("\n" + "=" * 60)
    print("  Training complete! Models saved to:", MODEL_DIR)
    print("  You can now start the Kafka consumer.")
    print("=" * 60)


if __name__ == "__main__":
    train_all_models()
