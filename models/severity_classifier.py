# models/severity_classifier.py
# ─────────────────────────────────────────────────────────────────────────────
# Severity Classifier: Converts an anomaly score (0–1) into a severity level.
#
# Two approaches combined:
#   1. Score-based: Compare combined_score against thresholds
#   2. Clinical-based: Hard rules for life-threatening vital ranges
#
# The clinical check overrides the score — even if the ML score is LOW,
# a heart rate of 180 bpm should still be flagged as HIGH.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SEVERITY_LOW_MAX, SEVERITY_MEDIUM_MAX, CLINICAL_THRESHOLDS


def classify_severity(combined_score: float) -> str:
    """
    Classify anomaly severity based on the combined ML score.

    Thresholds (configurable in config/settings.py):
        score < 0.4  → LOW    (minor deviation, monitor)
        score 0.4–0.7 → MEDIUM (notable anomaly, investigate)
        score > 0.7  → HIGH   (critical anomaly, alert immediately)

    Args:
        combined_score (float): Weighted average of Isolation Forest + Autoencoder scores.

    Returns:
        str: "LOW", "MEDIUM", or "HIGH"
    """
    if combined_score < SEVERITY_LOW_MAX:
        return "LOW"
    elif combined_score < SEVERITY_MEDIUM_MAX:
        return "MEDIUM"
    else:
        return "HIGH"


def is_clinically_critical(record: dict) -> bool:
    """
    Check if any vital sign falls outside safe clinical boundaries.

    These are hard thresholds based on medical standards:
      - Heart Rate: Normal range 60–100 bpm; critical if <30 or >150
      - SpO2: Should be >95%; dangerous if <85%
      - Body Temp: Normal 36.5–37.5°C; critical if <35 or >40
      - Systolic BP: Normal 90–120; critical if <70 or >180
      - Diastolic BP: Normal 60–80; critical if <40 or >120

    This function catches cases where ML models might miss obvious dangers.

    Args:
        record (dict): A single vital sign reading with these keys:
                       heart_rate, spo2, body_temperature, systolic_bp, diastolic_bp

    Returns:
        bool: True if ANY vital is in critical range (should be HIGH severity).
    """
    t = CLINICAL_THRESHOLDS  # Shorthand alias

    # ── Heart Rate ─────────────────────────────────────────────────────────────
    hr = record.get("heart_rate", 0)
    if hr < t["heart_rate_min"] or hr > t["heart_rate_max"]:
        return True

    # ── Oxygen Saturation (SpO2) ────────────────────────────────────────────────
    spo2 = record.get("spo2", 100)
    if spo2 < t["spo2_min"]:
        return True  # Hypoxemia — very dangerous

    # ── Body Temperature ────────────────────────────────────────────────────────
    temp = record.get("body_temperature", 37)
    if temp < t["body_temp_min"] or temp > t["body_temp_max"]:
        return True

    # ── Systolic Blood Pressure ─────────────────────────────────────────────────
    sbp = record.get("systolic_bp", 120)
    if sbp < t["systolic_bp_min"] or sbp > t["systolic_bp_max"]:
        return True

    # ── Diastolic Blood Pressure ────────────────────────────────────────────────
    dbp = record.get("diastolic_bp", 80)
    if dbp < t["diastolic_bp_min"] or dbp > t["diastolic_bp_max"]:
        return True

    return False  # All vitals are within safe ranges


def get_severity_color(severity: str) -> str:
    """
    Returns a hex color code for displaying severity in the frontend.

    Args:
        severity (str): "LOW", "MEDIUM", or "HIGH"

    Returns:
        str: Hex color string.
    """
    color_map = {
        "LOW":    "#22c55e",  # Green
        "MEDIUM": "#f59e0b",  # Amber/Orange
        "HIGH":   "#ef4444",  # Red
    }
    return color_map.get(severity, "#6b7280")  # Default: gray


def explain_severity(record: dict, severity: str) -> str:
    """
    Generate a human-readable explanation of why severity was assigned.

    Useful for displaying in the frontend dashboard or email alerts.

    Args:
        record (dict): Vital sign readings.
        severity (str): Assigned severity level.

    Returns:
        str: Explanation message.
    """
    issues = []

    hr = record.get("heart_rate", 0)
    spo2 = record.get("spo2", 100)
    temp = record.get("body_temperature", 37)
    sbp = record.get("systolic_bp", 120)
    dbp = record.get("diastolic_bp", 80)

    if hr < 60:
        issues.append(f"Bradycardia (HR={hr} bpm, low)")
    elif hr > 100:
        issues.append(f"Tachycardia (HR={hr} bpm, high)")

    if spo2 < 95:
        issues.append(f"Low oxygen saturation (SpO2={spo2}%)")

    if temp > 38.5:
        issues.append(f"Fever (Temp={temp}°C)")
    elif temp < 36.0:
        issues.append(f"Hypothermia risk (Temp={temp}°C)")

    if sbp > 140:
        issues.append(f"Hypertension (SBP={sbp} mmHg)")
    elif sbp < 90:
        issues.append(f"Hypotension (SBP={sbp} mmHg)")

    if not issues:
        return f"ML model detected statistical anomaly ({severity})"

    return f"{severity} Alert: " + "; ".join(issues)
