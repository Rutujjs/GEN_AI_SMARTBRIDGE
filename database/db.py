# database/db.py
# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL connection helper.
# Provides a reusable function to get a database connection using psycopg2.
# ─────────────────────────────────────────────────────────────────────────────

import psycopg2
import psycopg2.extras  # Allows dictionary-style row access
import sys
import os

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def get_connection():
    """
    Create and return a new PostgreSQL database connection.

    Returns:
        psycopg2.connection: Active database connection object.

    Raises:
        psycopg2.OperationalError: If connection fails (wrong credentials, DB not running, etc.)
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            # cursor_factory makes all queries return dicts instead of tuples
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"[DB ERROR] Could not connect to PostgreSQL: {e}")
        print("→ Make sure PostgreSQL is running and credentials in config/settings.py are correct.")
        raise


def insert_vital(conn, record: dict) -> int:
    """
    Insert a single vital sign record into the 'vitals' table.

    Args:
        conn: Active psycopg2 connection.
        record (dict): Keys = column names (patient_id, heart_rate, spo2, etc.)

    Returns:
        int: The auto-generated ID of the inserted row.
    """
    sql = """
        INSERT INTO vitals
            (patient_id, heart_rate, spo2, body_temperature, systolic_bp, diastolic_bp, timestamp)
        VALUES
            (%(patient_id)s, %(heart_rate)s, %(spo2)s, %(body_temperature)s,
             %(systolic_bp)s, %(diastolic_bp)s, %(timestamp)s)
        RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, record)
        vital_id = cur.fetchone()["id"]
        conn.commit()
    return vital_id


def insert_anomaly_result(conn, result: dict) -> int:
    """
    Insert an anomaly detection result into the 'anomaly_results' table.

    Args:
        conn: Active psycopg2 connection.
        result (dict): Contains vital_id, scores, severity, is_anomaly, and vital snapshot.

    Returns:
        int: The auto-generated ID of the inserted anomaly record.
    """
    sql = """
        INSERT INTO anomaly_results
            (vital_id, patient_id, isolation_forest_score, autoencoder_score,
             combined_score, severity, is_anomaly,
             heart_rate, spo2, body_temperature, systolic_bp, diastolic_bp)
        VALUES
            (%(vital_id)s, %(patient_id)s, %(isolation_forest_score)s, %(autoencoder_score)s,
             %(combined_score)s, %(severity)s, %(is_anomaly)s,
             %(heart_rate)s, %(spo2)s, %(body_temperature)s, %(systolic_bp)s, %(diastolic_bp)s)
        RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, result)
        anomaly_id = cur.fetchone()["id"]
        conn.commit()
    return anomaly_id


def log_email_alert(conn, anomaly_id: int, patient_id: int, severity: str,
                    email_sent_to: str, success: bool):
    """
    Log a sent email alert into the 'email_alerts' table.
    """
    sql = """
        INSERT INTO email_alerts (anomaly_id, patient_id, severity, email_sent_to, success)
        VALUES (%s, %s, %s, %s, %s);
    """
    with conn.cursor() as cur:
        cur.execute(sql, (anomaly_id, patient_id, severity, email_sent_to, success))
        conn.commit()
