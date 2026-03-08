# api/app.py
# ─────────────────────────────────────────────────────────────────────────────
# Flask REST API Backend
#
# Provides endpoints for the frontend dashboard to fetch monitoring data.
# Also serves the HTML frontend at the root URL ("/").
#
# Run: python api/app.py
# Dashboard: http://localhost:5000
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS  # Allow frontend to call API from different origin
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from database.db import get_connection

# ── Flask app setup ────────────────────────────────────────────────────────────
# template_folder points to our frontend HTML templates
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "static"),
)
CORS(app)  # Enable Cross-Origin Resource Sharing


# ─────────────────────────────────────────────────────────────────────────────
# ── Helper ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def serialize_row(row) -> dict:
    """
    Convert a psycopg2 RealDictRow to a JSON-serializable Python dict.
    Handles datetime objects (convert to ISO string).
    """
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


# ─────────────────────────────────────────────────────────────────────────────
# ── Routes ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard HTML page."""
    return render_template("index.html")


@app.route("/api/health")
def health_check():
    """
    Simple health check endpoint.
    Returns 200 OK with a status message.
    Useful for checking if the API is running.
    """
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/vitals")
def get_vitals():
    """
    GET /api/vitals?limit=50&patient_id=<optional>

    Returns the most recent vital sign records.

    Query params:
        limit      (int, default 50): Number of records to return.
        patient_id (int, optional):   Filter by specific patient.
    """
    limit      = request.args.get("limit",      50,   type=int)
    patient_id = request.args.get("patient_id", None, type=int)

    try:
        conn = get_connection()
        with conn.cursor() as cur:
            if patient_id:
                cur.execute(
                    """
                    SELECT * FROM vitals
                    WHERE patient_id = %s
                    ORDER BY received_at DESC
                    LIMIT %s
                    """,
                    (patient_id, limit)
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM vitals
                    ORDER BY received_at DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
            rows = cur.fetchall()
        conn.close()

        data = [serialize_row(r) for r in rows]
        return jsonify({"success": True, "count": len(data), "data": data})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/anomalies")
def get_anomalies():
    """
    GET /api/anomalies?limit=50&severity=<LOW|MEDIUM|HIGH>

    Returns the most recent anomaly detection results.

    Query params:
        limit    (int, default 50):    Number of records to return.
        severity (str, optional):      Filter by severity level.
    """
    limit    = request.args.get("limit",    50,   type=int)
    severity = request.args.get("severity", None, type=str)

    try:
        conn = get_connection()
        with conn.cursor() as cur:
            if severity:
                cur.execute(
                    """
                    SELECT * FROM anomaly_results
                    WHERE severity = %s
                    ORDER BY detected_at DESC
                    LIMIT %s
                    """,
                    (severity.upper(), limit)
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM anomaly_results
                    ORDER BY detected_at DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
            rows = cur.fetchall()
        conn.close()

        data = [serialize_row(r) for r in rows]
        return jsonify({"success": True, "count": len(data), "data": data})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stats")
def get_stats():
    """
    GET /api/stats

    Returns summary statistics:
      - Total records processed
      - Count per severity level (LOW, MEDIUM, HIGH)
      - Total anomalies detected
      - Latest 5 HIGH alerts
    """
    try:
        conn = get_connection()
        with conn.cursor() as cur:

            # ── Total vitals processed ────────────────────────────────────────
            cur.execute("SELECT COUNT(*) as total FROM vitals;")
            total_vitals = cur.fetchone()["total"]

            # ── Count by severity ─────────────────────────────────────────────
            cur.execute("""
                SELECT severity, COUNT(*) as count
                FROM anomaly_results
                GROUP BY severity;
            """)
            severity_rows = cur.fetchall()
            severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
            for row in severity_rows:
                severity_counts[row["severity"]] = row["count"]

            # ── Total anomalies (MEDIUM + HIGH) ───────────────────────────────
            total_anomalies = severity_counts["MEDIUM"] + severity_counts["HIGH"]

            # ── Latest HIGH alerts ────────────────────────────────────────────
            cur.execute("""
                SELECT patient_id, combined_score, heart_rate, spo2,
                       body_temperature, systolic_bp, diastolic_bp, detected_at
                FROM anomaly_results
                WHERE severity = 'HIGH'
                ORDER BY detected_at DESC
                LIMIT 5;
            """)
            high_alerts = [serialize_row(r) for r in cur.fetchall()]

            # ── Recent anomaly trend (last 20 records) ────────────────────────
            cur.execute("""
                SELECT severity, combined_score, detected_at
                FROM anomaly_results
                ORDER BY detected_at DESC
                LIMIT 20;
            """)
            trend = [serialize_row(r) for r in cur.fetchall()]

        conn.close()

        return jsonify({
            "success":        True,
            "total_vitals":   total_vitals,
            "total_anomalies": total_anomalies,
            "severity_counts": severity_counts,
            "high_alerts":    high_alerts,
            "trend":          trend,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/patient/<int:patient_id>")
def get_patient_history(patient_id: int):
    """
    GET /api/patient/<patient_id>

    Returns complete history for a specific patient:
    all their vitals and anomaly results, most recent first.
    """
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # Vitals history
            cur.execute(
                "SELECT * FROM vitals WHERE patient_id = %s ORDER BY timestamp DESC LIMIT 20",
                (patient_id,)
            )
            vitals = [serialize_row(r) for r in cur.fetchall()]

            # Anomaly history
            cur.execute(
                "SELECT * FROM anomaly_results WHERE patient_id = %s ORDER BY detected_at DESC LIMIT 20",
                (patient_id,)
            )
            anomalies = [serialize_row(r) for r in cur.fetchall()]

        conn.close()
        return jsonify({
            "success":   True,
            "patient_id": patient_id,
            "vitals":    vitals,
            "anomalies": anomalies,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ── Entry Point ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[API] Starting Flask server at http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"[API] Dashboard: http://localhost:{FLASK_PORT}")
    print("[API] Press Ctrl+C to stop.\n")
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
    )
