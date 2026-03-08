# notifications/email_alert.py
# ─────────────────────────────────────────────────────────────────────────────
# SMTP Email Alert System
#
# Sends HTML-formatted email alerts to medical staff when a patient's
# vital signs are classified as HIGH severity.
#
# Uses Gmail SMTP with TLS. For Gmail, you MUST use an "App Password"
# (not your regular password). Enable it at: myaccount.google.com/apppasswords
# ─────────────────────────────────────────────────────────────────────────────

import smtplib
import sys
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SMTP_HOST, SMTP_PORT, EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER


def build_html_email(record: dict, anomaly: dict) -> str:
    """
    Build a nicely formatted HTML email body for the HIGH severity alert.

    Args:
        record  (dict): The original vital sign record from Kafka.
        anomaly (dict): The anomaly detection results.

    Returns:
        str: Full HTML string for the email body.
    """
    timestamp = record.get("timestamp", datetime.now().isoformat())
    patient_id = record.get("patient_id", "N/A")
    combined_score = anomaly.get("combined_score", 0)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body       {{ font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }}
            .container {{ max-width: 600px; margin: auto; background: white;
                          border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .header    {{ background: #dc2626; color: white; padding: 24px; text-align: center; }}
            .header h1 {{ margin: 0; font-size: 22px; }}
            .header p  {{ margin: 4px 0 0; font-size: 14px; opacity: 0.9; }}
            .body      {{ padding: 24px; }}
            .badge     {{ display: inline-block; background: #dc2626; color: white;
                          padding: 4px 12px; border-radius: 999px; font-size: 13px;
                          font-weight: bold; margin-bottom: 16px; }}
            table      {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
            th         {{ background: #f1f5f9; text-align: left; padding: 10px;
                          font-size: 13px; color: #475569; border-bottom: 1px solid #e2e8f0; }}
            td         {{ padding: 10px; font-size: 14px; border-bottom: 1px solid #f1f5f9; }}
            tr:last-child td {{ border-bottom: none; }}
            .highlight {{ color: #dc2626; font-weight: bold; }}
            .footer    {{ background: #f8fafc; padding: 16px 24px; font-size: 12px;
                          color: #94a3b8; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <div class="header">
                <h1>🚨 CRITICAL PATIENT ALERT</h1>
                <p>AI Healthcare Monitoring System · Immediate Attention Required</p>
            </div>

            <!-- Body -->
            <div class="body">
                <span class="badge">HIGH SEVERITY</span>

                <p><strong>Patient ID:</strong> {patient_id} &nbsp;|&nbsp;
                   <strong>Detected at:</strong> {timestamp}</p>
                <p><strong>Anomaly Score:</strong>
                   <span class="highlight">{combined_score:.2f} / 1.00</span></p>

                <h3 style="margin-top:20px; margin-bottom:8px;">📊 Vital Signs at Time of Alert</h3>
                <table>
                    <tr>
                        <th>Vital Sign</th>
                        <th>Recorded Value</th>
                        <th>Normal Range</th>
                    </tr>
                    <tr>
                        <td>❤️ Heart Rate</td>
                        <td class="highlight">{record.get('heart_rate', 'N/A')} bpm</td>
                        <td>60 – 100 bpm</td>
                    </tr>
                    <tr>
                        <td>🫁 SpO2 (Oxygen Saturation)</td>
                        <td class="highlight">{record.get('spo2', 'N/A')}%</td>
                        <td>95 – 100%</td>
                    </tr>
                    <tr>
                        <td>🌡️ Body Temperature</td>
                        <td class="highlight">{record.get('body_temperature', 'N/A')} °C</td>
                        <td>36.1 – 37.2 °C</td>
                    </tr>
                    <tr>
                        <td>🩸 Systolic Blood Pressure</td>
                        <td class="highlight">{record.get('systolic_bp', 'N/A')} mmHg</td>
                        <td>90 – 120 mmHg</td>
                    </tr>
                    <tr>
                        <td>🩸 Diastolic Blood Pressure</td>
                        <td class="highlight">{record.get('diastolic_bp', 'N/A')} mmHg</td>
                        <td>60 – 80 mmHg</td>
                    </tr>
                </table>

                <h3 style="margin-top:20px; margin-bottom:8px;">🤖 ML Detection Scores</h3>
                <table>
                    <tr><th>Model</th><th>Score</th></tr>
                    <tr>
                        <td>Isolation Forest</td>
                        <td>{anomaly.get('isolation_forest_score', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Autoencoder (TensorFlow)</td>
                        <td>{anomaly.get('autoencoder_score', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td><strong>Combined Score</strong></td>
                        <td class="highlight"><strong>{combined_score:.4f}</strong></td>
                    </tr>
                </table>

                <p style="margin-top: 20px; padding: 12px; background:#fef2f2;
                   border-left: 4px solid #dc2626; border-radius: 4px; font-size: 14px;">
                    ⚠️ <strong>Action Required:</strong> Please review this patient immediately.
                    This is an automated alert from the AI monitoring system.
                </p>
            </div>

            <!-- Footer -->
            <div class="footer">
                AI-Driven Healthcare Anomaly Detection System &nbsp;·&nbsp;
                This is an automated message. Do not reply.
            </div>
        </div>
    </body>
    </html>
    """
    return html


def send_high_alert_email(record: dict, anomaly: dict) -> bool:
    """
    Send a HIGH severity alert email to the configured recipient.

    Args:
        record  (dict): Original vital sign record from Kafka.
        anomaly (dict): Anomaly detection result (scores, severity).

    Returns:
        bool: True if email was sent successfully, False otherwise.
    """
    patient_id = record.get("patient_id", "Unknown")

    # ── Build the email ───────────────────────────────────────────────────────
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 CRITICAL: HIGH Severity Alert — Patient {patient_id}"
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECEIVER

    # Attach HTML body
    html_body = build_html_email(record, anomaly)
    msg.attach(MIMEText(html_body, "html"))

    # ── Send via SMTP (Gmail with TLS) ─────────────────────────────────────────
    try:
        print(f"[Email] Sending HIGH alert for Patient {patient_id} to {EMAIL_RECEIVER}...")

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()                         # Identify ourselves to the server
            server.starttls()                     # Upgrade connection to TLS (secure)
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)   # Authenticate
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

        print(f"[Email] ✅ Alert sent successfully for Patient {patient_id}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("[Email] ❌ Authentication failed. Check EMAIL_SENDER and EMAIL_PASSWORD in settings.")
        print("         For Gmail, use an App Password: myaccount.google.com/apppasswords")
        return False

    except smtplib.SMTPException as e:
        print(f"[Email] ❌ SMTP error: {e}")
        return False

    except Exception as e:
        print(f"[Email] ❌ Unexpected error sending alert: {e}")
        return False
