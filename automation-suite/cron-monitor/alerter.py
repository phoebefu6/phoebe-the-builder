from __future__ import annotations

"""Email alerting for overdue jobs.

Sends via SMTP when `SMTP_HOST` (+ creds) are in the environment; otherwise runs
in **dry-run** mode and prints the alert. That means the service works out of the
box with no secrets - and you opt into real email by setting env vars.
"""

import os
import smtplib
from email.message import EmailMessage
from typing import List


def _smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST"))


def format_alert(job_names: List[str]) -> str:
    bullets = "\n".join(f"  - {n}" for n in job_names)
    return f"The following scheduled jobs are OVERDUE:\n{bullets}\n\nInvestigate now."


def send_alert(job_names: List[str], to_addr: str = "oncall@example.com") -> dict:
    """Send (or dry-run) an alert email for the given overdue jobs."""
    if not job_names:
        return {"sent": False, "reason": "no overdue jobs"}

    body = format_alert(job_names)
    subject = f"[cron-monitor] {len(job_names)} job(s) overdue"

    if not _smtp_configured():
        print("=" * 50)
        print(f"[DRY-RUN EMAIL] to={to_addr}  subject={subject}")
        print(body)
        print("=" * 50)
        return {"sent": False, "dry_run": True, "subject": subject, "jobs": job_names}

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("SMTP_FROM", "cron-monitor@example.com")
    msg["To"] = to_addr
    msg.set_content(body)

    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port) as server:
        server.starttls()
        user, pw = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
        if user and pw:
            server.login(user, pw)
        server.send_message(msg)
    return {"sent": True, "subject": subject, "jobs": job_names}
