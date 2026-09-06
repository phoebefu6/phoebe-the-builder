from __future__ import annotations

"""FastAPI service: register jobs, receive heartbeats, surface overdue jobs,
and fire email alerts.

    POST /jobs        {"name","interval_seconds","grace_seconds"}  register
    POST /ping/{name}                                              heartbeat
    GET  /status                                                   all jobs + state
    POST /check                                                    evaluate + alert overdue
    GET  /                                                         HTML dashboard

Run:  uvicorn api:app --reload
A scheduler (or external cron) should hit POST /check on an interval.
"""

import time
from typing import Any, Dict, Optional

from alerter import send_alert
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from monitor import Monitor
from pydantic import BaseModel

app = FastAPI(title="Cron Job Monitor", version="1.0.0")
mon = Monitor()


class RegisterRequest(BaseModel):
    name: str
    interval_seconds: float
    grace_seconds: float = 60.0


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs")
def register(req: RegisterRequest) -> Dict[str, Any]:
    job = mon.register(req.name, req.interval_seconds, req.grace_seconds)
    return {"registered": job.name, "interval_seconds": job.interval_seconds, "grace_seconds": job.grace_seconds}


@app.post("/ping/{name}")
def ping(name: str) -> Dict[str, Any]:
    job = mon.ping(name, now=time.time())
    return {"pinged": name, "last_ping": job.last_ping}


@app.get("/status")
def status() -> Dict[str, Any]:
    now = time.time()
    rows = mon.status(now)
    return {"now": now, "jobs": rows, "overdue_count": sum(r["state"] == "overdue" for r in rows)}


@app.post("/check")
def check(to: Optional[str] = "oncall@example.com") -> Dict[str, Any]:
    now = time.time()
    to_fire = mon.alerts_to_fire(now)
    result = send_alert([j.name for j in to_fire], to_addr=to or "oncall@example.com")
    return {"checked_at": now, "fired_for": [j.name for j in to_fire], "alert": result}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    now = time.time()
    rows = mon.status(now)
    badge = {"healthy": "🟢", "overdue": "🔴", "pending": "⚪"}
    body = "".join(
        f"<tr><td>{badge.get(r['state'],'')} {r['name']}</td><td>{r['state']}</td>"
        f"<td>{r['interval_seconds']:g}s</td>"
        f"<td>{'-' if r['seconds_since_ping'] is None else str(r['seconds_since_ping']) + 's ago'}</td></tr>"
        for r in rows
    )
    return f"""<!doctype html><html><head><title>Cron Monitor</title>
<meta http-equiv="refresh" content="10">
<style>body{{font-family:system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}}
table{{width:100%;border-collapse:collapse}}td,th{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}
code{{background:#f4f4f4;padding:.1rem .3rem;border-radius:4px}}</style></head><body>
<h1>🕒 Cron Job Monitor</h1>
<p>{len(rows)} job(s) registered. Auto-refresh every 10s.</p>
<table><tr><th>Job</th><th>State</th><th>Interval</th><th>Last ping</th></tr>{body}</table>
<p>Register: <code>POST /jobs</code> · Heartbeat: <code>POST /ping/&lt;name&gt;</code> · Evaluate: <code>POST /check</code></p>
</body></html>"""
