from __future__ import annotations

"""FastAPI service: collect standup updates, return the daily digest.

    POST /standup   {"name","yesterday","today","blockers","day"?}   submit/replace
    GET  /updates?day=YYYY-MM-DD                                     raw updates
    GET  /digest?day=YYYY-MM-DD&ai=true                              digest (md)
    GET  /                                                           submit form + digest

Run:  uvicorn api:app --reload
Set ANTHROPIC_API_KEY to enable the AI narrative digest (?ai=true); otherwise the
deterministic template is used.
"""

from datetime import date
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from standup import StandupStore, Update, build_digest, summarize_with_claude

app = FastAPI(title="Daily Standup Bot", version="1.0.0")
store = StandupStore()


class UpdateRequest(BaseModel):
    name: str
    yesterday: str = ""
    today: str = ""
    blockers: str = ""
    day: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/standup")
def submit(req: UpdateRequest) -> Dict[str, Any]:
    store.submit(Update(req.name, req.yesterday, req.today, req.blockers), day=req.day)
    day = req.day or date.today().isoformat()
    return {"ok": True, "day": day, "updates_today": len(store.get(day))}


@app.get("/updates")
def updates(day: Optional[str] = None) -> JSONResponse:
    ups = store.get(day)
    return JSONResponse({"day": day or date.today().isoformat(),
                         "updates": [u.__dict__ for u in ups]})


@app.get("/digest", response_class=PlainTextResponse)
def digest(day: Optional[str] = None, ai: bool = False) -> str:
    ups = store.get(day)
    return summarize_with_claude(ups, day) if ai else build_digest(ups, day)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html><html><head><title>Daily Standup Bot</title>
<style>body{font-family:system-ui;max-width:720px;margin:2rem auto;padding:0 1rem}
input,textarea{width:100%;padding:.5rem;margin:.25rem 0;box-sizing:border-box;font-size:.95rem}
textarea{height:48px}button{padding:.5rem 1rem;font-size:1rem;cursor:pointer}
pre{background:#f4f4f4;padding:1rem;border-radius:8px;white-space:pre-wrap}</style></head><body>
<h1>🧍 Daily Standup Bot</h1>
<p>One place for the team's standup. Submit yours, then refresh the digest.</p>
<input id="name" placeholder="your name">
<textarea id="yesterday" placeholder="Yesterday..."></textarea>
<textarea id="today" placeholder="Today..."></textarea>
<textarea id="blockers" placeholder="Blockers (or 'none')"></textarea>
<button onclick="submit()">Submit</button> <button onclick="refresh()">Refresh digest</button>
<h3>Today's digest</h3><pre id="digest">-</pre>
<script>
async function submit(){
  await fetch('/standup',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:name.value,yesterday:yesterday.value,today:today.value,blockers:blockers.value})});
  name.value=yesterday.value=today.value=blockers.value=''; refresh();
}
async function refresh(){
  const r=await fetch('/digest'); document.getElementById('digest').textContent=await r.text();
}
refresh();
</script></body></html>"""
