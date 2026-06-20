from __future__ import annotations

"""FastAPI microservice: format and lint SQL.

    POST /format  {"sql": "...", "keyword_case": "upper", "indent_width": 2}
    POST /lint     {"sql": "..."}
    POST /analyze  {"sql": "..."}   -> format + lint together
    GET  /                          -> built-in test form

Run:  uvicorn api:app --reload
"""

from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from formatter import analyze, format_sql, lint_sql

app = FastAPI(title="SQL Formatter & Linter", version="1.0.0")


class FormatRequest(BaseModel):
    sql: str
    keyword_case: str = "upper"
    indent_width: int = 2
    reindent: bool = True


class SqlRequest(BaseModel):
    sql: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/format")
def do_format(req: FormatRequest) -> Dict[str, str]:
    return {"formatted": format_sql(req.sql, keyword_case=req.keyword_case,
                                    indent_width=req.indent_width, reindent=req.reindent)}


@app.post("/lint")
def do_lint(req: SqlRequest) -> Dict[str, Any]:
    issues = lint_sql(req.sql)
    return {"issue_count": len(issues), "issues": [i.__dict__ for i in issues]}


@app.post("/analyze")
def do_analyze(req: SqlRequest) -> Dict[str, Any]:
    return analyze(req.sql)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html><html><head><title>SQL Formatter</title>
<style>body{font-family:system-ui;max-width:780px;margin:2rem auto;padding:0 1rem}
textarea{width:100%;height:160px;font-family:monospace;font-size:.9rem}
pre{background:#f4f4f4;padding:1rem;border-radius:6px;white-space:pre-wrap}
button{padding:.5rem 1rem;font-size:1rem;cursor:pointer}
.err{color:#b00}.warning{color:#b67000}.style{color:#557}</style></head><body>
<h1>🧹 SQL Formatter & Linter</h1>
<textarea id="sql">select id,name,email from users where status='active' and created_at>'2026-01-01'</textarea><br><br>
<button onclick="run()">Format + Lint</button>
<h3>Formatted</h3><pre id="out">-</pre>
<h3>Issues</h3><div id="issues">-</div>
<script>
async function run(){
  const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sql:document.getElementById('sql').value})});
  const d=await r.json();
  document.getElementById('out').textContent=d.formatted||'(empty)';
  const box=document.getElementById('issues');
  if(!d.issues.length){box.innerHTML='<p>✅ Clean - no issues.</p>';return;}
  box.innerHTML=d.issues.map(i=>`<p class="${i.severity}">[${i.severity}] <b>${i.rule}</b>: ${i.message}</p>`).join('');
}
</script></body></html>"""
