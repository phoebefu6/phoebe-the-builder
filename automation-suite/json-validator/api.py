from __future__ import annotations

"""FastAPI microservice: validate JSON payloads against a JSON Schema.

    POST /validate  {"schema": {...}, "data": {...}}  -> {valid, errors[]}
    POST /infer     {"sample": {...}}                 -> inferred Draft-7 schema
    GET  /health
    GET  /          -> tiny built-in test form

Run:  uvicorn api:app --reload
"""

from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from validator import infer_schema, validate_payload

app = FastAPI(title="JSON Schema Validator", version="1.0.0")


class ValidateRequest(BaseModel):
    # `schema` shadows BaseModel internals, so store under schema_ but accept the
    # JSON key "schema" via the alias.
    schema_: Dict[str, Any] = Field(alias="schema")
    data: Any

    model_config = {"populate_by_name": True}


class InferRequest(BaseModel):
    sample: Any


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/validate")
def validate(req: ValidateRequest) -> Dict[str, Any]:
    # Accept both {"schema": ...} and {"schema_": ...} via the alias-friendly model.
    return validate_payload(req.schema_, req.data)


@app.post("/infer")
def infer(req: InferRequest) -> Dict[str, Any]:
    return infer_schema(req.sample)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html><html><head><title>JSON Schema Validator</title>
<style>body{font-family:system-ui;max-width:720px;margin:2rem auto;padding:0 1rem}
textarea{width:100%;height:120px;font-family:monospace}pre{background:#f4f4f4;padding:1rem;border-radius:6px}
button{padding:.5rem 1rem;font-size:1rem;cursor:pointer}</style></head><body>
<h1>JSON Schema Validator</h1>
<p>Schema</p><textarea id="schema">{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}</textarea>
<p>Data</p><textarea id="data">{"id":"not-an-int"}</textarea><br><br>
<button onclick="run()">Validate</button>
<pre id="out">Result appears here.</pre>
<script>
async function run(){
  const body={schema:JSON.parse(document.getElementById('schema').value),
              data:JSON.parse(document.getElementById('data').value)};
  const r=await fetch('/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2);
}
</script></body></html>"""
