from __future__ import annotations

# Structured Output Enforcer - core logic.
#
# You ask an LLM for JSON and get back a code fence, a chatty preamble, a
# trailing comma, Python's True/None instead of true/null, or a missing field.
# Naive `json.loads()` throws and your pipeline dies. This module wraps the raw
# model text in a repair-then-validate pipeline: pull the JSON out of the
# noise, fix the common malformations deterministically, coerce values to the
# declared types, and check the schema. It returns *what it fixed* so you can
# log repair rates and decide when to force a model retry.
#
# Fully offline - no API keys. The "model output" in the demo is a set of
# realistically-messy strings; swap in your real Claude/LLM response text.
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Schema spec
# --------------------------------------------------------------------------- #

_TYPES = {"str": str, "int": int, "float": float, "bool": bool, "list": list}


@dataclass
class Field:
    """One expected output field: its type and whether it must be present."""

    name: str
    type: str = "str"
    required: bool = True


@dataclass
class Schema:
    """An ordered set of expected fields. Extra keys in the output are kept."""

    fields: List[Field]

    def required_names(self) -> List[str]:
        return [f.name for f in self.fields if f.required]

    def type_of(self, name: str) -> Optional[str]:
        for f in self.fields:
            if f.name == name:
                return f.type
        return None


@dataclass
class Result:
    """Outcome of enforcing one raw model output against a schema."""

    ok: bool
    data: Optional[Dict[str, Any]]
    repairs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        """True when the output only parsed *because* we fixed something."""
        return self.ok and len(self.repairs) > 0


# --------------------------------------------------------------------------- #
# Step 1 - pull a JSON object out of noisy text
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_block(raw: str) -> Tuple[str, List[str]]:
    """Strip markdown fences and any prose around the first balanced {...}.

    Returns the candidate JSON string plus a list of repairs applied.
    """
    repairs: List[str] = []
    text = raw.strip()

    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
        repairs.append("stripped markdown code fence")

    start = text.find("{")
    if start == -1:
        return text, repairs

    # Walk to the matching closing brace so trailing prose is dropped.
    depth = 0
    end = -1
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

    if end != -1 and (start > 0 or end < len(text) - 1):
        text = text[start : end + 1]
        repairs.append("removed surrounding prose")
    elif start > 0:
        text = text[start:]
        repairs.append("removed leading prose")

    return text, repairs


# --------------------------------------------------------------------------- #
# Step 2 - repair common JSON malformations
# --------------------------------------------------------------------------- #

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)")
_PY_LITERALS = [("True", "true"), ("False", "false"), ("None", "null")]


def repair_json_text(text: str) -> Tuple[str, List[str]]:
    """Apply deterministic fixes for the malformations LLMs produce most."""
    repairs: List[str] = []

    if _TRAILING_COMMA_RE.search(text):
        text = _TRAILING_COMMA_RE.sub(r"\1", text)
        repairs.append("removed trailing comma")

    for py, js in _PY_LITERALS:
        pat = re.compile(rf"(?<![\"\w]){py}(?![\"\w])")
        if pat.search(text):
            text = pat.sub(js, text)
            repairs.append(f"converted Python {py} -> {js}")

    # Single-quoted strings/keys -> double quotes (only if no double quotes yet).
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')
        repairs.append("converted single quotes to double quotes")

    if _UNQUOTED_KEY_RE.search(text):
        text = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', text)
        repairs.append("quoted unquoted keys")

    return text, repairs


# --------------------------------------------------------------------------- #
# Step 3 - coerce values to declared types
# --------------------------------------------------------------------------- #

def coerce_value(value: Any, target: str) -> Tuple[Any, bool]:
    """Best-effort coercion. Returns (value, changed)."""
    py_type = _TYPES.get(target)
    if py_type is None or isinstance(value, py_type) and not (
        target in ("int", "float") and isinstance(value, bool)
    ):
        return value, False

    try:
        if target == "int":
            return int(float(str(value).strip())), True
        if target == "float":
            return float(str(value).strip()), True
        if target == "bool":
            s = str(value).strip().lower()
            if s in ("true", "yes", "1"):
                return True, True
            if s in ("false", "no", "0"):
                return False, True
            return value, False
        if target == "str":
            return str(value), True
        if target == "list" and isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()], True
    except (ValueError, TypeError):
        return value, False
    return value, False


# --------------------------------------------------------------------------- #
# Step 4 - the full pipeline
# --------------------------------------------------------------------------- #

def enforce(raw: str, schema: Schema) -> Result:
    """Extract -> repair -> parse -> coerce -> validate against `schema`."""
    repairs: List[str] = []
    errors: List[str] = []

    candidate, r1 = extract_json_block(raw)
    repairs += r1

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        candidate, r2 = repair_json_text(candidate)
        repairs += r2
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return Result(ok=False, data=None, repairs=repairs,
                          errors=[f"unparseable JSON: {exc.msg}"])

    if not isinstance(data, dict):
        return Result(ok=False, data=None, repairs=repairs,
                      errors=[f"expected object, got {type(data).__name__}"])

    # Coerce declared fields to their types.
    for f in schema.fields:
        if f.name in data:
            new_val, changed = coerce_value(data[f.name], f.type)
            if changed:
                data[f.name] = new_val
                repairs.append(f"coerced '{f.name}' to {f.type}")

    # Validate required presence + final types.
    for f in schema.fields:
        if f.name not in data:
            if f.required:
                errors.append(f"missing required field '{f.name}'")
            continue
        py_type = _TYPES.get(f.type)
        val = data[f.name]
        bad_bool = f.type in ("int", "float") and isinstance(val, bool)
        if py_type and (not isinstance(val, py_type) or bad_bool):
            errors.append(
                f"field '{f.name}' should be {f.type}, got {type(val).__name__}"
            )

    return Result(ok=not errors, data=data if not errors else data,
                  repairs=repairs, errors=errors)


# --------------------------------------------------------------------------- #
# Sample data - realistically messy "model outputs"
# --------------------------------------------------------------------------- #

# Ticket-triage schema: what we asked the model to return for each ticket.
TICKET_SCHEMA = Schema([
    Field("category", "str"),
    Field("priority", "int"),
    Field("urgent", "bool"),
    Field("sentiment_score", "float"),
    Field("tags", "list"),
])

SAMPLE_OUTPUTS: List[Tuple[str, str]] = [
    ("clean", '{"category": "billing", "priority": 2, "urgent": false, '
              '"sentiment_score": -0.4, "tags": ["refund", "invoice"]}'),
    ("code fence", '```json\n{"category": "login", "priority": 1, '
                   '"urgent": true, "sentiment_score": -0.8, '
                   '"tags": ["auth", "outage"]}\n```'),
    ("chatty preamble", 'Sure! Here is the structured result:\n'
                        '{"category": "feature", "priority": 3, '
                        '"urgent": false, "sentiment_score": 0.2, '
                        '"tags": ["request"]}\nLet me know if you need more.'),
    ("trailing comma", '{"category": "bug", "priority": 2, "urgent": true, '
                       '"sentiment_score": -0.5, "tags": ["crash",],}'),
    ("python literals", "{'category': 'billing', 'priority': 2, "
                        "'urgent': True, 'sentiment_score': -0.3, "
                        "'tags': ['charge']}"),
    ("string numbers", '{"category": "shipping", "priority": "1", '
                       '"urgent": "yes", "sentiment_score": "-0.6", '
                       '"tags": ["delay"]}'),
    ("missing field", '{"category": "general", "urgent": false, '
                      '"sentiment_score": 0.1, "tags": []}'),
    ("not json", "I could not determine the category for this ticket."),
]


def run_sample() -> List[Dict[str, Any]]:
    """Enforce every sample output; return one summary row per case."""
    rows: List[Dict[str, Any]] = []
    for label, raw in SAMPLE_OUTPUTS:
        res = enforce(raw, TICKET_SCHEMA)
        rows.append({
            "case": label,
            "valid": res.ok,
            "repaired": res.repaired,
            "n_repairs": len(res.repairs),
            "repairs": "; ".join(res.repairs) or "-",
            "errors": "; ".join(res.errors) or "-",
        })
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    valid = sum(r["valid"] for r in rows)
    naive_ok = sum(1 for _, raw in SAMPLE_OUTPUTS if _naive_parses(raw))
    return {
        "total": total,
        "valid": valid,
        "valid_pct": round(100 * valid / total) if total else 0,
        "naive_ok": naive_ok,
        "naive_pct": round(100 * naive_ok / total) if total else 0,
        "recovered": valid - naive_ok,
    }


def _naive_parses(raw: str) -> bool:
    """Would a plain json.loads() have succeeded on the raw text?"""
    try:
        obj = json.loads(raw.strip())
        return isinstance(obj, dict)
    except (json.JSONDecodeError, ValueError):
        return False


if __name__ == "__main__":
    rows = run_sample()
    s = summarize(rows)
    print(f"Naive json.loads: {s['naive_ok']}/{s['total']} ({s['naive_pct']}%)")
    print(f"With enforcer:    {s['valid']}/{s['total']} ({s['valid_pct']}%)")
    print(f"Recovered:        +{s['recovered']} outputs\n")
    for r in rows:
        flag = "OK " if r["valid"] else "FAIL"
        print(f"[{flag}] {r['case']:<16} repairs: {r['repairs']}")
        if r["errors"] != "-":
            print(f"        errors: {r['errors']}")
