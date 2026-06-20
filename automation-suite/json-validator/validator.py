from __future__ import annotations

"""Core logic: validate a JSON payload against a JSON Schema, and infer a
starter schema from a sample payload.

API contracts break when a producer quietly changes a field's type or drops a
required key. This module gives you a fast yes/no with a list of *every* problem
(not just the first), plus a way to bootstrap a schema from an example so teams
don't have to hand-write one.
"""

from typing import Any, Dict, List

from jsonschema import Draft7Validator


def validate_payload(schema: Dict[str, Any], data: Any) -> Dict[str, Any]:
    """Validate `data` against `schema`. Returns every error, not just the first.

    A malformed schema is reported as an error rather than raised, so an API
    caller always gets a clean JSON response.
    """
    try:
        validator = Draft7Validator(schema)
    except Exception as exc:  # noqa: BLE001 - surface bad schemas to the caller
        return {"valid": False, "error_count": 1, "errors": [{"path": "<schema>", "message": f"invalid schema: {exc}"}]}

    errors: List[Dict[str, str]] = []
    for err in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        path = "$" + "".join(f"[{p!r}]" if isinstance(p, str) else f"[{p}]" for p in err.path)
        errors.append({"path": path, "message": err.message})

    return {"valid": not errors, "error_count": len(errors), "errors": errors}


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def infer_schema(sample: Any) -> Dict[str, Any]:
    """Infer a Draft-7 starter schema from a sample payload.

    Objects mark all present keys as `required` - a sensible default the user
    can loosen. Arrays infer from their first element.
    """
    t = _json_type(sample)
    if t == "object":
        props = {k: infer_schema(v) for k, v in sample.items()}
        return {
            "type": "object",
            "properties": props,
            "required": list(sample.keys()),
            "additionalProperties": True,
        }
    if t == "array":
        if sample:
            return {"type": "array", "items": infer_schema(sample[0])}
        return {"type": "array"}
    return {"type": t}
