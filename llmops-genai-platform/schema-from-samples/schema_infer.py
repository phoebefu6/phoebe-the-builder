"""Infer a JSON Schema from example LLM outputs - frequency-aware, not union-of-everything.

The naive approach (union every type seen, require nothing, enumerate nothing) produces a
schema that validates the broken outputs you were trying to catch. The strict approach
(require everything, enumerate every string) rejects valid outputs on day two.

This infers from *frequency*: how often a field appears, how stable its type is, how many
distinct values it takes. A field in 100% of samples is a contract. A field in 12% is
optional. A string with 4 distinct values seen 15 times each is an enum; a string with 60
distinct values is free text. Where the evidence does not support a constraint, the
inferencer abstains and says so in the findings report rather than guessing.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Tunables. Every one of these is a policy choice, so they live in one object
# and get reported alongside the schema instead of hiding in the code.
# ---------------------------------------------------------------------------


@dataclass
class Policy:
    """Thresholds that turn observed frequencies into schema constraints."""

    required_at: float = 0.98
    """Present in >= this share of samples -> listed in `required`."""

    rare_key_below: float = 0.20
    """Present in < this share -> optional, and flagged for human confirmation."""

    enum_max_distinct: int = 8
    """More distinct values than this -> free-form, never an enum."""

    enum_min_support: int = 2
    """Every enum member must be seen at least this many times."""

    enum_max_cardinality_ratio: float = 0.30
    """distinct / occurrences above this -> looks like an identifier, not an enum."""

    min_samples_for_frequency: int = 5
    """Below this, abstain from `required` and enums entirely - no frequency signal."""

    closed_objects: bool = True
    """Emit additionalProperties: false, so hallucinated keys fail validation."""

    numeric_bounds: bool = True
    """Emit minimum/maximum for numbers whose observed range is a clean interval."""


JSON_TYPES = ("null", "boolean", "integer", "number", "string", "array", "object")

_MISSING = object()


def json_type(value: Any) -> str:
    """JSON Schema type name for a Python value. bool before int - bool is an int."""
    if value is None:
        return "null"
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
    return "string"  # dates, Decimals - anything a JSON encoder would stringify


# ---------------------------------------------------------------------------
# Observation pass: walk every sample and accumulate evidence per JSON pointer
# ---------------------------------------------------------------------------


@dataclass
class FieldStats:
    """Everything observed at one location across all samples."""

    path: str
    parent: str
    key: str
    occurrences: int = 0
    """Samples where the containing object existed (the denominator)."""
    present: int = 0
    """Samples where the key was actually there."""
    types: Counter = field(default_factory=Counter)
    values: Counter = field(default_factory=Counter)
    distinct_unhashable: int = 0
    numbers: List[float] = field(default_factory=list)
    empty_arrays: int = 0
    nonempty_arrays: int = 0

    @property
    def presence(self) -> float:
        return self.present / self.occurrences if self.occurrences else 0.0

    @property
    def non_null_types(self) -> Set[str]:
        return {t for t in self.types if t != "null"}

    @property
    def nullable(self) -> bool:
        return self.types.get("null", 0) > 0


class Observation:
    """Accumulates FieldStats over a corpus of samples."""

    def __init__(self) -> None:
        self.fields: Dict[str, FieldStats] = {}
        self.root_types: Counter = Counter()
        self.n_samples = 0

    def _stat(self, parent: str, key: str) -> FieldStats:
        path = f"{parent}.{key}" if parent else key
        if path not in self.fields:
            self.fields[path] = FieldStats(path=path, parent=parent, key=key)
        return self.fields[path]

    def add(self, sample: Any) -> None:
        self.n_samples += 1
        self.root_types[json_type(sample)] += 1
        self._walk("", sample)

    def _walk(self, path: str, node: Any) -> None:
        if isinstance(node, dict):
            # Every key we have *ever* seen under this parent gets its denominator
            # bumped, so "absent here" is recorded, not just "present there".
            known = {p.key for p in self.fields.values() if p.parent == path}
            for key in known | set(node.keys()):
                stat = self._stat(path, key)
                stat.occurrences += 1
                value = node.get(key, _MISSING)
                if value is _MISSING:
                    continue
                stat.present += 1
                self._observe(stat, value)
                self._walk(stat.path, value)
        elif isinstance(node, list):
            for item in node:
                # Array items share one pseudo-path so item-level types merge.
                stat = self._stat(path, "[]")
                stat.occurrences += 1
                stat.present += 1
                self._observe(stat, item)
                self._walk(stat.path, item)

    @staticmethod
    def _observe(stat: FieldStats, value: Any) -> None:
        t = json_type(value)
        stat.types[t] += 1
        if t in ("string", "boolean", "integer", "number"):
            try:
                stat.values[value] += 1
            except TypeError:  # pragma: no cover - defensive
                stat.distinct_unhashable += 1
        if t in ("integer", "number"):
            stat.numbers.append(float(value))
        if t == "array":
            if value:
                stat.nonempty_arrays += 1
            else:
                stat.empty_arrays += 1


# ---------------------------------------------------------------------------
# Findings: what the inferencer noticed but could not or would not encode
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    path: str
    kind: str
    severity: str  # "block" | "warn" | "info"
    detail: str


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class SchemaInference:
    def __init__(self, schema: Dict[str, Any], findings: List[Finding], obs: Observation, policy: Policy):
        self.schema = schema
        self.findings = findings
        self.obs = obs
        self.policy = policy

    def findings_table(self) -> List[Dict[str, str]]:
        order = {"block": 0, "warn": 1, "info": 2}
        rows = sorted(self.findings, key=lambda f: (order[f.severity], f.path))
        return [{"path": f.path or "(root)", "kind": f.kind, "severity": f.severity, "detail": f.detail} for f in rows]

    def field_table(self) -> List[Dict[str, Any]]:
        rows = []
        for path, s in sorted(self.obs.fields.items()):
            if s.key == "[]":
                continue
            types = "|".join(t for t, _ in s.types.most_common())
            rows.append(
                {
                    "path": path,
                    "presence": round(s.presence, 3),
                    "seen": s.present,
                    "of": s.occurrences,
                    "types": types,
                    "distinct": len(s.values) + s.distinct_unhashable,
                    "decision": self._decision(s),
                }
            )
        return rows

    def _decision(self, s: FieldStats) -> str:
        if self.obs.n_samples < self.policy.min_samples_for_frequency:
            return "optional (too few samples)"
        if s.presence >= self.policy.required_at:
            return "required"
        if s.presence < self.policy.rare_key_below:
            return f"optional - rare ({s.presence:.0%})"
        return "optional"


def infer_schema(
    samples: Iterable[Any],
    policy: Optional[Policy] = None,
    title: str = "Inferred output contract",
) -> SchemaInference:
    """Infer a draft 2020-12 JSON Schema plus a findings report from example outputs."""
    policy = policy or Policy()
    obs = Observation()
    materialised = list(samples)
    for sample in materialised:
        obs.add(sample)

    findings: List[Finding] = []

    if obs.n_samples == 0:
        return SchemaInference(
            {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title},
            [Finding("", "no-samples", "block", "No samples supplied - nothing to infer.")],
            obs,
            policy,
        )

    if obs.n_samples < policy.min_samples_for_frequency:
        findings.append(
            Finding(
                "",
                "low-sample-abstain",
                "warn",
                f"{obs.n_samples} sample(s) < min_samples_for_frequency={policy.min_samples_for_frequency}. "
                "Nothing is marked required and no enums are inferred - one sample cannot tell an "
                "optional field from a missing one.",
            )
        )

    if len(obs.root_types) > 1:
        findings.append(
            Finding(
                "",
                "root-type-instability",
                "block",
                "Samples disagree on the root type: "
                + ", ".join(f"{t}x{n}" for t, n in obs.root_types.most_common())
                + ". Fix the generator before trusting any contract.",
            )
        )

    schema = _build(path="", obs=obs, policy=policy, findings=findings, root_type=obs.root_types.most_common(1)[0][0])
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, **schema}
    return SchemaInference(schema, findings, obs, policy)


def _children(obs: Observation, path: str) -> List[FieldStats]:
    return [s for s in obs.fields.values() if s.parent == path]


def _build(path: str, obs: Observation, policy: Policy, findings: List[Finding], root_type: str) -> Dict[str, Any]:
    if root_type == "object":
        return _build_object(path, obs, policy, findings)
    if root_type == "array":
        item = next((s for s in _children(obs, path) if s.key == "[]"), None)
        if item is None:
            return {"type": "array"}
        return {"type": "array", "items": _field_schema(item, obs, policy, findings)}
    return {"type": root_type}


def _build_object(path: str, obs: Observation, policy: Policy, findings: List[Finding]) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    required: List[str] = []
    use_frequency = obs.n_samples >= policy.min_samples_for_frequency

    for stat in sorted(_children(obs, path), key=lambda s: s.key):
        if stat.key == "[]":
            continue
        props[stat.key] = _field_schema(stat, obs, policy, findings)
        if use_frequency and stat.presence >= policy.required_at:
            required.append(stat.key)
        elif use_frequency and stat.presence < policy.rare_key_below:
            findings.append(
                Finding(
                    stat.path,
                    "rare-key",
                    "warn",
                    f"Present in {stat.present}/{stat.occurrences} ({stat.presence:.0%}). Kept optional. "
                    "Confirm it is a real optional field and not intermittent model drift.",
                )
            )

    out: Dict[str, Any] = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    if policy.closed_objects:
        out["additionalProperties"] = False
    return out


def _field_schema(stat: FieldStats, obs: Observation, policy: Policy, findings: List[Finding]) -> Dict[str, Any]:
    non_null = stat.non_null_types
    use_frequency = obs.n_samples >= policy.min_samples_for_frequency

    if not non_null:
        findings.append(
            Finding(
                stat.path,
                "always-null",
                "warn",
                f"null in all {stat.present} observations. Type unknowable - left unconstrained "
                "rather than pinned to null, which would reject the first real value.",
            )
        )
        return {}

    # More than one non-null type is a defect in the generator, not a fact about the
    # contract. Widening the schema to a union would bless the defect permanently.
    if len(non_null) > 1:
        numeric = non_null <= {"integer", "number"}
        if numeric:
            base: Dict[str, Any] = {"type": "number"}
        else:
            counts = ", ".join(f"{t}x{stat.types[t]}" for t in sorted(non_null, key=lambda t: -stat.types[t]))
            dominant = max(non_null, key=lambda t: stat.types[t])
            share = stat.types[dominant] / sum(stat.types[t] for t in non_null)
            findings.append(
                Finding(
                    stat.path,
                    "type-instability",
                    "block",
                    f"Mixed non-null types ({counts}). Pinned to the dominant type '{dominant}' "
                    f"({share:.0%}) rather than widened to a union - a union would make the "
                    "minority form valid forever. Fix the generator or split the field.",
                )
            )
            base = {"type": dominant}
            non_null = {dominant}
    else:
        base = {"type": next(iter(non_null))}

    if stat.nullable:
        base["type"] = [base["type"], "null"]

    t = base["type"][0] if isinstance(base["type"], list) else base["type"]

    if t == "object":
        nested = _build_object(stat.path, obs, policy, findings)
        nested.pop("type", None)
        base.update(nested)
        return base

    if t == "array":
        item = next((s for s in _children(obs, stat.path) if s.key == "[]"), None)
        if item is None or stat.nonempty_arrays == 0:
            findings.append(
                Finding(
                    stat.path,
                    "empty-array-abstain",
                    "warn",
                    f"Empty in all {stat.empty_arrays} observations. Item type unknown, so `items` is "
                    "left open. An inferred item type here would be invention, not evidence.",
                )
            )
            return base
        base["items"] = _field_schema(item, obs, policy, findings)
        if stat.nonempty_arrays < max(3, 0.25 * stat.present):
            findings.append(
                Finding(
                    stat.path,
                    "thin-array-evidence",
                    "info",
                    f"Item type inferred from only {stat.nonempty_arrays} non-empty array(s) "
                    f"out of {stat.present}.",
                )
            )
        return base

    if t == "string" and use_frequency:
        enum = _maybe_enum(stat, policy)
        if enum is not None:
            base["enum"] = enum + ([None] if stat.nullable else [])
            base.pop("type", None)  # enum already pins the domain; type would be noise
        return base

    if t in ("integer", "number") and policy.numeric_bounds and stat.numbers:
        lo, hi = min(stat.numbers), max(stat.numbers)
        if math.isfinite(lo) and math.isfinite(hi) and lo != hi:
            # Bounds are widened to a round interval - the observed min is a sample
            # minimum, not a specification minimum.
            base["minimum"], base["maximum"] = _round_out(lo, hi)
    return base


def _maybe_enum(stat: FieldStats, policy: Policy) -> Optional[List[Any]]:
    values = stat.values
    if not values:
        return None
    distinct = len(values)
    total = sum(values.values())
    if distinct > policy.enum_max_distinct:
        return None
    if min(values.values()) < policy.enum_min_support:
        return None
    if distinct / total > policy.enum_max_cardinality_ratio:
        # Looks like an identifier: nearly every value is unique.
        return None
    return sorted(values, key=lambda v: (-values[v], str(v)))


def _round_out(lo: float, hi: float) -> Tuple[float, float]:
    span = hi - lo
    step = 10 ** math.floor(math.log10(span)) if span > 0 else 1.0
    out_lo = math.floor(lo / step) * step
    out_hi = math.ceil(hi / step) * step
    return (round(out_lo, 6), round(out_hi, 6))


# ---------------------------------------------------------------------------
# Comparison gates - what people actually ship, so the numbers mean something
# ---------------------------------------------------------------------------


def loose_schema(samples: Iterable[Any]) -> Dict[str, Any]:
    """Union-of-everything: every field optional, types unioned, no enums, open objects."""
    obs = Observation()
    for s in samples:
        obs.add(s)
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **_loose(obs, "")}


def _loose(obs: Observation, path: str) -> Dict[str, Any]:
    props = {}
    for stat in _children(obs, path):
        if stat.key == "[]":
            continue
        types = sorted(stat.types)
        node: Dict[str, Any] = {"type": types[0] if len(types) == 1 else types}
        if "object" in stat.types:
            node = {"type": node["type"], **{k: v for k, v in _loose(obs, stat.path).items() if k != "type"}}
        props[stat.key] = node
    return {"type": "object", "properties": props}


def strict_schema(samples: Iterable[Any]) -> Dict[str, Any]:
    """Everything required, every string enumerated to values seen. What over-fitting looks like."""
    obs = Observation()
    for s in samples:
        obs.add(s)
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **_strict(obs, "")}


def _strict(obs: Observation, path: str) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    required: List[str] = []
    for stat in _children(obs, path):
        if stat.key == "[]":
            continue
        required.append(stat.key)
        t = max(stat.types, key=lambda k: stat.types[k])
        if t == "string" and stat.values:
            props[stat.key] = {"enum": sorted(stat.values)}
        elif t == "object":
            props[stat.key] = _strict(obs, stat.path)
        else:
            props[stat.key] = {"type": t}
    return {"type": "object", "properties": props, "required": sorted(required), "additionalProperties": False}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def validate(schema: Dict[str, Any], sample: Any) -> Optional[str]:
    """Return None if valid, else the first validation error message."""
    try:
        import jsonschema
    except ImportError:  # pragma: no cover
        return _fallback_validate(schema, sample)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(sample), key=lambda e: list(e.absolute_path))
    if not errors:
        return None
    e = errors[0]
    where = ".".join(str(p) for p in e.absolute_path) or "(root)"
    return f"{where}: {e.message}"


def _fallback_validate(schema: Dict[str, Any], sample: Any) -> Optional[str]:  # pragma: no cover
    """Minimal type/required/enum check so the tool still runs without jsonschema."""
    if schema.get("type") == "object" and isinstance(sample, dict):
        for key in schema.get("required", []):
            if key not in sample:
                return f"{key}: required property missing"
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in sample:
                if key not in props:
                    return f"{key}: additional property not allowed"
        for key, sub in props.items():
            if key in sample:
                err = _fallback_validate(sub, sample[key])
                if err:
                    return f"{key}.{err}" if "." in err else f"{key}: {err}"
    if "enum" in schema and sample not in schema["enum"]:
        return f"{sample!r} is not one of {schema['enum']}"
    if "type" in schema:
        allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        t = json_type(sample)
        if t not in allowed and not (t == "integer" and "number" in allowed):
            return f"{t} is not of type {allowed}"
    return None


def evaluate(
    schema: Dict[str, Any],
    valid_holdout: List[Any],
    drifted: List[Tuple[str, Any]],
) -> Dict[str, Any]:
    """Score a schema: how much valid traffic it rejects, how much drift it catches."""
    false_rejects = [(i, validate(schema, s)) for i, s in enumerate(valid_holdout)]
    false_rejects = [(i, e) for i, e in false_rejects if e is not None]
    caught, missed = [], []
    for label, sample in drifted:
        err = validate(schema, sample)
        (caught if err else missed).append((label, err))
    n_valid = max(len(valid_holdout), 1)
    n_drift = max(len(drifted), 1)
    return {
        "false_reject_rate": len(false_rejects) / n_valid,
        "false_rejects": false_rejects,
        "catch_rate": len(caught) / n_drift,
        "caught": caught,
        "missed": [label for label, _ in missed],
    }


def to_json(schema: Dict[str, Any]) -> str:
    return json.dumps(schema, indent=2)


# ---------------------------------------------------------------------------
# Sample corpus: extracted support tickets, the shape an LLM actually returns
# ---------------------------------------------------------------------------

_CATEGORIES = ["billing", "technical", "account", "shipping"]
_PRIORITIES = ["low", "medium", "high"]
_FIRST = ["Ada", "Rui", "Sam", "Mei", "Jo", "Kai", "Lena", "Omar"]
_LAST = ["Chen", "Tan", "Okafor", "Lin", "Alvarez", "Berg", "Silva"]


def sample_corpus(n: int = 60) -> List[Dict[str, Any]]:
    """Deterministic mock of n well-formed extraction outputs (no RNG, no API)."""
    out = []
    for i in range(n):
        name = f"{_FIRST[i % len(_FIRST)]} {_LAST[(i * 3) % len(_LAST)]}"
        rec: Dict[str, Any] = {
            "ticket_id": f"TK-{4100 + i * 7}",
            "category": _CATEGORIES[i % 4],
            "priority": _PRIORITIES[(i // 2) % 3],
            "customer": {"name": name, "email": name.split()[0].lower() + f"{i}@example.com"},
            "sentiment_score": round(0.05 + (i % 19) * 0.05, 3),
            "tags": [_CATEGORIES[i % 4], "auto"][: 1 + i % 2],
            "resolved_at": None if i % 5 in (0, 1) else f"2026-07-{(i % 28) + 1:02d}T09:30:00Z",
            "attachments": [f"receipt_{i}.pdf"] if i % 8 == 0 else [],
            "related_ids": [],
        }
        if i % 9 == 0:  # legitimately rare optional block
            rec["escalation"] = {"tier": 2, "owner": "queue-ops"}
        if i % 2 == 0:  # optional, roughly half the time
            rec["estimated_hours"] = round(0.5 + (i % 7) * 0.75, 2)
        out.append(rec)
    return out


def drifted_corpus() -> List[Tuple[str, Dict[str, Any]]]:
    """Six labelled outputs that are wrong in six different ways."""
    base = sample_corpus(1)[0]

    def mutate(**kw: Any) -> Dict[str, Any]:
        rec = json.loads(json.dumps(base))
        for k, v in kw.items():
            if v is _MISSING:
                rec.pop(k, None)
            else:
                rec[k] = v
        return rec

    return [
        ("new enum value: priority='URGENT'", mutate(priority="URGENT")),
        ("type drift: sentiment_score as string", mutate(sentiment_score="0.8")),
        ("required field dropped: category", mutate(category=_MISSING)),
        ("shape collapse: customer as string", mutate(customer="Ada Chen <ada@example.com>")),
        ("hallucinated key: confidence_explanation", mutate(confidence_explanation="I am fairly sure")),
        ("container drift: tags as string", mutate(tags="billing, auto")),
    ]
