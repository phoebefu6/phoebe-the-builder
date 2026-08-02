"""Generate demo.ipynb for schema-from-samples. Run once, then nbconvert --execute."""

from __future__ import annotations

import json

nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}


def md(src: str) -> None:
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src: str) -> None:
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})


BASE = "llmops-genai-platform/schema-from-samples"

md(f"""# Schema from Samples

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/{BASE}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath={BASE}/demo.ipynb)

> Your LLM pipeline returns JSON. Nothing checks it. This notebook infers the contract you should have written - from the outputs you already have - and shows why the two schemas people usually ship both fail.

**What this covers**

1. The problem: structured outputs drift, and "it parsed" is not a contract
2. Observation: walk a corpus of outputs and count evidence per field
3. Inference: turn frequencies into `required`, enums, bounds - or abstain
4. The findings report: what the evidence could **not** support, and why
5. Benchmark: inferred vs. loose vs. strict on held-out valid traffic + 6 labelled drift cases
6. Try your own

*Fully offline - the corpus is generated, no API key needed.*""")

md("""## 1. The problem

An extraction pipeline returns `{"category": "billing", "priority": "high", ...}` a thousand times a day. Then a model update ships and `priority` starts arriving as `"URGENT"`, or `sentiment_score` comes back as the string `"0.8"`, or a helpful new `confidence_explanation` key appears. Everything still parses. `json.loads` is not a contract.

The fix is a JSON Schema validated on every response. But who writes it? In practice one of two things gets shipped:

- **Loose** - union every type ever seen, require nothing. Validates the broken outputs you were trying to catch.
- **Strict** - require every field, enumerate every string seen. Rejects legitimate traffic the first time an optional field is absent.

The information to do better is already sitting in your logs: *frequency*. A field present in 100% of samples is a contract. A field in 12% is optional. A string with 3 distinct values across 45 samples is an enum; a string with 40 distinct values is free text. This notebook infers from that evidence - and **abstains, visibly,** where the evidence is thin.""")

md("""## 2. A corpus of real-shaped outputs

60 support-ticket extractions, deterministic (no RNG), with the texture real outputs have: a nullable timestamp, a rarely-present `escalation` block, an optional `estimated_hours`, an always-empty `related_ids` array, and arrays that are usually empty.""")

code("""from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# --- deterministic mock corpus: support-ticket extraction outputs -----------
_CATEGORIES = ["billing", "technical", "account", "shipping"]
_PRIORITIES = ["low", "medium", "high"]
_FIRST = ["Ada", "Rui", "Sam", "Mei", "Jo", "Kai", "Lena", "Omar"]
_LAST = ["Chen", "Tan", "Okafor", "Lin", "Alvarez", "Berg", "Silva"]


def sample_corpus(n: int = 60) -> List[Dict[str, Any]]:
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
        if i % 9 == 0:
            rec["escalation"] = {"tier": 2, "owner": "queue-ops"}
        if i % 2 == 0:
            rec["estimated_hours"] = round(0.5 + (i % 7) * 0.75, 2)
        out.append(rec)
    return out


corpus = sample_corpus(60)
train, holdout = corpus[:45], corpus[45:]
print(f"{len(train)} training samples, {len(holdout)} held out")
print(json.dumps(train[0], indent=2))""")

md("""## 3. Observation: count evidence per field

One walk over the corpus accumulates, for every JSON path: how often the containing object existed (the denominator), how often the key was present, the types seen, the distinct values, the numeric range, and how many arrays were empty.

One subtlety that matters: when an object appears *without* a key we have seen elsewhere, that absence must be **recorded**, not skipped - presence = present / occurrences is the single number the whole inference turns on.""")

code("""JSON_TYPES = ("null", "boolean", "integer", "number", "string", "array", "object")
_MISSING = object()


def json_type(value: Any) -> str:
    \"\"\"JSON Schema type name. bool checked before int - in Python, bool IS an int.\"\"\"
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
    return "string"


@dataclass
class FieldStats:
    path: str
    parent: str
    key: str
    occurrences: int = 0   # samples where the containing object existed
    present: int = 0       # samples where the key was actually there
    types: Counter = field(default_factory=Counter)
    values: Counter = field(default_factory=Counter)
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
            # bump the denominator for every key EVER seen under this parent,
            # so absence is evidence too
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
                stat = self._stat(path, "[]")  # array items share one pseudo-path
                stat.occurrences += 1
                stat.present += 1
                self._observe(stat, item)
                self._walk(stat.path, item)

    @staticmethod
    def _observe(stat: FieldStats, value: Any) -> None:
        t = json_type(value)
        stat.types[t] += 1
        if t in ("string", "boolean", "integer", "number"):
            stat.values[value] += 1
        if t in ("integer", "number"):
            stat.numbers.append(float(value))
        if t == "array":
            if value:
                stat.nonempty_arrays += 1
            else:
                stat.empty_arrays += 1


obs = Observation()
for s in train:
    obs.add(s)

print(f"{'path':<22}{'presence':>9}{'seen':>6}{'of':>5}  {'types':<14}{'distinct':>8}")
for path, s in sorted(obs.fields.items()):
    if s.key == "[]":
        continue
    types = "|".join(t for t, _ in s.types.most_common())
    print(f"{path:<22}{s.presence:>9.0%}{s.present:>6}{s.occurrences:>5}  {types:<14}{len(s.values):>8}")""")

md("""Read that table like a detective. `category` is at 100% presence with 4 distinct values - a required enum. `estimated_hours` sits at 51% - clearly optional. `escalation` at 11% - optional, but suspicious enough to flag. `resolved_at` shows `string|null` - nullable, not drift. `related_ids` has been an empty array 45 times - its item type is **unknowable** from this data.

## 4. Inference: frequency → constraint, or abstention

Every threshold is a named policy choice, reported alongside the schema:""")

code("""@dataclass
class Policy:
    required_at: float = 0.98             # presence >= this -> required
    rare_key_below: float = 0.20          # presence < this -> flagged for review
    enum_max_distinct: int = 8            # more distinct values -> free text
    enum_min_support: int = 2             # each enum member seen >= this
    enum_max_cardinality_ratio: float = 0.30  # distinct/total above this -> identifier
    min_samples_for_frequency: int = 5    # below this, abstain from required + enums
    closed_objects: bool = True           # additionalProperties: false
    numeric_bounds: bool = True           # rounded-out min/max for numerics


@dataclass
class Finding:
    path: str
    kind: str
    severity: str  # block | warn | info
    detail: str


def _children(obs: Observation, path: str) -> List[FieldStats]:
    return [s for s in obs.fields.values() if s.parent == path]


def _round_out(lo: float, hi: float) -> Tuple[float, float]:
    # observed min is a SAMPLE minimum, not a spec minimum - widen to a round interval
    span = hi - lo
    step = 10 ** math.floor(math.log10(span)) if span > 0 else 1.0
    return (round(math.floor(lo / step) * step, 6), round(math.ceil(hi / step) * step, 6))


def _maybe_enum(stat: FieldStats, policy: Policy) -> Optional[List[Any]]:
    values = stat.values
    if not values:
        return None
    distinct, total = len(values), sum(values.values())
    if distinct > policy.enum_max_distinct:
        return None
    if min(values.values()) < policy.enum_min_support:
        return None
    if distinct / total > policy.enum_max_cardinality_ratio:
        return None  # nearly every value unique -> identifier, not enum
    return sorted(values, key=lambda v: (-values[v], str(v)))


def _field_schema(stat, obs, policy, findings):
    non_null = stat.non_null_types
    use_freq = obs.n_samples >= policy.min_samples_for_frequency

    if not non_null:
        findings.append(Finding(stat.path, "always-null", "warn",
            f"null in all {stat.present} observations - type unknowable, left unconstrained."))
        return {}

    if len(non_null) > 1:
        if non_null <= {"integer", "number"}:
            base = {"type": "number"}
        else:
            dominant = max(non_null, key=lambda t: stat.types[t])
            share = stat.types[dominant] / sum(stat.types[t] for t in non_null)
            findings.append(Finding(stat.path, "type-instability", "block",
                f"Mixed non-null types - pinned to dominant '{dominant}' ({share:.0%}), NOT widened "
                "to a union. A union would bless the defect forever. Fix the generator."))
            base = {"type": dominant}
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
            findings.append(Finding(stat.path, "empty-array-abstain", "warn",
                f"Empty in all {stat.empty_arrays} observations - `items` left open. "
                "Inferring an item type here would be invention, not evidence."))
            return base
        base["items"] = _field_schema(item, obs, policy, findings)
        if stat.nonempty_arrays < max(3, 0.25 * stat.present):
            findings.append(Finding(stat.path, "thin-array-evidence", "info",
                f"Item type from only {stat.nonempty_arrays} non-empty array(s) of {stat.present}."))
        return base

    if t == "string" and use_freq:
        enum = _maybe_enum(stat, policy)
        if enum is not None:
            base["enum"] = enum + ([None] if stat.nullable else [])
            base.pop("type", None)
        return base

    if t in ("integer", "number") and policy.numeric_bounds and stat.numbers:
        lo, hi = min(stat.numbers), max(stat.numbers)
        if lo != hi:
            base["minimum"], base["maximum"] = _round_out(lo, hi)
    return base


def _build_object(path, obs, policy, findings):
    props, required = {}, []
    use_freq = obs.n_samples >= policy.min_samples_for_frequency
    for stat in sorted(_children(obs, path), key=lambda s: s.key):
        if stat.key == "[]":
            continue
        props[stat.key] = _field_schema(stat, obs, policy, findings)
        if use_freq and stat.presence >= policy.required_at:
            required.append(stat.key)
        elif use_freq and stat.presence < policy.rare_key_below:
            findings.append(Finding(stat.path, "rare-key", "warn",
                f"Present in {stat.present}/{stat.occurrences} ({stat.presence:.0%}) - kept optional. "
                "Confirm it is a real optional field, not intermittent drift."))
    out = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    if policy.closed_objects:
        out["additionalProperties"] = False
    return out


def infer_schema(samples: Iterable[Any], policy: Optional[Policy] = None):
    policy = policy or Policy()
    obs = Observation()
    for s in samples:
        obs.add(s)
    findings: List[Finding] = []
    if obs.n_samples < policy.min_samples_for_frequency:
        findings.append(Finding("", "low-sample-abstain", "warn",
            f"Only {obs.n_samples} sample(s) - nothing marked required, no enums inferred. "
            "One sample cannot tell an optional field from a missing one."))
    if len(obs.root_types) > 1:
        findings.append(Finding("", "root-type-instability", "block",
            "Samples disagree on the root type. Fix the generator before trusting any contract."))
    schema = _build_object("", obs, policy, findings)
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}, findings, obs


schema, findings, obs = infer_schema(train)
print(json.dumps(schema, indent=2))""")

md("""## 5. The findings report

The schema is only half the output. The other half is every place the inferencer noticed something and **refused to encode it** - each with the reason. This is the part a human should read before shipping the contract:""")

code("""for f in sorted(findings, key=lambda f: {"block": 0, "warn": 1, "info": 2}[f.severity]):
    print(f"[{f.severity.upper():5}] {f.path or '(root)'} - {f.kind}")
    print(f"        {f.detail}\\n")""")

md("""Three abstentions, three different reasons - and each one is a decision a naive inferencer silently gets wrong:

- **`escalation` (rare-key):** 11% presence *could* be a legitimate optional block or *could* be the model occasionally inventing structure. Frequency alone cannot distinguish those. The schema keeps it optional; the finding routes the question to a human.
- **`related_ids` (empty-array-abstain):** 45 empty arrays contain zero bits of information about item type. Guessing `string` because "arrays are usually strings" would reject the first legitimate `[123]`.
- **`attachments` (thin-array-evidence):** typed from 6 non-empty observations - enough to act, thin enough to say so.

## 6. Benchmark: does frequency-awareness actually matter?

Build the two contenders and score all three on (a) the 15 **held-out valid** samples - the false-reject rate that pages you at 2am, and (b) **6 labelled drift cases**, one for each way structured outputs actually break.""")

code("""def loose_schema(samples):
    \"\"\"Union-of-everything: types unioned, nothing required, objects open.\"\"\"
    o = Observation()
    for s in samples:
        o.add(s)
    def build(path):
        props = {}
        for stat in _children(o, path):
            if stat.key == "[]":
                continue
            types = sorted(stat.types)
            node = {"type": types[0] if len(types) == 1 else types}
            if "object" in stat.types:
                node = {"type": node["type"], **{k: v for k, v in build(stat.path).items() if k != "type"}}
            props[stat.key] = node
        return {"type": "object", "properties": props}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **build("")}


def strict_schema(samples):
    \"\"\"Everything required, every string enumerated. Textbook over-fitting.\"\"\"
    o = Observation()
    for s in samples:
        o.add(s)
    def build(path):
        props, req = {}, []
        for stat in _children(o, path):
            if stat.key == "[]":
                continue
            req.append(stat.key)
            t = max(stat.types, key=lambda k: stat.types[k])
            if t == "string" and stat.values:
                props[stat.key] = {"enum": sorted(stat.values)}
            elif t == "object":
                props[stat.key] = build(stat.path)
            else:
                props[stat.key] = {"type": t}
        return {"type": "object", "properties": props, "required": sorted(req), "additionalProperties": False}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **build("")}


def drifted_corpus():
    \"\"\"Six labelled outputs, wrong in six different ways.\"\"\"
    base = sample_corpus(1)[0]
    def mutate(**kw):
        rec = json.loads(json.dumps(base))
        for k, v in kw.items():
            rec.pop(k, None) if v is _MISSING else rec.__setitem__(k, v)
        return rec
    return [
        ("new enum value: priority='URGENT'", mutate(priority="URGENT")),
        ("type drift: sentiment_score as string", mutate(sentiment_score="0.8")),
        ("required field dropped: category", mutate(category=_MISSING)),
        ("shape collapse: customer as string", mutate(customer="Ada Chen <ada@example.com>")),
        ("hallucinated key: confidence_explanation", mutate(confidence_explanation="I am fairly sure")),
        ("container drift: tags as string", mutate(tags="billing, auto")),
    ]


import jsonschema


def validate(schema, sample):
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(sample),
                    key=lambda e: list(e.absolute_path))
    if not errors:
        return None
    e = errors[0]
    where = ".".join(str(p) for p in e.absolute_path) or "(root)"
    return f"{where}: {e.message}"


drift = drifted_corpus()
gates = {
    "inferred": schema,
    "loose": loose_schema(train),
    "strict": strict_schema(train),
}

results = {}
for name, g in gates.items():
    fr = [validate(g, s) for s in holdout]
    fr = [e for e in fr if e]
    caught = [(lbl, validate(g, s)) for lbl, s in drift]
    missed = [lbl for lbl, e in caught if e is None]
    results[name] = {"fr_rate": len(fr) / len(holdout), "catch_rate": 1 - len(missed) / len(drift),
                     "missed": missed, "first_fr": fr[0] if fr else None}

print(f"{'gate':<10}{'valid rejected':>15}{'drift caught':>14}   missed / first false reject")
for name, r in results.items():
    note = ("MISSED: " + "; ".join(r["missed"])) if r["missed"] else (r["first_fr"] or "clean")
    print(f"{name:<10}{r['fr_rate']:>15.0%}{r['catch_rate']:>14.0%}   {note}")""")

md("""**The loose schema misses exactly the drift you built the gate for** - the new enum value, the dropped field, the hallucinated key. It only catches outright type swaps. **The strict schema catches everything and rejects 100% of legitimate traffic**, because every holdout sample either lacks `escalation` or lacks `estimated_hours` - both of which it made required. It would be turned off within a day, and then nothing is validated at all.

The inferred schema is the only one of the three you could actually leave running.

## 7. Visualize the trade-off""")

code("""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
fig.suptitle("One schema you can ship, two you cannot", fontsize=14, fontweight="bold")

names = list(results)
colors = {"inferred": "#2E7D32", "loose": "#F9A825", "strict": "#C62828"}
x = np.arange(len(names))

fr = [results[n]["fr_rate"] for n in names]
cr = [results[n]["catch_rate"] for n in names]

ax1.bar(x, [v * 100 for v in fr], color=[colors[n] for n in names], width=0.6)
ax1.set_title("Valid holdout rejected (lower = better)")
ax1.set_ylabel("% of legitimate outputs rejected")
ax1.set_xticks(x, names)
ax1.set_ylim(0, 105)
for i, v in enumerate(fr):
    ax1.text(i, v * 100 + 2, f"{v:.0%}", ha="center", fontweight="bold")

ax2.bar(x, [v * 100 for v in cr], color=[colors[n] for n in names], width=0.6)
ax2.set_title("Labelled drift caught (higher = better)")
ax2.set_ylabel("% of 6 drift cases caught")
ax2.set_xticks(x, names)
ax2.set_ylim(0, 105)
for i, v in enumerate(cr):
    ax2.text(i, v * 100 + 2, f"{v:.0%}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("schema_benchmark.png", dpi=150, bbox_inches="tight")
plt.show()
print("saved schema_benchmark.png")""")

md("""## 8. Summary

| Gate | Valid holdout rejected | Drift caught | Verdict |
|---|---|---|---|
| **inferred (frequency-aware)** | **0%** | **6/6** | ship it |
| loose (union-of-everything) | 0% | 3/6 | a smoke detector with no battery |
| strict (require + enum all) | 100% | 6/6 | disabled by lunchtime |

What made the difference is not cleverness - it is three habits:

1. **Presence is a number, not a boolean.** 100% → required, 51% → optional, 11% → optional *and flagged*.
2. **Enums need support.** Distinct-value count, per-value support, and a cardinality ratio separate `priority` (3 values, dozens of sightings each) from `customer.name` (an identifier wearing a string type).
3. **Abstain out loud.** An always-empty array, an always-null field, a 2-sample corpus - the honest schema leaves them open and says so in the findings, instead of inventing a constraint that will fire on legitimate traffic.

And one deliberate sharp edge: on **mixed types** the inferencer pins the dominant type rather than widening to a union - a union would make the minority form *valid forever*, which is exactly backwards. That decision is a `block`-severity finding, because the right fix is in the generator, not the schema.

## 9. Try your own

Paste your own outputs below - one dict per element - and re-run:""")

code("""# my_samples = [
#     {"answer": "yes", "confidence": 0.92, "sources": ["doc_1"]},
#     {"answer": "no", "confidence": 0.71, "sources": []},
#     # ... paste 20+ real outputs for meaningful frequencies ...
# ]
# my_schema, my_findings, _ = infer_schema(my_samples)
# print(json.dumps(my_schema, indent=2))
# for f in my_findings:
#     print(f"[{f.severity}] {f.path}: {f.detail}")
#
# # then wire it into the pipeline:
# # err = validate(my_schema, new_output)
# # if err: retry_or_alert(err)""")

md(f"""---

Part of the **[phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder)** portfolio - one 30-minute AI product per day.

**Streamlit version:** `streamlit run app.py` in [`{BASE}/`](https://github.com/phoebefu6/phoebe-the-builder/tree/main/{BASE}) - paste JSONL, tune the policy thresholds live, download `schema.json`.""")

with open("demo.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote demo.ipynb with {len(nb['cells'])} cells")
