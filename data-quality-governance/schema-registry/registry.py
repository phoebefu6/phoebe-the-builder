from __future__ import annotations

# Schema Registry - register schema versions per dataset and check whether a
# PROPOSED schema is safe to ship against the latest registered version. A
# schema is a list of fields (name, type, nullable, required). When a producer
# proposes a change, the engine diffs it field-by-field and classifies the
# compatibility verdict - BACKWARD, FORWARD, FULL, or BREAKING - with a plain
# reason for every change, so a data steward can approve or block the change
# BEFORE three downstream jobs break silently.
# Fully offline, standard python/pandas only - no API keys.

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


# --- data model -----------------------------------------------------------

@dataclass
class Field:
    """One column in a schema."""

    name: str
    type: str                # "int" | "long" | "float" | "double" | "string" | "bool" ...
    nullable: bool = False    # can this field hold null?
    required: bool = True     # must a producer supply this field (no default)?


@dataclass
class Schema:
    """A named, versioned list of fields for one dataset."""

    dataset: str
    version: int
    fields: List[Field]

    def field_map(self) -> Dict[str, Field]:
        return {f.name: f for f in self.fields}


@dataclass
class Change:
    """One field-level difference between two schema versions, with a reason."""

    field: str
    kind: str        # "added" | "removed" | "type-changed" | "nullability-changed" | "required-changed"
    detail: str
    # Directional safety of THIS single change. A change can break backward
    # readers, forward readers, both, or neither.
    breaks_backward: bool
    breaks_forward: bool
    reason: str


# Type-widening lattice: reading a value stored as the KEY type is safe when the
# reader expects any type in its VALUE set (the reader's type is wider). So
# int -> long is a widening (safe); long -> int is a NARROWING (breaking,
# because a long value may not fit in an int). This is the Avro/Confluent
# promotion rule, kept deliberately small and explicit.
_WIDENS_TO: Dict[str, List[str]] = {
    "int": ["int", "long", "float", "double"],
    "long": ["long", "float", "double"],
    "float": ["float", "double"],
    "double": ["double"],
    "string": ["string"],
    "bool": ["bool"],
}


def _is_widening(old_type: str, new_type: str) -> bool:
    """True if a value of old_type always fits in new_type (safe promotion)."""
    return new_type in _WIDENS_TO.get(old_type, [old_type])


# --- the compare engine ---------------------------------------------------

def _field_changes(old: Field, new: Field) -> List[Change]:
    """Diff a field that exists in BOTH schemas (same name)."""
    changes: List[Change] = []

    # 1) Type change. Widening is safe both ways; narrowing / incompatible is BREAKING.
    if old.type != new.type:
        if _is_widening(old.type, new.type):
            # WHY safe: every old value fits the new (wider) type, so new
            # consumers read old data fine (backward ok); old consumers reading
            # new data see values that were always in-range (forward ok).
            changes.append(Change(
                new.name, "type-changed",
                f"{old.type} -> {new.type} (widening)",
                breaks_backward=False, breaks_forward=False,
                reason=(f"type widened {old.type}->{new.type}; every value still "
                        f"fits, safe both directions"),
            ))
        else:
            # WHY breaking: a narrowing (long->int) or an incompatible switch
            # (string->int) means some value cannot be represented for one side
            # or the other - neither reader is guaranteed to parse the bytes.
            changes.append(Change(
                new.name, "type-changed",
                f"{old.type} -> {new.type} (narrowing/incompatible)",
                breaks_backward=True, breaks_forward=True,
                reason=(f"type {old.type}->{new.type} is not a safe widening; "
                        f"values may not fit / parse - BREAKING both ways"),
            ))

    # 2) Nullability change.
    if old.nullable != new.nullable:
        if old.nullable and not new.nullable:
            # WHY breaking: old data (and old producers) can emit nulls; a reader
            # on the new schema forbids null, so it chokes on existing/old rows.
            changes.append(Change(
                new.name, "nullability-changed",
                "nullable -> required (non-null)",
                breaks_backward=True, breaks_forward=False,
                reason=("field was nullable, now non-null; new consumers reject "
                        "the nulls present in old data - BREAKING (backward)"),
            ))
        else:
            # WHY safe: widening to allow null never invalidates old non-null data.
            changes.append(Change(
                new.name, "nullability-changed",
                "required (non-null) -> nullable",
                breaks_backward=False, breaks_forward=False,
                reason=("field made nullable; old non-null values are still valid, "
                        "safe both directions"),
            ))

    # 3) Required-ness change (does a producer have to supply the field?).
    if old.required != new.required:
        if not old.required and new.required:
            # WHY forward-breaking: old producers may omit this field; a new
            # consumer that now demands it has no value and no default.
            changes.append(Change(
                new.name, "required-changed",
                "optional -> required",
                breaks_backward=False, breaks_forward=True,
                reason=("field became required with no default; old producers omit "
                        "it, so new consumers have nothing to read - forward-breaking"),
            ))
        else:
            changes.append(Change(
                new.name, "required-changed",
                "required -> optional",
                breaks_backward=False, breaks_forward=False,
                reason=("field relaxed to optional; every reader still copes, safe"),
            ))
    return changes


def _added_change(f: Field) -> Change:
    """A field present in NEW but not OLD."""
    if f.required and not f.nullable:
        # WHY forward-breaking: old data has no value for this new mandatory
        # field, so a consumer built on the new schema cannot read old rows.
        return Change(
            f.name, "added",
            f"added required field '{f.name}' ({f.type})",
            breaks_backward=False, breaks_forward=True,
            reason=("added a required field with no default; old data / old "
                    "producers omit it - new consumers break reading old data "
                    "(forward-breaking)"),
        )
    # WHY safe: an optional/nullable new field can be absent; old data simply
    # has null/no value there and every reader tolerates it.
    return Change(
        f.name, "added",
        f"added optional/nullable field '{f.name}' ({f.type})",
        breaks_backward=False, breaks_forward=False,
        reason=("added an optional/nullable field; absent in old data is fine, "
                "backward-compatible and safe"),
    )


def _removed_change(f: Field) -> Change:
    """A field present in OLD but not NEW."""
    # WHY backward-breaking: a consumer on the NEW schema that dropped the field
    # will not find it - but more precisely, a consumer still expecting the old
    # field (old consumer reading new data that lacks it) is the break. We model
    # removal as backward-breaking (new-schema readers lose data the old schema
    # guaranteed) and forward-safe (old readers ignore extra/missing gracefully
    # only if it was optional). We treat any removal as backward-breaking.
    return Change(
        f.name, "removed",
        f"removed field '{f.name}' ({f.type})",
        breaks_backward=True, breaks_forward=False,
        reason=("removed a field; consumers still expecting it lose data they "
                "relied on - backward-breaking, but old producers' extra field "
                "is ignored so forward reads survive"),
    )


def _verdict(changes: List[Change]) -> str:
    """Roll field-level directional breaks up into ONE compatibility verdict.

    BACKWARD  = new consumers can read OLD data (no backward breaks).
    FORWARD   = old consumers can read NEW data (no forward breaks).
    FULL      = both hold. BREAKING = at least one direction is broken both ways
    (i.e. neither BACKWARD nor FORWARD holds).
    """
    if not changes:
        return "FULL"
    any_back = any(c.breaks_backward for c in changes)
    any_fwd = any(c.breaks_forward for c in changes)
    if not any_back and not any_fwd:
        return "FULL"
    if not any_back:
        return "BACKWARD"   # only forward is broken -> backward still holds
    if not any_fwd:
        return "FORWARD"    # only backward is broken -> forward still holds
    return "BREAKING"       # both directions broken


def compare(old: Optional[Schema], new: Schema) -> Dict[str, object]:
    """Diff `new` against `old` and classify compatibility.

    Edge case: no prior version (old is None) -> the first schema is always
    accepted as "initial, compatible".
    """
    if old is None:
        return {
            "verdict": "FULL",
            "changes": [],
            "summary": "initial, compatible - first registered version, nothing to break",
        }

    old_map, new_map = old.field_map(), new.field_map()
    changes: List[Change] = []

    # Fields in both -> per-field diff.
    for name in old_map:
        if name in new_map:
            changes.extend(_field_changes(old_map[name], new_map[name]))
    # Added (in new only) and removed (in old only).
    for name, f in new_map.items():
        if name not in old_map:
            changes.append(_added_change(f))
    for name, f in old_map.items():
        if name not in new_map:
            changes.append(_removed_change(f))

    verdict = _verdict(changes)
    drivers = [c for c in changes if c.breaks_backward or c.breaks_forward]
    if not changes:
        summary = "no field changes - FULL compatibility"
    elif verdict == "FULL":
        summary = f"{len(changes)} change(s), all safe both directions - FULL"
    elif verdict == "BREAKING":
        summary = f"{len(drivers)} breaking change(s) in both directions - BREAKING"
    else:
        summary = (f"{verdict}-compatible; {len(drivers)} change(s) break the "
                   f"other direction only")
    return {"verdict": verdict, "changes": changes, "summary": summary}


# --- the registry ---------------------------------------------------------

class Registry:
    """In-memory version store: one growing list of Schemas per dataset."""

    def __init__(self) -> None:
        self._store: Dict[str, List[Schema]] = {}

    def register(self, dataset: str, fields: List[Field]) -> Schema:
        """Append a new version. Version numbers start at 1 and increment."""
        versions = self._store.setdefault(dataset, [])
        schema = Schema(dataset, len(versions) + 1, fields)
        versions.append(schema)
        return schema

    def versions(self, dataset: str) -> List[Schema]:
        return list(self._store.get(dataset, []))

    def latest(self, dataset: str) -> Optional[Schema]:
        versions = self._store.get(dataset, [])
        return versions[-1] if versions else None

    def datasets(self) -> List[str]:
        return list(self._store.keys())


def changes_frame(changes: List[Change]) -> pd.DataFrame:
    """Flat table of every change, for display / export."""
    rows = [{
        "field": c.field, "kind": c.kind, "detail": c.detail,
        "breaks_backward": c.breaks_backward, "breaks_forward": c.breaks_forward,
        "reason": c.reason,
    } for c in changes]
    cols = ["field", "kind", "detail", "breaks_backward", "breaks_forward", "reason"]
    return pd.DataFrame(rows, columns=cols)


def make_sample_registry() -> Registry:
    """A `customer_events` dataset with two registered versions.

    v1 -> v2 adds an optional field and widens a type (safe). Two PROPOSED v3
    schemas are returned via the helper below: one BREAKING, one safe.
    """
    reg = Registry()
    reg.register("customer_events", [
        Field("event_id", "long", nullable=False, required=True),
        Field("user_id", "long", nullable=False, required=True),
        Field("amount", "int", nullable=False, required=True),
        Field("country", "string", nullable=True, required=False),
    ])
    # v2: add an optional field (backward-safe) + widen amount int->long (safe).
    reg.register("customer_events", [
        Field("event_id", "long", nullable=False, required=True),
        Field("user_id", "long", nullable=False, required=True),
        Field("amount", "long", nullable=False, required=True),   # widened
        Field("country", "string", nullable=True, required=False),
        Field("device", "string", nullable=True, required=False),  # new optional
    ])
    return reg


def sample_proposals() -> Dict[str, List[Field]]:
    """Two proposed v3 schemas against customer_events v2."""
    breaking = [
        Field("event_id", "long", nullable=False, required=True),
        Field("user_id", "long", nullable=False, required=True),
        Field("amount", "int", nullable=False, required=True),      # long->int NARROWING = BREAKING
        Field("country", "string", nullable=False, required=True),  # nullable->required = BREAKING
        Field("device", "string", nullable=True, required=False),
    ]
    safe = [
        Field("event_id", "long", nullable=False, required=True),
        Field("user_id", "long", nullable=False, required=True),
        Field("amount", "double", nullable=False, required=True),    # long->double WIDENING = safe
        Field("country", "string", nullable=True, required=False),
        Field("device", "string", nullable=True, required=False),
        Field("session_id", "string", nullable=True, required=False),  # new optional = safe
    ]
    return {"breaking": breaking, "safe": safe}


# --- CLI -------------------------------------------------------------------

def _print_history(reg: Registry, dataset: str) -> None:
    print(f"=== version history: {dataset} ===")
    for s in reg.versions(dataset):
        fields = ", ".join(
            f"{f.name}:{f.type}"
            f"{'?' if f.nullable else ''}{'' if f.required else '(opt)'}"
            for f in s.fields
        )
        print(f"  v{s.version}: {fields}")
    print()


def _print_report(reg: Registry, dataset: str, name: str, fields: List[Field]) -> None:
    latest = reg.latest(dataset)
    proposed = Schema(dataset, (latest.version + 1) if latest else 1, fields)
    result = compare(latest, proposed)
    against = f"v{latest.version}" if latest else "(none)"
    print(f"--- proposed change '{name}' vs latest {against} ---")
    print(f"VERDICT: {result['verdict']}  |  {result['summary']}")
    changes = result["changes"]
    if not changes:
        print("  (no field changes)\n")
        return
    for c in changes:
        flags = []
        if c.breaks_backward:
            flags.append("breaks-backward")
        if c.breaks_forward:
            flags.append("breaks-forward")
        flag = f" [{', '.join(flags)}]" if flags else " [safe]"
        print(f"  - {c.kind}: {c.detail}{flag}")
        print(f"      WHY: {c.reason}")
    print()


def _cli() -> None:
    reg = make_sample_registry()
    print("=== Schema Registry ===\n")
    _print_history(reg, "customer_events")
    proposals = sample_proposals()
    _print_report(reg, "customer_events", "breaking", proposals["breaking"])
    _print_report(reg, "customer_events", "safe", proposals["safe"])


if __name__ == "__main__":
    _cli()
