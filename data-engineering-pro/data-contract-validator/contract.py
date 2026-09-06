from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import yaml

TYPE_CHECKS = {
    "int": pd.api.types.is_integer_dtype,
    "float": lambda s: pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s),
    "string": lambda s: pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s),
    "datetime": pd.api.types.is_datetime64_any_dtype,
    "bool": pd.api.types.is_bool_dtype,
}


@dataclass
class Violation:
    level: str  # error | warning
    column: str
    rule: str
    detail: str


def load_contract(text: str) -> Dict:
    return yaml.safe_load(text)


def validate_dataset(df: pd.DataFrame, contract: Dict) -> List[Violation]:
    """Check a dataset against its contract. Errors fail the build; warnings inform."""
    v: List[Violation] = []
    cols = {c["name"]: c for c in contract.get("columns", [])}

    for name, spec in cols.items():
        if name not in df.columns:
            v.append(Violation("error", name, "missing-column", "column absent from dataset"))
            continue
        s = df[name]
        ctype = spec.get("type", "string")
        check = TYPE_CHECKS.get(ctype)
        series = s
        if ctype == "datetime" and pd.api.types.is_object_dtype(s):
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.99:
                series = parsed
        if check and not check(series):
            v.append(Violation("error", name, "type-mismatch",
                               f"expected {ctype}, got {s.dtype}"))
            continue
        if not spec.get("nullable", True):
            nulls = int(s.isna().sum())
            if nulls:
                v.append(Violation("error", name, "null-violation",
                                   f"{nulls} nulls in non-nullable column"))
        if spec.get("unique"):
            dupes = int(s.duplicated().sum())
            if dupes:
                v.append(Violation("error", name, "uniqueness", f"{dupes} duplicate values"))
        allowed = spec.get("allowed")
        if allowed:
            bad = s.dropna()[~s.dropna().isin(allowed)]
            if len(bad):
                v.append(Violation("error", name, "allowed-values",
                                   f"{len(bad)} rows outside {allowed} (e.g. {bad.iloc[0]!r})"))
        if "min" in spec:
            below = int((pd.to_numeric(s, errors="coerce") < spec["min"]).sum())
            if below:
                v.append(Violation("error", name, "min-bound", f"{below} rows below min={spec['min']}"))
        if "max" in spec:
            above = int((pd.to_numeric(s, errors="coerce") > spec["max"]).sum())
            if above:
                v.append(Violation("error", name, "max-bound", f"{above} rows above max={spec['max']}"))

    extra = [c for c in df.columns if c not in cols]
    for name in extra:
        v.append(Violation("warning", name, "undeclared-column",
                           "column present but not in contract — declare it or drop it"))

    fresh = contract.get("freshness")
    if fresh:
        col, max_age_h = fresh["column"], fresh["max_age_hours"]
        if col in df.columns:
            newest = pd.to_datetime(df[col], errors="coerce", format="mixed").max()
            asof = pd.Timestamp(fresh.get("as_of")) if fresh.get("as_of") else pd.Timestamp.now()
            age_h = (asof - newest).total_seconds() / 3600
            if age_h > max_age_h:
                v.append(Violation("error", col, "freshness",
                                   f"newest row is {age_h:.1f}h old (max {max_age_h}h)"))
    return v


def diff_contracts(old: Dict, new: Dict) -> List[Violation]:
    """Compare contract versions: what would break consumers vs what's safe."""
    v: List[Violation] = []
    old_cols = {c["name"]: c for c in old.get("columns", [])}
    new_cols = {c["name"]: c for c in new.get("columns", [])}

    for name, spec in old_cols.items():
        if name not in new_cols:
            v.append(Violation("error", name, "breaking:removed", "column removed — consumers will break"))
            continue
        ns = new_cols[name]
        if ns.get("type") != spec.get("type"):
            v.append(Violation("error", name, "breaking:type-change",
                               f"{spec.get('type')} → {ns.get('type')}"))
        if spec.get("nullable", True) is False and ns.get("nullable", True) is True:
            v.append(Violation("error", name, "breaking:nullable-loosened",
                               "non-nullable became nullable — consumer assumptions break"))
        if ns.get("nullable", True) is False and spec.get("nullable", True) is True:
            v.append(Violation("warning", name, "tightened:nullable",
                               "nullable became non-nullable — safe for consumers, hard for producer"))
    for name, ns in new_cols.items():
        if name in old_cols:
            continue
        if ns.get("nullable", True):
            v.append(Violation("warning", name, "safe:added-optional", "new optional column — non-breaking"))
        else:
            v.append(Violation("error", name, "breaking:added-required",
                               "new required column — old producers will fail validation"))
    return v


def exit_code(violations: List[Violation]) -> int:
    """CI semantics: 1 if any error, else 0."""
    return 1 if any(x.level == "error" for x in violations) else 0


def report(violations: List[Violation]) -> str:
    if not violations:
        return "✅ contract check passed — no violations"
    lines = []
    for x in sorted(violations, key=lambda x: (x.level != "error", x.column)):
        icon = "✗" if x.level == "error" else "⚠"
        lines.append(f"{icon} [{x.level:7s}] {x.column}: {x.rule} — {x.detail}")
    n_err = sum(1 for x in violations if x.level == "error")
    lines.append(f"\n{n_err} error(s), {len(violations) - n_err} warning(s) → exit {exit_code(violations)}")
    return "\n".join(lines)
