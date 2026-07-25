from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ColumnTests:
    """The dbt tests suggested for one column, with the profile evidence behind them."""

    name: str
    tests: list[str] = field(default_factory=list)
    accepted_values: list = field(default_factory=list)
    relationship: Optional[str] = None  # inferred FK target, e.g. 'ref(users).id'
    reason: dict = field(default_factory=dict)  # test -> why


def _uniqueness(series: pd.Series) -> float:
    n = series.dropna().shape[0]
    return series.dropna().nunique() / n if n else 0.0


def suggest_tests(
    name: str,
    series: pd.Series,
    unique_threshold: float = 0.99,
    category_max: int = 12,
) -> ColumnTests:
    """Pick dbt tests for a column from its profile. Deterministic and explainable.

    - not_null: fired when the column has (nearly) no nulls - it looks required.
    - unique: fired when values are ~all distinct - it looks like a key.
    - accepted_values: fired for low-cardinality text - a natural enum.
    - relationships: inferred when the name looks like a foreign key (*_id).
    """
    ct = ColumnTests(name=name)
    n = len(series)
    null_frac = series.isna().sum() / n if n else 1.0
    uniq = _uniqueness(series)
    nunique = series.dropna().nunique()

    if null_frac == 0:
        ct.tests.append("not_null")
        ct.reason["not_null"] = "0% null in sample"
    elif null_frac < 0.02:
        ct.tests.append("not_null")
        ct.reason["not_null"] = f"{null_frac:.1%} null (looks required)"

    is_key_name = bool(re.search(r"(^id$|_id$|^id_|uuid|key$)", name, re.I))
    if uniq >= unique_threshold and (is_key_name or name.lower() == "id"):
        ct.tests.append("unique")
        ct.reason["unique"] = f"{uniq:.0%} distinct + key-like name"
    elif uniq >= unique_threshold:
        ct.tests.append("unique")
        ct.reason["unique"] = f"{uniq:.0%} distinct values"

    # accepted_values for low-cardinality text - but NOT when it's near-unique (that's a key,
    # e.g. an email, not an enum).
    if (
        not pd.api.types.is_numeric_dtype(series)
        and 1 < nunique <= category_max
        and uniq < unique_threshold
    ):
        vals = sorted(str(v) for v in series.dropna().unique())
        ct.accepted_values = vals
        ct.tests.append("accepted_values")
        ct.reason["accepted_values"] = f"{nunique} distinct categories"

    # relationship inference from name (not id itself)
    m = re.match(r"^(.*)_id$", name, re.I)
    if m and name.lower() != "id":
        target = _pluralize(m.group(1))
        ct.relationship = f"ref('{target}')"
        ct.tests.append("relationships")
        ct.reason["relationships"] = f"name '{name}' implies FK to {target}.{m.group(1)}_id or id"

    return ct


def _pluralize(word: str) -> str:
    if word.endswith("y"):
        return word[:-1] + "ies"
    if word.endswith("s"):
        return word
    return word + "s"


def generate_model_tests(df: pd.DataFrame, **kwargs) -> list[ColumnTests]:
    return [suggest_tests(col, df[col], **kwargs) for col in df.columns]


def to_schema_yml(model_name: str, columns: list[ColumnTests], description: str = "") -> str:
    """Render suggested tests as a dbt schema.yml - paste-ready into a dbt project."""
    lines = ["version: 2", "", "models:", f"  - name: {model_name}"]
    if description:
        lines.append(f"    description: \"{description}\"")
    lines.append("    columns:")
    for ct in columns:
        lines.append(f"      - name: {ct.name}")
        simple = [t for t in ct.tests if t not in ("accepted_values", "relationships")]
        has_complex = ct.accepted_values or ct.relationship
        if not simple and not has_complex:
            continue
        lines.append("        tests:")
        for t in simple:
            lines.append(f"          - {t}")
        if ct.accepted_values:
            vals = ", ".join(f'"{v}"' for v in ct.accepted_values)
            lines.append("          - accepted_values:")
            lines.append(f"              values: [{vals}]")
        if ct.relationship:
            lines.append("          - relationships:")
            lines.append(f"              to: {ct.relationship}")
            lines.append("              field: id")
    return "\n".join(lines)


def coverage(columns: list[ColumnTests]) -> dict:
    """How many columns got at least one test - the headline number for 'do we have tests?'."""
    tested = sum(1 for c in columns if c.tests)
    total = len(columns)
    by_test: dict = {}
    for c in columns:
        for t in c.tests:
            by_test[t] = by_test.get(t, 0) + 1
    return {"tested": tested, "total": total, "pct": round(100 * tested / total) if total else 0, "by_test": by_test}


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "user_id": [10, 11, 10, 12, 11, 13],
            "status": ["paid", "paid", "refunded", "paid", "pending", "paid"],
            "plan": ["pro", "free", "pro", "team", "free", "pro"],
            "amount": [100, 0, 50, 200, 0, 120],
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com", "f@x.com"],
        }
    )
