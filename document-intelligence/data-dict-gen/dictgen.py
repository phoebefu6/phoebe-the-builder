from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ColumnDoc:
    """Profile + generated documentation for one column."""

    name: str
    dtype: str
    semantic_type: str  # id | email | date | currency | category | boolean | numeric | text
    null_pct: float
    unique_count: int
    sample_values: list = field(default_factory=list)
    is_pii: bool = False
    description: str = ""


# name-based hints for semantic typing
_NAME_HINTS = [
    (re.compile(r"(^id$|_id$|^id_|uuid|guid)", re.I), "id"),
    (re.compile(r"email|e_mail", re.I), "email"),
    (re.compile(r"phone|mobile|tel", re.I), "phone"),
    (re.compile(r"date|_at$|_on$|timestamp|created|updated|dob|birth", re.I), "date"),
    (re.compile(r"price|amount|cost|revenue|salary|balance|fee|usd|total", re.I), "currency"),
    (re.compile(r"(^is_|^has_|_flag$|active|enabled)", re.I), "boolean"),
    (re.compile(r"name|first|last|address|ssn|zip|postal", re.I), "pii_text"),
]

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PII_SEMANTIC = {"email", "phone", "pii_text"}


def _infer_semantic_type(name: str, series: pd.Series) -> str:
    """Combine column-name hints with value inspection to classify the column."""
    for pat, label in _NAME_HINTS:
        if pat.search(name):
            return "pii_text" if label == "pii_text" else label

    non_null = series.dropna()
    if non_null.empty:
        return "text"

    # value-based checks
    sample = non_null.astype(str).head(50)
    if sample.map(lambda v: bool(_EMAIL_RE.match(v))).mean() > 0.8:
        return "email"

    if pd.api.types.is_bool_dtype(series) or set(non_null.unique()) <= {0, 1, True, False}:
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    # low-cardinality object -> category
    if non_null.nunique() <= max(20, int(0.1 * len(non_null))):
        return "category"

    return "text"


def _heuristic_description(col: ColumnDoc) -> str:
    """Plain-English description from profile signals - no LLM."""
    base = {
        "id": f"Unique identifier ({col.unique_count} distinct values).",
        "email": "Email address of the record.",
        "phone": "Phone number contact field.",
        "date": "Date/time value.",
        "currency": "Monetary amount.",
        "boolean": "True/false flag.",
        "category": f"Categorical field with {col.unique_count} distinct categories.",
        "numeric": "Numeric measure.",
        "pii_text": "Free-text field containing personal information.",
        "text": "Free-text field.",
    }.get(col.semantic_type, "Field.")
    if col.null_pct > 0:
        base += f" {col.null_pct:.0f}% missing."
    if col.sample_values:
        preview = ", ".join(str(v) for v in col.sample_values[:3])
        base += f" e.g. {preview}."
    return base


def profile_column(name: str, series: pd.Series) -> ColumnDoc:
    """Profile a single column into a ColumnDoc (no description yet)."""
    n = len(series)
    null_pct = 100.0 * series.isna().sum() / n if n else 0.0
    semantic = _infer_semantic_type(name, series)
    samples = [v for v in series.dropna().unique()[:5]]
    is_pii = semantic in _PII_SEMANTIC
    col = ColumnDoc(
        name=name,
        dtype=str(series.dtype),
        semantic_type=semantic,
        null_pct=round(null_pct, 1),
        unique_count=int(series.nunique()),
        sample_values=samples,
        is_pii=is_pii,
    )
    return col


def heuristic_dictionary(df: pd.DataFrame) -> list[ColumnDoc]:
    """Build a full data dictionary from a DataFrame with no LLM."""
    docs = []
    for name in df.columns:
        col = profile_column(name, df[name])
        col.description = _heuristic_description(col)
        docs.append(col)
    return docs


def llm_dictionary(df: pd.DataFrame, api_key: str, table_name: str = "table") -> list[ColumnDoc]:
    """Use Claude to write richer, business-context column descriptions from the profile."""
    import anthropic

    profiles = heuristic_dictionary(df)  # reuse profiling; Claude only rewrites descriptions
    compact = [
        {
            "name": c.name,
            "semantic_type": c.semantic_type,
            "null_pct": c.null_pct,
            "unique_count": c.unique_count,
            "samples": [str(v) for v in c.sample_values[:3]],
        }
        for c in profiles
    ]
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Write clear one-sentence business descriptions for each column in table "
                    f"'{table_name}'. Base descriptions ONLY on the profile - do not invent domain "
                    "facts. Respond with ONLY valid JSON, no markdown fences:\n"
                    '{"descriptions": {"column_name": "description", ...}}\n\n'
                    f"Column profiles:\n{json.dumps(compact, indent=2)}"
                ),
            }
        ],
    )
    data = json.loads(response.content[0].text.strip())
    descs = data.get("descriptions", {})
    for c in profiles:
        if c.name in descs:
            c.description = descs[c.name]
    return profiles


def generate_dictionary(
    df: pd.DataFrame, api_key: Optional[str] = None, table_name: str = "table"
) -> list[ColumnDoc]:
    """Generate a data dictionary for a DataFrame.

    Uses Claude for descriptions if an API key is available, else heuristics."""
    if df.empty:
        return []
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return heuristic_dictionary(df)
    return llm_dictionary(df, api_key, table_name=table_name)


def dictionary_to_dataframe(docs: list[ColumnDoc]) -> pd.DataFrame:
    rows = []
    for c in docs:
        d = asdict(c)
        d["sample_values"] = ", ".join(str(v) for v in c.sample_values[:3])
        rows.append(d)
    return pd.DataFrame(rows)


def dictionary_to_markdown(docs: list[ColumnDoc], table_name: str = "table") -> str:
    lines = [f"# Data Dictionary — `{table_name}`", "", "| Column | Type | Semantic | Null % | Unique | PII | Description |", "|---|---|---|---|---|---|---|"]
    for c in docs:
        pii = "⚠️ yes" if c.is_pii else "no"
        lines.append(
            f"| `{c.name}` | {c.dtype} | {c.semantic_type} | {c.null_pct} | {c.unique_count} | {pii} | {c.description} |"
        )
    return "\n".join(lines)


def sample_dataframe() -> pd.DataFrame:
    """A small messy table so the tool has something to document out of the box."""
    return pd.DataFrame(
        {
            "customer_id": [1001, 1002, 1003, 1004, 1005],
            "email": ["a@x.com", "b@y.com", None, "d@z.com", "e@w.com"],
            "signup_date": ["2026-01-05", "2026-02-11", "2026-02-28", "2026-03-15", "2026-04-01"],
            "plan": ["pro", "free", "pro", "team", "free"],
            "mrr_usd": [29.0, 0.0, 29.0, 59.0, 0.0],
            "is_active": [True, False, True, True, False],
            "full_name": ["Ada Lovelace", "Alan Turing", "Grace Hopper", "Ken Thompson", "Katherine Johnson"],
        }
    )
