from __future__ import annotations

# Lightweight Data Catalog - point it at one or more DataFrames/CSVs and it
# auto-profiles each as a catalog entry: table shape plus per-column metadata
# (dtype, null %, distinct count, sample values, and an INFERRED semantic type
# like id / email / date / category / numeric). Humans then attach the context a
# machine can't guess - owner, description, tags - and everyone can search across
# it and export a Markdown data dictionary. Fully offline, pandas/numpy only.
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnMeta:
    """Profiled metadata for one column. The semantic_type is a GUESS from the
    data - a human should confirm it (see README Impact Note)."""

    name: str
    dtype: str
    null_pct: float
    distinct: int
    semantic_type: str          # id | email | date | category | numeric | text | boolean
    samples: List[str] = field(default_factory=list)


@dataclass
class CatalogEntry:
    """One table in the catalog: auto-profiled shape + columns, plus the human
    metadata (owner / description / tags) that makes it discoverable."""

    name: str
    n_rows: int
    n_cols: int
    columns: List[ColumnMeta] = field(default_factory=list)
    owner: str = "unassigned"
    description: str = ""
    tags: List[str] = field(default_factory=list)


# Regexes are cheap, offline heuristics - not validators. They classify the
# SHAPE of sampled values so a steward gets a starting point, not a verdict.
_EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
_ID_HINT = ("id", "uuid", "key", "code")


def _infer_semantic_type(name: str, s: pd.Series) -> str:
    """Guess what a column MEANS from its name + values. Order matters: check
    the most specific / cheapest signals first, fall back to broad dtype kinds."""
    non_null = s.dropna()
    if len(non_null) == 0:
        return "unknown"                     # all-null: nothing to infer from

    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"

    lname = name.lower()
    # Name hint + high uniqueness = identifier. High uniqueness alone isn't
    # enough (a numeric measure can be unique too), so we require the name hint.
    uniq_ratio = non_null.nunique() / len(non_null)
    if any(h in lname for h in _ID_HINT) and uniq_ratio > 0.9:
        return "id"

    if pd.api.types.is_numeric_dtype(s):
        return "numeric"

    # Object/string columns: sniff a sample for emails, then dates, else decide
    # category vs free text by how repetitive the values are.
    sample = non_null.astype(str).head(200)
    if sample.str.match(_EMAIL_RE).mean() > 0.8:
        return "email"
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    if parsed.notna().mean() > 0.8:
        return "date"
    # Low distinct-ratio => a bounded set of labels (category); otherwise text.
    return "category" if uniq_ratio < 0.5 else "text"


def _profile_column(name: str, s: pd.Series) -> ColumnMeta:
    n = len(s)
    null_pct = round(float(s.isna().mean()) * 100, 1) if n else 0.0
    distinct = int(s.dropna().nunique())
    # Show up to 3 real values so a reader recognizes the column at a glance.
    samples = [str(v) for v in s.dropna().unique()[:3]]
    sem = _infer_semantic_type(name, s)
    return ColumnMeta(name, str(s.dtype), null_pct, distinct, sem, samples)


def profile_dataframe(
    df: pd.DataFrame,
    name: str,
    owner: str = "unassigned",
    description: str = "",
    tags: Optional[List[str]] = None,
) -> CatalogEntry:
    """Auto-profile a DataFrame into a catalog entry. Empty frames are handled
    gracefully - you still get an entry (0 rows / columns), never a crash."""
    cols = [_profile_column(c, df[c]) for c in df.columns]
    return CatalogEntry(
        name=name,
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        columns=cols,
        owner=owner,
        description=description,
        tags=list(tags) if tags else [],
    )


class Catalog:
    """A registry of catalog entries with search and Markdown export."""

    def __init__(self) -> None:
        self.entries: Dict[str, CatalogEntry] = {}

    def add(self, entry: CatalogEntry) -> None:
        self.entries[entry.name] = entry     # name is the key; re-adding updates

    def search(self, query: str) -> List[CatalogEntry]:
        """Free-text search across table names, column names, descriptions, and
        tags - the whole point is that nobody should have to ping three people
        to find where a field lives. Case-insensitive, substring match."""
        q = query.strip().lower()
        if not q:
            return list(self.entries.values())
        hits: List[CatalogEntry] = []
        for e in self.entries.values():
            haystack = [e.name, e.description, " ".join(e.tags)]
            haystack += [c.name for c in e.columns]
            haystack += [c.semantic_type for c in e.columns]
            if any(q in field.lower() for field in haystack):
                hits.append(e)
        return hits

    def to_markdown(self) -> str:
        """Export the whole catalog as a Markdown data dictionary - paste it into
        a wiki so the knowledge outlives any one person's memory."""
        lines: List[str] = ["# Data Catalog", ""]
        lines.append(f"_{len(self.entries)} tables catalogued._")
        lines.append("")
        for e in self.entries.values():
            lines.append(f"## {e.name}")
            tags = ", ".join(e.tags) if e.tags else "-"
            lines.append(f"- **Owner:** {e.owner}")
            lines.append(f"- **Description:** {e.description or '-'}")
            lines.append(f"- **Tags:** {tags}")
            lines.append(f"- **Shape:** {e.n_rows:,} rows x {e.n_cols} columns")
            lines.append("")
            lines.append("| Column | Type | Semantic | Null % | Distinct | Samples |")
            lines.append("|---|---|---|---|---|---|")
            for c in e.columns:
                samples = ", ".join(c.samples) if c.samples else "-"
                lines.append(
                    f"| {c.name} | {c.dtype} | {c.semantic_type} | "
                    f"{c.null_pct}% | {c.distinct} | {samples} |"
                )
            lines.append("")
        return "\n".join(lines)


def make_sample_tables() -> Dict[str, pd.DataFrame]:
    """Two related tables (customers, orders) so search + relationships are
    visible in the demo. customer_id is the shared key across both."""
    rng = np.random.default_rng(7)
    n_cust = 50
    customers = pd.DataFrame({
        "customer_id": [f"C{1000 + i}" for i in range(n_cust)],
        "email": [f"user{i}@shop.com" for i in range(n_cust)],
        "signup_date": pd.to_datetime("2024-01-01")
        + pd.to_timedelta(rng.integers(0, 400, n_cust), unit="D"),
        "region": rng.choice(["North", "South", "East", "West"], n_cust),
        "lifetime_value": rng.normal(500, 150, n_cust).round(2),
    })

    n_ord = 200
    orders = pd.DataFrame({
        "order_id": [f"O{5000 + i}" for i in range(n_ord)],
        "customer_id": rng.choice(customers["customer_id"], n_ord),
        "order_date": pd.to_datetime("2024-02-01")
        + pd.to_timedelta(rng.integers(0, 300, n_ord), unit="D"),
        "amount": rng.normal(120, 40, n_ord).round(2),
        "status": rng.choice(["shipped", "pending", "returned"], n_ord,
                             p=[0.7, 0.2, 0.1]),
    })
    return {"customers": customers, "orders": orders}


def _cli() -> None:
    tables = make_sample_tables()
    cat = Catalog()
    # Attach the human metadata a machine can't guess.
    cat.add(profile_dataframe(
        tables["customers"], "customers", owner="growth-team",
        description="One row per registered customer with contact + value.",
        tags=["crm", "pii", "core"],
    ))
    cat.add(profile_dataframe(
        tables["orders"], "orders", owner="commerce-team",
        description="One row per order placed, linked to customers by customer_id.",
        tags=["transactions", "core"],
    ))

    print("=== Lightweight Data Catalog ===\n")
    for e in cat.entries.values():
        print(f"{e.name}: {e.n_rows:,} rows x {e.n_cols} cols  "
              f"(owner: {e.owner}, tags: {', '.join(e.tags)})")
        for c in e.columns:
            print(f"    - {c.name:<16} {c.semantic_type:<9} "
                  f"null={c.null_pct}%  distinct={c.distinct}")
        print()

    print('--- search: "email" ---')
    for e in cat.search("email"):
        cols = [c.name for c in e.columns if "email" in c.name.lower()
                or c.semantic_type == "email"]
        print(f"  {e.name}  ->  matched columns: {cols or '(name/desc/tag match)'}")


if __name__ == "__main__":
    _cli()
