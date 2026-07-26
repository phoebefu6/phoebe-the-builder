from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class Panel:
    """One recommended dashboard panel: a chart type + its encoding + why it was chosen."""

    title: str
    chart: str            # kpi | line | bar | grouped_bar | scatter | histogram
    x: str = ""
    y: str = ""
    series: str = ""
    agg: str = "sum"
    rationale: str = ""


@dataclass
class ColumnRole:
    name: str
    role: str  # temporal | categorical | measure | high_card_id


def _is_temporal(name: str, series: pd.Series) -> bool:
    if re.search(r"(date|_at$|_on$|time|day|month|year|week|quarter)", name, re.I):
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    # try parsing a sample of object values as dates
    if series.dtype == object:
        sample = series.dropna().astype(str).head(10)
        hits = sum(bool(re.match(r"\d{4}-\d{2}(-\d{2})?", v)) for v in sample)
        return hits >= max(1, len(sample) // 2)
    return False


def classify_columns(df: pd.DataFrame) -> list[ColumnRole]:
    """Assign each column a role - the basis for every chart recommendation."""
    roles = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        if _is_temporal(col, s):
            roles.append(ColumnRole(col, "temporal"))
        elif pd.api.types.is_numeric_dtype(s):
            # numeric but very high-cardinality unique ints that look like ids
            if re.search(r"(^id$|_id$|key$)", col, re.I) and s.nunique() > 0.9 * n:
                roles.append(ColumnRole(col, "high_card_id"))
            else:
                roles.append(ColumnRole(col, "measure"))
        else:
            if s.nunique() > 0.5 * n and n > 20:
                roles.append(ColumnRole(col, "high_card_id"))
            else:
                roles.append(ColumnRole(col, "categorical"))
    return roles


def recommend_dashboard(df: pd.DataFrame, max_panels: int = 6) -> list[Panel]:
    """Recommend a coherent set of panels from the data's shape, using encoding best practices."""
    roles = classify_columns(df)
    temporal = [r.name for r in roles if r.role == "temporal"]
    cats = [r.name for r in roles if r.role == "categorical"]
    measures = [r.name for r in roles if r.role == "measure"]

    panels: list[Panel] = []

    # 1) KPI cards for the top measures (big-number summary)
    for m in measures[:2]:
        panels.append(Panel(
            title=f"Total {m}", chart="kpi", y=m, agg="sum",
            rationale="A headline number gives the dashboard an at-a-glance summary."))

    # 2) trend: temporal x measure -> line
    if temporal and measures:
        panels.append(Panel(
            title=f"{measures[0]} over time", chart="line", x=temporal[0], y=measures[0], agg="sum",
            rationale="A time column + a measure is a trend - lines show change over time best."))

    # 3) category breakdown: categorical x measure -> bar
    if cats and measures:
        panels.append(Panel(
            title=f"{measures[0]} by {cats[0]}", chart="bar", x=cats[0], y=measures[0], agg="sum",
            rationale="A category vs a measure compares magnitudes - bars beat pies for that."))

    # 4) two categoricals + measure -> grouped bar
    if len(cats) >= 2 and measures:
        panels.append(Panel(
            title=f"{measures[0]} by {cats[0]} and {cats[1]}", chart="grouped_bar",
            x=cats[0], y=measures[0], series=cats[1], agg="sum",
            rationale="Two dimensions over one measure - group bars by the second dimension."))

    # 5) two measures -> scatter (relationship)
    if len(measures) >= 2:
        panels.append(Panel(
            title=f"{measures[0]} vs {measures[1]}", chart="scatter", x=measures[0], y=measures[1],
            rationale="Two measures invite a correlation check - scatter reveals the relationship."))

    # 6) distribution of the primary measure -> histogram
    if measures:
        panels.append(Panel(
            title=f"Distribution of {measures[0]}", chart="histogram", x=measures[0],
            rationale="A single measure's shape (skew, outliers) is best seen as a histogram."))

    return panels[:max_panels]


def spec_to_dict(panels: list[Panel]) -> list[dict]:
    out = []
    for p in panels:
        d = {"title": p.title, "chart": p.chart}
        for k in ("x", "y", "series", "agg"):
            v = getattr(p, k)
            if v:
                d[k] = v
        d["rationale"] = p.rationale
        out.append(d)
    return out


def sample_dataframe() -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(3)
    n = 120
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "region": rng.choice(["US", "EU", "APAC"], n),
            "channel": rng.choice(["organic", "paid", "referral"], n),
            "revenue": np.round(rng.gamma(3, 120, n), 2),
            "sessions": rng.integers(50, 500, n),
        }
    )
