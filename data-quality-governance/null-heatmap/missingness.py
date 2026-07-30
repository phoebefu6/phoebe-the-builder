from __future__ import annotations

# Null Heatmap - core logic.
#
# "Where's the missing data?" gets answered with df.isna().sum() - a per-column
# count that hides the only thing that matters: whether nulls are INDEPENDENT or
# CORRELATED.
#
# 8% missing spread evenly across rows is a nuisance you impute. 8% missing that
# always lands on the same rows is a broken join or a late-arriving source, and
# dropna() will silently delete that entire population - which is how a cohort
# disappears from a model's training data without anyone noticing.
#
# This module reports:
#   - per-column completeness and null counts
#   - the co-missingness matrix (which columns go missing together)
#   - row-level patterns (the distinct null signatures, ranked)
#   - the dropna() cost, and WHO gets dropped (bias check against a segment)
#   - a mechanism guess per column: structural / correlated / scattered
#
# Fully offline: pandas + numpy, works on any DataFrame.
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# A column missing in ~everything or ~nothing is not interesting; the middle is.
STRUCTURAL_MIN_JACCARD = 0.9   # co-missing this often => same root cause
SCATTERED_MAX_JACCARD = 0.25   # co-missing this rarely => independent


@dataclass
class ColumnReport:
    column: str
    n_missing: int
    completeness: float
    mechanism: str          # "complete" | "structural" | "correlated" | "scattered"
    partner: Optional[str]  # the column it most co-occurs with
    partner_jaccard: float
    note: str = ""
    # Segment dependence is a SEPARATE axis from column-pair lockstep: a column
    # can be in perfect lockstep with a sibling (structural) and also land almost
    # entirely on one segment. The first fact tells you how many root causes
    # there are; the second tells you who pays for them.
    segment_spread: float = 0.0
    segment_note: str = ""


# Completeness gap across segment values above which the nulls are not
# population-neutral and dropna() becomes a sampling decision.
SEGMENT_SKEW_THRESHOLD = 0.30


def column_report(df: pd.DataFrame, segment: Optional[str] = None) -> List[ColumnReport]:
    """Per-column completeness plus a guess at the missingness mechanism."""
    miss = df.isna()
    jac = co_missing_matrix(df)
    out: List[ColumnReport] = []

    seg_completeness = None
    if segment and segment in df.columns:
        # Group the boolean null-mask directly rather than groupby().apply() - the
        # apply form needs pandas>=2.2 for include_groups and is slower besides.
        mask = miss.copy()
        mask.insert(0, "__seg__", df[segment].to_numpy())
        seg_completeness = 1 - mask.groupby("__seg__", observed=True).mean()

    for col in df.columns:
        n = int(miss[col].sum())
        completeness = round(1 - n / len(df), 4) if len(df) else 1.0
        if n == 0:
            out.append(ColumnReport(col, 0, completeness, "complete", None, 0.0,
                                    "no nulls"))
            continue

        # Strongest co-missing partner, excluding itself and fully-complete columns.
        others = [c for c in df.columns if c != col and miss[c].any()]
        if others:
            partner = max(others, key=lambda c: jac.loc[col, c])
            pj = float(jac.loc[col, partner])
        else:
            partner, pj = None, 0.0

        if pj >= STRUCTURAL_MIN_JACCARD:
            mech = "structural"
            note = f"missing in lockstep with '{partner}' - one root cause, not two"
        elif pj > SCATTERED_MAX_JACCARD:
            mech = "correlated"
            note = f"overlaps '{partner}' ({pj:.0%}) - check the shared upstream source"
        else:
            mech = "scattered"
            note = "independent of other columns - imputable"

        spread, seg_note = 0.0, ""
        if seg_completeness is not None and col in seg_completeness.columns:
            vals = seg_completeness[col]
            spread = float(vals.max() - vals.min())
            if spread >= SEGMENT_SKEW_THRESHOLD:
                worst = vals.idxmin()
                seg_note = (
                    f"segment-skewed: '{worst}' is only {vals.min():.0%} complete vs "
                    f"{vals.max():.0%} elsewhere - dropping these rows removes a population"
                )

        out.append(ColumnReport(col, n, completeness, mech, partner, round(pj, 3), note,
                                round(spread, 3), seg_note))

    return sorted(out, key=lambda r: r.completeness)


def co_missing_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Jaccard overlap of null positions between every pair of columns.

    Jaccard, not phi correlation. Both agree on perfect lockstep (each returns
    1.0), but phi is base-rate dependent: two columns sharing half their nulls
    score phi=0.49 at a 2% null rate and phi=0.29 at a 30% rate - the same
    overlap, two different numbers. Since every pair in a real table has a
    different base rate, phi values are not comparable across the matrix.
    Jaccard returns 0.33 for both, because it asks only the question being
    asked: of the rows missing in either column, what share are missing in both?
    """
    miss = df.isna()
    cols = list(df.columns)
    m = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i:]:
            union = int((miss[a] | miss[b]).sum())
            inter = int((miss[a] & miss[b]).sum())
            # No nulls in either column: they are perfectly consistent, but there
            # is nothing to overlap. 0/0 is 0 here, not 1 - calling it a perfect
            # match would flag every clean column pair as "structural".
            val = round(inter / union, 4) if union else 0.0
            m.loc[a, b] = val
            m.loc[b, a] = val
    return m


def row_patterns(df: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """Distinct null signatures across rows, ranked by frequency.

    This is the view df.isna().sum() cannot give you: whether the nulls form a
    handful of repeated shapes (a systematic cause) or thousands of unique ones
    (random noise).
    """
    miss = df.isna()
    sig = miss.apply(
        lambda r: ", ".join(c for c in df.columns if r[c]) or "(complete)", axis=1
    )
    counts = sig.value_counts().head(top)
    return pd.DataFrame({
        "missing_columns": counts.index,
        "rows": counts.to_numpy(),
        "share": (counts / len(df)).round(4).to_numpy(),
        "n_cols_missing": [0 if s == "(complete)" else len(s.split(", ")) for s in counts.index],
    })


def dropna_cost(
    df: pd.DataFrame, segment: Optional[str] = None
) -> Dict[str, object]:
    """What dropna() actually costs - and who pays it.

    The headline number (rows lost) is the easy part. The part that matters for
    governance is whether the loss is evenly distributed: if one segment loses
    40% of its rows and another loses 2%, every downstream model is now trained
    on a skewed population.
    """
    complete = df.dropna()
    kept, total = len(complete), len(df)
    out: Dict[str, object] = {
        "rows_total": total,
        "rows_kept": kept,
        "rows_dropped": total - kept,
        "share_dropped": round((total - kept) / total, 4) if total else 0.0,
    }
    if segment and segment in df.columns:
        before = df[segment].value_counts(normalize=True)
        after = complete[segment].value_counts(normalize=True)
        rows = []
        for value in before.index:
            b = float(before.get(value, 0.0))
            a = float(after.get(value, 0.0))
            n_before = int((df[segment] == value).sum())
            n_after = int((complete[segment] == value).sum())
            rows.append({
                "segment": value,
                "rows_before": n_before,
                "rows_after": n_after,
                "retained": round(n_after / n_before, 4) if n_before else 0.0,
                "share_before": round(b, 4),
                "share_after": round(a, 4),
                "share_shift_pp": round((a - b) * 100, 2),
            })
        seg = pd.DataFrame(rows).sort_values("retained")
        out["segment_impact"] = seg
        worst = seg.iloc[0]
        best = seg.iloc[-1]
        out["bias_warning"] = (
            f"'{worst['segment']}' keeps {worst['retained']:.0%} of its rows while "
            f"'{best['segment']}' keeps {best['retained']:.0%} - dropna() does not "
            f"drop rows, it drops a population."
        ) if worst["retained"] < best["retained"] - 0.15 else None
    return out


def completeness_by_segment(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Per-column completeness split by a segment - where the gap actually lives."""
    if segment not in df.columns:
        raise KeyError(f"{segment!r} is not a column")
    cols = [c for c in df.columns if c != segment]
    rows = []
    for value, chunk in df.groupby(segment, observed=True):
        row = {"segment": value, "rows": len(chunk)}
        for c in cols:
            row[c] = round(1 - chunk[c].isna().mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


def summary(df: pd.DataFrame, segment: Optional[str] = None) -> Dict[str, object]:
    reports = column_report(df, segment)
    cells = df.size
    nulls = int(df.isna().sum().sum())
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "cell_completeness": round(1 - nulls / cells, 4) if cells else 1.0,
        "columns_with_nulls": sum(1 for r in reports if r.n_missing),
        "mechanisms": {
            m: sum(1 for r in reports if r.mechanism == m)
            for m in ("complete", "structural", "correlated", "scattered")
        },
        "reports": reports,
        "dropna": dropna_cost(df, segment),
    }


# --------------------------------------------------------------------------
# Sample data - three DIFFERENT mechanisms in one frame, on purpose
# --------------------------------------------------------------------------


def sample_frame(n: int = 800) -> pd.DataFrame:
    """A customer table where the same 6% null rate means three different things.

    - scattered:   `age` nulls hit random rows (a genuinely optional field)
    - structural:  `card_last4` + `card_expiry` go missing together (one join failed)
    - segment-skewed: `last_login`/`sessions` are ALSO in lockstep with each
                   other, but their nulls land almost entirely on one signup
                   channel (a source that came online late). Same headline rate,
                   completely different consequence - dropna() deletes that
                   channel's customers, not a random 15% of rows.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    channel = rng.choice(["web", "mobile", "partner"], size=n, p=[0.5, 0.35, 0.15])

    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "channel": channel,
        "signup_date": pd.date_range("2026-01-01", periods=n, freq="3h").astype(str),
        "age": rng.integers(18, 78, n).astype(float),
        "country": rng.choice(["SG", "MY", "ID", "TH"], n),
        "card_last4": [f"{x:04d}" for x in rng.integers(0, 9999, n)],
        "card_expiry": [f"{m:02d}/2{rng.integers(7, 9)}" for m in rng.integers(1, 13, n)],
        "last_login": pd.date_range("2026-07-01", periods=n, freq="1h").astype(str),
        "sessions": rng.integers(1, 60, n).astype(float),
        "lifetime_value": (rng.gamma(2.0, 120.0, n)).round(2),
    })

    # 1 - scattered: random 6% of rows lose `age`
    df.loc[rng.choice(n, size=int(0.06 * n), replace=False), "age"] = np.nan

    # 2 - structural: the payment-service join failed for 9% of rows, so BOTH
    #     card columns vanish on exactly the same rows.
    card_fail = rng.choice(n, size=int(0.09 * n), replace=False)
    df.loc[card_fail, ["card_last4", "card_expiry"]] = np.nan

    # 3 - correlated + biased: the activity source came online late and barely
    #     covers the `partner` channel. Same headline rate, completely different
    #     consequence - and dropna() will delete most partner customers.
    p_partner = np.where(channel == "partner", 0.80, 0.04)
    activity_fail = rng.random(n) < p_partner
    df.loc[activity_fail, ["last_login", "sessions"]] = np.nan

    return df


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    df = sample_frame()
    s = summary(df, segment="channel")

    print("=" * 88)
    print(f"NULL HEATMAP - {s['rows']} rows x {s['columns']} columns, "
          f"{s['cell_completeness']:.2%} of cells populated")
    print("=" * 88)

    print("\nPER-COLUMN (the view df.isna().sum() gives you, plus the mechanism)")
    rep = pd.DataFrame([{
        "column": r.column, "n_missing": r.n_missing, "completeness": r.completeness,
        "mechanism": r.mechanism, "partner": r.partner or "-",
        "jaccard": r.partner_jaccard, "seg_spread": r.segment_spread,
    } for r in s["reports"]])
    print(rep.to_string(index=False))

    print("\nMECHANISM MIX:", s["mechanisms"])
    print("  Column-pair lockstep and segment skew are different axes:")
    for r in s["reports"]:
        if r.segment_note:
            print(f"    {r.column:<14} {r.segment_note}")

    print("\n" + "-" * 88)
    print("CO-MISSINGNESS (Jaccard overlap of null positions, nulls-only columns)")
    jac = co_missing_matrix(df)
    nulled = [c for c in df.columns if df[c].isna().any()]
    print(jac.loc[nulled, nulled].to_string())

    print("\n" + "-" * 88)
    print("ROW PATTERNS (distinct null signatures)")
    print(row_patterns(df).to_string(index=False))

    print("\n" + "-" * 88)
    d = s["dropna"]
    print(f"DROPNA COST: {d['rows_dropped']} of {d['rows_total']} rows "
          f"({d['share_dropped']:.1%}) would be deleted")
    if "segment_impact" in d:
        print(d["segment_impact"].to_string(index=False))
    if d.get("bias_warning"):
        print(f"\n  BIAS: {d['bias_warning']}")

    print("\n" + "-" * 88)
    print("COMPLETENESS BY SEGMENT")
    print(completeness_by_segment(df, "channel").to_string(index=False))


if __name__ == "__main__":
    main()
