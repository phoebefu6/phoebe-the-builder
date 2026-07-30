from __future__ import annotations

# Pivot Narrator - core logic.
#
# A crosstab is a wall of numbers. The three facts a reader actually needs -
# what dominates, what moved, and where the interaction is - are all in there
# and none of them are legible.
#
# This module reads a pivot table and writes the paragraph: totals and
# concentration, the biggest movers, the row/column effects, and the cells that
# deviate from what the margins predict. That last one is the point - the
# interaction is the insight, and it is the one thing eyeballing a grid reliably
# misses.
#
# Deterministic and offline: no LLM. Every sentence is arithmetic with a
# threshold, so the same pivot always produces the same narration and every
# claim can be traced to a number.
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# A cell must beat this |lift| vs its expected value AND hold at least this
# share of the grand total before it is worth a sentence. The second guard is
# what stops a tiny cell with a 400% lift from leading the narration.
LIFT_THRESHOLD = 0.25
MIN_CELL_SHARE = 0.01
# Concentration above this and "the average" is the wrong summary statistic.
CONCENTRATION_ALERT = 0.5


@dataclass
class Narration:
    headline: str
    paragraphs: List[str]
    bullets: List[str]
    facts: Dict[str, object]

    def as_text(self) -> str:
        out = [self.headline, ""]
        out.extend(self.paragraphs)
        if self.bullets:
            out.append("")
            out.extend(f"- {b}" for b in self.bullets)
        return "\n".join(out)


def _fmt(v: float, unit: str = "") -> str:
    """Human-readable magnitude - a narration full of 1234567.89 is unreadable."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    a = abs(v)
    if unit == "%":
        return f"{v:.1%}"
    if a >= 1_000_000:
        return f"{unit}{v / 1_000_000:.2f}M"
    if a >= 1_000:
        return f"{unit}{v / 1_000:.1f}k"
    if a >= 10:
        return f"{unit}{v:,.0f}"
    return f"{unit}{v:,.2f}"


def expected_matrix(pivot: pd.DataFrame) -> pd.DataFrame:
    """What each cell would be if rows and columns were independent.

    row_total * col_total / grand_total - the same expectation a chi-square test
    uses. Deviation from this is the interaction: the part of a cell that its
    row and column margins do not already explain.
    """
    row_tot = pivot.sum(axis=1)
    col_tot = pivot.sum(axis=0)
    grand = float(pivot.to_numpy().sum())
    if grand == 0:
        return pd.DataFrame(np.zeros(pivot.shape), index=pivot.index, columns=pivot.columns)
    return pd.DataFrame(
        np.outer(row_tot.to_numpy(), col_tot.to_numpy()) / grand,
        index=pivot.index,
        columns=pivot.columns,
    )


def lift_matrix(pivot: pd.DataFrame) -> pd.DataFrame:
    """(actual - expected) / expected, per cell."""
    exp = expected_matrix(pivot)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = (pivot - exp) / exp.replace(0, np.nan)
    return lift


def concentration(series: pd.Series) -> Dict[str, object]:
    """How much of the total sits in the top few members."""
    s = series.sort_values(ascending=False)
    total = float(s.sum())
    if total == 0:
        return {"top1_share": 0.0, "top1": None, "n_for_half": 0, "n": len(s)}
    cum = s.cumsum() / total
    n_for_half = int((cum < 0.5).sum() + 1)
    return {
        "top1_share": round(float(s.iloc[0] / total), 4),
        "top1": s.index[0],
        "top2_share": round(float(s.iloc[1] / total), 4) if len(s) > 1 else 0.0,
        "top2": s.index[1] if len(s) > 1 else None,
        "n_for_half": n_for_half,
        "n": len(s),
    }


def notable_cells(
    pivot: pd.DataFrame,
    lift_threshold: float = LIFT_THRESHOLD,
    min_share: float = MIN_CELL_SHARE,
    top: int = 4,
) -> pd.DataFrame:
    """Cells that deviate most from what the margins predict, worth-reading only.

    Two guards, both necessary. The lift threshold finds the interaction; the
    share threshold keeps a 3-row cell with a 500% lift from being reported as
    the story. Without the second one this function is a noise generator.
    """
    exp = expected_matrix(pivot)
    lift = lift_matrix(pivot)
    grand = float(pivot.to_numpy().sum())
    rows = []
    for r in pivot.index:
        for c in pivot.columns:
            actual = float(pivot.loc[r, c])
            e = float(exp.loc[r, c])
            lv = float(lift.loc[r, c]) if not pd.isna(lift.loc[r, c]) else 0.0
            share = actual / grand if grand else 0.0
            if abs(lv) >= lift_threshold and share >= min_share:
                rows.append({
                    "row": r, "column": c, "actual": round(actual, 2),
                    "expected": round(e, 2), "lift": round(lv, 4),
                    "share_of_total": round(share, 4),
                    "direction": "over" if lv > 0 else "under",
                })
    if not rows:
        return pd.DataFrame(
            columns=["row", "column", "actual", "expected", "lift", "share_of_total",
                     "direction"]
        )
    out = pd.DataFrame(rows)
    out["abs_lift"] = out["lift"].abs()
    return out.sort_values("abs_lift", ascending=False).drop(columns="abs_lift").head(top)


def compare_periods(
    current: pd.DataFrame, previous: pd.DataFrame, top: int = 3
) -> pd.DataFrame:
    """Cell-level movement between two pivots of the same shape.

    Reindexed to the union of both, so a row that appears or disappears is
    reported as a change rather than silently dropped - a new segment showing up
    is usually the most interesting line in the table.
    """
    idx = current.index.union(previous.index)
    cols = current.columns.union(previous.columns)
    cur = current.reindex(index=idx, columns=cols).fillna(0.0)
    prev = previous.reindex(index=idx, columns=cols).fillna(0.0)
    delta = cur - prev
    rows = []
    for r in idx:
        for c in cols:
            d = float(delta.loc[r, c])
            base = float(prev.loc[r, c])
            if d == 0:
                continue
            rows.append({
                "row": r, "column": c,
                "previous": round(base, 2), "current": round(float(cur.loc[r, c]), 2),
                "delta": round(d, 2),
                # No previous value means growth is undefined, not infinite - a
                # naive pct_change emits inf here and it renders as "+inf%" on a
                # slide. None (stored as NaN in a float column) plus the explicit
                # is_new flag below is what the narration reads, so it says "new"
                # instead of quoting a ratio that does not exist.
                "pct_change": round(d / base, 4) if base else None,
                "is_new": base == 0,
            })
    if not rows:
        return pd.DataFrame(columns=["row", "column", "previous", "current", "delta",
                                     "pct_change", "is_new"])
    out = pd.DataFrame(rows)
    out["abs_delta"] = out["delta"].abs()
    return out.sort_values("abs_delta", ascending=False).drop(columns="abs_delta").head(top)


def narrate(
    pivot: pd.DataFrame,
    metric: str = "value",
    unit: str = "",
    previous: Optional[pd.DataFrame] = None,
    row_label: Optional[str] = None,
    col_label: Optional[str] = None,
) -> Narration:
    """Turn a pivot table into the paragraph a reader actually wants."""
    row_label = row_label or (pivot.index.name or "row")
    col_label = col_label or (pivot.columns.name or "column")

    grand = float(pivot.to_numpy().sum())
    row_tot = pivot.sum(axis=1).sort_values(ascending=False)
    col_tot = pivot.sum(axis=0).sort_values(ascending=False)
    rconc = concentration(row_tot)
    cconc = concentration(col_tot)

    headline = (
        f"{metric}: {_fmt(grand, unit)} across {len(pivot.index)} {row_label}s "
        f"x {len(pivot.columns)} {col_label}s"
    )

    paragraphs: List[str] = []

    # 1 - shape and concentration
    p1 = (
        f"Total {metric} is {_fmt(grand, unit)}. "
        f"The largest {row_label} is {rconc['top1']} at {_fmt(row_tot.iloc[0], unit)} "
        f"({rconc['top1_share']:.0%} of the total)"
    )
    if rconc["top2"] is not None:
        p1 += f", followed by {rconc['top2']} at {rconc['top2_share']:.0%}"
    p1 += (
        f". Half the total sits in {rconc['n_for_half']} of {rconc['n']} {row_label}s. "
        f"By {col_label}, {cconc['top1']} leads with {cconc['top1_share']:.0%}."
    )
    paragraphs.append(p1)

    if rconc["top1_share"] >= CONCENTRATION_ALERT:
        paragraphs.append(
            f"{rconc['top1']} alone is {rconc['top1_share']:.0%} of {metric}, so the "
            f"average across {row_label}s describes almost nothing - read the rows, not "
            f"the mean."
        )

    # 2 - the interaction, which is the part eyeballing a grid misses
    notable = notable_cells(pivot)
    if len(notable):
        bits = []
        for _, n in notable.iterrows():
            bits.append(
                f"{n['row']} x {n['column']} is {abs(n['lift']):.0%} "
                f"{'above' if n['direction'] == 'over' else 'below'} expectation "
                f"({_fmt(n['actual'], unit)} vs {_fmt(n['expected'], unit)})"
            )
        paragraphs.append(
            "Against what the row and column totals alone predict: " + "; ".join(bits) + "."
        )
        top_cell = notable.iloc[0]
        paragraphs.append(
            f"The strongest interaction is {top_cell['row']} x {top_cell['column']}. "
            f"Neither {top_cell['row']}'s size nor {top_cell['column']}'s size explains it - "
            f"it is {abs(top_cell['lift']):.0%} "
            f"{'higher' if top_cell['direction'] == 'over' else 'lower'} than the margins "
            f"imply, which is the kind of thing a grid of numbers hides in plain sight."
        )
    else:
        paragraphs.append(
            f"No cell deviates more than {LIFT_THRESHOLD:.0%} from what its row and column "
            f"totals predict, so {metric} is close to independent across these two "
            f"dimensions - the margins tell the whole story and the cells add nothing."
        )

    # 3 - movement, if a comparison period was supplied
    movers = pd.DataFrame()
    if previous is not None:
        prev_grand = float(previous.to_numpy().sum())
        change = (grand - prev_grand) / prev_grand if prev_grand else None
        direction = "up" if change and change > 0 else "down"
        paragraphs.append(
            f"Against the comparison period, {metric} is {direction} "
            f"{abs(change):.1%} ({_fmt(prev_grand, unit)} to {_fmt(grand, unit)})."
            if change is not None else
            f"No comparable prior total for {metric}."
        )
        movers = compare_periods(pivot, previous)
        if len(movers):
            mbits = []
            for _, m in movers.iterrows():
                if m["is_new"]:
                    mbits.append(
                        f"{m['row']} x {m['column']} is new at {_fmt(m['current'], unit)}"
                    )
                else:
                    mbits.append(
                        f"{m['row']} x {m['column']} moved {_fmt(m['delta'], unit)} "
                        f"({m['pct_change']:+.0%})"
                    )
            paragraphs.append("Biggest movers: " + "; ".join(mbits) + ".")

    bullets = [
        f"Top {row_label}: {rconc['top1']} ({rconc['top1_share']:.0%})",
        f"Top {col_label}: {cconc['top1']} ({cconc['top1_share']:.0%})",
        f"Concentration: half of {metric} in {rconc['n_for_half']}/{rconc['n']} {row_label}s",
    ]
    if len(notable):
        t = notable.iloc[0]
        bullets.append(
            f"Strongest interaction: {t['row']} x {t['column']} "
            f"({t['lift']:+.0%} vs expected)"
        )
    if len(movers):
        m = movers.iloc[0]
        # Computed outside the f-string: nesting the same quote style inside an
        # f-string only parses on Python 3.12+.
        mover_change = "new" if m["is_new"] else f"{m['pct_change']:+.0%}"
        bullets.append(f"Biggest mover: {m['row']} x {m['column']} ({mover_change})")

    return Narration(
        headline=headline,
        paragraphs=paragraphs,
        bullets=bullets,
        facts={
            "grand_total": round(grand, 2),
            "row_totals": row_tot,
            "col_totals": col_tot,
            "row_concentration": rconc,
            "col_concentration": cconc,
            "notable_cells": notable,
            "movers": movers,
            "expected": expected_matrix(pivot),
            "lift": lift_matrix(pivot),
        },
    )


# --------------------------------------------------------------------------
# Sample data - a revenue crosstab with one planted interaction
# --------------------------------------------------------------------------


def sample_pivots() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Revenue by region x product, current and prior quarter.

    The planted structure: `partner` is a small channel overall, but it is
    wildly over-indexed on one product in one region - an interaction the row
    and column totals cannot show. There is also a brand-new cell in the
    current period, to exercise the divide-by-zero path.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    regions = ["Singapore", "Malaysia", "Indonesia", "Thailand", "Vietnam"]
    products = ["Core", "Pro", "Enterprise", "Add-ons"]

    # Base: region size x product mix, multiplicative and therefore independent.
    region_w = np.array([0.40, 0.22, 0.20, 0.11, 0.07])
    product_w = np.array([0.45, 0.28, 0.19, 0.08])
    base = np.outer(region_w, product_w) * 2_400_000

    prev = base * rng.normal(1.0, 0.04, base.shape)
    cur = base * rng.normal(1.06, 0.05, base.shape)

    prev_df = pd.DataFrame(prev.round(2), index=regions, columns=products)
    cur_df = pd.DataFrame(cur.round(2), index=regions, columns=products)

    # The planted interaction: Indonesia x Enterprise runs far above what the
    # margins imply, and it grew hard.
    prev_df.loc["Indonesia", "Enterprise"] *= 2.1
    cur_df.loc["Indonesia", "Enterprise"] *= 3.0

    # A cell that did not exist last quarter.
    prev_df.loc["Vietnam", "Enterprise"] = 0.0
    cur_df.loc["Vietnam", "Enterprise"] = 61_500.0

    for d in (prev_df, cur_df):
        d.index.name = "region"
        d.columns.name = "product"
    return cur_df.round(2), prev_df.round(2)


def main() -> None:
    pd.set_option("display.width", 200)
    cur, prev = sample_pivots()

    print("=" * 86)
    print("THE PIVOT TABLE (revenue by region x product, current quarter)")
    print("=" * 86)
    print(cur.to_string())
    print("\nrow totals:", {k: _fmt(v, "$") for k, v in cur.sum(axis=1).items()})
    print("col totals:", {k: _fmt(v, "$") for k, v in cur.sum(axis=0).items()})

    print("\n" + "=" * 86)
    print("THE NARRATION")
    print("=" * 86)
    n = narrate(cur, metric="revenue", unit="$", previous=prev,
                row_label="region", col_label="product")
    print(n.as_text())

    print("\n" + "-" * 86)
    print("EXPECTED (if region and product were independent)")
    print(expected_matrix(cur).round(0).to_string())

    print("\nLIFT (actual vs expected)")
    print((lift_matrix(cur) * 100).round(1).to_string())

    print("\nNOTABLE CELLS")
    print(n.facts["notable_cells"].to_string(index=False))

    print("\nBIGGEST MOVERS")
    print(n.facts["movers"].to_string(index=False))


if __name__ == "__main__":
    main()
