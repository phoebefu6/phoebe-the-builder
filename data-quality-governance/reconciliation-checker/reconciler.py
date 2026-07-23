from __future__ import annotations

# Source-to-Target Reconciliation - prove that a table copied from a SOURCE
# system landed correctly in a TARGET system (e.g. warehouse) at the SAME point
# in time. This is cross-system CORRECTNESS at one instant, not change over
# time: "did the copy match?" We compare on a chosen key column and report
# row-count delta, keys MISSING in target, keys EXTRA in target, cell-level
# value mismatches on shared keys, and optional aggregate checks (a numeric sum
# must agree within a tolerance). Every finding is explainable so a steward can
# defend the pass/fail verdict. Fully offline, standard pandas/numpy - no API keys.

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Absolute tolerance for numeric cell comparisons. Floats rarely survive a
# system hop bit-for-bit (a DECIMAL becomes a float, a re-rounded currency),
# so a tiny epsilon avoids drowning the steward in false mismatches. Anything
# above this is a REAL value drift worth reporting.
DEFAULT_TOLERANCE = 1e-6


@dataclass
class CellMismatch:
    """One shared key where a column's value differs between the two systems."""

    key: object
    column: str
    source_value: object
    target_value: object
    reason: str


@dataclass
class AggCheck:
    """A column-level aggregate (e.g. sum) compared source vs target."""

    column: str
    agg: str            # "sum"
    source_value: float
    target_value: float
    delta: float
    tolerance: float
    passed: bool
    reason: str


@dataclass
class ReconResult:
    """The full reconciliation verdict - each field is independently explainable."""

    ok: bool                                  # False only on a hard error (bad key / structure)
    error: Optional[str] = None
    key: str = ""
    source_rows: int = 0
    target_rows: int = 0
    row_delta: int = 0                        # target_rows - source_rows
    shared_keys: int = 0
    missing_keys: List[object] = field(default_factory=list)   # in source, absent in target
    extra_keys: List[object] = field(default_factory=list)     # in target, absent in source
    compared_columns: List[str] = field(default_factory=list)  # columns checked cell-by-cell
    cell_mismatches: List[CellMismatch] = field(default_factory=list)
    agg_checks: List[AggCheck] = field(default_factory=list)
    match_rate: float = 0.0                   # shared keys with zero mismatches / source rows
    passed: bool = False                      # overall pass/fail verdict


def _err(msg: str, key: str = "") -> ReconResult:
    """A graceful failure result - never raise into the caller's face."""
    return ReconResult(ok=False, error=msg, key=key)


def _values_match(a: object, b: object, tolerance: float) -> bool:
    """Compare one cell across systems, tolerant of the ways a copy blurs values.

    Two NaNs are treated as equal (both systems agree the value is absent).
    Numbers compare within an absolute tolerance so a harmless float re-round
    is not screamed about. Everything else compares as trimmed strings, which
    catches the common " 12 " vs "12" whitespace drift without inventing
    mismatches that aren't really there.
    """
    a_null = a is None or (isinstance(a, float) and np.isnan(a)) or pd.isna(a)
    b_null = b is None or (isinstance(b, float) and np.isnan(b)) or pd.isna(b)
    if a_null and b_null:
        return True
    if a_null or b_null:
        return False
    # Numeric path: compare magnitudes within tolerance.
    a_num = isinstance(a, (int, float, np.integer, np.floating))
    b_num = isinstance(b, (int, float, np.integer, np.floating))
    if a_num and b_num:
        return abs(float(a) - float(b)) <= tolerance
    return str(a).strip() == str(b).strip()


def reconcile(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    key: str,
    tolerance: float = DEFAULT_TOLERANCE,
    agg_columns: Optional[List[str]] = None,
) -> ReconResult:
    """Reconcile a source table against its target copy on `key`.

    Steps, each producing an explainable finding:
      1. row-count delta (did the same number of rows land?)
      2. keys MISSING in target (rows that did not make the trip)
      3. keys EXTRA in target (rows that shouldn't be there)
      4. cell-level value mismatches on the SHARED keys, column by column
      5. optional aggregate checks - a numeric sum must agree within tolerance,
         a cheap independent proof that no value silently drifted at scale.

    Edge cases handled: key absent in either frame -> graceful error result;
    empty frames -> a clean zero-row verdict, not a crash.
    """
    # --- structural guards: fail loud but graceful, never raise ---
    if source_df is None or target_df is None:
        return _err("source_df and target_df must both be provided", key)
    if key not in source_df.columns:
        return _err(f"key column '{key}' not found in source", key)
    if key not in target_df.columns:
        return _err(f"key column '{key}' not found in target", key)

    result = ReconResult(ok=True, key=key)
    result.source_rows = len(source_df)
    result.target_rows = len(target_df)
    result.row_delta = result.target_rows - result.source_rows

    # Empty-frame shortcut: nothing to compare, report the row picture honestly.
    if source_df.empty and target_df.empty:
        result.match_rate = 1.0
        result.passed = True
        return result

    # Index by key for set logic and aligned lookups. `first()` keeps behaviour
    # defined if a key repeats (dupes are a data issue, but we won't explode).
    src = source_df.drop_duplicates(subset=[key]).set_index(key)
    tgt = target_df.drop_duplicates(subset=[key]).set_index(key)

    src_keys = set(src.index)
    tgt_keys = set(tgt.index)

    # 2 + 3: which keys didn't land, which showed up unexpectedly.
    result.missing_keys = sorted([k for k in (src_keys - tgt_keys)], key=str)
    result.extra_keys = sorted([k for k in (tgt_keys - src_keys)], key=str)

    shared = src_keys & tgt_keys
    result.shared_keys = len(shared)

    # 4: cell-level comparison on columns present in BOTH systems (minus the
    # key itself). We only compare what both sides claim to hold - a column that
    # exists on one side only is a schema difference, not a value mismatch.
    common_cols = [c for c in source_df.columns if c in target_df.columns and c != key]
    result.compared_columns = common_cols

    shared_sorted = sorted(shared, key=str)
    src_shared = src.loc[list(shared_sorted)] if shared_sorted else src.iloc[0:0]
    tgt_shared = tgt.loc[list(shared_sorted)] if shared_sorted else tgt.iloc[0:0]

    keys_with_mismatch = set()
    for k in shared_sorted:
        for col in common_cols:
            sv = src_shared.at[k, col]
            tv = tgt_shared.at[k, col]
            if not _values_match(sv, tv, tolerance):
                keys_with_mismatch.add(k)
                reason = (
                    f"key {k!r}, column '{col}': source={sv!r} vs target={tv!r} "
                    f"(differ beyond tolerance {tolerance:g})"
                )
                result.cell_mismatches.append(
                    CellMismatch(k, col, sv, tv, reason)
                )

    # 5: aggregate checks. Default to every numeric common column so the steward
    # gets a scale-level cross-check for free; a mismatching sum on otherwise
    # clean keys is a strong hint that rows are missing or values drifted.
    if agg_columns is None:
        agg_columns = [
            c for c in common_cols
            if pd.api.types.is_numeric_dtype(source_df[c])
            and pd.api.types.is_numeric_dtype(target_df[c])
        ]
    for col in agg_columns:
        if col not in source_df.columns or col not in target_df.columns:
            continue
        s_sum = float(pd.to_numeric(source_df[col], errors="coerce").sum())
        t_sum = float(pd.to_numeric(target_df[col], errors="coerce").sum())
        delta = t_sum - s_sum
        # Sums span whole tables, so scale the tolerance by row count - a per-cell
        # epsilon would be far too strict against a column-wide total.
        agg_tol = max(tolerance, tolerance * max(result.source_rows, 1))
        passed = abs(delta) <= agg_tol
        reason = (
            f"sum('{col}'): source={s_sum:g} vs target={t_sum:g}, "
            f"delta={delta:g} (tolerance {agg_tol:g}) -> "
            f"{'within' if passed else 'OUTSIDE'} tolerance"
        )
        result.agg_checks.append(
            AggCheck(col, "sum", s_sum, t_sum, delta, agg_tol, passed, reason)
        )

    # Match rate: share of SOURCE rows that arrived AND matched on every cell.
    # Anchored on source because the source is the system of record - the target
    # is only correct insofar as it faithfully reproduces the source.
    clean_shared = len(shared) - len(keys_with_mismatch)
    denom = result.source_rows if result.source_rows else 1
    result.match_rate = round(clean_shared / denom, 4)

    # Overall verdict: a copy PASSES only if every dimension agrees.
    result.passed = (
        not result.missing_keys
        and not result.extra_keys
        and not result.cell_mismatches
        and all(a.passed for a in result.agg_checks)
    )
    return result


def summarize(result: ReconResult) -> pd.DataFrame:
    """One-glance scorecard the steward can paste into a ticket."""
    if not result.ok:
        return pd.DataFrame([{"metric": "error", "value": result.error}])
    rows = [
        ("key_column", result.key),
        ("source_rows", result.source_rows),
        ("target_rows", result.target_rows),
        ("row_delta", result.row_delta),
        ("shared_keys", result.shared_keys),
        ("missing_in_target", len(result.missing_keys)),
        ("extra_in_target", len(result.extra_keys)),
        ("cell_mismatches", len(result.cell_mismatches)),
        ("agg_checks_failed", sum(1 for a in result.agg_checks if not a.passed)),
        ("match_rate", f"{result.match_rate:.2%}"),
        ("verdict", "PASS" if result.passed else "FAIL"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def mismatches_frame(result: ReconResult) -> pd.DataFrame:
    """Flat cell-mismatch table for export / review."""
    cols = ["key", "column", "source_value", "target_value", "reason"]
    rows = [
        {"key": m.key, "column": m.column, "source_value": m.source_value,
         "target_value": m.target_value, "reason": m.reason}
        for m in result.cell_mismatches
    ]
    return pd.DataFrame(rows, columns=cols)


def make_sample_data(seed: int = 42) -> Dict[str, pd.DataFrame]:
    """A source orders table and its imperfect target copy, with planted drift.

    Planted so the demo shows every finding type:
      - 3 rows MISSING in target (dropped in transit)
      - 1 EXTRA row in target (a stray insert)
      - 2 cell value mismatches on shared keys (a re-rounded amount, a changed status)
      - the amount sum lands just OUTSIDE tolerance because of the missing rows
    """
    rng = np.random.default_rng(seed)
    n = 50
    order_id = np.arange(1000, 1000 + n)
    amount = rng.normal(120, 30, n).round(2)
    status = rng.choice(["paid", "pending", "refunded"], n, p=[0.7, 0.2, 0.1])
    region = rng.choice(["North", "South", "East", "West"], n)

    source = pd.DataFrame({
        "order_id": order_id,
        "amount": amount,
        "status": status,
        "region": region,
    })

    # Target starts as a faithful copy, then we corrupt it like a real bad load.
    target = source.copy()

    # 3 rows never made it into the warehouse.
    target = target[~target["order_id"].isin([1005, 1020, 1044])].reset_index(drop=True)

    # 1 stray row that shouldn't exist in the target.
    stray = pd.DataFrame([{
        "order_id": 9999, "amount": 88.0, "status": "paid", "region": "North",
    }])
    target = pd.concat([target, stray], ignore_index=True)

    # 2 silent cell drifts on rows that DID land.
    target.loc[target["order_id"] == 1002, "amount"] = (
        float(source.loc[source["order_id"] == 1002, "amount"].iloc[0]) + 5.0
    )  # a value drift beyond any float epsilon
    target.loc[target["order_id"] == 1010, "status"] = "pending"  # status flipped

    return {"source": source, "target": target}


def _cli() -> None:
    data = make_sample_data()
    source, target = data["source"], data["target"]
    result = reconcile(source, target, key="order_id")

    print("=== Source-to-Target Reconciliation ===\n")
    print(summarize(result).to_string(index=False))

    print("\n--- keys missing in target (did not land) ---")
    print(result.missing_keys or "(none)")
    print("\n--- keys extra in target (should not exist) ---")
    print(result.extra_keys or "(none)")

    print("\n--- cell-level value mismatches ---")
    mm = mismatches_frame(result)
    print(mm.to_string(index=False) if not mm.empty else "(none)")

    print("\n--- aggregate checks ---")
    for a in result.agg_checks:
        print(f"  [{'PASS' if a.passed else 'FAIL'}] {a.reason}")

    print(f"\nVERDICT: {'PASS' if result.passed else 'FAIL'} "
          f"(match rate {result.match_rate:.2%})")


if __name__ == "__main__":
    _cli()
