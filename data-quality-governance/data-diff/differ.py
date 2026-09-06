from __future__ import annotations

# Dataset Snapshot Diff - compare TWO snapshots of the SAME dataset taken at
# different times and answer the one question nobody can answer by eyeballing:
# "what actually changed since the last run?" We diff on a key column and report
# ADDED rows (new keys), REMOVED rows (gone keys), MODIFIED rows (shared key with
# at least one column changed - naming which columns and the old -> new value),
# UNCHANGED count, SCHEMA DRIFT (columns added/removed), and change velocity.
#
# This is CHANGE-OVER-TIME within one system (temporal auditing / CDC-style),
# NOT cross-system reconciliation. Every finding is explainable so a steward can
# judge intent - a diff shows WHAT changed, never WHY. Offline, pandas/numpy only.
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class CellChange:
    """One column that changed for one key, with the before/after values."""

    column: str
    before: object
    after: object


@dataclass
class ModifiedRow:
    """A key present in both snapshots whose row changed in >=1 column."""

    key: object
    changes: List[CellChange] = field(default_factory=list)


@dataclass
class SchemaDrift:
    """Columns that appeared or disappeared between the two snapshots."""

    added_columns: List[str] = field(default_factory=list)
    removed_columns: List[str] = field(default_factory=list)


@dataclass
class DiffResult:
    """The full temporal diff - the steward's 'what changed' answer."""

    key: str
    added: pd.DataFrame                 # rows whose key is new in `after`
    removed: pd.DataFrame               # rows whose key vanished from `before`
    modified: List[ModifiedRow]         # shared keys with changed values
    unchanged: int                      # shared keys with no change at all
    schema_drift: SchemaDrift
    n_before: int
    n_after: int

    @property
    def velocity(self) -> Dict[str, float]:
        """Change-velocity stats - the % of the prior snapshot that moved.

        Denominator is the BEFORE row count: velocity answers "of what we had
        yesterday, how much churned?" A brand-new empty history is treated as
        0% so we never divide by zero.
        """
        base = self.n_before if self.n_before else 1
        n_add, n_rem, n_mod = len(self.added), len(self.removed), len(self.modified)
        changed = n_add + n_rem + n_mod
        return {
            "added": n_add,
            "removed": n_rem,
            "modified": n_mod,
            "unchanged": self.unchanged,
            "pct_changed": round(100.0 * changed / base, 2),
            "pct_added": round(100.0 * n_add / base, 2),
            "pct_removed": round(100.0 * n_rem / base, 2),
            "pct_modified": round(100.0 * n_mod / base, 2),
        }


def _cells_equal(a: object, b: object) -> bool:
    """True when two cell values are the SAME for diff purposes.

    We treat NaN == NaN as equal (a null that stays null did not "change"),
    and compare everything else as strings so 5 vs 5.0 or int vs numpy-int
    don't register as spurious edits - a steward cares about real value moves,
    not dtype noise from a re-serialized CSV.
    """
    a_null = a is None or (isinstance(a, float) and np.isnan(a)) or pd.isna(a)
    b_null = b is None or (isinstance(b, float) and np.isnan(b)) or pd.isna(b)
    if a_null and b_null:
        return True
    if a_null or b_null:
        return False
    return str(a) == str(b)


def diff_snapshots(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    key: str,
) -> DiffResult:
    """Diff two snapshots of the same dataset on `key`.

    Edge cases handled up front so the caller gets a clear error, not a crash:
      - key missing in either snapshot -> ValueError naming the offender
      - duplicate keys within a snapshot -> ValueError (a key must be unique to
        diff by it; otherwise "which row changed?" is ambiguous)
      - empty snapshot -> everything reads as pure adds or pure removes
    """
    if key not in before_df.columns:
        raise ValueError(f"key column '{key}' is missing from the BEFORE snapshot")
    if key not in after_df.columns:
        raise ValueError(f"key column '{key}' is missing from the AFTER snapshot")
    if before_df[key].duplicated().any():
        raise ValueError(f"key '{key}' is not unique in the BEFORE snapshot")
    if after_df[key].duplicated().any():
        raise ValueError(f"key '{key}' is not unique in the AFTER snapshot")

    # Schema drift first - which columns exist in one snapshot but not the other.
    before_cols = list(before_df.columns)
    after_cols = list(after_df.columns)
    drift = SchemaDrift(
        added_columns=[c for c in after_cols if c not in before_cols],
        removed_columns=[c for c in before_cols if c not in after_cols],
    )

    # Index by key so membership tests and per-key lookups are cheap and clear.
    b_idx = before_df.set_index(key)
    a_idx = after_df.set_index(key)
    b_keys = set(b_idx.index)
    a_keys = set(a_idx.index)

    added_keys = [k for k in after_df[key] if k not in b_keys]
    removed_keys = [k for k in before_df[key] if k not in a_keys]
    shared_keys = [k for k in before_df[key] if k in a_keys]

    added = after_df[after_df[key].isin(added_keys)].reset_index(drop=True)
    removed = before_df[before_df[key].isin(removed_keys)].reset_index(drop=True)

    # Only compare columns present in BOTH snapshots - a column that was added or
    # dropped is schema drift, not a per-row modification, and we report it there.
    shared_cols = [c for c in before_cols if c in after_cols and c != key]

    modified: List[ModifiedRow] = []
    unchanged = 0
    for k in shared_keys:
        b_row = b_idx.loc[k]
        a_row = a_idx.loc[k]
        changes: List[CellChange] = []
        for c in shared_cols:
            bv, av = b_row[c], a_row[c]
            if not _cells_equal(bv, av):
                changes.append(CellChange(c, bv, av))
        if changes:
            modified.append(ModifiedRow(k, changes))
        else:
            unchanged += 1

    return DiffResult(
        key=key,
        added=added,
        removed=removed,
        modified=modified,
        unchanged=unchanged,
        schema_drift=drift,
        n_before=len(before_df),
        n_after=len(after_df),
    )


def modified_frame(result: DiffResult) -> pd.DataFrame:
    """Flatten modified rows to one row per changed cell - export/review table."""
    rows = []
    for m in result.modified:
        for ch in m.changes:
            rows.append({
                "key": m.key,
                "column": ch.column,
                "before": ch.before,
                "after": ch.after,
            })
    cols = ["key", "column", "before", "after"]
    return pd.DataFrame(rows, columns=cols)


def summary_frame(result: DiffResult) -> pd.DataFrame:
    """One-line change-breakdown table - the steward's headline numbers."""
    v = result.velocity
    return pd.DataFrame([{
        "rows_before": result.n_before,
        "rows_after": result.n_after,
        "added": v["added"],
        "removed": v["removed"],
        "modified": v["modified"],
        "unchanged": v["unchanged"],
        "pct_changed": v["pct_changed"],
    }])


def make_sample_snapshots(seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Two snapshots of a products catalog with planted, known changes.

    Between the 'before' (yesterday) and 'after' (today) snapshots we plant:
      - 2 ADDED products (new SKUs P021, P022)
      - 2 REMOVED products (discontinued P004, P015)
      - several MODIFIED rows across different columns (price, stock, status)
      - 1 SCHEMA change: a new 'discount_pct' column appears in `after`
    so the diff has something real to find and the demo is verifiable.
    """
    rng = np.random.default_rng(seed)
    n = 20
    skus = [f"P{i:03d}" for i in range(1, n + 1)]
    categories = rng.choice(["Home", "Tech", "Outdoor", "Kitchen"], n)
    price = rng.uniform(9.99, 199.99, n).round(2)
    stock = rng.integers(0, 500, n)
    status = np.where(stock > 0, "active", "out_of_stock").astype(object)

    before = pd.DataFrame({
        "sku": skus,
        "category": categories,
        "price": price,
        "stock": stock,
        "status": status,
    })

    # Build `after` as a copy of `before`, then apply temporal changes.
    after = before.copy()

    # MODIFIED: price moves on P002; restock flips P007 back to active; a price
    # cut on P010; stock drawn down on P012; category re-classified on P018.
    after.loc[after["sku"] == "P002", "price"] = 149.99
    after.loc[after["sku"] == "P007", "stock"] = 120
    after.loc[after["sku"] == "P007", "status"] = "active"
    after.loc[after["sku"] == "P010", "price"] = round(
        float(before.loc[before["sku"] == "P010", "price"].iloc[0]) * 0.8, 2
    )
    after.loc[after["sku"] == "P012", "stock"] = 3
    after.loc[after["sku"] == "P018", "category"] = "Tech"

    # REMOVED: two SKUs discontinued (gone from today's snapshot).
    after = after[~after["sku"].isin(["P004", "P015"])].copy()

    # ADDED: two new SKUs onboarded today.
    new_rows = pd.DataFrame({
        "sku": ["P021", "P022"],
        "category": ["Tech", "Kitchen"],
        "price": [59.99, 24.99],
        "stock": [80, 200],
        "status": ["active", "active"],
    })
    after = pd.concat([after, new_rows], ignore_index=True)

    # SCHEMA DRIFT: a new column shows up in today's snapshot only.
    after["discount_pct"] = rng.integers(0, 30, len(after))

    return {"before": before, "after": after}


def _cli() -> None:
    snaps = make_sample_snapshots()
    result = diff_snapshots(snaps["before"], snaps["after"], key="sku")

    print("=== Dataset Snapshot Diff (temporal change tracking) ===\n")
    print(summary_frame(result).to_string(index=False))

    v = result.velocity
    print(
        f"\nChange velocity: {v['pct_changed']}% of the prior snapshot moved "
        f"({v['added']} added, {v['removed']} removed, {v['modified']} modified, "
        f"{v['unchanged']} unchanged)."
    )

    drift = result.schema_drift
    if drift.added_columns or drift.removed_columns:
        print("\n--- schema drift ---")
        if drift.added_columns:
            print(f"  columns ADDED in after:   {drift.added_columns}")
        if drift.removed_columns:
            print(f"  columns REMOVED in after: {drift.removed_columns}")
    else:
        print("\nNo schema drift - column set is identical.")

    if len(result.added):
        print(f"\n--- added rows ({len(result.added)}) ---")
        print(result.added.to_string(index=False))
    if len(result.removed):
        print(f"\n--- removed rows ({len(result.removed)}) ---")
        print(result.removed.to_string(index=False))

    mf = modified_frame(result)
    if not mf.empty:
        print(f"\n--- modified cells (sample) - {len(result.modified)} rows changed ---")
        print(mf.head(12).to_string(index=False))


if __name__ == "__main__":
    _cli()
