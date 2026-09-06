from __future__ import annotations

# DQ Rules Engine - declarative data-quality checks that run against a pandas
# DataFrame. You describe the rules as plain dicts (not_null, unique, in_range,
# regex_match, allowed_values, foreign_key, row_count, expression); the engine
# runs them and returns per-rule results: pass/fail, how many rows violated, a
# few sample offending values, and a plain-English message. The point is to move
# quality rules OUT of a wiki nobody enforces and INTO code that runs on every
# batch - so a bad batch is caught before it lands in a report, not after.
# Fully offline, standard pandas/numpy only - no API keys.
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RuleResult:
    """Outcome of one rule against one DataFrame."""

    name: str
    rule_type: str
    column: Optional[str]
    severity: str            # "error" | "warn"
    passed: bool
    n_checked: int           # rows the rule could evaluate
    n_violations: int
    samples: List[object]    # a few offending values, for a human to eyeball
    message: str             # plain-English verdict


SAMPLE_LIMIT = 5  # how many offending values to surface - enough to spot a pattern


def _severity(rule: Dict) -> str:
    # Default to "error": a rule is worth writing because breaking it matters.
    # Downgrade to "warn" explicitly when a violation is a smell, not a stop.
    sev = str(rule.get("severity", "error")).lower()
    return sev if sev in ("error", "warn") else "error"


def _missing_column(rule: Dict, df: pd.DataFrame) -> Optional[RuleResult]:
    """A rule pointing at a column that isn't there is a config error, not a
    crash. Return a failing result so the whole run still completes and the
    steward sees exactly which rule is misconfigured."""
    col = rule.get("column")
    if col is not None and col not in df.columns:
        return RuleResult(
            name=rule.get("name", rule.get("type", "rule")),
            rule_type=rule.get("type", "?"),
            column=col,
            severity=_severity(rule),
            passed=False,
            n_checked=0,
            n_violations=0,
            samples=[],
            message=f"column '{col}' not found in data - check the rule definition",
        )
    return None


def _samples(df: pd.DataFrame, column: Optional[str], mask: pd.Series) -> List[object]:
    """A few offending values (or row indices when the rule isn't column-bound)."""
    idx = mask[mask].index[:SAMPLE_LIMIT]
    if column is not None and column in df.columns:
        return [df.loc[i, column] for i in idx]
    return [f"row {int(i)}" for i in idx]


def _run_one(rule: Dict, df: pd.DataFrame) -> RuleResult:
    rtype = rule["type"]
    col = rule.get("column")
    n = len(df)
    sev = _severity(rule)
    name = rule.get("name", rtype)

    # --- rules that don't need a column present ---------------------------
    if rtype == "row_count":
        lo = rule.get("min")
        hi = rule.get("max")
        bad = (lo is not None and n < lo) or (hi is not None and n > hi)
        bounds = f"[{lo if lo is not None else '-'}, {hi if hi is not None else '-'}]"
        return RuleResult(
            name, rtype, None, sev, not bad, n, 1 if bad else 0, [],
            f"row count {n} within {bounds}" if not bad
            else f"row count {n} outside expected {bounds}")

    if rtype == "expression":
        # A custom pandas expression evaluated per row; it must return True for
        # GOOD rows (so violations are where it's False). This is the escape
        # hatch for cross-column business rules the fixed types can't express.
        expr = rule["expr"]
        try:
            good = df.eval(expr)
        except Exception as exc:  # a bad expression is a config error, not a crash
            return RuleResult(name, rtype, None, sev, False, 0, 0, [],
                              f"expression failed to evaluate: {exc}")
        mask = ~good.astype(bool)
        n_bad = int(mask.sum())
        return RuleResult(name, rtype, None, sev, n_bad == 0, n, n_bad,
                          _samples(df, None, mask),
                          f"all {n} rows satisfy `{expr}`" if n_bad == 0
                          else f"{n_bad} of {n} rows fail `{expr}`")

    # --- rules that need their column to exist ----------------------------
    miss = _missing_column(rule, df)
    if miss is not None:
        return miss

    s = df[col]

    if rtype == "not_null":
        mask = s.isna()
    elif rtype == "unique":
        # duplicated(keep=False) marks EVERY row in a dup group, so the sample
        # shows the actual colliding values, not just the second occurrence.
        mask = s.duplicated(keep=False) & s.notna()
    elif rtype == "in_range":
        lo, hi = rule.get("min"), rule.get("max")
        num = pd.to_numeric(s, errors="coerce")
        below = num < lo if lo is not None else pd.Series(False, index=s.index)
        above = num > hi if hi is not None else pd.Series(False, index=s.index)
        # A value that won't parse as a number also violates a numeric range.
        unparseable = num.isna() & s.notna()
        mask = (below | above | unparseable) & s.notna()
    elif rtype == "allowed_values":
        allowed = set(rule["values"])
        mask = ~s.isin(allowed) & s.notna()
    elif rtype == "regex_match":
        pattern = rule["pattern"]
        # NaN never matches; only judge non-null values against the pattern.
        matched = s.astype(str).str.match(pattern)
        mask = ~matched & s.notna()
    elif rtype == "foreign_key":
        # Referential check: every value in `column` must exist in a reference
        # set (e.g. order.customer_id must exist in the customers table).
        ref = set(rule["reference"])
        mask = ~s.isin(ref) & s.notna()
    else:
        return RuleResult(name, rtype, col, sev, False, 0, 0, [],
                          f"unknown rule type '{rtype}'")

    n_checked = int(s.notna().sum()) if rtype != "not_null" else n
    n_bad = int(mask.sum())
    passed = n_bad == 0
    samples = _samples(df, col, mask)
    verb = {
        "not_null": "are null",
        "unique": "are duplicated",
        "in_range": "fall outside the allowed range",
        "allowed_values": "are not in the allowed set",
        "regex_match": "do not match the required pattern",
        "foreign_key": "have no matching reference key",
    }[rtype]
    msg = (f"all {n_checked} values in '{col}' pass ({rtype})" if passed
           else f"{n_bad} value(s) in '{col}' {verb}")
    return RuleResult(name, rtype, col, sev, passed, n_checked, n_bad, samples, msg)


def run_rules(df: pd.DataFrame, rules: List[Dict]) -> List[RuleResult]:
    """Run every rule against `df`. Edge case: an empty frame still runs - most
    rules pass vacuously, but row_count with a `min` correctly fails, which is
    exactly the alarm you want when an upstream job delivered nothing."""
    if df is None:
        df = pd.DataFrame()
    return [_run_one(rule, df) for rule in rules]


def summarize(results: List[RuleResult]) -> pd.DataFrame:
    """One row per rule - the steward's pass/fail triage table."""
    rows = []
    for r in results:
        rows.append({
            "rule": r.name,
            "type": r.rule_type,
            "column": r.column if r.column is not None else "-",
            "severity": r.severity,
            "status": "PASS" if r.passed else "FAIL",
            "violations": r.n_violations,
            "checked": r.n_checked,
        })
    cols = ["rule", "type", "column", "severity", "status", "violations", "checked"]
    return pd.DataFrame(rows, columns=cols)


def rollup(results: List[RuleResult]) -> Dict:
    """Overall verdict. A failing WARN rule does not block; a failing ERROR does.
    That distinction is what lets a pipeline gate on real breaks while still
    logging the softer smells."""
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    errors = [r for r in failed if r.severity == "error"]
    warns = [r for r in failed if r.severity == "warn"]
    status = "PASS" if not failed else ("FAIL" if errors else "WARN")
    return {
        "total": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "errors_failed": len(errors),
        "warns_failed": len(warns),
        "overall_status": status,
    }


def make_sample_data(seed: int = 42) -> pd.DataFrame:
    """Realistic orders table with planted rule violations for the demo."""
    rng = np.random.default_rng(seed)
    n = 200
    order_id = np.arange(1000, 1000 + n)
    # Plant a duplicate order_id - the classic broken-primary-key bug.
    order_id[50] = order_id[49]

    customer_id = rng.integers(1, 41, n)   # customers 1..40 exist
    customer_id[7] = 999                    # orphan FK - no such customer
    customer_id[123] = 777                  # another orphan

    amount = rng.normal(120, 40, n).round(2)
    amount[10] = -5.0                        # impossible negative
    amount[80] = 20000.0                     # out-of-range spike

    status = rng.choice(["paid", "pending", "refunded"], n, p=[.7, .2, .1]).astype(object)
    status[30] = "PAID"                      # wrong case - not in allowed set
    status[31] = "cancelled"                 # value nobody approved

    email = np.array([f"user{i}@shop.com" for i in range(n)], dtype=object)
    email[3] = None                          # missing contact
    email[4] = "not-an-email"                # fails regex
    email[5] = None

    return pd.DataFrame({
        "order_id": order_id,
        "customer_id": customer_id,
        "amount": amount,
        "status": status,
        "email": email,
    })


def sample_rules() -> List[Dict]:
    """The rule set as plain dicts - this is what would live in a YAML file or a
    governance table instead of a wiki page. Each dict IS the enforceable rule."""
    valid_customers = list(range(1, 41))
    return [
        {"name": "order_id present", "type": "not_null", "column": "order_id"},
        {"name": "order_id unique", "type": "unique", "column": "order_id"},
        {"name": "customer exists", "type": "foreign_key", "column": "customer_id",
         "reference": valid_customers},
        {"name": "amount in range", "type": "in_range", "column": "amount",
         "min": 0, "max": 10000},
        {"name": "status allowed", "type": "allowed_values", "column": "status",
         "values": ["paid", "pending", "refunded"]},
        {"name": "email format", "type": "regex_match", "column": "email",
         "pattern": r"^[^@\s]+@[^@\s]+\.[^@\s]+$", "severity": "warn"},
        {"name": "email present", "type": "not_null", "column": "email",
         "severity": "warn"},
        {"name": "batch not empty", "type": "row_count", "min": 1},
        # Cross-column business rule via the expression escape hatch:
        {"name": "paid orders positive", "type": "expression",
         "expr": "not (status == 'paid' and amount <= 0)"},
        # Deliberately misconfigured rule - references a column that doesn't
        # exist, to show the engine reports it gracefully instead of crashing.
        {"name": "discount sane", "type": "in_range", "column": "discount_pct",
         "min": 0, "max": 100, "severity": "warn"},
    ]


def _cli() -> None:
    df = make_sample_data()
    rules = sample_rules()
    results = run_rules(df, rules)

    print("=== DQ Rules Engine ===\n")
    print(summarize(results).to_string(index=False))

    roll = rollup(results)
    print(
        f"\nOVERALL: {roll['overall_status']}  "
        f"({roll['passed']}/{roll['total']} passed, "
        f"{roll['errors_failed']} error-fails, {roll['warns_failed']} warn-fails)")

    print("\n--- failing rules (with sample offending values) ---")
    for r in results:
        if r.passed:
            continue
        tag = "ERROR" if r.severity == "error" else "WARN "
        print(f"[{tag}] {r.name}: {r.message}")
        if r.samples:
            print(f"         e.g. {r.samples}")


if __name__ == "__main__":
    _cli()
