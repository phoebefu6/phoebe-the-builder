from __future__ import annotations

# Data Retention Enforcer - evaluate every record against the retention policy
# for its data_class and produce an ACTION PLAN: what is past retention (with the
# action due), what is approaching expiry (inside a warning window), and what is
# still within policy. Every row is explainable (class, age, policy, verdict) so
# a steward - and legal - can trust the plan instead of guessing.
#
# This is a DEFENSIVE data-minimization tool. It PLANS actions for human review;
# it NEVER auto-deletes or auto-anonymizes anything. Legal owns the policy.
# Fully offline, standard pandas/numpy only - no API keys.
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Reproducibility: age is measured against a FIXED baseline, not the wall clock,
# so the sample data and every rendered output stay stable across runs/machines.
AS_OF = pd.Timestamp("2026-07-01")

# How close to expiry counts as "approaching" - a heads-up window so teams can
# act (or file a legal-hold exception) BEFORE a record is past its purpose.
DEFAULT_WARNING_DAYS = 30

# Verdict labels - kept as constants so the app and notebook agree on wording.
V_PAST = "past_retention"      # age > max_age_days -> action is due now
V_APPROACHING = "approaching"  # inside the warning window before expiry
V_WITHIN = "within_policy"     # comfortably inside retention
V_NO_POLICY = "no_policy"      # data_class has no policy -> flag, do not crash


@dataclass
class RetentionPolicy:
    """The retention rule for one data_class. `action` is what SHOULD happen
    once a record is past retention - reviewed by a human, never auto-run."""

    data_class: str
    max_age_days: int
    action: str                 # "delete" | "anonymize" | "review"
    warning_days: int = DEFAULT_WARNING_DAYS
    basis: str = ""             # WHY this policy exists (purpose / legal basis)


@dataclass
class Verdict:
    """One record's evaluation - carries enough context to defend the decision."""

    record_id: object
    data_class: str
    age_days: int
    verdict: str                # one of the V_* constants
    action_due: str             # concrete action, or "" / "none"
    reason: str


# Urgency ordering for sorting the plan - most pressing at the top.
_VERDICT_RANK = {V_PAST: 0, V_APPROACHING: 1, V_NO_POLICY: 2, V_WITHIN: 3}
# Within "past_retention", delete is the most consequential action to review.
_ACTION_RANK = {"delete": 0, "anonymize": 1, "review": 2, "none": 3, "": 3}


def _policies_to_map(policies: List[RetentionPolicy]) -> Dict[str, RetentionPolicy]:
    return {p.data_class: p for p in policies}


def evaluate(
    records_df: pd.DataFrame,
    policies: List[RetentionPolicy],
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Evaluate every record against its class policy; return the action plan.

    Expects columns `record_id`, `created_date`, `data_class`. Age is computed
    against `as_of` (defaults to the fixed AS_OF baseline for reproducibility).
    Edge cases: an empty frame returns an empty plan (same columns); an unknown
    data_class is flagged V_NO_POLICY rather than raising - a missing policy is
    itself a governance finding, not a crash.
    """
    cols = [
        "record_id", "data_class", "created_date", "as_of", "age_days",
        "max_age_days", "policy_action", "verdict", "action_due", "reason",
    ]
    if records_df is None or records_df.empty:
        return pd.DataFrame(columns=cols)

    as_of = AS_OF if as_of is None else pd.Timestamp(as_of)
    pmap = _policies_to_map(policies)
    created = pd.to_datetime(records_df["created_date"])

    rows: List[dict] = []
    for i, rec in records_df.reset_index(drop=True).iterrows():
        rid = rec["record_id"]
        dclass = rec["data_class"]
        cdate = created.iloc[i]
        # Age in whole days against the fixed baseline. A future-dated record
        # yields a negative age - surfaced honestly rather than clamped, since
        # it usually signals a bad timestamp worth a human look.
        age_days = int((as_of.normalize() - cdate.normalize()).days)

        policy = pmap.get(dclass)
        if policy is None:
            rows.append({
                "record_id": rid, "data_class": dclass,
                "created_date": cdate, "as_of": as_of, "age_days": age_days,
                "max_age_days": np.nan, "policy_action": "none",
                "verdict": V_NO_POLICY, "action_due": "define_policy",
                "reason": (
                    f"data_class '{dclass}' has no retention policy - cannot "
                    f"prove this data is within purpose; define a policy"
                ),
            })
            continue

        max_age = policy.max_age_days
        warn_at = max_age - policy.warning_days
        if age_days > max_age:
            verdict, action = V_PAST, policy.action
            over = age_days - max_age
            reason = (
                f"age {age_days}d exceeds {max_age}d limit by {over}d "
                f"({policy.basis or 'policy'}); action due: {action}"
            )
        elif age_days >= warn_at:
            verdict, action = V_APPROACHING, "none"
            left = max_age - age_days
            reason = (
                f"age {age_days}d is within {policy.warning_days}d of the "
                f"{max_age}d limit ({left}d left); prepare to {policy.action}"
            )
        else:
            verdict, action = V_WITHIN, "none"
            left = max_age - age_days
            reason = f"age {age_days}d is within the {max_age}d limit ({left}d left)"

        rows.append({
            "record_id": rid, "data_class": dclass,
            "created_date": cdate, "as_of": as_of, "age_days": age_days,
            "max_age_days": max_age, "policy_action": policy.action,
            "verdict": verdict, "action_due": action, "reason": reason,
        })

    plan = pd.DataFrame(rows, columns=cols)
    # Sort by urgency: verdict first, then the weight of the action due.
    plan["_v"] = plan["verdict"].map(_VERDICT_RANK).fillna(9)
    plan["_a"] = plan["action_due"].map(_ACTION_RANK).fillna(9)
    plan = plan.sort_values(["_v", "_a", "age_days"], ascending=[True, True, False])
    return plan.drop(columns=["_v", "_a"]).reset_index(drop=True)


def summarize(plan: pd.DataFrame) -> pd.DataFrame:
    """Rollup for the plan: records and rough size by action due. Size uses a
    nominal 5 KB/record estimate so leadership can see storage-at-stake in
    GB-ish terms without needing the real byte counts."""
    cols = ["action_due", "records", "approx_gb"]
    if plan is None or plan.empty:
        return pd.DataFrame(columns=cols)
    kb_per_record = 5.0
    grp = plan.groupby("action_due").size().rename("records").reset_index()
    grp["approx_gb"] = (grp["records"] * kb_per_record / (1024 * 1024)).round(4)
    order = {"delete": 0, "anonymize": 1, "review": 2, "define_policy": 3, "none": 4}
    grp["_o"] = grp["action_due"].map(order).fillna(9)
    return grp.sort_values("_o").drop(columns=["_o"])[cols].reset_index(drop=True)


def make_sample_data() -> tuple[pd.DataFrame, List[RetentionPolicy]]:
    """Sample records + policy set with planted cases so the demo shows every
    verdict. Dates are anchored to AS_OF so ages are deterministic - NO call to
    datetime.now(), on purpose."""
    policies = [
        RetentionPolicy("marketing_lead", 365, "delete", 30,
                        "consent-based marketing, 1yr purpose"),
        RetentionPolicy("transaction", 2555, "anonymize", 90,
                        "7yr tax/finance retention, then de-identify"),
        RetentionPolicy("support_ticket", 730, "review", 60,
                        "2yr service history, review before disposal"),
        RetentionPolicy("audit_log", 3650, "review", 90,
                        "10yr compliance evidence, legal review to dispose"),
    ]

    def d(days_old: int) -> pd.Timestamp:
        # created_date = AS_OF minus an age, so age_days == days_old exactly.
        return AS_OF - pd.Timedelta(days=days_old)

    # (record_id, data_class, days_old) - planted to hit each verdict/action.
    planted = [
        # Past retention -> delete (marketing_lead limit 365)
        ("ML-001", "marketing_lead", 500),
        ("ML-002", "marketing_lead", 400),
        # Approaching -> marketing_lead (within 30d of 365)
        ("ML-003", "marketing_lead", 350),
        # Within policy -> marketing_lead
        ("ML-004", "marketing_lead", 120),
        # Past retention -> anonymize (transaction limit 2555)
        ("TX-001", "transaction", 2800),
        # Approaching -> transaction (within 90d of 2555)
        ("TX-002", "transaction", 2500),
        # Within policy -> transaction
        ("TX-003", "transaction", 300),
        # Past retention -> review (support_ticket limit 730)
        ("ST-001", "support_ticket", 900),
        # Approaching -> support_ticket (within 60d of 730)
        ("ST-002", "support_ticket", 700),
        # Within policy -> support_ticket
        ("ST-003", "support_ticket", 200),
        # Within policy -> audit_log (limit 3650, kept long on purpose)
        ("AL-001", "audit_log", 1000),
        # Approaching -> audit_log (within 90d of 3650)
        ("AL-002", "audit_log", 3600),
        # Unknown class -> no_policy (governance gap, must not crash)
        ("XX-001", "biometric_scan", 50),
    ]
    records = pd.DataFrame(
        [(rid, dc, d(age)) for rid, dc, age in planted],
        columns=["record_id", "data_class", "created_date"],
    )
    return records, policies


def _cli() -> None:
    records, policies = make_sample_data()
    plan = evaluate(records, policies, as_of=AS_OF)

    print("=== Data Retention Enforcer ===")
    print(f"AS_OF baseline: {AS_OF.date()}  |  records: {len(records)}  "
          f"|  policies: {len(policies)}\n")

    print("--- Action Plan (sorted by urgency) ---")
    show = plan[["record_id", "data_class", "age_days", "max_age_days",
                 "verdict", "action_due", "reason"]].copy()
    show["max_age_days"] = show["max_age_days"].astype("Int64")
    print(show.to_string(index=False))

    print("\n--- Rollup summary (records / approx GB by action due) ---")
    print(summarize(plan).to_string(index=False))

    # Edge case demo: an empty frame yields an empty plan, no exception.
    empty = evaluate(pd.DataFrame(), policies, as_of=AS_OF)
    print(f"\nedge case - empty input -> {len(empty)} rows (no crash)")


if __name__ == "__main__":
    _cli()
