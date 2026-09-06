from __future__ import annotations

# Data Access Auditor - ingest access-grant records (who has what permission on
# which dataset) and flag governance risks so a human access review can start
# from evidence instead of a week of digging. Every finding says WHY it fired,
# so a data steward can trust the signal. This is a DEFENSIVE least-privilege
# hygiene tool - it surfaces review candidates, it does not revoke anything.
# Fully offline, standard pandas/numpy only - no API keys.
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

# --- Tunable governance bar - edit for your own policy tolerance. ---
STALE_DAYS = 90            # a grant unused this long is stale access to review
EXPOSURE_MAX_USERS = 5     # more than this many users on ONE restricted dataset
                           # is over-exposure of sensitive data
# Sensitivity levels ordered least -> most sensitive; "restricted" is the top.
SENSITIVITY_ORDER = ["public", "internal", "confidential", "restricted"]
# Write/admin on restricted data is only expected for these privileged roles.
PRIVILEGED_ROLES = {"data_admin", "dba", "security_admin", "platform_admin"}
# Roles that should never appear on restricted data at all - orphaned grants.
DISALLOWED_ON_RESTRICTED = {"contractor", "intern", "guest", "vendor"}
# Pairs of roles that create a segregation-of-duties conflict when one user
# holds both (e.g. the person who approves also executes).
SOD_CONFLICTS = [
    ("data_engineer", "auditor"),      # builds the pipeline AND audits it
    ("payments_admin", "approver"),    # moves money AND signs off on it
]

_SEV_RANK = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Finding:
    """One flagged grant (or user-level issue) with the reason it fired."""

    rule: str        # "over-privileged" | "stale" | "exposure" | "orphaned" | "sod"
    severity: str    # "high" | "medium" | "low"
    user: str
    dataset: str     # "*" for user-level findings that span datasets
    reason: str


def _is_restricted(level: str) -> bool:
    return str(level).strip().lower() == "restricted"


def _over_privileged(df: pd.DataFrame) -> List[Finding]:
    # Write/admin on restricted data by a non-privileged role = too much power.
    # WHY least privilege: a role should get the minimum access to do its job,
    # so a non-admin role with write/admin on restricted data is the first
    # thing an access review should trim.
    out: List[Finding] = []
    for _, r in df.iterrows():
        perm = str(r["permission"]).strip().lower()
        role = str(r["role"]).strip().lower()
        if (_is_restricted(r["sensitivity"]) and perm in {"write", "admin"}
                and role not in PRIVILEGED_ROLES):
            sev = "high" if perm == "admin" else "medium"
            out.append(Finding(
                "over-privileged", sev, str(r["user"]), str(r["dataset"]),
                f"role '{role}' holds {perm} on RESTRICTED dataset "
                f"'{r['dataset']}' but is not a privileged role "
                f"({', '.join(sorted(PRIVILEGED_ROLES))}); violates least privilege",
            ))
    return out


def _stale(df: pd.DataFrame, as_of: pd.Timestamp) -> List[Finding]:
    # Grants unused for STALE_DAYS - access that piled up and nobody uses.
    # Measured vs `as_of` (passed in, not datetime.now()) for reproducibility; a
    # never-used grant counts stale from its grant date; higher sensitivity ->
    # higher severity (a stale restricted grant is a bigger liability).
    out: List[Finding] = []
    for _, r in df.iterrows():
        last = pd.to_datetime(r.get("last_used_date"), errors="coerce")
        ref = last if pd.notna(last) else pd.to_datetime(
            r.get("granted_date"), errors="coerce")
        if pd.isna(ref):
            continue
        days = (as_of - ref).days
        if days >= STALE_DAYS:
            never = "never used" if pd.isna(last) else f"last used {days}d ago"
            sev = "high" if _is_restricted(r["sensitivity"]) else (
                "medium" if days >= STALE_DAYS * 2 else "low")
            out.append(Finding(
                "stale", sev, str(r["user"]), str(r["dataset"]),
                f"grant on '{r['dataset']}' ({r['sensitivity']}) {never}, "
                f">= {STALE_DAYS}d idle bar; candidate to revoke",
            ))
    return out


def _exposure(df: pd.DataFrame) -> List[Finding]:
    # Too many distinct users on ONE restricted dataset = over-exposure.
    # WHY: sensitive data should have a small, named audience; when the distinct
    # user count crosses the bar the whole dataset (not one grant) needs review,
    # so this is a dataset-level finding.
    out: List[Finding] = []
    restricted = df[df["sensitivity"].astype(str).str.lower() == "restricted"]
    if restricted.empty:
        return out
    for dataset, grp in restricted.groupby("dataset"):
        users = grp["user"].astype(str).nunique()
        if users > EXPOSURE_MAX_USERS:
            sev = "high" if users > EXPOSURE_MAX_USERS * 2 else "medium"
            out.append(Finding(
                "exposure", sev, "*", str(dataset),
                f"{users} distinct users can access RESTRICTED dataset "
                f"'{dataset}' (bar is {EXPOSURE_MAX_USERS}); shrink the audience",
            ))
    return out


def _orphaned(df: pd.DataFrame) -> List[Finding]:
    # Roles that should never touch restricted data (contractors, interns...).
    # WHY: an orphaned grant is one policy says should not exist at all; these
    # often survive a role change or offboarding gap and are pure risk.
    out: List[Finding] = []
    for _, r in df.iterrows():
        role = str(r["role"]).strip().lower()
        if _is_restricted(r["sensitivity"]) and role in DISALLOWED_ON_RESTRICTED:
            out.append(Finding(
                "orphaned", "high", str(r["user"]), str(r["dataset"]),
                f"role '{role}' should never hold access to RESTRICTED data, "
                f"yet has {str(r['permission']).lower()} on '{r['dataset']}'; "
                f"likely a leftover / offboarding gap",
            ))
    return out


def _sod(df: pd.DataFrame) -> List[Finding]:
    # Segregation-of-duties: one user holding two conflicting roles.
    # WHY: SoD stops one person from both performing and checking an action; a
    # user whose roles contain a conflicting pair is a control gap (user-level).
    out: List[Finding] = []
    roles_by_user: Dict[str, set] = {}
    for _, r in df.iterrows():
        roles_by_user.setdefault(str(r["user"]), set()).add(
            str(r["role"]).strip().lower())
    for user, roles in roles_by_user.items():
        for a, b in SOD_CONFLICTS:
            if a in roles and b in roles:
                out.append(Finding(
                    "sod", "high", user, "*",
                    f"user holds conflicting roles '{a}' + '{b}'; "
                    f"segregation-of-duties conflict, split the duties",
                ))
    return out


def audit_grants(df: pd.DataFrame,
                 as_of: Optional[pd.Timestamp] = None) -> List[Finding]:
    """Run every rule over a grants frame. Edge case: empty frame -> no findings.

    `as_of` is the reference "today" for staleness; pass it in for reproducible
    audits (the sample bakes a fixed date). Defaults to the max date in the data.
    """
    if df is None or df.empty:
        return []
    if as_of is None:
        dates = pd.to_datetime(
            pd.concat([df.get("last_used_date"), df.get("granted_date")]),
            errors="coerce")
        as_of = dates.max() if dates.notna().any() else pd.Timestamp.today()
    findings: List[Finding] = []
    findings += _over_privileged(df)
    findings += _stale(df, as_of)
    findings += _exposure(df)
    findings += _orphaned(df)
    findings += _sod(df)
    return findings


def findings_frame(findings: List[Finding]) -> pd.DataFrame:
    """Flat table of every finding, sorted most-severe first, for review/export."""
    cols = ["rule", "severity", "user", "dataset", "reason"]
    rows = [
        {"rule": f.rule, "severity": f.severity, "user": f.user,
         "dataset": f.dataset, "reason": f.reason}
        for f in findings
    ]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values(
            "severity", key=lambda s: s.map(_SEV_RANK), kind="stable"
        ).reset_index(drop=True)
    return df


def summarize(findings: List[Finding]) -> pd.DataFrame:
    """One row per rule with counts by severity - the steward's triage rollup."""
    rules = ["over-privileged", "stale", "exposure", "orphaned", "sod"]
    rows = []
    for rule in rules:
        sub = [f for f in findings if f.rule == rule]
        rows.append({
            "rule": rule,
            "findings": len(sub),
            "high": sum(f.severity == "high" for f in sub),
            "medium": sum(f.severity == "medium" for f in sub),
            "low": sum(f.severity == "low" for f in sub),
        })
    return pd.DataFrame(rows)


# Fixed reference date so the sample audit is reproducible (no datetime.now()).
AS_OF = pd.Timestamp("2026-07-01")


def make_sample_grants(seed: int = 42) -> pd.DataFrame:
    """Realistic access-grant table with PLANTED governance risks for the demo.

    Planted so each rule earns its keep:
      - over-privileged: analyst has ADMIN on a restricted table
      - stale: a grant unused for ~200 days
      - exposure: a restricted dataset (`customer_pii`) with many users
      - orphaned: a contractor holding access to restricted data
      - sod: one user holds conflicting data_engineer + auditor roles
    """
    rows = [
        # user, role, dataset, sensitivity, permission, granted, last_used
        # --- exposure: 7 users all on the restricted customer_pii table ---
        ("alice",   "analyst",       "customer_pii", "restricted", "read",  "2026-05-01", "2026-06-20"),
        ("bob",     "analyst",       "customer_pii", "restricted", "read",  "2026-05-01", "2026-06-18"),
        ("carol",   "analyst",       "customer_pii", "restricted", "read",  "2026-05-02", "2026-06-25"),
        ("dan",     "analyst",       "customer_pii", "restricted", "read",  "2026-05-03", "2026-06-15"),
        ("erin",    "analyst",       "customer_pii", "restricted", "read",  "2026-05-04", "2026-06-10"),
        ("frank",   "analyst",       "customer_pii", "restricted", "read",  "2026-05-05", "2026-06-01"),
        # over-privileged: an analyst with ADMIN on restricted data
        ("grace",   "analyst",       "customer_pii", "restricted", "admin", "2026-04-01", "2026-06-28"),
        # a legitimately privileged admin - should NOT be over-privileged flagged
        ("heidi",   "data_admin",    "customer_pii", "restricted", "admin", "2026-04-01", "2026-06-29"),
        # --- stale: unused ~200 days (last used 2025-12-13 vs AS_OF 2026-07-01) ---
        ("ivan",    "analyst",       "sales_summary", "internal",  "read",  "2025-06-01", "2025-12-13"),
        # stale on restricted -> escalates to high
        ("judy",    "analyst",       "customer_pii", "restricted", "read",  "2025-05-01", "2025-11-20"),
        # never-used grant (no last_used) older than the bar -> stale
        ("mallory", "analyst",       "billing",      "confidential", "read", "2026-01-01", None),
        # --- orphaned: a contractor on restricted data ---
        ("oscar",   "contractor",    "customer_pii", "restricted", "read",  "2026-06-01", "2026-06-27"),
        # --- sod: peggy holds two conflicting roles across datasets ---
        ("peggy",   "data_engineer", "sales_summary", "internal",  "write", "2026-03-01", "2026-06-26"),
        ("peggy",   "auditor",       "audit_log",     "confidential", "read", "2026-03-01", "2026-06-26"),
        # --- clean, low-risk grants (should NOT be flagged) ---
        ("trent",   "analyst",       "sales_summary", "internal",  "read",  "2026-06-01", "2026-06-28"),
        ("wendy",   "viewer",        "public_docs",   "public",    "read",  "2026-06-10", "2026-06-30"),
    ]
    df = pd.DataFrame(rows, columns=[
        "user", "role", "dataset", "sensitivity",
        "permission", "granted_date", "last_used_date",
    ])
    df["granted_date"] = pd.to_datetime(df["granted_date"])
    df["last_used_date"] = pd.to_datetime(df["last_used_date"])
    return df


def _cli() -> None:
    df = make_sample_grants()
    findings = audit_grants(df, as_of=AS_OF)
    print("=== Data Access Auditor ===")
    print(f"(reference date AS_OF = {AS_OF.date()}, {len(df)} grants)\n")
    print("--- findings (most severe first) ---")
    flat = findings_frame(findings)
    if flat.empty:
        print("No governance risks found under the current policy bar.")
    else:
        with pd.option_context("display.max_colwidth", 70):
            print(flat.to_string(index=False))
    print("\n--- summary rollup by rule ---")
    print(summarize(findings).to_string(index=False))


if __name__ == "__main__":
    _cli()
