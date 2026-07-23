from __future__ import annotations

# Consent & Purpose Tracker - cross-check what you're actually DOING with personal
# data against what each subject CONSENTED to, and flag the gaps before a regulator
# (or the subject) asks you to prove your lawful basis. Every finding is explainable:
# it names the purpose, the dataset, the subject, and WHY the basis is broken -
# no-consent, withdrawn, expired, or purpose-mismatch. Fully offline, pandas/numpy
# only, no API keys. This is a compliance review queue, not legal advice.

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Reproducibility: sample expiry checks are evaluated as-of this fixed date, never
# datetime.now(), so the demo output is identical on every machine and every day.
AS_OF = pd.Timestamp("2026-07-01")


@dataclass
class Finding:
    """One processing activity that lacks a valid consent basis, with the reason."""

    activity_id: str
    subject_id: str
    purpose: str
    dataset: str
    issue: str        # "no-consent" | "withdrawn" | "expired" | "purpose-mismatch"
    severity: str     # "high" | "medium" | "low"
    reason: str


# Severity policy - tune to your own risk appetite. Processing personal data with no
# basis at all, or under a consent the subject actively pulled, is the worst case;
# a lapsed (expired) consent is serious but usually a renewal gap, not a betrayal;
# purpose-mismatch depends on how far the new use strays from the original one.
_SEVERITY: Dict[str, str] = {
    "no-consent": "high",
    "withdrawn": "high",
    "expired": "medium",
    "purpose-mismatch": "medium",
}

_SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


def _latest_consent(consent_df: pd.DataFrame, subject_id: str, purpose: str) -> Optional[pd.Series]:
    """Return the most recent consent row for one subject+purpose, or None.

    Consent is a log, not a single fact - a subject can grant, withdraw, and
    re-grant over time. Only the LATEST record by timestamp reflects their
    current wish, so we always reason about that one, never a stale earlier row.
    """
    mask = (consent_df["subject_id"] == subject_id) & (consent_df["purpose"] == purpose)
    matches = consent_df[mask]
    if matches.empty:
        return None
    return matches.sort_values("timestamp").iloc[-1]


def audit_consent(
    consent_df: pd.DataFrame,
    processing_df: pd.DataFrame,
    as_of: pd.Timestamp = AS_OF,
) -> List[Finding]:
    """Cross-check every processing activity against the consent log.

    For each (activity, subject) pair we ask, in order:
      1. Is there ANY consent for this exact purpose? If not -> no-consent, unless
         a consent exists for a DIFFERENT purpose (that is purpose-mismatch: data
         gathered under basis A is being reused for basis B).
      2. If the latest consent for this purpose is withdrawn -> withdrawn.
      3. If it is granted but expired as-of the audit date -> expired.
      4. Otherwise the basis is valid and no finding is raised.
    Edge case: empty inputs yield an empty finding list, not an error.
    """
    findings: List[Finding] = []
    if consent_df is None or processing_df is None:
        return findings
    if consent_df.empty and processing_df.empty:
        return findings
    if processing_df.empty:
        return findings

    # Normalise timestamp columns once so comparisons are dtype-safe.
    consent = consent_df.copy()
    if not consent.empty:
        consent["timestamp"] = pd.to_datetime(consent["timestamp"])
        # expiry may be blank (consent with no fixed end) - keep NaT, treated as
        # "never expires" below rather than "already expired".
        consent["expiry"] = pd.to_datetime(consent["expiry"], errors="coerce")

    for _, act in processing_df.iterrows():
        activity_id = str(act["activity_id"])
        purpose = str(act["purpose"])
        dataset = str(act["dataset"])
        # A single activity can touch many subjects - the subjects_touched cell is a
        # semicolon-delimited list so one processing row can fan out to many checks.
        subjects = _split_subjects(act["subjects_touched"])

        for subject_id in subjects:
            record = _latest_consent(consent, subject_id, purpose) if not consent.empty else None

            if record is None:
                # No consent for THIS purpose. Distinguish "we have nothing on this
                # person" from "we have consent, but for something else entirely".
                other = _has_other_purpose_consent(consent, subject_id, purpose)
                if other is not None:
                    findings.append(Finding(
                        activity_id, subject_id, purpose, dataset,
                        "purpose-mismatch", _SEVERITY["purpose-mismatch"],
                        f"subject consented to '{other}' but data is being processed "
                        f"for '{purpose}' - purpose limitation breached (collected for "
                        f"one basis, reused for another)",
                    ))
                else:
                    findings.append(Finding(
                        activity_id, subject_id, purpose, dataset,
                        "no-consent", _SEVERITY["no-consent"],
                        f"no consent record on file for subject '{subject_id}' and "
                        f"purpose '{purpose}' - processing has no lawful basis to point to",
                    ))
                continue

            status = str(record["status"]).lower()
            legal_basis = str(record.get("legal_basis", ""))

            if status == "withdrawn":
                findings.append(Finding(
                    activity_id, subject_id, purpose, dataset,
                    "withdrawn", _SEVERITY["withdrawn"],
                    f"subject WITHDREW consent for '{purpose}' on "
                    f"{record['timestamp'].date()} (basis '{legal_basis}') yet "
                    f"processing continues - withdrawal must stop the processing",
                ))
                continue

            if status == "expired":
                findings.append(Finding(
                    activity_id, subject_id, purpose, dataset,
                    "expired", _SEVERITY["expired"],
                    f"consent for '{purpose}' is marked expired (basis '{legal_basis}') "
                    f"but the data is still being used - re-consent required",
                ))
                continue

            # status granted: still valid ONLY if not past its expiry as-of the audit.
            expiry = record["expiry"]
            if pd.notna(expiry) and expiry < as_of:
                findings.append(Finding(
                    activity_id, subject_id, purpose, dataset,
                    "expired", _SEVERITY["expired"],
                    f"consent for '{purpose}' expired on {expiry.date()} (audited "
                    f"as-of {as_of.date()}) but processing continues - re-consent required",
                ))
                continue
            # else: granted, not withdrawn, not past expiry -> valid basis, no finding.

    return findings


def _split_subjects(cell: object) -> List[str]:
    """Parse a subjects_touched cell into a clean list of subject ids."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    return [s.strip() for s in str(cell).split(";") if s.strip()]


def _has_other_purpose_consent(
    consent: pd.DataFrame, subject_id: str, purpose: str
) -> Optional[str]:
    """If the subject has a GRANTED consent for some other purpose, name it.

    This is what separates a genuine purpose-mismatch (we hold a valid basis, just
    for the wrong thing) from a plain no-consent (we hold nothing at all).
    """
    if consent.empty:
        return None
    mask = (consent["subject_id"] == subject_id) & (consent["purpose"] != purpose)
    others = consent[mask]
    if others.empty:
        return None
    granted = others[others["status"].str.lower() == "granted"]
    pick = granted if not granted.empty else others
    return str(pick.sort_values("timestamp").iloc[-1]["purpose"])


def findings_frame(findings: List[Finding]) -> pd.DataFrame:
    """Flat table of every finding, sorted by severity - the reviewer's queue."""
    cols = ["activity_id", "subject_id", "purpose", "dataset", "issue", "severity", "reason"]
    if not findings:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame([f.__dict__ for f in findings], columns=cols)
    return df.sort_values("severity", key=lambda s: s.map(_SEV_ORDER)).reset_index(drop=True)


def compliance_summary(
    processing_df: pd.DataFrame, findings: List[Finding]
) -> Dict[str, object]:
    """Headline numbers: how much processing rests on a valid basis?

    The denominator is (activity, subject) pairs, not raw activity rows, because one
    activity touching five subjects is five separate lawful-basis questions.
    """
    total_checks = 0
    if processing_df is not None and not processing_df.empty:
        for _, act in processing_df.iterrows():
            total_checks += len(_split_subjects(act["subjects_touched"]))

    flagged = len(findings)
    valid = max(total_checks - flagged, 0)
    pct_valid = (valid / total_checks * 100.0) if total_checks else 0.0

    by_issue: Dict[str, int] = {}
    for f in findings:
        by_issue[f.issue] = by_issue.get(f.issue, 0) + 1

    return {
        "total_checks": total_checks,
        "valid_basis": valid,
        "flagged": flagged,
        "pct_valid_basis": round(pct_valid, 1),
        "by_issue": by_issue,
    }


def make_sample_data() -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Two small tables with planted issues AND clean cases, for the demo.

    Planted, one of each kind:
      - no-consent:        s005 processed for analytics, nothing on file
      - withdrawn:         s002 withdrew marketing, still being emailed
      - expired:           s003 marketing consent lapsed before the audit date
      - purpose-mismatch:  s004 consented to analytics, used for personalization
    Clean valid cases (must NOT be flagged): s001 marketing (granted, current),
      s003 analytics (granted, no expiry), s006 personalization (granted, future expiry).
    """
    consent = pd.DataFrame([
        # subject, purpose, status, legal_basis, timestamp, expiry
        ("s001", "marketing",       "granted",   "consent", "2026-01-10", "2027-01-10"),
        ("s002", "marketing",       "withdrawn", "consent", "2026-05-01", ""),
        ("s002", "marketing",       "granted",   "consent", "2026-01-01", "2027-01-01"),
        ("s003", "marketing",       "granted",   "consent", "2025-01-15", "2026-01-15"),  # expired
        ("s003", "analytics",       "granted",   "legitimate-interest", "2026-02-01", ""),
        ("s004", "analytics",       "granted",   "consent", "2026-03-01", "2027-03-01"),
        ("s006", "personalization", "granted",   "consent", "2026-04-01", "2027-04-01"),
    ], columns=["subject_id", "purpose", "status", "legal_basis", "timestamp", "expiry"])

    processing = pd.DataFrame([
        # activity_id, purpose, dataset, subjects_touched (semicolon list)
        ("act01", "marketing",       "email_campaign_q3", "s001;s002;s003"),
        ("act02", "analytics",       "product_usage_logs", "s003;s005"),
        ("act03", "personalization", "recommender_features", "s004;s006"),
    ], columns=["activity_id", "purpose", "dataset", "subjects_touched"])

    return consent, processing


def _cli() -> None:
    consent, processing = make_sample_data()
    findings = audit_consent(consent, processing, AS_OF)
    summary = compliance_summary(processing, findings)

    print("=== Consent & Purpose Tracker ===")
    print(f"(audited as-of {AS_OF.date()})\n")

    print("--- findings by severity ---")
    frame = findings_frame(findings)
    if frame.empty:
        print("no findings - every processing activity has a valid basis")
    else:
        show = frame[["severity", "issue", "activity_id", "subject_id", "purpose", "dataset"]]
        print(show.to_string(index=False))
        print("\n--- why (first lines) ---")
        for _, r in frame.iterrows():
            print(f"[{r['severity']:6}] {r['subject_id']}/{r['purpose']}: {r['reason']}")

    print("\n--- compliance summary ---")
    print(f"processing checks (activity x subject): {summary['total_checks']}")
    print(f"valid lawful basis:                     {summary['valid_basis']}")
    print(f"flagged for review:                     {summary['flagged']}")
    print(f"% with valid basis:                     {summary['pct_valid_basis']}%")
    print(f"breakdown by issue:                     {summary['by_issue']}")


if __name__ == "__main__":
    _cli()
