"""Policy-driven PII redaction with re-identification scoring.

UI-free core so it can be imported by app.py, a notebook, or a pipeline.

The premise: hiding direct identifiers is the easy half. The hard half is
(a) keeping the extract useful - joins must still work - and (b) knowing
that a row with no name in it can still be re-identified from the columns
you left behind.

Strategies
  keep        leave as-is
  drop        remove the column entirely
  nullify     keep the column, blank every value
  mask        partial reveal, human readable, NOT join-safe
  hash        one-way SHA-256, not joinable across salts
  tokenize    keyed HMAC-SHA256, deterministic -> joins survive
  generalize  coarsen into bands (age, zip prefix, numeric band)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

STRATEGIES = (
    "keep",
    "drop",
    "nullify",
    "mask",
    "hash",
    "tokenize",
    "generalize",
)

# --------------------------------------------------------------- primitives


def tokenize_value(value: object, salt: bytes, length: int = 12) -> Optional[str]:
    """Deterministic keyed pseudonym. Same input + same salt -> same token,
    which is what lets a redacted extract still join.

    Keyed (HMAC) rather than a plain hash: an unkeyed hash of an email is
    trivially reversed by hashing a wordlist.
    """
    if pd.isna(value):
        return None
    digest = hmac.new(salt, str(value).encode("utf-8"), hashlib.sha256).digest()
    token = base64.b32encode(digest).decode("ascii").rstrip("=")
    return f"tok_{token[:length].lower()}"


def hash_value(value: object, length: int = 12) -> Optional[str]:
    """One-way, unkeyed. Not joinable across systems, not reversible by you
    either - use when you never need the value back."""
    if pd.isna(value):
        return None
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return digest[:length]


def mask_value(
    value: object, keep_last: int = 4, keep_domain: bool = False
) -> Optional[str]:
    """Partial reveal for human eyeballs. Every distinct input can produce
    the same output, so masked columns cannot be joined or counted
    distinctly - that is the tradeoff."""
    if pd.isna(value):
        return None
    text = str(value)
    if keep_domain and "@" in text:
        local, _, domain = text.partition("@")
        head = local[:1] if local else ""
        return f"{head}{'*' * max(1, len(local) - 1)}@{domain}"
    digits = re.sub(r"\D", "", text)
    if digits and len(digits) > keep_last:
        return f"{'*' * (len(digits) - keep_last)}{digits[-keep_last:]}"
    if len(text) > keep_last:
        return f"{'*' * (len(text) - keep_last)}{text[-keep_last:]}"
    return "*" * len(text)


def generalize_age_band(dob: object, width: int = 10, today: str = "2026-07-29") -> Optional[str]:
    """Date of birth -> age band. Keeps the column analytically useful while
    collapsing the single most powerful quasi-identifier."""
    if pd.isna(dob):
        return None
    born = pd.to_datetime(dob, errors="coerce")
    if pd.isna(born):
        return None
    age = int((pd.Timestamp(today) - born).days // 365.25)
    low = (age // width) * width
    return f"{low}-{low + width - 1}"


def generalize_zip(value: object, digits: int = 3) -> Optional[str]:
    """Postal code -> leading digits. US ZIP3 is the classic HIPAA
    generalization; ZIP5 + DOB + gender is famously near-unique."""
    if pd.isna(value):
        return None
    text = re.sub(r"\D", "", str(value))
    if not text:
        return None
    return f"{text[:digits]}{'*' * max(0, len(text[:5]) - digits)}"


def generalize_numeric(value: object, width: float = 10000.0) -> Optional[str]:
    """Continuous value -> band label."""
    if pd.isna(value):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    low = int(np.floor(num / width) * width)
    return f"{low:,}-{int(low + width):,}"


# ------------------------------------------------------------------ policy


def apply_policy(
    df: pd.DataFrame, policy: Dict[str, dict], salt: bytes
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a per-column redaction policy.

    Returns (redacted_frame, audit_log). The audit log is the artefact a
    reviewer actually reads: one row per column saying what was done, how
    many values it touched, and whether the result is still joinable.
    """
    out = df.copy()
    records: List[dict] = []

    for column in df.columns:
        rule = policy.get(column, {"strategy": "keep"})
        strategy = rule.get("strategy", "keep")
        params = rule.get("params", {}) or {}
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r} for column {column!r}")

        before_distinct = int(df[column].nunique(dropna=True))

        if strategy == "keep":
            pass
        elif strategy == "drop":
            out = out.drop(columns=[column])
        elif strategy == "nullify":
            out[column] = None
        elif strategy == "mask":
            out[column] = df[column].map(lambda v: mask_value(v, **params))
        elif strategy == "hash":
            out[column] = df[column].map(lambda v: hash_value(v, **params))
        elif strategy == "tokenize":
            out[column] = df[column].map(lambda v: tokenize_value(v, salt, **params))
        elif strategy == "generalize":
            kind = params.get("kind", "numeric_band")
            if kind == "age_band":
                out[column] = df[column].map(
                    lambda v: generalize_age_band(v, params.get("width", 10))
                )
            elif kind == "zip_prefix":
                out[column] = df[column].map(
                    lambda v: generalize_zip(v, params.get("digits", 3))
                )
            elif kind == "numeric_band":
                out[column] = df[column].map(
                    lambda v: generalize_numeric(v, params.get("width", 10000.0))
                )
            else:
                raise ValueError(f"unknown generalize kind {kind!r}")

        after_distinct = (
            0 if strategy == "drop" else int(out[column].nunique(dropna=True))
        )
        records.append(
            {
                "column": column,
                "strategy": strategy,
                "distinct_before": before_distinct,
                "distinct_after": after_distinct,
                "join_safe": strategy in ("keep", "tokenize", "hash"),
                "reversible_by_key_holder": strategy == "tokenize",
            }
        )

    return out, pd.DataFrame(records)


# -------------------------------------------------- re-identification risk


def equivalence_classes(df: pd.DataFrame, quasi_ids: Iterable[str]) -> pd.Series:
    """Group sizes over the quasi-identifier combination."""
    cols = [c for c in quasi_ids if c in df.columns]
    if not cols:
        return pd.Series(dtype=int)
    return df.groupby(cols, dropna=False).size()


def k_anonymity(df: pd.DataFrame, quasi_ids: Iterable[str]) -> Dict[str, object]:
    """k-anonymity over a quasi-identifier set.

    k = the size of the smallest equivalence class. k=1 means at least one
    row is unique on those columns, so anyone holding the same columns from
    another source can single that person out - even with every direct
    identifier stripped.
    """
    sizes = equivalence_classes(df, quasi_ids)
    if sizes.empty:
        return {"k": None, "singletons": 0, "rows": len(df), "classes": 0}
    singletons = int((sizes == 1).sum())
    return {
        "k": int(sizes.min()),
        "singletons": singletons,
        "singleton_share": singletons / len(df) if len(df) else 0.0,
        "rows": len(df),
        "classes": int(len(sizes)),
        "median_class_size": float(sizes.median()),
        "quasi_ids": [c for c in quasi_ids if c in df.columns],
    }


def suppress_below_k(
    df: pd.DataFrame, quasi_ids: Iterable[str], k: int = 5
) -> Tuple[pd.DataFrame, int]:
    """Drop rows whose equivalence class is smaller than k.

    The last resort when generalization alone will not reach the threshold.
    Returns (kept_frame, rows_suppressed). Suppression is not free: it
    biases the extract toward common cases, which is worth stating in the
    handover note.
    """
    cols = [c for c in quasi_ids if c in df.columns]
    if not cols:
        return df.copy(), 0
    sizes = df.groupby(cols, dropna=False)[cols[0]].transform("size")
    keep = sizes >= k
    return df[keep].copy(), int((~keep).sum())


def generalization_ladder(
    df: pd.DataFrame,
    levels: Optional[List[dict]] = None,
    target_k: int = 5,
) -> pd.DataFrame:
    """Walk progressively coarser generalizations and score k at each rung.

    The point of a ladder rather than a single setting: k-anonymity is a dial
    with a real utility cost, and the rung where k clears the threshold is
    usually coarser than people guess. Each level says which quasi-identifiers
    it keeps and how coarsely.
    """
    if levels is None:
        levels = [
            {"label": "raw", "zip": 5, "age": 1, "gender": True},
            {"label": "zip3 + 10y bands", "zip": 3, "age": 10, "gender": True},
            {"label": "zip2 + 10y bands", "zip": 2, "age": 10, "gender": True},
            {"label": "zip2 + 20y bands", "zip": 2, "age": 20, "gender": True},
            {"label": "zip1 + 20y bands", "zip": 1, "age": 20, "gender": True},
            {"label": "zip1 + 20y, no gender", "zip": 1, "age": 20, "gender": False},
            {"label": "region only + 20y", "zip": 0, "age": 20, "gender": False},
        ]

    rows = []
    for level in levels:
        work = pd.DataFrame(index=df.index)
        quasi: List[str] = []

        if level["zip"] > 0:
            work["postal_code"] = df["postal_code"].map(
                lambda v: generalize_zip(v, level["zip"])
            )
            quasi.append("postal_code")

        if level["age"] <= 1:
            work["date_of_birth"] = pd.to_datetime(
                df["date_of_birth"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        else:
            work["date_of_birth"] = df["date_of_birth"].map(
                lambda v: generalize_age_band(v, level["age"])
            )
        quasi.append("date_of_birth")

        if level["gender"]:
            work["gender"] = df["gender"]
            quasi.append("gender")

        score = k_anonymity(work, quasi)
        _, suppressed = suppress_below_k(work, quasi, target_k)
        rows.append(
            {
                "level": level["label"],
                "quasi_ids": len(quasi),
                "classes": score["classes"],
                "k": score["k"],
                "singletons": score["singletons"],
                "singleton_share": score["singleton_share"],
                f"rows_suppressed_for_k{target_k}": suppressed,
                f"suppression_share": suppressed / len(df) if len(df) else 0.0,
                f"clears_k{target_k}": bool(
                    score["k"] is not None and score["k"] >= target_k
                ),
            }
        )
    return pd.DataFrame(rows)


def dictionary_attack(
    tokens: pd.Series, candidates: Iterable[object], salt: bytes, length: int = 12
) -> Dict[str, object]:
    """Recover a tokenized column by brute-forcing a candidate list.

    This is the failure mode people miss: tokenization is only as strong as
    the input space. Token a column with 3 distinct values and an attacker
    who knows the salt needs 3 guesses. Deterministic tokens on
    low-cardinality columns are theatre.
    """
    table = {tokenize_value(c, salt, length): c for c in candidates}
    recovered = {t: table[t] for t in tokens.dropna().unique() if t in table}
    distinct = int(tokens.nunique(dropna=True))
    return {
        "distinct_tokens": distinct,
        "guesses_needed": len(list(candidates)),
        "recovered": recovered,
        "recovered_share": (len(recovered) / distinct) if distinct else 0.0,
    }


def low_cardinality_warnings(
    df: pd.DataFrame, audit: pd.DataFrame, threshold: int = 50
) -> List[str]:
    """Flag tokenized or hashed columns whose value space is small enough to
    brute-force."""
    notes: List[str] = []
    for row in audit.itertuples():
        if row.strategy in ("tokenize", "hash") and row.distinct_before <= threshold:
            notes.append(
                f"'{row.column}' was {row.strategy}d but has only "
                f"{row.distinct_before} distinct values. Anyone who can guess "
                f"the value space (and, for tokenize, holds the salt) reverses "
                f"it in {row.distinct_before} attempts. Generalize, drop, or "
                f"accept it as clear text with eyes open."
            )
    return notes


def verify_join(
    left_before: pd.DataFrame,
    right_before: pd.DataFrame,
    left_after: pd.DataFrame,
    right_after: pd.DataFrame,
    key: str,
) -> Dict[str, object]:
    """Prove the redacted extract is still usable.

    A redaction that silently breaks referential integrity gets discovered
    by an analyst three weeks later, so check it at redaction time.
    """
    before = len(left_before.merge(right_before, on=key, how="inner"))
    after = (
        len(left_after.merge(right_after, on=key, how="inner"))
        if key in left_after.columns and key in right_after.columns
        else 0
    )
    return {
        "key": key,
        "rows_before": before,
        "rows_after": after,
        "preserved": before == after and before > 0,
    }


# ------------------------------------------------------------ sample data


def sample_data(n: int = 400, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Two synthetic related tables: members and their claims.

    Entirely generated - no real person's data. Deliberately shaped so that
    zip + date_of_birth + gender leaves most rows unique, which is the whole
    point of the k-anonymity section.
    """
    rng = np.random.default_rng(seed)
    first = ["Alex", "Bianca", "Chen", "Divya", "Ekow", "Farah", "Goh", "Hana",
             "Ivan", "Jia", "Kofi", "Lena", "Mei", "Nadia", "Omar", "Priya"]
    last = ["Tan", "Lim", "Okafor", "Silva", "Novak", "Haddad", "Wong", "Reyes",
            "Ahmed", "Ivanov", "Mbeki", "Costa", "Nair", "Kaur", "Yusuf", "Park"]
    domains = ["example.com", "mail.test", "corp.example", "sample.org"]

    members = pd.DataFrame(
        {
            "member_id": [f"M{100000 + i}" for i in range(n)],
            "full_name": [
                f"{rng.choice(first)} {rng.choice(last)}" for _ in range(n)
            ],
            "email": [
                f"user{i:04d}@{rng.choice(domains)}" for i in range(n)
            ],
            "phone": [
                f"+65 {rng.integers(8000, 9999)}{rng.integers(1000, 9999)}"
                for _ in range(n)
            ],
            "national_id": [
                f"S{rng.integers(1000000, 9999999)}{rng.choice(list('ABCDEFG'))}"
                for _ in range(n)
            ],
            "date_of_birth": pd.to_datetime("1960-01-01")
            + pd.to_timedelta(rng.integers(0, 16000, n), unit="D"),
            "postal_code": [f"{rng.integers(10000, 99999)}" for _ in range(n)],
            "gender": rng.choice(["F", "M", "X"], n, p=[0.49, 0.49, 0.02]),
            "annual_income": rng.normal(78000, 26000, n).round(-2).clip(20000),
            "plan_tier": rng.choice(["basic", "plus", "premium"], n),
        }
    )

    claim_rows = rng.integers(1, 4, n)
    claims = pd.DataFrame(
        {
            "member_id": np.repeat(members["member_id"].values, claim_rows),
            "claim_id": [f"C{200000 + i}" for i in range(int(claim_rows.sum()))],
            "diagnosis_code": rng.choice(
                ["E11", "I10", "J45", "M54", "F32", "K21"], int(claim_rows.sum())
            ),
            "amount_sgd": rng.gamma(3, 400, int(claim_rows.sum())).round(2),
        }
    )
    return members, claims


DEFAULT_POLICY: Dict[str, dict] = {
    "member_id": {"strategy": "tokenize"},
    "full_name": {"strategy": "drop"},
    "email": {"strategy": "tokenize"},
    "phone": {"strategy": "mask", "params": {"keep_last": 4}},
    "national_id": {"strategy": "hash"},
    "date_of_birth": {"strategy": "generalize", "params": {"kind": "age_band", "width": 10}},
    "postal_code": {"strategy": "generalize", "params": {"kind": "zip_prefix", "digits": 2}},
    "gender": {"strategy": "keep"},
    "annual_income": {
        "strategy": "generalize",
        "params": {"kind": "numeric_band", "width": 25000.0},
    },
    "plan_tier": {"strategy": "keep"},
}

QUASI_IDS = ["postal_code", "date_of_birth", "gender"]
