from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# --- Normalization: make different spellings of the same entity collide -------

def normalize_email(v: object) -> Optional[str]:
    if pd.isna(v) or not str(v).strip():
        return None
    email = str(v).strip().lower()
    local, _, domain = email.partition("@")
    local = local.split("+", 1)[0].replace(".", "") if domain in ("gmail.com", "googlemail.com") else local
    return f"{local}@{domain}"


def normalize_phone(v: object) -> Optional[str]:
    if pd.isna(v) or not str(v).strip():
        return None
    digits = re.sub(r"\D", "", str(v))
    return digits[-10:] if len(digits) >= 10 else (digits or None)


def normalize_name(v: object) -> Optional[str]:
    if pd.isna(v) or not str(v).strip():
        return None
    return re.sub(r"\s+", " ", str(v).strip().lower())


# --- Survivorship rules: per-field, pick the winning value in a cluster -------

def most_recent(group: pd.DataFrame, col: str, ts_col: str) -> Tuple[object, int]:
    g = group[group[col].notna()]
    if g.empty:
        return None, -1
    idx = g[ts_col].idxmax()
    return g.loc[idx, col], idx


def source_priority(priority: List[str]) -> Callable:
    rank = {s: i for i, s in enumerate(priority)}

    def rule(group: pd.DataFrame, col: str, ts_col: str) -> Tuple[object, int]:
        g = group[group[col].notna()].copy()
        if g.empty:
            return None, -1
        g["_rank"] = g["source"].map(lambda s: rank.get(s, len(rank)))
        idx = g.sort_values(["_rank", ts_col], ascending=[True, False]).index[0]
        return g.loc[idx, col], idx
    rule.__name__ = f"source_priority({'>'.join(priority)})"
    return rule


def longest_value(group: pd.DataFrame, col: str, ts_col: str) -> Tuple[object, int]:
    g = group[group[col].notna()]
    if g.empty:
        return None, -1
    idx = g[col].astype(str).str.len().idxmax()
    return g.loc[idx, col], idx


def most_complete_record(group: pd.DataFrame, col: str, ts_col: str) -> Tuple[object, int]:
    g = group[group[col].notna()]
    if g.empty:
        return None, -1
    idx = group.notna().sum(axis=1).loc[g.index].idxmax()
    return g.loc[idx, col], idx


@dataclass
class DedupReport:
    input_rows: int
    clusters: int
    golden_rows: int
    dup_rows_merged: int
    field_provenance: pd.DataFrame  # cluster_id, field, winning_source, rule


def dedupe(df: pd.DataFrame, match_keys: List[str], rules: Dict[str, Callable],
           default_rule: Callable = most_recent, ts_col: str = "updated_at",
           id_col: str = "record_id") -> Tuple[pd.DataFrame, DedupReport]:
    """Cluster rows whose normalized match_keys collide; merge each cluster into a golden record."""
    work = df.copy().reset_index(drop=True)
    work["_match"] = work[match_keys].astype(str).agg("|".join, axis=1)

    golden_rows: List[Dict] = []
    provenance: List[Dict] = []
    audit_cols = {id_col, "_match", ts_col, "source", *match_keys}
    data_cols = [c for c in df.columns if c not in audit_cols]

    for cluster_id, (_, group) in enumerate(work.groupby("_match", sort=False)):
        golden: Dict[str, object] = {"cluster_id": cluster_id, "cluster_size": len(group),
                                     "member_ids": ",".join(group[id_col].astype(str))}
        for col in data_cols:
            rule = rules.get(col, default_rule)
            value, win_idx = rule(group, col, ts_col)
            golden[col] = value
            if len(group) > 1 and win_idx >= 0:
                provenance.append({"cluster_id": cluster_id, "field": col,
                                   "winning_source": group.loc[win_idx, "source"],
                                   "winning_record": group.loc[win_idx, id_col],
                                   "rule": rule.__name__ if hasattr(rule, "__name__") else "custom"})
        golden_rows.append(golden)

    golden_df = pd.DataFrame(golden_rows)
    report = DedupReport(
        input_rows=len(df),
        clusters=len(golden_df),
        golden_rows=len(golden_df),
        dup_rows_merged=len(df) - len(golden_df),
        field_provenance=pd.DataFrame(provenance),
    )
    return golden_df, report


def make_sample_customers(seed: int = 3) -> pd.DataFrame:
    """90 raw rows from 3 systems describing 60 real people — with messy variants."""
    import random
    rng = random.Random(seed)
    first = ["ana", "ben", "carla", "dev", "elena", "farid", "gina", "hugo", "iris", "jon"]
    last = ["li", "novak", "santos", "kim", "okafor", "muller"]
    rows = []
    rid = 0
    for i in range(60):
        f, ln = rng.choice(first), rng.choice(last)
        email = f"{f}.{ln}{i}@gmail.com"
        phone = f"+1 (555) {rng.randint(100, 999)}-{1000 + i}"
        base = {"name": f"{f.title()} {ln.title()}", "email": email, "phone": phone,
                "address": f"{rng.randint(1, 999)} Oak St", "loyalty_tier": rng.choice(["gold", "silver", None])}
        n_copies = 1 if rng.random() < 0.6 else rng.choice([2, 2, 3])
        for c in range(n_copies):
            rid += 1
            row = dict(base)
            row["record_id"] = f"r{rid:03d}"
            row["source"] = rng.choice(["crm", "web", "import"])
            row["updated_at"] = f"2026-0{rng.randint(1, 6)}-{rng.randint(10, 28)}"
            if c > 0:  # mess up the duplicates
                if rng.random() < 0.5:
                    row["email"] = email.replace(".", "") if "@gmail" in email else email.upper()
                if rng.random() < 0.5:
                    row["phone"] = re.sub(r"\D", "", phone)
                if rng.random() < 0.4:
                    row["address"] = None
                if rng.random() < 0.3:
                    row["name"] = row["name"].upper()
                if rng.random() < 0.3:
                    row["loyalty_tier"] = "platinum"
            rows.append(row)
    return pd.DataFrame(rows)


def add_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["email_norm"] = out["email"].map(normalize_email)
    out["phone_norm"] = out["phone"].map(normalize_phone)
    return out
