from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass
class LoadStats:
    cycle: int
    source_rows: int
    extracted: int
    inserted: int
    updated: int
    deleted: int
    watermark: str

    @property
    def scanned_pct(self) -> float:
        return round(100 * self.extracted / self.source_rows, 1) if self.source_rows else 0.0


class WatermarkStore:
    """Persists the high-watermark per table so each run picks up where the last one stopped."""

    def __init__(self, path: str = "watermarks.json") -> None:
        self.path = Path(path)
        self._data: Dict[str, str] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, table: str) -> Optional[str]:
        return self._data.get(table)

    def set(self, table: str, value: str) -> None:
        self._data[table] = value
        self.path.write_text(json.dumps(self._data, indent=2))


def extract_increment(source: pd.DataFrame, watermark_col: str, last_watermark: Optional[str]) -> pd.DataFrame:
    """Rows changed since the last watermark — the whole point: never rescan the full table."""
    if last_watermark is None:
        return source.copy()
    return source[source[watermark_col] > last_watermark].copy()


def upsert(target: pd.DataFrame, increment: pd.DataFrame, key: str,
           soft_delete_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Merge an increment into the target by key: update matches, insert new, apply tombstones."""
    counts = {"inserted": 0, "updated": 0, "deleted": 0}
    if increment.empty:
        return target, counts

    inc = increment.copy()
    if soft_delete_col and soft_delete_col in inc.columns:
        tombstones = inc[inc[soft_delete_col].fillna(False).astype(bool)]
        counts["deleted"] = int(target[key].isin(tombstones[key]).sum())
        target = target[~target[key].isin(tombstones[key])]
        inc = inc[~inc[soft_delete_col].fillna(False).astype(bool)]

    existing = set(target[key])
    counts["updated"] = int(inc[key].isin(existing).sum())
    counts["inserted"] = len(inc) - counts["updated"]

    merged = pd.concat([target[~target[key].isin(inc[key])], inc], ignore_index=True)
    return merged.sort_values(key).reset_index(drop=True), counts


def run_incremental_load(source: pd.DataFrame, target: pd.DataFrame, key: str, watermark_col: str,
                         store: WatermarkStore, table: str = "default", cycle: int = 1,
                         soft_delete_col: Optional[str] = None) -> Tuple[pd.DataFrame, LoadStats]:
    """One load cycle: read watermark → extract increment → upsert → advance watermark."""
    last = store.get(table)
    increment = extract_increment(source, watermark_col, last)
    merged, counts = upsert(target, increment, key, soft_delete_col)
    new_watermark = str(source[watermark_col].max()) if len(source) else (last or "")
    store.set(table, new_watermark)
    stats = LoadStats(cycle=cycle, source_rows=len(source), extracted=len(increment),
                      watermark=new_watermark, **counts)
    return merged, stats


def simulate_source_day(source: pd.DataFrame, day: str, n_inserts: int, n_updates: int,
                        n_deletes: int, key: str = "id", seed: int = 42) -> pd.DataFrame:
    """Mutate a fake source table for one 'day': inserts, in-place updates, soft deletes."""
    rng = pd.Series(range(len(source))).sample(frac=1, random_state=seed).index
    df = source.copy()
    ts = f"{day} 09:00:00"

    live = df[~df["deleted"].fillna(False).astype(bool)]
    upd_keys = live[key].sample(min(n_updates, len(live)), random_state=seed)
    df.loc[df[key].isin(upd_keys), ["amount", "updated_at"]] = [
        round(100 + 900 * (seed % 7) / 7, 2), ts]

    remaining = df[~df[key].isin(upd_keys) & ~df["deleted"].fillna(False).astype(bool)]
    del_keys = remaining[key].sample(min(n_deletes, len(remaining)), random_state=seed + 1)
    df.loc[df[key].isin(del_keys), ["deleted", "updated_at"]] = [True, ts]

    start_id = int(df[key].str.replace("cust-", "").astype(int).max()) + 1
    new = pd.DataFrame({
        key: [f"cust-{start_id + i:04d}" for i in range(n_inserts)],
        "amount": [round(50.0 + 13.7 * i, 2) for i in range(n_inserts)],
        "updated_at": ts,
        "deleted": False,
    })
    _ = rng
    return pd.concat([df, new], ignore_index=True)


def make_initial_source(n: int = 500, day: str = "2026-07-01") -> pd.DataFrame:
    return pd.DataFrame({
        "id": [f"cust-{i:04d}" for i in range(1, n + 1)],
        "amount": [round(20 + (i * 37.3) % 480, 2) for i in range(1, n + 1)],
        "updated_at": f"{day} 08:00:00",
        "deleted": False,
    })
