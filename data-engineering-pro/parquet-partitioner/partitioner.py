from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


@dataclass
class ConversionReport:
    rows: int
    csv_bytes: int
    parquet_bytes: int
    n_partitions: int
    n_files: int
    partition_cols: List[str]

    @property
    def compression_ratio(self) -> float:
        return round(self.csv_bytes / self.parquet_bytes, 1) if self.parquet_bytes else 0.0


def infer_partition_cols(df: pd.DataFrame, max_cardinality: int = 50) -> List[str]:
    """Pick partition columns: derive year/month from the first date-like column,
    else use a low-cardinality categorical column."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return [f"{col}__year", f"{col}__month"]
        if df[col].dtype == object:
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.9:
                return [f"{col}__year", f"{col}__month"]
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() <= max_cardinality:
            return [col]
    return []


def add_derived_partitions(df: pd.DataFrame, partition_cols: List[str]) -> pd.DataFrame:
    """Materialize year/month columns referenced as '<col>__year' / '<col>__month'."""
    out = df.copy()
    for pcol in partition_cols:
        if "__" in pcol:
            src, part = pcol.rsplit("__", 1)
            dt = pd.to_datetime(out[src], errors="coerce", format="mixed")
            out[pcol] = getattr(dt.dt, part).astype("Int64")
    return out


def convert_to_partitioned_parquet(df: pd.DataFrame, out_dir: str,
                                   partition_cols: Optional[List[str]] = None,
                                   compression: str = "snappy",
                                   csv_bytes: Optional[int] = None) -> ConversionReport:
    """Write a DataFrame as a hive-partitioned parquet dataset and report the win."""
    partition_cols = partition_cols if partition_cols is not None else infer_partition_cols(df)
    prepared = add_derived_partitions(df, partition_cols)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(prepared, preserve_index=False)
    pq.write_to_dataset(table, root_path=str(out), partition_cols=partition_cols or None,
                        compression=compression)

    files = list(out.rglob("*.parquet"))
    partitions = {f.parent for f in files}
    if csv_bytes is None:
        csv_bytes = len(df.to_csv(index=False).encode())
    return ConversionReport(
        rows=len(df),
        csv_bytes=csv_bytes,
        parquet_bytes=sum(f.stat().st_size for f in files),
        n_partitions=len(partitions),
        n_files=len(files),
        partition_cols=partition_cols,
    )


def compare_compressions(df: pd.DataFrame, base_dir: str,
                         codecs: Tuple[str, ...] = ("snappy", "gzip", "zstd")) -> Dict[str, int]:
    """Bytes on disk per codec, unpartitioned single file."""
    sizes: Dict[str, int] = {}
    table = pa.Table.from_pandas(df, preserve_index=False)
    for codec in codecs:
        path = Path(base_dir) / f"single_{codec}.parquet"
        pq.write_table(table, path, compression=codec)
        sizes[codec] = path.stat().st_size
    return sizes


def timed_query(dataset_dir: str, filter_expr, columns: Optional[List[str]] = None) -> Tuple[int, float]:
    """Read with a partition filter; returns (rows, seconds). Pruning skips whole directories."""
    start = time.perf_counter()
    dataset = ds.dataset(dataset_dir, format="parquet", partitioning="hive")
    table = dataset.to_table(filter=filter_expr, columns=columns)
    return table.num_rows, time.perf_counter() - start


def make_sample_events(n: int = 200_000, seed: int = 7) -> pd.DataFrame:
    """Fake clickstream: 6 months of events, realistic-ish skew."""
    import numpy as np
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01")
    days = rng.integers(0, 181, n)
    seconds = rng.integers(0, 86_400, n)
    return pd.DataFrame({
        "event_time": start + pd.to_timedelta(days, unit="D") + pd.to_timedelta(seconds, unit="s"),
        "user_id": rng.integers(1, 20_000, n),
        "event_type": rng.choice(["view", "click", "add_to_cart", "purchase"], n, p=[0.6, 0.25, 0.1, 0.05]),
        "amount": (rng.random(n) * 200).round(2),
        "country": rng.choice(["US", "DE", "JP", "BR", "IN"], n, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
    })
