# CSV/JSON to Parquet Partitioner

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-engineering-pro/parquet-partitioner/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-engineering-pro/parquet-partitioner/demo.ipynb)

> Our data lake is a pile of CSVs — convert flat files into hive-partitioned, compressed parquet and prove the query speedup with a stopwatch.

## Business Impact
- **Before:** Every query parses whole CSVs; storage is untyped and uncompressed; "one month of data" means scanning all of it.
- **After:** Typed, compressed, directory-partitioned datasets that Spark/DuckDB/Athena/Trino prune natively — one-month queries touch one partition.
- **Estimated ROI:** ~2x+ storage reduction and order-of-magnitude query pruning on time-filtered workloads.

## Tech Stack
Python 3.10+, pyarrow (datasets + parquet), pandas, Streamlit, matplotlib. Hive-style partition layout. Runs fully offline.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Partition inference** — first date-like column becomes derived `year`/`month` partition columns; else the first low-cardinality string column; override with `partition_cols=`.
2. **Hive layout** — `pq.write_to_dataset` produces `col__year=2026/col__month=3/part-0.parquet` directories, the layout every lake engine prunes.
3. **Codec comparison** — same table written with snappy/gzip/zstd, sizes reported against the CSV baseline.
4. **Pruning proof** — `timed_query` filters on a partition column and is timed against a full scan; the demo reads 1 of 6 months ~an order of magnitude faster.

Sample: 200k clickstream events → 6 monthly partitions, ~2x smaller than CSV, one-month query in single-digit milliseconds.

## Learning Connection
Built while studying data lake layout and incremental patterns (Month 7: Data Engineering Pro).
Applies: pyarrow dataset API, hive partitioning, compression trade-offs, and benchmark-driven engineering claims.

## Impact Note
- **Who benefits:** Data engineers migrating file dumps into a queryable lake; analysts whose ad-hoc queries stop scanning everything.
- **Potential risks:** Over-partitioning (high-cardinality keys) creates thousands of tiny files and makes performance worse — the inference caps cardinality for this reason.
