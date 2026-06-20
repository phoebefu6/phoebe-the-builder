from __future__ import annotations

"""CLI: clean a messy CSV.

    python main.py messy.csv                 # -> messy.cleaned.csv
    python main.py messy.csv -o tidy.csv     # custom output
    python main.py --demo                    # write + clean a bundled messy sample
"""

import argparse
import sys
from pathlib import Path

from cleaner import clean_csv

SAMPLE = """Order ID , Customer Name ,Amount,Amount,Status,Notes
1001, Alice  ,12.50,12.50,Paid,
1002,Bob,N/A,N/A,paid ,
1002,Bob,N/A,N/A,paid ,
,,,,,
1003, Carol ,99.00,99.00, Refunded ,-
1004,Dan ,  7.25 ,7.25,PENDING,#N/A
"""


def _write_sample(path: Path) -> None:
    path.write_text(SAMPLE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean a messy CSV export.")
    parser.add_argument("input", nargs="?", help="Path to the input CSV.")
    parser.add_argument("-o", "--output", help="Output path (default: <input>.cleaned.csv).")
    parser.add_argument("--demo", action="store_true", help="Generate and clean a bundled messy sample.")
    args = parser.parse_args(argv)

    if args.demo:
        sample = Path("sample_messy.csv")
        _write_sample(sample)
        args.input = str(sample)

    if not args.input:
        parser.error("provide an input CSV path, or use --demo")

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"error: file not found: {in_path}", file=sys.stderr)
        return 1

    out_path = Path(args.output) if args.output else in_path.with_suffix(".cleaned.csv")

    try:
        report = clean_csv(str(in_path), str(out_path))
    except Exception as exc:  # noqa: BLE001 - surface any parse failure cleanly to the CLI user
        print(f"error: could not clean {in_path}: {exc}", file=sys.stderr)
        return 1

    print(f"Cleaned {in_path}  ->  {out_path}")
    print("-" * 40)
    print(f"  rows: {report['rows_in']} -> {report['rows_out']}")
    print(f"  empty rows dropped:     {report['empty_rows_dropped']}")
    print(f"  empty cols dropped:     {report['empty_cols_dropped']}")
    print(f"  duplicate rows dropped: {report['duplicate_rows_dropped']}")
    print(f"  numeric cols coerced:   {report['columns_coerced_numeric']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
