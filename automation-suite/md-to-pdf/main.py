from __future__ import annotations

"""CLI: turn a Markdown report into a branded PDF (or HTML fallback).

    python main.py report.md                      # -> report.pdf (or .html fallback)
    python main.py report.md -o weekly.pdf --title "Weekly Metrics" --author "Phoebe"
    python main.py report.md --html-only          # force styled HTML output
    python main.py --demo                          # generate + convert a sample report
"""

import argparse
import sys
from pathlib import Path

from report import convert

SAMPLE = """# Overview

This is an **auto-formatted** weekly report. Write Markdown, get a branded document -
no more hand-formatting in Word every week.

## Key metrics

| Metric | This week | Last week | Change |
|--------|-----------|-----------|--------|
| Active users | 12,480 | 11,900 | +4.9% |
| Revenue | $84.2k | $79.1k | +6.4% |
| Churn | 1.8% | 2.1% | -0.3pp |

## Highlights
- Onboarding flow shipped; activation up 4.9%.
- Two enterprise deals in late-stage.

## Risks
> Data pipeline latency crept up midweek - monitoring added.

## Next week
1. Ship the export feature.
2. Close the Q3 forecast.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a Markdown report to a branded PDF.")
    parser.add_argument("input", nargs="?", help="Markdown file to convert.")
    parser.add_argument("-o", "--output", help="Output path (.pdf or .html).")
    parser.add_argument("--title", help="Report title (default: from filename).")
    parser.add_argument("--subtitle", default="", help="Optional subtitle.")
    parser.add_argument("--author", default="", help="Optional author for the cover.")
    parser.add_argument("--html-only", action="store_true", help="Emit styled HTML instead of PDF.")
    parser.add_argument("--demo", action="store_true", help="Generate and convert a sample report.")
    args = parser.parse_args(argv)

    if args.demo:
        sample = Path("sample_report.md")
        sample.write_text(SAMPLE)
        args.input = str(sample)
        args.title = args.title or "Weekly Metrics Report"
        args.subtitle = args.subtitle or "Product & Growth"
        args.author = args.author or "Data Team"

    if not args.input:
        parser.error("provide a Markdown file, or use --demo")

    src = Path(args.input)
    if not src.exists():
        print(f"error: file not found: {src}", file=sys.stderr)
        return 1

    written, info = convert(
        str(src), args.output, title=args.title, subtitle=args.subtitle,
        author=args.author, html_only=args.html_only,
    )
    print(f"Wrote {written}  (engine: {info['engine']})")
    if info.get("fallback"):
        print("note: PDF engine unavailable, wrote styled HTML instead.", file=sys.stderr)
        print(f"      {info.get('reason', '')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
