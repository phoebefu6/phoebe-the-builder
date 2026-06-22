from __future__ import annotations

"""CLI: find (and optionally delete) duplicate files.

    python main.py /path/to/folder                  # report duplicates
    python main.py /path/to/folder --min-size 1024  # ignore files under 1KB
    python main.py /path/to/folder --json            # machine-readable output
    python main.py /path/to/folder --delete          # delete extras (keeps 1 per group)
    python main.py --demo                            # build a sample tree and scan it

Deletion is opt-in and always keeps the first file in each group. Without --delete
it is a pure report (dry run).
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from finder import find_duplicates, human_bytes, summarize


def _make_demo() -> str:
    root = Path(tempfile.mkdtemp(prefix="dupdemo_"))
    (root / "a").mkdir()
    (root / "b").mkdir()
    # Two identical "report" files + a third copy in another folder.
    content = b"quarterly report data " * 500
    (root / "report.csv").write_bytes(content)
    (root / "a" / "report_copy.csv").write_bytes(content)
    (root / "b" / "report_final.csv").write_bytes(content)
    # A separate duplicate pair.
    logo = b"PNGDATA" * 1000
    (root / "logo.png").write_bytes(logo)
    (root / "a" / "logo (1).png").write_bytes(logo)
    # A unique file (never a dup).
    (root / "unique.txt").write_bytes(b"one of a kind")
    return str(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find duplicate files by content.")
    parser.add_argument("path", nargs="?", help="Directory to scan.")
    parser.add_argument("--min-size", type=int, default=1, help="Ignore files smaller than this (bytes).")
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    parser.add_argument("--delete", action="store_true", help="Delete extras, keeping one per group.")
    parser.add_argument("--demo", action="store_true", help="Create a sample tree and scan it.")
    args = parser.parse_args(argv)

    if args.demo:
        args.path = _make_demo()

    if not args.path:
        parser.error("provide a directory to scan, or use --demo")

    try:
        groups = find_duplicates(args.path, min_size=args.min_size)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stats = summarize(groups)

    if args.json:
        print(json.dumps({
            "summary": stats,
            "groups": [{"hash": g.file_hash[:12], "size": g.size, "wasted": g.wasted,
                        "paths": g.paths} for g in groups],
        }, indent=2))
    else:
        print(f"Scanned: {args.path}")
        print(f"Duplicate groups: {stats['duplicate_groups']}  |  "
              f"redundant files: {stats['redundant_files']}  |  "
              f"reclaimable: {human_bytes(stats['wasted_bytes'])}")
        print("-" * 50)
        for g in groups:
            print(f"[{human_bytes(g.size)} each x {g.count}  -> save {human_bytes(g.wasted)}]")
            for i, p in enumerate(g.paths):
                tag = "keep" if i == 0 else "dup "
                print(f"  {tag} {p}")
            print()

    if args.delete:
        removed, freed = 0, 0
        for g in groups:
            for p in g.paths[1:]:  # keep the first, delete the rest
                try:
                    os.remove(p)
                    removed += 1
                    freed += g.size
                except OSError as exc:
                    print(f"  could not delete {p}: {exc}", file=sys.stderr)
        print(f"Deleted {removed} duplicate file(s), freed {human_bytes(freed)}.")
    elif groups:
        print("(dry run - rerun with --delete to remove extras, keeping one per group)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
