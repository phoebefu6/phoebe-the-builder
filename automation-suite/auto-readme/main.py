from __future__ import annotations

"""CLI: scaffold a README for any repo.

    python main.py .                      # scan cwd, write README to stdout
    python main.py /path/to/repo -o README.md
    python main.py . --llm                # use Claude if ANTHROPIC_API_KEY is set

Designed to run in CI (GitHub Actions): scan the checked-out repo and open a PR
with the generated README.
"""

import argparse
import sys
from pathlib import Path

from generator import generate_readme_claude, render_readme_template
from scanner import scan_repo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-generate a README from a repo's contents.")
    parser.add_argument("path", nargs="?", default=".", help="Repo path to scan (default: current dir).")
    parser.add_argument("-o", "--output", help="Write to this file (default: print to stdout).")
    parser.add_argument("--llm", action="store_true", help="Use Claude (needs ANTHROPIC_API_KEY); falls back to template.")
    args = parser.parse_args(argv)

    try:
        profile = scan_repo(args.path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    readme = generate_readme_claude(profile) if args.llm else render_readme_template(profile)

    if args.output:
        Path(args.output).write_text(readme)
        print(f"Wrote {len(readme)} chars to {args.output}")
    else:
        print(readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
