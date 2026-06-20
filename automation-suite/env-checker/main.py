from __future__ import annotations

"""CLI: validate env vars against a schema before deploy.

    python main.py --schema .env.schema                  # check the live environment
    python main.py --schema .env.schema --env-file .env  # check a .env file instead
    python main.py --demo                                # run a bundled example

Exit code is 1 when any required var is missing or any value is invalid - so it
fails a CI step or a deploy script.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

from checker import check_env, is_passing, parse_dotenv, parse_schema

DEMO_SCHEMA = """# Deploy requirements
DATABASE_URL=required:^postgres://     # primary database
API_KEY=required                        # third-party API key
PORT=optional:^\\d+$                     # web port, digits only
DEBUG=optional:^(true|false)$           # feature flag
"""

DEMO_ENV = """DATABASE_URL=mysql://localhost/app
PORT=eighty-eighty
# API_KEY intentionally missing
DEBUG=true
"""


def _print_report(report: Dict) -> None:
    def line(items, mark):
        for it in items:
            hint = f"  ({it['hint']})" if it.get("hint") else ""
            reason = f" - {it['reason']}" if it.get("reason") else ""
            print(f"  {mark} {it['name']}{reason}{hint}")

    print("Environment check")
    print("-" * 40)
    if report["ok"]:
        print(f"OK ({len(report['ok'])}):")
        line(report["ok"], "✅")
    if report["optional_missing"]:
        print(f"Optional, not set ({len(report['optional_missing'])}):")
        line(report["optional_missing"], "•")
    if report["invalid"]:
        print(f"INVALID ({len(report['invalid'])}):")
        line(report["invalid"], "⚠")
    if report["missing"]:
        print(f"MISSING ({len(report['missing'])}):")
        line(report["missing"], "❌")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate env vars against a schema before deploy.")
    parser.add_argument("--schema", help="Path to the .env schema file.")
    parser.add_argument("--env-file", help="Check this .env file instead of the live environment.")
    parser.add_argument("--demo", action="store_true", help="Run a bundled example (writes sample files).")
    args = parser.parse_args(argv)

    if args.demo:
        Path(".env.schema").write_text(DEMO_SCHEMA)
        Path(".env.demo").write_text(DEMO_ENV)
        args.schema, args.env_file = ".env.schema", ".env.demo"

    if not args.schema:
        parser.error("provide --schema PATH, or use --demo")

    schema_path = Path(args.schema)
    if not schema_path.exists():
        print(f"error: schema not found: {schema_path}", file=sys.stderr)
        return 2

    specs = parse_schema(schema_path.read_text())

    if args.env_file:
        env_path = Path(args.env_file)
        if not env_path.exists():
            print(f"error: env file not found: {env_path}", file=sys.stderr)
            return 2
        env = parse_dotenv(env_path.read_text())
    else:
        env = dict(os.environ)

    report = check_env(specs, env)
    _print_report(report)
    print("-" * 40)

    if is_passing(report):
        print("PASS - environment looks good.")
        return 0
    print(f"FAIL - {len(report['missing'])} missing, {len(report['invalid'])} invalid.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
