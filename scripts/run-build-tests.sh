#!/bin/bash
# Run the same thing CI runs, locally, without pushing.
#
#   DIRS=$'"'"'automation-suite/slug-collider\ndata-science-cookbook/interference-check'"'"' scripts/run-build-tests.sh
#   DIRS="$(git diff --name-only HEAD^ | awk -F/ '"'"'NF>=3 {print $1"/"$2}'"'"' | sort -u)" scripts/run-build-tests.sh
#
# This is the body of .github/workflows/builds.yml "Test them", driven by DIRS
# instead of the step output, so the two cannot drift without someone noticing.
# Verified 2026-09-06 against four cases: two passing builds (one pytest, one
# self-running) -> exit 0; a deliberately failing pytest assertion -> exit 1; a
# deliberately failing self-running script -> exit 1; MAX_BUILDS=1 over three
# builds -> names both skipped directories.  A runner that cannot report red is
# not a runner.
set -uo pipefail
MAX_BUILDS="${MAX_BUILDS:-12}"
ran=0; failed=0; skipped=0
total=$(echo "$DIRS" | grep -c . || true)
if [ "$total" -gt "$MAX_BUILDS" ]; then
  echo "WARN: this push touches $total build directories; testing the first $MAX_BUILDS only."
fi
while IFS= read -r d; do
  [ -n "$d" ] || continue
  [ -d "$d" ] || continue
  ls "$d"/test_*.py >/dev/null 2>&1 || continue
  if [ "$ran" -ge "$MAX_BUILDS" ]; then echo "SKIPPED BY CAP, NOT TESTED: $d"; skipped=$((skipped+1)); continue; fi
  echo "--- $d"
  ran=$((ran+1))
  for f in "$d"/test_*.py; do
    b="$(basename "$f")"
    if grep -qE '^[[:space:]]*def test_' "$f"; then
      ( cd "$d" && python3 -m pytest -q "$b" >/dev/null 2>&1 ) || { echo "  ERROR: $d/$b failed (pytest)"; failed=$((failed+1)); }
    else
      ( cd "$d" && python3 "$b" >/dev/null 2>&1 ) || { echo "  ERROR: $d/$b failed (self-running)"; failed=$((failed+1)); }
    fi
  done
done <<< "$DIRS"
echo "ran $ran build suite(s), $failed failed, $skipped skipped by the cap"
[ "$failed" -eq 0 ]
