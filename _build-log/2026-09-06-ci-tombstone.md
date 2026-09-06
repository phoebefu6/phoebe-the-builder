# Tombstone: the per-build CI workflows (removed 2026-09-06)

## What was there

Every build directory carried its own workflow file:

```
<product-line>/<slug>/.github/workflows/ci.yml
```

108 of them, one per build that had reached the point in the template where that
file gets written. Each ran `ruff check .`, that build's pytest suite, and its
`evidence.py`, path-filtered to its own directory.

## Why they are gone

**GitHub only reads workflows from the repository root**, `.github/workflows/`.
A workflow file nested inside a subdirectory is an ordinary text file. None of
those 108 workflows had ever run - not once, across 168 builds and roughly six
months. `gh workflow list` showed exactly two registered workflows the whole
time: the root `CI` and `pages-build-deployment`.

They were not broken. They were decorative, and they looked exactly like working
CI in every build's file listing, which is worse than having none.

## What replaced them

`.github/workflows/builds.yml` at the repository root. It diffs the push, maps
changed paths to `<product-line>/<slug>` directories, and runs the test suite of
each one that has `test_*.py` - installing that build's own `requirements.txt`
first. A normal daily push touches one build, so it runs one suite.

Lint is not in that job. It is in the root `CI` workflow, once, against the single
`./ruff.toml` - see the header comment in that file for why there used to be eight
lint configurations and now there is one.

## What else changed on the same day

- `./ruff.toml` created: one lint configuration for the whole repository.
- The seven per-build `ruff.toml` files deleted (they said the same thing the
  root now says, and the root CI's `--select E,F,I` command-line flag had been
  contradicting all of them at ruff's default 88 columns).
- An **eighth** config found afterwards: `analytics-accelerator/kpi-tracker/pyproject.toml`.
  The first sweep searched for files named `ruff.toml` and missed it, and it was
  only caught because the pushed CI disagreed with a clean local run (24 UP006
  findings a local `ruff check` could not see). Deleted, and `ci.yml` now fails
  loudly if any `pyproject.toml` / `ruff.toml` / `.ruff.toml` reappears outside
  the root.
- Ruff is now **pinned** in CI (`ruff==0.15.18`). An unpinned install gave CI a
  different rule set from the local one - the same class of local-vs-CI
  disagreement the `--select` flag used to cause.
- The root `CI` workflow stopped passing `--select` on the command line, so a
  local `ruff check` and CI can no longer disagree.
- 9,232 lint findings resolved to zero: ~8,500 by fixing the configuration to
  match how the code was actually written, ~380 by autofix, and 29 by hand
  (ambiguous `l` names, semicolon statements, two lambda assignments, and six
  dead locals - one of which was a 100-bootstrap distribution fit whose result
  was thrown away).

## If you want per-build CI back

Do not put a workflow inside the build directory. Add the build to the matrix in
`builds.yml`, or let the path-diff pick it up automatically, which it already does.
