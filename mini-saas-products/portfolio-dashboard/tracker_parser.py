"""Tombstone. This module was removed on 2026-08-25.

It regex-parsed `TRACKER.md` to produce burn-up statistics - builds shipped,
builds per calendar day, busiest day - for the old version of this tool, which
was a progress dashboard.

Two problems with it:

  * The catalog is not a progress bar. The tool now answers what it covers for
    a team, not how much of a plan is done. See `capability.py`.
  * It was the *second* regex parser over `TRACKER.md`, alongside
    `one-data-platform/homepage/build_site.py`. Two readers of one file with
    their own parsers are two sources of truth wearing a costume, and they
    disagree eventually. `capability.py` reads the generated
    `one-data-platform/homepage/catalog.json` instead, so `build_site.py` is
    the only thing that parses the tracker.

The original is in git history: `git log --follow -- mini-saas-products/portfolio-dashboard/tracker_parser.py`
"""

from __future__ import annotations

RAISE_MESSAGE = (
    "tracker_parser was removed on 2026-08-25. Use capability.load(), which reads the "
    "generated one-data-platform/homepage/catalog.json. See this file's docstring."
)


def __getattr__(name: str):  # pragma: no cover - guidance path only
    raise AttributeError(f"{RAISE_MESSAGE} (requested: {name!r})")
