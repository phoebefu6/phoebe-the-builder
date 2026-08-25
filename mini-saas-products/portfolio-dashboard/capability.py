"""What this catalog can cover for a team.

The old version of this tool was a burn-up chart: builds shipped, builds per
day, busiest day, capped at day 60.  It measured completion of a task list,
which is not a question anybody outside the project has.

The question people actually arrive with is one of three:

  * *Do you have something for this problem?*      -> by task
  * *What does this give my data engineer?*         -> by role
  * *Walk me through what you would use, in order.* -> by scenario

This module answers those from `one-data-platform/homepage/catalog.json`,
which `build_site.py` generates from the tracker.  It deliberately does not
re-parse the tracker: two scripts with their own regex over one file are two
sources of truth wearing a costume, and this tool used to be the second one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_URL = "https://github.com/phoebefu6/phoebe-the-builder"

CATALOG_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "one-data-platform" / "homepage" / "catalog.json",
    Path("catalog.json"),
)

#: Task order, and the question each one answers. Mirrors `TASKS` in
#: build_site.py, which owns the taxonomy; this is display order only.
TASK_ORDER: Tuple[str, ...] = (
    "ingest", "shape", "trust", "govern", "observe", "explore",
    "measure", "infer", "predict", "understand", "evaluate", "automate", "decide",
)

#: Which tasks each role spends its week inside. A tool can serve several
#: roles; a role that shares no task with another still shares tools.
ROLES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("Analyst", "Answers questions with data somebody else moved.",
     ("explore", "measure", "infer")),
    ("Analytics engineer", "Owns the layer between the warehouse and the dashboard.",
     ("shape", "trust", "measure", "govern")),
    ("Data engineer", "Owns the data arriving, correctly, on time.",
     ("ingest", "shape", "trust", "observe")),
    ("Data scientist", "Turns a question into a defensible estimate.",
     ("explore", "infer", "predict")),
    ("ML / AI engineer", "Puts a model or an LLM in front of real traffic.",
     ("predict", "understand", "evaluate")),
    ("Platform / governance", "Owns who may see what, and proving it.",
     ("govern", "observe", "automate", "decide")),
)

#: A situation, and the tasks it runs through in order. The point is that the
#: tools compose: a real incident crosses four of these in one afternoon.
SCENARIOS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("A new source lands",
     "Somebody hands you an export and a deadline.",
     ("ingest", "shape", "trust", "govern")),
    ("Two dashboards disagree",
     "Same metric, two numbers, and both owners are certain.",
     ("measure", "trust", "shape")),
    ("The nightly job failed",
     "It is 8am, the dashboard is empty, and nobody was paged.",
     ("observe", "trust", "ingest")),
    ("A model is going to production",
     "It scores well offline. That is not the same as working.",
     ("predict", "evaluate", "observe")),
    ("An auditor is coming",
     "Who touched what, under which policy, and can you show it.",
     ("govern", "trust", "observe")),
    ("Nobody can read the documents",
     "The answer is in a PDF nobody will open.",
     ("understand", "evaluate", "govern")),
    ("A decision has to be made",
     "The number is on the screen. Someone still has to choose.",
     ("measure", "infer", "decide")),
)


@dataclass(frozen=True)
class Tool:
    slug: str
    name: str
    task: str
    task_title: str
    problem: str
    repo_url: str
    colab_url: str
    product_slug: str


class CatalogMissing(RuntimeError):
    """catalog.json has not been generated yet."""


def catalog_path() -> Optional[Path]:
    return next((p for p in CATALOG_CANDIDATES if p.exists()), None)


@lru_cache(maxsize=1)
def load(path: Optional[str] = None) -> Tuple[Tool, ...]:
    """Every shipped tool, from the generated catalog."""
    p = Path(path) if path else catalog_path()
    if p is None or not p.exists():
        raise CatalogMissing(
            "catalog.json not found - run one-data-platform/homepage/build_site.py first"
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    tools = [
        Tool(b["slug"], b["name"], b.get("task") or "", b.get("task_title") or "",
             b.get("problem") or "", b.get("repo_url") or "", b.get("colab_url") or "",
             b.get("product_slug") or "")
        for b in raw
        if b.get("status") == "done"
    ]
    return tuple(tools)


def by_task(tools: Sequence[Tool] = ()) -> Dict[str, List[Tool]]:
    tools = tools or load()
    out: Dict[str, List[Tool]] = {t: [] for t in TASK_ORDER}
    for tool in tools:
        out.setdefault(tool.task, []).append(tool)
    for group in out.values():
        group.sort(key=lambda t: t.name.lower())
    return out


def task_titles(tools: Sequence[Tool] = ()) -> Dict[str, str]:
    tools = tools or load()
    return {t.task: t.task_title for t in tools if t.task}


def for_role(role: str, tools: Sequence[Tool] = ()) -> List[Tool]:
    tasks = next((r[2] for r in ROLES if r[0] == role), ())
    grouped = by_task(tools)
    out: List[Tool] = []
    for t in tasks:
        out += grouped.get(t, [])
    return out


def for_scenario(name: str, tools: Sequence[Tool] = ()) -> List[Tuple[str, List[Tool]]]:
    """Ordered steps, each with the tools that apply at that step."""
    tasks = next((s[2] for s in SCENARIOS if s[0] == name), ())
    grouped = by_task(tools)
    return [(t, grouped.get(t, [])) for t in tasks]


def role_overlap(tools: Sequence[Tool] = ()) -> Dict[Tuple[str, str], int]:
    """Tools two roles both reach for. Shows where the handoffs are."""
    tools = tools or load()
    sets = {r[0]: {t.slug for t in for_role(r[0], tools)} for r in ROLES}
    out: Dict[Tuple[str, str], int] = {}
    names = [r[0] for r in ROLES]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[(a, b)] = len(sets[a] & sets[b])
    return out


def haystack(tool: Tool) -> str:
    """Everything a query may legitimately match, hyphens opened up.

    The slug is the only place some vocabulary lives - nothing in
    `line-ending-detector`'s problem line contains the word "CRLF" - so the
    slug is indexed with its hyphens turned into spaces, which is what lets a
    two-word query like "line ending" hit it.
    """
    return " ".join((
        tool.problem, tool.name, tool.slug, tool.slug.replace("-", " "),
        tool.task_title, tool.product_slug.replace("-", " "),
    )).lower()


def search(query: str, tools: Sequence[Tool] = ()) -> List[Tool]:
    """Match on the problem text first - that is what people describe.

    Every whitespace-separated token must appear somewhere, so a longer query
    narrows instead of returning nothing. Substring matching, deliberately: no
    stemming, so a miss is a real gap in the catalog rather than a scoring
    artefact you cannot reason about.
    """
    tools = tools or load()
    tokens = query.lower().split()
    if not tokens:
        return list(tools)
    return [t for t in tools if all(tok in haystack(t) for tok in tokens)]


def unclassified(tools: Sequence[Tool] = ()) -> List[Tool]:
    """Tools with no task. Should always be empty; build_site.py enforces it."""
    return [t for t in (tools or load()) if not t.task]
