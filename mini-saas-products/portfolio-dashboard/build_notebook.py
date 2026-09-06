"""Generate demo.ipynb.

The notebook carries a snapshot of the catalog so it runs on Colab and Binder
with nothing checked out, and prefers the real generated file when one is
present. The previous version of this notebook computed burn-up statistics from
a regex over TRACKER.md; both the parser and the framing are gone.

    python build_notebook.py  ->  demo.ipynb (unexecuted)
"""

from __future__ import annotations

import json

import capability as C
import nbformat as nbf

REPO = "phoebefu6/phoebe-the-builder"
PATH = "mini-saas-products/portfolio-dashboard"


def md(t: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(t.strip("\n"))


def code(t: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(t.strip("\n"))


def snapshot() -> str:
    tools = C.load()
    rows = [
        {"slug": t.slug, "name": t.name, "task": t.task, "task_title": t.task_title,
         "problem": t.problem, "repo_url": t.repo_url}
        for t in tools
    ]
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = []

    c.append(md(f"""
# What this covers for a team

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A catalog is only useful if somebody arriving with a problem can find the thing
that solves it. Three ways in:

| | question |
|---|---|
| **by job** | *Do you have something for this?* |
| **by role** | *What does this give my data engineer?* |
| **by situation** | *Walk me through what you would use, in order.* |

Sourced from `one-data-platform/homepage/catalog.json`, which is generated from
the tracker. This notebook does not re-parse the tracker: two readers of one
file with their own parsers are two sources of truth wearing a costume, and this
tool used to be the second one.
"""))

    c.append(md("""
## The catalog

A snapshot is embedded so this runs anywhere. If you have the repo checked out it
loads the live file instead and tells you which one you got.
"""))
    c.append(code(f'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

SNAPSHOT = json.loads(r"""{snapshot()}""")

live = None
for p in (Path("../../one-data-platform/homepage/catalog.json"),
          Path("one-data-platform/homepage/catalog.json")):
    if p.exists():
        live = [b for b in json.loads(p.read_text()) if b.get("status") == "done"]
        break

TOOLS = live or SNAPSHOT
print(f"{{len(TOOLS)}} tools loaded from "
      f"{{'the live catalog' if live else 'the embedded snapshot'}}.")
if live and len(live) != len(SNAPSHOT):
    print(f"(The snapshot in this notebook holds {{len(SNAPSHOT)}}; the repo has moved on.)")
'''))

    c.append(md("""
## 1. By the job you arrived with

Thirteen jobs. The technology a tool happens to use is rarely how anyone looks
for it.
"""))
    c.append(code(f'''
TASK_ORDER = {list(C.TASK_ORDER)!r}

grouped: Dict[str, List[dict]] = {{t: [] for t in TASK_ORDER}}
for t in TOOLS:
    grouped.setdefault(t["task"], []).append(t)
for g in grouped.values():
    g.sort(key=lambda x: x["name"].lower())

TITLES = {{t["task"]: t["task_title"] for t in TOOLS if t.get("task")}}

for task in TASK_ORDER:
    print(f"\\n{{TITLES.get(task, task).upper()}}  ({{len(grouped[task])}})")
    for t in grouped[task][:4]:
        print(f"    {{t['name'][:44]:<46}} {{t['problem'][:72]}}")
    if len(grouped[task]) > 4:
        print(f"    ... and {{len(grouped[task]) - 4}} more")
'''))

    c.append(md("""
## 2. By role

Which of the thirteen jobs a role spends its week inside - and where two roles
reach for the same tool, which is where the ownership argument happens.
"""))
    c.append(code(f'''
ROLES = {[(r[0], r[2]) for r in C.ROLES]!r}


def for_role(role: str) -> List[dict]:
    tasks = next(t for r, t in ROLES if r == role)
    return [x for task in tasks for x in grouped.get(task, [])]


print(f"{{'role':<24}}{{'tools':>7}}   jobs")
print("-" * 78)
for role, tasks in ROLES:
    print(f"{{role:<24}}{{len(for_role(role)):>7}}   "
          f"{{', '.join(TITLES.get(t, t) for t in tasks)}}")

names = [r for r, _t in ROLES]
sets = {{r: {{x['slug'] for x in for_role(r)}} for r in names}}
print(f"\\n{{'':<24}}" + "".join(f"{{n[:9]:>11}}" for n in names))
for a in names:
    row = "".join(f"{{(len(sets[a] & sets[b]) if a != b else len(sets[a])):>11}}" for b in names)
    print(f"{{a:<24}}{{row}}")
print("\\nDiagonal is the role's own tools; off-diagonal is what two roles share.")
print("A zero means the two roles touch none of the same jobs - that is a clean handoff,")
print("and a large number is a shared surface somebody has to own.")
'''))

    c.append(md("""
## 3. By situation

A real afternoon crosses three or four jobs. This is the order they arrive in.
"""))
    c.append(code(f'''
SCENARIOS = {[(s[0], s[1], s[2]) for s in C.SCENARIOS]!r}

for name, blurb, tasks in SCENARIOS:
    print(f"\\n{{name.upper()}}")
    print(f"  {{blurb}}")
    for i, task in enumerate(tasks, 1):
        picks = grouped.get(task, [])[:3]
        print(f"  {{i}}. {{TITLES.get(task, task)}}")
        for p in picks:
            print(f"       {{p['name'][:60]}}")
'''))

    c.append(md("""
## The picture
"""))
    c.append(code('''
import matplotlib.pyplot as plt
import numpy as np

INK, GRIDC, PAPER, COOL = "#1d1a17", "#e3ddd5", "#faf7f2", "#2f6f8f"
plt.rcParams.update({"figure.facecolor": PAPER, "axes.facecolor": PAPER,
                     "text.color": INK, "font.size": 9})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6),
                               gridspec_kw={"width_ratios": [1.05, 1]})

order = sorted(TASK_ORDER, key=lambda t: len(grouped[t]))
y = np.arange(len(order))
ax1.barh(y, [len(grouped[t]) for t in order], color=COOL, height=0.7)
ax1.set_yticks(y)
ax1.set_yticklabels([TITLES.get(t, t) for t in order])
ax1.set_xlabel("tools behind this job")
ax1.set_title("How deep each job goes", loc="left", fontweight="bold")
for s in ("top", "right", "left"):
    ax1.spines[s].set_visible(False)
ax1.tick_params(length=0)

grid = np.array([[len(sets[a] & sets[b]) if a != b else len(sets[a])
                  for b in names] for a in names], dtype=float)
im = ax2.imshow(grid, cmap="YlGnBu", aspect="auto")
for i in range(len(names)):
    for j in range(len(names)):
        v = int(grid[i, j])
        ax2.text(j, i, str(v), ha="center", va="center", fontsize=8,
                 color="white" if v > grid.max() * 0.55 else INK,
                 fontweight="bold" if i == j else "normal")
ax2.set_xticks(range(len(names)))
ax2.set_yticks(range(len(names)))
ax2.set_xticklabels(names, rotation=32, ha="right", fontsize=8)
ax2.set_yticklabels(names, fontsize=8)
ax2.set_title("Where the handoffs are", loc="left", fontweight="bold")
for s in ("top", "right", "left", "bottom"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(length=0)
plt.colorbar(im, ax=ax2, fraction=0.045, pad=0.02).outline.set_visible(False)

fig.tight_layout()
fig.savefig("notebook_capability.png", dpi=140, facecolor=PAPER)
plt.show()
'''))

    c.append(md("""
## Try your own

Search on the **symptom**, not the technology. The catalog is indexed by what was
going wrong.
"""))
    c.append(code('''
def haystack(t: dict) -> str:
    """Hyphens opened up, so a two-word query can hit a slug."""
    return " ".join((t["problem"], t["name"], t["slug"],
                     t["slug"].replace("-", " "), t["task_title"])).lower()


def search(q: str) -> List[dict]:
    """Every token must appear, so a longer query narrows instead of failing."""
    tokens = q.lower().split()
    if not tokens:
        return list(TOOLS)
    return [t for t in TOOLS if all(tok in haystack(t) for tok in tokens)]


for q in ("stale", "line ending", "collation", "cost", "CRLF"):
    hits = search(q)
    jobs = sorted({t["task_title"] for t in hits})
    print(f"\\n{q!r} -> {len(hits)} tools across {len(jobs)} jobs")
    for t in hits[:4]:
        print(f"    [{t['task_title']}] {t['name'][:52]}")
    if not hits:
        print("    nothing. Matching is plain substring over the problem text, the name,")
        print("    the slug and the job - no stemming - so a miss is a real gap in what the")
        print("    catalog SAYS, not a scoring artefact. 'CRLF' is a case in point: the tool")
        print("    exists (try 'line ending') and its own description never uses the word.")

# MY_SYMPTOM = "the join tripled the revenue"
# for t in search(MY_SYMPTOM):
#     print(f"[{t['task_title']}] {t['name']}\\n    {t['problem']}\\n")
'''))

    c.append(md(f"""
## What changed here

This tool used to be a **portfolio dashboard**: builds shipped, builds per calendar
day, busiest day, a burn-up against a one-a-day pace line, capped at day 60. It
measured completion of a task list, which is not a question anybody outside the
project has - and it had gone stale at 60 while the catalog passed 150.

It now answers what the catalog covers. Two structural changes came with that:

* **It reads the generated catalog, not the tracker.** It used to run its own
  regex over `TRACKER.md` alongside `build_site.py` - two parsers over one file,
  which disagree eventually. `tracker_parser.py` is now a tombstone pointing here.
* **It gained tests.** The taxonomy is a contract now: if a job is renamed or
  added upstream, `test_capability.py` fails rather than a tab rendering empty.

---

**Interactive:** `streamlit run app.py` - search by symptom, then browse by job,
role or situation.

Part of [phoebe-the-builder](https://github.com/{REPO}).
"""))

    nb["cells"] = c
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


if __name__ == "__main__":
    nbf.write(build(), "demo.ipynb")
    print(f"wrote demo.ipynb ({len(build()['cells'])} cells)")
