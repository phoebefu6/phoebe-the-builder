from __future__ import annotations

"""Generate the One Data Platform homepage from the portfolio tracker.

Single source of truth = ../../TRACKER.md (the 60-build checklist). This script
parses it, figures out each build's product line + GitHub/Colab links, and writes:

    homepage/catalog.json   machine-readable list of every build
    homepage/index.html     the homepage (all completed builds as cards + roadmap)

Because it reads the tracker, the homepage stays correct automatically: every time a
day is marked [x], re-running this adds exactly one card. Run it at the end of each
daily build (the daily-build skill calls it), or by hand:

    python homepage/build_catalog.py
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO = "phoebefu6/phoebe-the-builder"
HERE = Path(__file__).resolve().parent
TRACKER = HERE.parent.parent / "TRACKER.md"

# Day ranges -> (folder slug, display name) for each product line.
PRODUCT_LINES = [
    (1, 10, "data-infra-toolkit", "Data Infrastructure Toolkit"),
    (11, 20, "automation-suite", "Business Automation Suite"),
    (21, 30, "analytics-accelerator", "Analytics Accelerator"),
    (31, 40, "document-intelligence", "Document Intelligence"),
    (41, 50, "ai-agent-workshop", "AI Agent Workshop"),
    (51, 60, "mini-saas-products", "Mini SaaS Products"),
]

# Matches:  - [x] Day 1 — csv-loader: CSV to PostgreSQL Loader (2026-06-07)
#           - [ ] Day 18 — md-to-pdf: Markdown to PDF Report
LINE_RE = re.compile(
    r"^- \[(?P<done>[ x])\]\s*Day\s*(?P<day>\d+)\s*[—-]\s*(?P<slug>[\w-]+):\s*"
    r"(?P<name>.+?)\s*(?:\((?P<date>\d{4}-\d{2}-\d{2})\))?\s*$"
)


def product_line(day: int):
    for lo, hi, slug, name in PRODUCT_LINES:
        if lo <= day <= hi:
            return slug, name
    return "misc", "Other"


def parse_tracker() -> List[Dict[str, object]]:
    if not TRACKER.exists():
        raise FileNotFoundError(f"tracker not found: {TRACKER}")
    builds: List[Dict[str, object]] = []
    for line in TRACKER.read_text().splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        day = int(m.group("day"))
        pl_slug, pl_name = product_line(day)
        done = m.group("done") == "x"
        slug = m.group("slug")
        builds.append(
            {
                "day": day,
                "slug": slug,
                "name": m.group("name").strip(),
                "status": "done" if done else "planned",
                "date": m.group("date"),
                "product_line": pl_name,
                "product_slug": pl_slug,
                "repo_url": f"https://github.com/{REPO}/tree/main/{pl_slug}/{slug}" if done else None,
                "colab_url": (
                    f"https://colab.research.google.com/github/{REPO}/blob/main/{pl_slug}/{slug}/demo.ipynb"
                    if done else None
                ),
            }
        )
    return builds


def write_catalog(builds: List[Dict[str, object]]) -> None:
    (HERE / "catalog.json").write_text(json.dumps(builds, indent=2))


# ── HTML rendering ─────────────────────────────────────────────────────────────

def _card(b: Dict[str, object]) -> str:
    links = ""
    if b["status"] == "done":
        links = (
            f"<div class='links'>"
            f"<a href='{b['repo_url']}' target='_blank' rel='noopener'>code →</a>"
            f"<a href='{b['colab_url']}' target='_blank' rel='noopener'>notebook →</a>"
            f"</div>"
        )
    cls = "card done" if b["status"] == "done" else "card planned"
    badge = f"Day {b['day']}"
    return (
        f"<div class='{cls}'>"
        f"<div class='top'><span class='day'>{badge}</span>"
        f"<span class='dot'></span></div>"
        f"<h3>{b['name']}</h3>"
        f"<code>{b['slug']}</code>"
        f"{links}"
        f"</div>"
    )


def render_html(builds: List[Dict[str, object]]) -> str:
    done = [b for b in builds if b["status"] == "done"]
    planned = [b for b in builds if b["status"] == "planned"]
    total = len(builds)
    n_done = len(done)
    pct = round(100 * n_done / total) if total else 0

    # Group done builds by product line, in plan order, newest day first within.
    sections = []
    for _, _, pl_slug, pl_name in PRODUCT_LINES:
        items = sorted([b for b in done if b["product_slug"] == pl_slug], key=lambda x: x["day"])
        if not items:
            continue
        cards = "".join(_card(b) for b in items)
        sections.append(f"<section><h2>{pl_name} <span class='count'>{len(items)}</span></h2>"
                        f"<div class='grid'>{cards}</div></section>")

    # Roadmap = next few planned.
    upcoming = sorted(planned, key=lambda x: x["day"])[:6]
    roadmap = "".join(
        f"<li><span class='day'>Day {b['day']}</span> {b['name']} <code>{b['slug']}</code></li>"
        for b in upcoming
    )

    updated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>One Data Platform</title>
<style>
:root{{--ink:#13131f;--muted:#6b6b80;--accent:#3d34d6;--accent2:#6c63ff;--line:#e7e7f0;--bg:#fbfbfe;--done:#2e9e5b;}}
*{{box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;color:var(--ink);background:var(--bg);line-height:1.5}}
.wrap{{max-width:1040px;margin:0 auto;padding:0 1.2rem}}
header.hero{{padding:3.2rem 0 2rem;border-bottom:1px solid var(--line)}}
.kicker{{color:var(--accent);font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:.78rem}}
h1{{font-size:2.7rem;margin:.4rem 0 .6rem;letter-spacing:-.02em}}
.tag{{color:var(--muted);font-size:1.05rem;max-width:640px}}
.stats{{display:flex;gap:1.6rem;margin-top:1.6rem;flex-wrap:wrap}}
.stat{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:.9rem 1.2rem;min-width:120px}}
.stat b{{display:block;font-size:1.8rem;color:var(--accent)}}
.stat span{{font-size:.8rem;color:var(--muted)}}
.bar{{height:10px;background:#ececf6;border-radius:999px;margin-top:1.4rem;overflow:hidden}}
.bar>i{{display:block;height:100%;width:{pct}%;background:linear-gradient(90deg,var(--accent),var(--accent2))}}
.nav{{display:flex;gap:1rem;margin-top:1.6rem;flex-wrap:wrap}}
.nav a{{color:var(--accent);text-decoration:none;font-weight:600;font-size:.92rem;border:1px solid var(--line);padding:.45rem .9rem;border-radius:8px;background:#fff}}
.nav a:hover{{border-color:var(--accent)}}
section{{padding:2rem 0 .5rem}}
h2{{font-size:1.25rem;margin:0 0 1rem;display:flex;align-items:center;gap:.6rem}}
h2 .count{{font-size:.8rem;background:#eeedfb;color:var(--accent);border-radius:999px;padding:.1rem .6rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:1rem}}
.card{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1.1rem;transition:.15s}}
.card:hover{{transform:translateY(-2px);box-shadow:0 8px 24px rgba(60,52,214,.08)}}
.card .top{{display:flex;justify-content:space-between;align-items:center}}
.card .day{{font-size:.72rem;font-weight:700;color:var(--muted);letter-spacing:.04em}}
.card .dot{{width:9px;height:9px;border-radius:50%;background:var(--done)}}
.card h3{{margin:.5rem 0 .3rem;font-size:1.02rem}}
.card code{{font-size:.76rem;color:var(--accent);background:#f3f2fc;padding:.1rem .4rem;border-radius:5px}}
.card .links{{margin-top:.8rem;display:flex;gap:.9rem}}
.card .links a{{font-size:.82rem;color:var(--accent);text-decoration:none;font-weight:600}}
.roadmap{{background:#fff;border:1px dashed var(--line);border-radius:14px;padding:1.2rem 1.4rem;margin-top:1rem}}
.roadmap ul{{list-style:none;padding:0;margin:.4rem 0 0}}
.roadmap li{{padding:.4rem 0;border-bottom:1px solid #f2f2f8;font-size:.92rem;color:var(--muted)}}
.roadmap .day{{color:var(--accent);font-weight:700;margin-right:.4rem}}
.roadmap code{{font-size:.76rem;color:var(--accent)}}
footer{{color:var(--muted);font-size:.82rem;padding:2.4rem 0;border-top:1px solid var(--line);margin-top:2rem}}
</style></head>
<body><div class="wrap">
<header class="hero">
  <div class="kicker">One Data Platform</div>
  <h1>One governed home for the whole data team.</h1>
  <p class="tag">Connect to any source, process, explore, and build &amp; share dashboards,
  models, and AI products - with access control over all of it. Built one day at a time.</p>
  <div class="stats">
    <div class="stat"><b>{n_done}</b><span>builds shipped</span></div>
    <div class="stat"><b>{total}</b><span>planned total</span></div>
    <div class="stat"><b>{pct}%</b><span>complete</span></div>
  </div>
  <div class="bar"><i></i></div>
  <nav class="nav">
    <a href="../docs/01-why-a-platform.md">Why a platform</a>
    <a href="../docs/02-architecture.md">Architecture</a>
    <a href="../docs/00-glossary.md">Glossary</a>
    <a href="../docs/03-build-log.md">Build log</a>
    <a href="../gateway/app.py">The shell (gateway)</a>
  </nav>
</header>
<main>
{''.join(sections)}
<section>
  <h2>Roadmap <span class="count">next</span></h2>
  <div class="roadmap"><ul>{roadmap}</ul></div>
</section>
</main>
<footer>One Data Platform - the governed shell + app catalog. Homepage auto-generated
from TRACKER.md. Last updated {updated}. A new build is added every day.</footer>
</div></body></html>"""


def main() -> None:
    builds = parse_tracker()
    write_catalog(builds)
    (HERE / "index.html").write_text(render_html(builds))
    n_done = sum(1 for b in builds if b["status"] == "done")
    print(f"Catalog built: {n_done} shipped / {len(builds)} total -> homepage/index.html")


if __name__ == "__main__":
    main()
