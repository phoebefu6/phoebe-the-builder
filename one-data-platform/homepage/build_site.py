from __future__ import annotations

"""Generate the whole One Data Platform site from source.

Two outputs, one command:
  - homepage/index.html        the catalog homepage (cards from ../../TRACKER.md)
  - homepage/wiki/*.html       every wiki doc in docs/ rendered to styled HTML
  - homepage/wiki/decisions/*  ADRs rendered too

Markdown stays the editable source in docs/; this script renders the pretty HTML.
Run after each daily build (the daily-fde-build skill calls it):

    python homepage/build_site.py
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import markdown
import yaml

REPO = "phoebefu6/phoebe-the-builder"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # one-data-platform/
TRACKER = ROOT.parent / "TRACKER.md"
DOCS = ROOT / "docs"
WIKI_OUT = HERE / "wiki"
SHELL_FILE = ROOT / "shell.yaml"

PRODUCT_LINES = [
    (1, 10, "data-infra-toolkit", "Data Infrastructure Toolkit"),
    (11, 20, "automation-suite", "Business Automation Suite"),
    (21, 30, "analytics-accelerator", "Analytics Accelerator"),
    (31, 40, "document-intelligence", "Document Intelligence"),
    (41, 50, "ai-agent-workshop", "AI Agent Workshop"),
    (51, 60, "mini-saas-products", "Mini SaaS Products"),
]

# Wiki nav order: (source filename, short nav label)
WIKI_NAV = [
    ("01-why-a-platform.md", "Why"),
    ("02-architecture.md", "Architecture"),
    ("00-glossary.md", "Glossary"),
    ("10-gateway-login.md", "Gateway"),
    ("11-rbac-registry.md", "RBAC"),
    ("12-audit-log.md", "Audit"),
    ("13-connector-layer.md", "Connectors"),
    ("03-build-log.md", "Build log"),
]

LINE_RE = re.compile(
    r"^- \[(?P<done>[ x])\]\s*Day\s*(?P<day>\d+)\s*[—-]\s*(?P<slug>[\w-]+):\s*"
    r"(?P<name>.+?)\s*(?:\((?P<date>\d{4}-\d{2}-\d{2})\))?\s*$"
)

# ── Shared theme ───────────────────────────────────────────────────────────────
THEME = """
:root{--ink:#13131f;--muted:#6b6b80;--accent:#3d34d6;--accent2:#6c63ff;--line:#e7e7f0;
--bg:#fbfbfe;--surface:#fff;--code-bg:#f3f2fc;--done:#2e9e5b;}
*{box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;color:var(--ink);
background:var(--bg);line-height:1.6}
.wrap{max-width:1040px;margin:0 auto;padding:0 1.2rem}
.topbar{position:sticky;top:0;z-index:10;background:rgba(251,251,254,.92);backdrop-filter:blur(8px);
border-bottom:1px solid var(--line)}
.topbar .wrap{display:flex;align-items:center;gap:.4rem;flex-wrap:wrap;padding-top:.7rem;padding-bottom:.7rem}
.brand{font-weight:800;color:var(--accent);margin-right:.6rem;text-decoration:none;letter-spacing:-.01em}
.topbar a.nav{color:var(--muted);text-decoration:none;font-size:.86rem;font-weight:600;padding:.35rem .7rem;
border-radius:7px}
.topbar a.nav:hover{background:#eeedfb;color:var(--accent)}
.topbar a.nav.active{background:#eeedfb;color:var(--accent)}
"""

WIKI_CSS = THEME + """
.doc{padding:2.4rem 0 3rem;max-width:780px}
.doc h1{font-size:2.1rem;letter-spacing:-.02em;margin:.2rem 0 1rem;padding-bottom:.6rem;border-bottom:2px solid var(--line)}
.doc h2{font-size:1.4rem;margin:2rem 0 .8rem}
.doc h3{font-size:1.1rem;margin:1.5rem 0 .5rem}
.doc p,.doc li{font-size:1rem}
.doc a{color:var(--accent);text-decoration:none;border-bottom:1px solid #d7d4f7}
.doc a:hover{border-color:var(--accent)}
.doc code{background:var(--code-bg);color:var(--accent);padding:.12rem .4rem;border-radius:5px;
font-size:.86em;font-family:'SF Mono',Menlo,Consolas,monospace}
.doc pre{background:#1a1830;color:#e8e6ff;padding:1rem 1.2rem;border-radius:10px;overflow:auto;font-size:.84rem;
line-height:1.5}
.doc pre code{background:none;color:inherit;padding:0}
.doc table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem}
.doc th,.doc td{border:1px solid var(--line);padding:.55rem .75rem;text-align:left;vertical-align:top}
.doc th{background:#f3f2fc;color:var(--accent);font-weight:700}
.doc tr:nth-child(even) td{background:#fafaff}
.doc blockquote{margin:1.2rem 0;padding:.6rem 1.1rem;background:#f5f5ff;border-left:4px solid var(--accent2);
border-radius:0 8px 8px 0;color:#34324f}
.doc blockquote p{margin:.3rem 0}
.doc hr{border:0;border-top:1px solid var(--line);margin:2rem 0}
.doc ul,.doc ol{padding-left:1.3rem}
.doc li{margin:.3rem 0}
.crumbs{font-size:.82rem;color:var(--muted);padding-top:1.4rem}
.crumbs a{color:var(--accent);text-decoration:none}
footer{color:var(--muted);font-size:.82rem;padding:2rem 0;border-top:1px solid var(--line)}
"""

HOME_CSS = THEME + """
header.hero{padding:3rem 0 2rem;border-bottom:1px solid var(--line)}
.kicker{color:var(--accent);font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:.78rem}
h1{font-size:2.7rem;margin:.4rem 0 .6rem;letter-spacing:-.02em}
.tag{color:var(--muted);font-size:1.05rem;max-width:640px}
.stats{display:flex;gap:1.6rem;margin-top:1.6rem;flex-wrap:wrap}
.stat{background:#fff;border:1px solid var(--line);border-radius:12px;padding:.9rem 1.2rem;min-width:120px}
.stat b{display:block;font-size:1.8rem;color:var(--accent)}
.stat span{font-size:.8rem;color:var(--muted)}
.bar{height:10px;background:#ececf6;border-radius:999px;margin-top:1.4rem;overflow:hidden}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2))}
section{padding:2rem 0 .5rem}
h2{font-size:1.25rem;margin:0 0 1rem;display:flex;align-items:center;gap:.6rem}
h2 .count{font-size:.8rem;background:#eeedfb;color:var(--accent);border-radius:999px;padding:.1rem .6rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:1rem}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1.1rem;transition:.15s}
.card:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(60,52,214,.08)}
.card .top{display:flex;justify-content:space-between;align-items:center}
.card .day{font-size:.72rem;font-weight:700;color:var(--muted);letter-spacing:.04em}
.card .dot{width:9px;height:9px;border-radius:50%;background:var(--done)}
.card h3{margin:.5rem 0 .3rem;font-size:1.02rem}
.card code{font-size:.76rem;color:var(--accent);background:#f3f2fc;padding:.1rem .4rem;border-radius:5px}
.card .links{margin-top:.8rem;display:flex;gap:.9rem}
.card .links a{font-size:.82rem;color:var(--accent);text-decoration:none;font-weight:600}
.roadmap{background:#fff;border:1px dashed var(--line);border-radius:14px;padding:1.2rem 1.4rem;margin-top:1rem}
.roadmap ul{list-style:none;padding:0;margin:.4rem 0 0}
.roadmap li{padding:.4rem 0;border-bottom:1px solid #f2f2f8;font-size:.92rem;color:var(--muted)}
.roadmap .day{color:var(--accent);font-weight:700;margin-right:.4rem}
.roadmap code{font-size:.76rem;color:var(--accent)}
.shell-tag{color:var(--muted);font-size:.92rem;margin:.2rem 0 1rem}
.modgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:.8rem}
.mod{border-radius:12px;padding:.9rem 1rem;border:1px solid var(--line)}
.mod .mtop{display:flex;justify-content:space-between;align-items:center}
.mod .step{font-size:.72rem;font-weight:700;color:var(--muted);letter-spacing:.04em}
.mod .mtag{font-size:.72rem;font-weight:700;border-radius:999px;padding:.1rem .55rem}
.mod h3{margin:.45rem 0 .25rem;font-size:1rem}
.mod h3 a{color:inherit;text-decoration:none;border-bottom:1px solid #cdd9ff}
.mod .concept{font-size:.82rem;color:#555}
.mod.done{background:#eef6f0;border-color:#cbe8d6}
.mod.done .mtag{background:#d6f0df;color:#1c7a44}
.mod.planned{background:#fafafa;opacity:.85}
.mod.planned .mtag{background:#ececec;color:#888}
footer{color:var(--muted);font-size:.82rem;padding:2.4rem 0;border-top:1px solid var(--line);margin-top:2rem}
"""


def topnav(active: str, prefix: str) -> str:
    """Shared top bar. `prefix` is the path back to homepage/ from the current page."""
    links = [f'<a class="brand" href="{prefix}index.html">One Data Platform</a>',
             f'<a class="nav{" active" if active=="home" else ""}" href="{prefix}index.html">Home</a>']
    for fname, label in WIKI_NAV:
        html_name = fname.replace(".md", ".html")
        cls = " active" if active == fname else ""
        links.append(f'<a class="nav{cls}" href="{prefix}wiki/{html_name}">{label}</a>')
    return f'<div class="topbar"><div class="wrap">{"".join(links)}</div></div>'


# ── Tracker -> catalog ─────────────────────────────────────────────────────────
def product_line(day: int):
    for lo, hi, slug, name in PRODUCT_LINES:
        if lo <= day <= hi:
            return slug, name
    return "misc", "Other"


def parse_tracker() -> List[Dict[str, object]]:
    builds: List[Dict[str, object]] = []
    for line in TRACKER.read_text().splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        day = int(m.group("day"))
        pl_slug, pl_name = product_line(day)
        done = m.group("done") == "x"
        slug = m.group("slug")
        builds.append({
            "day": day, "slug": slug, "name": m.group("name").strip(),
            "status": "done" if done else "planned", "date": m.group("date"),
            "product_line": pl_name, "product_slug": pl_slug,
            "repo_url": f"https://github.com/{REPO}/tree/main/{pl_slug}/{slug}" if done else None,
            "colab_url": (f"https://colab.research.google.com/github/{REPO}/blob/main/{pl_slug}/{slug}/demo.ipynb"
                          if done else None),
        })
    return builds


def _card(b: Dict[str, object]) -> str:
    links = ""
    if b["status"] == "done":
        links = (f"<div class='links'><a href='{b['repo_url']}' target='_blank' rel='noopener'>code →</a>"
                 f"<a href='{b['colab_url']}' target='_blank' rel='noopener'>notebook →</a></div>")
    return (f"<div class='card done'><div class='top'><span class='day'>Day {b['day']}</span>"
            f"<span class='dot'></span></div><h3>{b['name']}</h3><code>{b['slug']}</code>{links}</div>")


def render_shell_section() -> str:
    """The platform's own modules (the governance spine) from shell.yaml."""
    if not SHELL_FILE.exists():
        return ""
    data = yaml.safe_load(SHELL_FILE.read_text()) or {}
    modules = data.get("modules", [])
    if not modules:
        return ""
    n_done = sum(1 for m in modules if m.get("status") == "done")
    cards = []
    for m in modules:
        live = m.get("status") == "done"
        cls = "mod done" if live else "mod planned"
        tag = "✅ built" if live else "planned"
        doc = m.get("doc") or ""
        name = (f"<a href='{doc}'>{m['name']}</a>" if (live and doc) else m["name"])
        cards.append(
            f"<div class='{cls}'><div class='mtop'><span class='step'>Step {m['step']}</span>"
            f"<span class='mtag'>{tag}</span></div>"
            f"<h3>{name}</h3><span class='concept'>{m['concept']}</span></div>"
        )
    return (f"<section><h2>Platform shell <span class='count'>{n_done}/{len(modules)} built</span></h2>"
            f"<p class='shell-tag'>The governance spine we build ourselves - the control plane every "
            f"app plugs into.</p><div class='modgrid'>{''.join(cards)}</div></section>")


def render_home(builds: List[Dict[str, object]]) -> str:
    done = [b for b in builds if b["status"] == "done"]
    planned = [b for b in builds if b["status"] == "planned"]
    total, n_done = len(builds), len(done)
    pct = round(100 * n_done / total) if total else 0
    shell_section = render_shell_section()
    sections = []
    for _, _, pl_slug, pl_name in PRODUCT_LINES:
        items = sorted([b for b in done if b["product_slug"] == pl_slug], key=lambda x: x["day"])
        if not items:
            continue
        sections.append(f"<section><h2>{pl_name} <span class='count'>{len(items)}</span></h2>"
                        f"<div class='grid'>{''.join(_card(b) for b in items)}</div></section>")
    upcoming = sorted(planned, key=lambda x: x["day"])[:6]
    roadmap = "".join(f"<li><span class='day'>Day {b['day']}</span> {b['name']} <code>{b['slug']}</code></li>"
                      for b in upcoming)
    updated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex">
<title>One Data Platform</title><style>{HOME_CSS}</style></head><body>
{topnav("home", "")}
<div class="wrap"><header class="hero">
<div class="kicker">One Data Platform</div>
<h1>One governed home for the whole data team.</h1>
<p class="tag">Connect to any source, process, explore, and build &amp; share dashboards,
models, and AI products - with access control over all of it. Built one day at a time.</p>
<div class="stats"><div class="stat"><b>{n_done}</b><span>builds shipped</span></div>
<div class="stat"><b>{total}</b><span>planned total</span></div>
<div class="stat"><b>{pct}%</b><span>complete</span></div></div>
<div class="bar"><i style="width:{pct}%"></i></div></header>
<main>{shell_section}
{''.join(sections)}
<section><h2>Roadmap <span class="count">next</span></h2><div class="roadmap"><ul>{roadmap}</ul></div></section>
</main>
<footer>One Data Platform - the governed shell + app catalog. Homepage auto-generated
from TRACKER.md. Last updated {updated}. A new build is added every day.</footer>
</div></body></html>"""


# ── Markdown -> styled HTML ────────────────────────────────────────────────────
def render_doc(md_path: Path, prefix: str) -> str:
    raw = md_path.read_text()
    # First H1 becomes the page title.
    title_m = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    title = title_m.group(1).strip() if title_m else md_path.stem

    html_body = markdown.markdown(raw, extensions=["tables", "fenced_code", "sane_lists", "toc"])
    # Rewrite intra-wiki links: .md -> .html (keep external + decisions/ relative).
    html_body = re.sub(r'href="([^"]+?)\.md"', r'href="\1.html"', html_body)
    # Fix links that point up to gateway/registry source -> GitHub (they're code, not docs).
    html_body = re.sub(
        r'href="\.\./(gateway|registry|connectors)/([^"]+)"',
        rf'href="https://github.com/{REPO}/tree/main/one-data-platform/\1/\2" target="_blank" rel="noopener"',
        html_body,
    )

    active = md_path.name
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex">
<title>{title} - One Data Platform</title><style>{WIKI_CSS}</style></head><body>
{topnav(active, prefix)}
<div class="wrap"><div class="crumbs"><a href="{prefix}index.html">← Home</a></div>
<article class="doc">{html_body}</article>
<footer>One Data Platform wiki · rendered from <code>docs/{md_path.name}</code></footer>
</div></body></html>"""


def render_wiki() -> int:
    if WIKI_OUT.exists():
        shutil.rmtree(WIKI_OUT)
    (WIKI_OUT / "decisions").mkdir(parents=True)
    count = 0
    # Top-level docs -> homepage/wiki/<name>.html  (prefix back to homepage = ../)
    for md in sorted(DOCS.glob("*.md")):
        (WIKI_OUT / f"{md.stem}.html").write_text(render_doc(md, prefix="../"))
        count += 1
    # Decision records -> homepage/wiki/decisions/<name>.html  (prefix = ../../)
    for md in sorted((DOCS / "decisions").glob("*.md")):
        (WIKI_OUT / "decisions" / f"{md.stem}.html").write_text(render_doc(md, prefix="../../"))
        count += 1
    return count


def main() -> None:
    builds = parse_tracker()
    (HERE / "catalog.json").write_text(json.dumps(builds, indent=2))
    (HERE / "index.html").write_text(render_home(builds))
    n_docs = render_wiki()
    n_done = sum(1 for b in builds if b["status"] == "done")
    print(f"Site built: homepage ({n_done}/{len(builds)} builds) + {n_docs} wiki pages -> homepage/")


if __name__ == "__main__":
    main()
