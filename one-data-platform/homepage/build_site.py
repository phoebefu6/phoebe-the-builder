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
    (61, 70, "data-engineering-pro", "Data Engineering Pro"),
    (71, 80, "ml-engineering-toolkit", "ML Engineering Toolkit"),
    (81, 90, "llmops-genai-platform", "LLMOps & GenAI Platform"),
    (91, 100, "data-quality-governance", "Data Quality & Governance Suite"),
    (101, 110, "analytics-engineering-bi", "Analytics Engineering & BI"),
    (111, 120, "data-science-cookbook", "Data Science Cookbook"),
]

# Capability domains: how a data specialist actually browses the catalog.
# (id, title, one-line, accent color, [product_slugs included])
DOMAINS = [
    ("data-engineering", "Data Engineering",
     "Move, load, transform, and orchestrate data at scale.", "#3d34d6",
     ["data-infra-toolkit", "data-engineering-pro"]),
    ("governance", "Data Quality & Governance",
     "Trust, validate, catalog, and control your data.", "#0e7c66",
     ["data-quality-governance"]),
    ("analytics-bi", "Analytics & BI",
     "Metrics, dashboards, and self-serve analytics.", "#c2410c",
     ["analytics-accelerator", "analytics-engineering-bi"]),
    ("ml-ds", "Machine Learning & Data Science",
     "Model training, evaluation, and the DS toolkit.", "#7c3aed",
     ["ml-engineering-toolkit", "data-science-cookbook"]),
    ("ai-llm", "AI, LLM & Agents",
     "RAG, agents, LLMOps, and document intelligence.", "#be185d",
     ["document-intelligence", "ai-agent-workshop", "llmops-genai-platform"]),
    ("automation", "Automation & Apps",
     "Workflow automation and shippable data apps.", "#0369a1",
     ["automation-suite", "mini-saas-products"]),
]

# Short pill labels per product line (the sub-category shown on each card).
LINE_LABEL = {
    "data-infra-toolkit": "Infra Toolkit",
    "data-engineering-pro": "Data Engineering",
    "data-quality-governance": "Quality & Governance",
    "analytics-accelerator": "Analytics Accelerator",
    "analytics-engineering-bi": "Analytics Eng & BI",
    "ml-engineering-toolkit": "ML Engineering",
    "data-science-cookbook": "Data Science",
    "document-intelligence": "Document Intelligence",
    "ai-agent-workshop": "AI Agents",
    "llmops-genai-platform": "LLMOps & GenAI",
    "automation-suite": "Automation",
    "mini-saas-products": "Mini SaaS",
}

# Wiki nav order: (source filename, short nav label)
WIKI_NAV = [
    ("01-why-a-platform.md", "Why"),
    ("02-architecture.md", "Architecture"),
    ("00-glossary.md", "Glossary"),
    ("10-gateway-login.md", "Gateway"),
    ("11-rbac-registry.md", "RBAC"),
    ("12-audit-log.md", "Audit"),
    ("13-connector-layer.md", "Connectors"),
    ("14-mount-app.md", "Mount app"),
    ("15-orchestration.md", "Orchestration"),
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
header.hero{padding:2.6rem 0 1.6rem}
.kicker{color:var(--accent);font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:.78rem}
h1{font-size:2.4rem;margin:.4rem 0 .6rem;letter-spacing:-.02em;max-width:760px}
.tag{color:var(--muted);font-size:1.05rem;max-width:680px}
.meta{margin-top:1rem;color:var(--muted);font-size:.86rem;font-weight:600}
.meta b{color:var(--ink)}
/* controls: search + domain filters (sticky under topbar) */
.controls{position:sticky;top:52px;z-index:9;background:rgba(251,251,254,.94);backdrop-filter:blur(8px);
border-bottom:1px solid var(--line);padding:.8rem 0}
.controls .wrap{display:flex;flex-direction:column;gap:.7rem}
.search{position:relative;max-width:100%}
.search input{width:100%;padding:.7rem .9rem .7rem 2.3rem;border:1px solid var(--line);border-radius:10px;
font-size:.95rem;background:#fff;color:var(--ink)}
.search input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px rgba(61,52,214,.1)}
.search svg{position:absolute;left:.8rem;top:50%;transform:translateY(-50%);opacity:.45}
.chips{display:flex;gap:.5rem;flex-wrap:wrap}
.chip{border:1px solid var(--line);background:#fff;color:var(--muted);font-size:.82rem;font-weight:600;
padding:.4rem .8rem;border-radius:999px;cursor:pointer;transition:.12s;display:inline-flex;align-items:center;gap:.4rem}
.chip:hover{border-color:var(--accent);color:var(--accent)}
.chip.active{background:var(--accent);border-color:var(--accent);color:#fff}
.chip .c{font-size:.72rem;opacity:.7}
.chip.active .c{opacity:.85}
.chip .swatch{width:9px;height:9px;border-radius:50%}
main{padding-bottom:1rem}
.domain{padding:2rem 0 .5rem}
.domain-head{display:flex;align-items:baseline;gap:.7rem;margin:0 0 .2rem;border-left:4px solid var(--dc);
padding-left:.7rem}
.domain-head h2{font-size:1.35rem;margin:0}
.domain-head .count{font-size:.78rem;color:var(--muted);font-weight:600}
.domain-sub{color:var(--muted);font-size:.9rem;margin:.1rem 0 1rem .95rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(255px,1fr));gap:1rem}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1.05rem 1.1rem;transition:.15s;
display:flex;flex-direction:column}
.card:hover{transform:translateY(-2px);box-shadow:0 10px 26px rgba(60,52,214,.09);border-color:#d6d2f5}
.pill{align-self:flex-start;font-size:.68rem;font-weight:700;letter-spacing:.02em;text-transform:uppercase;
padding:.18rem .5rem;border-radius:6px;color:#fff;background:var(--dc)}
.card h3{margin:.6rem 0 .35rem;font-size:1.05rem;line-height:1.35}
.card code{font-size:.74rem;color:var(--muted);font-family:'SF Mono',Menlo,Consolas,monospace}
.card .links{margin-top:auto;padding-top:.85rem;display:flex;gap:.9rem}
.card .links a{font-size:.82rem;color:var(--accent);text-decoration:none;font-weight:600}
.card .links a:hover{text-decoration:underline}
.noresults{display:none;text-align:center;color:var(--muted);padding:3rem 0;font-size:1rem}
footer{color:var(--muted);font-size:.82rem;padding:2.4rem 0;border-top:1px solid var(--line);margin-top:2rem}
@media(max-width:560px){h1{font-size:1.9rem}.controls{top:96px}}
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


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _card(b: Dict[str, object], domain_id: str) -> str:
    cat = LINE_LABEL.get(b["product_slug"], b["product_line"])
    search = _esc(f"{b['name']} {b['slug']} {cat} {b['product_line']}".lower())
    links = (f"<div class='links'><a href='{b['repo_url']}' target='_blank' rel='noopener'>Code</a>"
             f"<a href='{b['colab_url']}' target='_blank' rel='noopener'>Notebook</a></div>")
    return (f"<article class='card' data-domain='{domain_id}' data-search='{search}'>"
            f"<span class='pill'>{_esc(cat)}</span>"
            f"<h3>{_esc(b['name'])}</h3><code>{_esc(b['slug'])}</code>{links}</article>")


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
    by_slug: Dict[str, List[Dict[str, object]]] = {}
    for b in done:
        by_slug.setdefault(b["product_slug"], []).append(b)

    # domain filter chips
    chips = ['<button class="chip active" data-domain="all">All <span class="c">'
             f'{len(done)}</span></button>']
    for did, title, _sub, color, slugs in DOMAINS:
        n = sum(len(by_slug.get(s, [])) for s in slugs)
        if not n:
            continue
        chips.append(f'<button class="chip" data-domain="{did}" style="--sw:{color}">'
                     f'<span class="swatch" style="background:{color}"></span>{_esc(title)} '
                     f'<span class="c">{n}</span></button>')

    # domain sections
    sections = []
    for did, title, sub, color, slugs in DOMAINS:
        items: List[Dict[str, object]] = []
        for s in slugs:
            items += sorted(by_slug.get(s, []), key=lambda x: x["day"])
        if not items:
            continue
        cards = "".join(_card(b, did) for b in items)
        sections.append(
            f'<section class="domain" data-domain="{did}" style="--dc:{color}">'
            f'<div class="domain-head"><h2>{_esc(title)}</h2>'
            f'<span class="count">{len(items)} tools</span></div>'
            f'<p class="domain-sub">{_esc(sub)}</p>'
            f'<div class="grid">{cards}</div></section>'
        )

    n_domains = sum(1 for _d, _t, _s, _c, sl in DOMAINS if any(by_slug.get(x) for x in sl))
    updated = datetime.now().strftime("%Y-%m-%d")
    script = """
<script>
(function(){
  var q=document.getElementById('q');
  var chips=Array.prototype.slice.call(document.querySelectorAll('.chip'));
  var cards=Array.prototype.slice.call(document.querySelectorAll('.card'));
  var secs=Array.prototype.slice.call(document.querySelectorAll('.domain'));
  var none=document.getElementById('none');
  var dom='all';
  function apply(){
    var t=q.value.trim().toLowerCase(), shown=0;
    cards.forEach(function(c){
      var okD=(dom==='all'||c.dataset.domain===dom);
      var okT=(!t||c.dataset.search.indexOf(t)!==-1);
      var v=okD&&okT;
      c.style.display=v?'':'none';
      if(v)shown++;
    });
    secs.forEach(function(s){
      var any=Array.prototype.slice.call(s.querySelectorAll('.card')).some(function(c){return c.style.display!=='none';});
      s.style.display=any?'':'none';
    });
    none.style.display=shown?'none':'block';
  }
  q.addEventListener('input',apply);
  chips.forEach(function(ch){
    ch.addEventListener('click',function(){
      chips.forEach(function(x){x.classList.remove('active');});
      ch.classList.add('active');
      dom=ch.dataset.domain;
      apply();
      if(dom!=='all'){
        var s=document.querySelector('.domain[data-domain="'+dom+'"]');
        if(s)window.scrollTo({top:s.offsetTop-150,behavior:'smooth'});
      }
    });
  });
})();
</script>"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex">
<title>Data &amp; AI Toolkit — One Data Platform</title><style>{HOME_CSS}</style></head><body>
{topnav("home", "")}
<div class="wrap"><header class="hero">
<div class="kicker">One Data Platform</div>
<h1>An open, growing catalog of data &amp; AI tools for data teams.</h1>
<p class="tag">Small, focused, runnable tools across the data lifecycle — each with source code and a
one-click notebook. Search it, or browse by capability. New tools added whenever there's an
opportunity.</p>
<div class="meta"><b>{len(done)}</b> tools and counting · <b>{n_domains}</b> capability domains · open source</div>
</header></div>
<div class="controls"><div class="wrap">
<div class="search">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>
<input id="q" type="search" placeholder="Search tools — e.g. forecast, dbt, RAG, chi-square, lineage…"
autocomplete="off"></div>
<div class="chips">{''.join(chips)}</div>
</div></div>
<div class="wrap"><main>
{''.join(sections)}
<div class="noresults" id="none">No tools match your search. Try another term or clear the filter.</div>
</main>
<footer>One Data Platform · catalog auto-generated from the build log · updated {updated}.
Browse the <a href="wiki/02-architecture.html" style="color:var(--accent)">architecture</a> and
<a href="wiki/01-why-a-platform.html" style="color:var(--accent)">platform notes</a> in the wiki.</footer>
</div>{script}</body></html>"""


# ── Markdown -> styled HTML ────────────────────────────────────────────────────
def render_doc(md_path: Path, prefix: str) -> str:
    raw = md_path.read_text()
    # Strip the file-order prefix ("00 - ", "10 - ", etc.) from the H1 title so the
    # displayed page heading reads cleanly. Source files keep their numbers (ordering).
    raw = re.sub(r"^(#\s+)\d+\s*[-–]\s*", r"\1", raw, count=1, flags=re.MULTILINE)
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
