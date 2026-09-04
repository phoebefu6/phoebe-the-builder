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

# ── Task taxonomy ─────────────────────────────────────────────────────────────
# The catalog is organised by the JOB somebody is trying to do, not by when a
# tool happened to get built. Build order is an accident of the calendar; the
# task is what a person arrives with.
#
# (id, verb-led title, the question the practitioner actually asks)
TASKS = [
    ("ingest", "Move data in",
     "It lives somewhere else and it has to land here, on a schedule, without losing rows."),
    ("shape", "Make raw values usable",
     "The bytes arrived. What they MEAN is a decision, and every layer decides differently."),
    ("trust", "Prove it is right",
     "Something changed upstream. Find out before a dashboard does."),
    ("govern", "Control who sees what",
     "Who owns this column, who may read it, and can you show an auditor?"),
    ("observe", "Know when it breaks",
     "The pipeline fails at 3am. The question is whether anyone finds out before the meeting."),
    ("explore", "Find out what is in it",
     "A new dataset landed. Two hours of profiling before you can say anything about it."),
    ("measure", "Turn it into a number people act on",
     "One metric, three dashboards, three answers. Define it once and serve it."),
    ("infer", "Decide what is actually true",
     "The line went up. Whether that means anything is a separate question."),
    ("predict", "Learn from it",
     "Fit something, then find out honestly whether it is better than doing nothing."),
    ("understand", "Get answers out of documents",
     "The answer is in a 200-page PDF nobody will read."),
    ("evaluate", "Check the AI is any good",
     "It sounds right. Sounding right is not a measurement."),
    ("automate", "Make it run itself",
     "The task is small, correct, and done by hand every single week."),
    ("decide", "Choose, and be able to defend it",
     "The number is on the screen. Someone still has to decide, and later justify it."),
]

# Every built tool is assigned exactly one task. `main()` fails loudly on an
# unmapped slug rather than silently dropping it from the catalog.
TASK_OF = {
    # Move data in
    "csv-loader": "ingest", "api-warehouse-connector": "ingest", "api-paginator": "ingest",
    "incremental-loader": "ingest", "streaming-aggregator": "ingest",
    "parquet-partitioner": "ingest", "backfill-planner": "ingest",
    # Make raw values usable
    "boolean-parser": "shape", "number-parser-locale": "shape",
    "line-ending-detector": "shape", "sort-order-drift": "shape",
    "unicode-width-truncator": "shape", "duration-parser": "shape",
    "header-casing": "shape", "csv-dialect-sniffer": "shape",
    "fixed-width-parser": "shape", "timezone-normalizer": "shape",
    "type-inferencer": "shape", "json-flattener": "shape", "currency-rounder": "shape",
    "percent-recomputer": "shape", "slug-collider": "shape", "filename-sanitiser": "shape",
    "csv-cleaner": "shape", "markdown-tabler": "shape", "dedup-pipeline": "shape",
    "duplicate-finder": "shape",
    # Prove it is right
    "data-contract-validator": "trust", "dq-rules-engine": "trust",
    "gx-config-generator": "trust", "data-quality-scorecard": "trust",
    "reconciliation-checker": "trust", "data-diff": "trust", "schema-diff": "trust",
    "schema-registry": "trust", "dbt-test-gen": "trust", "json-validator": "trust",
    "null-heatmap": "trust", "anomaly-detector": "trust",
    # Control who sees what
    "pii-detector": "govern", "pii-redactor": "govern", "dsar-extractor": "govern",
    "consent-tracker": "govern", "retention-enforcer": "govern", "access-auditor": "govern",
    "compliance-checker": "govern", "data-catalog": "govern", "business-glossary": "govern",
    "data-dict-gen": "govern", "column-lineage": "govern", "data-lineage-viz": "govern",
    "erd-generator": "govern", "privacy-policy-gen": "govern", "model-card-gen": "govern",
    # Know when it breaks
    "data-freshness-monitor": "observe", "pipeline-sla-monitor": "observe",
    "cron-monitor": "observe", "db-health-dashboard": "observe", "log-parser": "observe",
    "metric-alerting": "observe", "model-drift-detector": "observe",
    "pipeline-monitor-agent": "observe", "incident-agent": "observe", "env-checker": "observe",
    # Find out what is in it
    "auto-eda": "explore", "correlation-explorer": "explore", "outlier-explainer": "explore",
    "dim-reducer": "explore", "customer-segments": "explore", "survey-analyzer": "explore",
    "feedback-analyzer": "explore", "topic-modeler": "explore",
    # Turn it into a number people act on
    "kpi-tracker": "measure", "kpi-tree": "measure", "metric-catalog": "measure",
    "metrics-layer": "measure", "metric-diff": "measure", "dashboard-spec": "measure",
    "self-serve-explorer": "measure", "funnel-analyzer": "measure",
    "cohort-analysis": "measure", "sparkline-gen": "measure", "pivot-narrator": "measure",
    "report-scheduler": "measure", "nl-to-sql": "measure", "query-optimizer": "measure",
    "sales-forecast": "measure",
    # Decide what is actually true
    "ab-test-calc": "infer", "sample-size-calc": "infer", "stat-test-advisor": "infer",
    "peeking-cost": "infer", "srm-detector": "infer", "cuped-variance": "infer",
    "crosstab-chi2": "infer", "distribution-fitter": "infer",
    # Learn from it
    "baseline-model": "predict", "feature-factory": "predict",
    "feature-importance": "predict", "feature-binner": "predict",
    "hyperparam-tuner": "predict", "train-eval-harness": "predict",
    "model-registry": "predict", "batch-scorer": "predict", "leakage-detector": "predict",
    "imbalance-toolkit": "predict", "calibration-checker": "predict",
    "threshold-explorer": "predict", "churn-predictor": "predict", "recommender": "predict",
    "ts-forecaster": "predict", "text-classifier": "predict",
    # Get answers out of documents
    "pdf-qa-bot": "understand", "semantic-search": "understand",
    "knowledge-base": "understand", "faq-generator": "understand",
    "contract-extractor": "understand", "meeting-summarizer": "understand",
    "competitive-intel": "understand", "resume-screener": "understand",
    "ner-extractor": "understand", "structured-extractor": "understand",
    "schema-from-samples": "understand", "embedding-dedup": "understand",
    "chunk-optimizer": "understand",
    # Check the AI is any good
    "rag-eval": "evaluate", "agent-eval-dashboard": "evaluate",
    "hallucination-checker": "evaluate", "llm-guardrails": "evaluate",
    "prompt-linter": "evaluate", "prompt-registry": "evaluate",
    "fewshot-selector": "evaluate", "llm-router": "evaluate",
    "llm-cost-tracker": "evaluate", "token-cost-estimator": "evaluate",
    "semantic-cache": "evaluate",
    # Make it run itself
    "email-agent": "automate", "code-review-agent": "automate",
    "onboarding-agent": "automate", "report-agent": "automate",
    "slack-qa-agent": "automate", "ticket-router": "automate", "standup-bot": "automate",
    "auto-readme": "automate", "dbt-model-generator": "automate",
    "airflow-dag-gen": "automate", "md-to-pdf": "automate", "sql-formatter": "automate",
    "cron-explainer": "automate", "retry-schedule": "automate",
    "accessibility-checker": "automate",
    # Choose, and be able to defend it
    "decision-log": "decide", "pre-mortem": "decide",
    "expected-value-calc": "decide", "cost-of-delay": "decide",
    "target-setter": "decide", "guardrail-metric": "decide",
    "warehouse-cost-attribution": "decide", "goodhart-detector": "decide",
    "leading-indicator-finder": "decide",
    "portfolio-dashboard": "decide",
    "feature-prioritizer": "decide", "idea-validator": "decide",
    "okr-tracker": "decide", "roadmap-viz": "decide", "retro-generator": "decide",
    "user-story-gen": "decide",
}

TASK_TITLE = {t[0]: t[1] for t in TASKS}


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
/* One row at every width. A wrapping topbar changes its own height, which
   breaks the sticky offset of anything docked beneath it. */
.topbar .wrap{display:flex;align-items:center;gap:.4rem;flex-wrap:nowrap;overflow-x:auto;
scrollbar-width:none;-webkit-overflow-scrolling:touch;padding-top:.7rem;padding-bottom:.7rem}
.topbar .wrap::-webkit-scrollbar{display:none}
.brand{font-weight:800;color:var(--accent);margin-right:.6rem;text-decoration:none;letter-spacing:-.01em;
flex:0 0 auto;white-space:nowrap}
.topbar a.nav{color:var(--muted);text-decoration:none;font-size:.86rem;font-weight:600;padding:.35rem .7rem;
border-radius:7px;flex:0 0 auto;white-space:nowrap}
.topbar a.nav:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
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
header.hero{padding:3.4rem 0 2rem;max-width:780px}
.kicker{color:var(--accent);font-weight:700;letter-spacing:.09em;text-transform:uppercase;font-size:.74rem}
h1{font-size:2.55rem;margin:.55rem 0 .75rem;letter-spacing:-.028em;line-height:1.12}
.tag{color:var(--muted);font-size:1.04rem;line-height:1.6;margin:0}
/* controls: search + task filters (sticky under topbar) */
.controls{position:sticky;top:var(--topbar-h,52px);z-index:9;background:rgba(251,251,254,.94);
backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:.8rem 0}
.controls .wrap{display:flex;flex-direction:column;gap:.7rem}
.search{position:relative}
.search input{width:100%;padding:.72rem .9rem .72rem 2.3rem;border:1px solid var(--line);border-radius:10px;
font-size:.95rem;background:#fff;color:var(--ink)}
.search input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px rgba(61,52,214,.1)}
.search svg{position:absolute;left:.8rem;top:50%;transform:translateY(-50%);opacity:.45}
/* One scrollable row: 14 filters must not wrap into a wall of sticky chrome. */
.chips{display:flex;gap:.42rem;flex-wrap:nowrap;overflow-x:auto;scrollbar-width:none;
-webkit-overflow-scrolling:touch;padding-bottom:2px;
-webkit-mask-image:linear-gradient(90deg,#000 calc(100% - 28px),transparent);
mask-image:linear-gradient(90deg,#000 calc(100% - 28px),transparent)}
.chips::-webkit-scrollbar{display:none}
.chip{border:1px solid var(--line);background:#fff;color:var(--muted);font-size:.8rem;font-weight:600;
padding:.36rem .74rem;border-radius:999px;cursor:pointer;transition:.12s;white-space:nowrap;flex:0 0 auto}
.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.chip:hover{border-color:var(--accent);color:var(--accent)}
.chip.active{background:var(--accent);border-color:var(--accent);color:#fff}
main{padding-bottom:1rem}
.task{padding:2.9rem 0 .3rem;border-top:1px solid var(--line)}
.task:first-of-type{border-top:0;padding-top:2rem}
.task h2{font-size:1.55rem;margin:0;letter-spacing:-.022em}
.task-q{color:var(--muted);font-size:.95rem;margin:.4rem 0 1.35rem;max-width:640px;line-height:1.55}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(288px,1fr));gap:.95rem}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1.05rem 1.1rem;
display:flex;flex-direction:column;gap:.45rem;transition:.15s}
.card:hover{transform:translateY(-2px);box-shadow:0 10px 26px rgba(60,52,214,.09);border-color:#d6d2f5}
.card h3{margin:0;font-size:1rem;line-height:1.3;letter-spacing:-.012em}
.problem{margin:0;color:#4c4c63;font-size:.875rem;line-height:1.58}
.foot{margin-top:auto;padding-top:.8rem;display:flex;align-items:center;justify-content:space-between;
gap:.6rem;border-top:1px solid #f1f0f9}
.foot code{font-size:.72rem;color:var(--muted);font-family:'SF Mono',Menlo,Consolas,monospace;
overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.links{display:flex;gap:.75rem;flex-shrink:0}
.links a{font-size:.79rem;color:var(--accent);text-decoration:none;font-weight:600}
.links a:hover{text-decoration:underline}
/* platform spine */
.spine{padding:2.9rem 0 .3rem;border-top:1px solid var(--line)}
.spine h2{font-size:1.55rem;margin:0;letter-spacing:-.022em}
.spine .task-q{margin-bottom:1.35rem}
.modgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:.95rem}
.mod{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1rem 1.05rem}
.mod h3{margin:0 0 .3rem;font-size:.98rem;letter-spacing:-.012em}
.mod h3 a{color:var(--ink);text-decoration:none;border-bottom:1px solid #d7d4f7}
.mod h3 a:hover{color:var(--accent)}
.mod .concept{color:var(--muted);font-size:.83rem;line-height:1.5;display:block}
.noresults{display:none;text-align:center;color:var(--muted);padding:3.5rem 0;font-size:1rem}
footer{color:var(--muted);font-size:.82rem;padding:2.4rem 0;border-top:1px solid var(--line);margin-top:2.6rem}
@media(max-width:620px){h1{font-size:2rem}header.hero{padding:2.4rem 0 1.4rem}
.grid,.modgrid{grid-template-columns:1fr}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}
.card:hover{transform:none}}
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


# Backlog builds (Day 121+) sit outside the month ranges, so their tracker line
# names the folder instead: "slug: Name (ml-engineering-toolkit/)".
FOLDER_RE = re.compile(r"\s*\((?P<folder>[\w-]+)/\)\s*$")


def split_folder(name: str, day: int):
    """Return (clean_name, slug, line_name), honouring an explicit folder tag."""
    m = FOLDER_RE.search(name)
    if not m:
        return name, *product_line(day)
    folder = m.group("folder")
    clean = FOLDER_RE.sub("", name).strip()
    for _lo, _hi, slug, line_name in PRODUCT_LINES:
        if slug == folder:
            return clean, slug, line_name
    return clean, folder, folder.replace("-", " ").title()


def parse_tracker() -> List[Dict[str, object]]:
    builds: List[Dict[str, object]] = []
    for line in TRACKER.read_text().splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        day = int(m.group("day"))
        name, pl_slug, pl_name = split_folder(m.group("name").strip(), day)
        done = m.group("done") == "x"
        slug = m.group("slug")
        builds.append({
            "day": day, "slug": slug, "name": name,
            "task": TASK_OF.get(slug), "task_title": TASK_TITLE.get(TASK_OF.get(slug, ""), ""),
            "problem": problem_line(pl_slug, slug) if done else "",
            "status": "done" if done else "planned", "date": m.group("date"),
            "product_line": pl_name, "product_slug": pl_slug,
            "repo_url": f"https://github.com/{REPO}/tree/main/{pl_slug}/{slug}" if done else None,
            "colab_url": (f"https://colab.research.google.com/github/{REPO}/blob/main/{pl_slug}/{slug}/demo.ipynb"
                          if done else None),
        })
    return builds


BLOCKQUOTE_RE = re.compile(r"^>\s*(.+?)\s*$", re.M)
_MD_STRIP = re.compile(r"\*\*|__|\*|`|\[([^\]]+)\]\([^)]*\)")


def problem_line(product_slug: str, slug: str, limit: int = 190) -> str:
    """The tool's own one-line statement of the problem, from its README.

    Every README opens with a blockquote naming the thing that was actually
    going wrong. That sentence is the most honest description of the tool
    there is - far better than any label a taxonomy could hang on it - so
    the card leads with it rather than with a category.
    """
    readme = ROOT.parent / product_slug / slug / "README.md"
    if not readme.exists():
        return ""
    m = BLOCKQUOTE_RE.search(readme.read_text(encoding="utf-8", errors="replace"))
    if not m:
        return ""
    text = _MD_STRIP.sub(lambda x: x.group(1) or "", m.group(1)).strip()
    text = re.sub(r"\s+", " ", text)
    # House typography: spaced hyphen, not an em or en dash. Purely a
    # rendering normalisation - the README keeps whatever it was written with.
    text = re.sub(r"\s*[\u2014\u2013]\s*", " - ", text)
    if len(text) <= limit:
        return text
    # Take whole sentences while they fit, so the card never ends mid-clause.
    # These openings are written punchline-first, so the first sentence alone
    # is usually the sharpest thing on the card.
    parts = re.split(r"(?<=[.!?])\s+", text)
    out, taken = "", 0
    for part in parts:
        if out and len(out) + 1 + len(part) > limit:
            break
        out = f"{out} {part}".strip()
        taken += 1
    # A punchline-first opener can leave a one-line card sitting next to
    # five-line neighbours. If the whole-sentence rule kept very little, pull
    # in the next sentence and cut it on a word instead.
    floor = int(limit * 0.42)
    if out and len(out) < floor and taken < len(parts):
        extra = parts[taken]
        room = limit - len(out) - 1
        if room > 24:
            clipped = extra[:room].rsplit(" ", 1)[0].rstrip(" -,;:")
            out = f"{out} {clipped}\u2026"
    # A single sentence can still overrun the budget; cap it so one long
    # opener cannot blow out a card or a table row.
    hard = int(limit * 1.3)
    if out and len(out) <= hard:
        return out
    source = out or text
    return source[:hard].rsplit(" ", 1)[0].rstrip(" -,;:") + "\u2026"


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _card(b: Dict[str, object]) -> str:
    problem = str(b.get("problem") or "")
    search = _esc(f"{b['name']} {b['slug']} {b.get('task_title', '')} {problem}".lower())
    links = (f"<a href='{b['repo_url']}' target='_blank' rel='noopener'>Code</a>"
             f"<a href='{b['colab_url']}' target='_blank' rel='noopener'>Notebook</a>")
    body = f"<p class='problem'>{_esc(problem)}</p>" if problem else ""
    return (f"<article class='card' data-task='{b['task']}' data-search='{search}'>"
            f"<h3>{_esc(b['name'])}</h3>{body}"
            f"<div class='foot'><code>{_esc(b['slug'])}</code>"
            f"<div class='links'>{links}</div></div></article>")


def render_shell_section() -> str:
    """The platform's own modules - the control plane every tool plugs into.

    Only modules that exist are rendered. A roadmap of unbuilt steps belongs
    in the build log, not on a page whose job is to show what runs.
    """
    if not SHELL_FILE.exists():
        return ""
    data = yaml.safe_load(SHELL_FILE.read_text()) or {}
    modules = [m for m in (data.get("modules") or []) if m.get("status") == "done"]
    if not modules:
        return ""
    cards = []
    for m in modules:
        doc = m.get("doc") or ""
        name = f"<a href='{doc}'>{_esc(m['name'])}</a>" if doc else _esc(m["name"])
        cards.append(f"<div class='mod'><h3>{name}</h3>"
                     f"<span class='concept'>{_esc(m['concept'])}</span></div>")
    return ("<section class='spine'><h2>The spine underneath</h2>"
            "<p class='task-q'>The control plane the tools plug into: identity, permissions, "
            "audit, connectors, orchestration.</p>"
            f"<div class='modgrid'>{''.join(cards)}</div></section>")


def render_home(builds: List[Dict[str, object]]) -> str:
    done = [b for b in builds if b["status"] == "done"]
    by_task: Dict[str, List[Dict[str, object]]] = {}
    for b in done:
        by_task.setdefault(str(b["task"]), []).append(b)

    chips = ['<button class="chip active" data-task="all">Everything</button>']
    for tid, title, _q in TASKS:
        if by_task.get(tid):
            chips.append(f'<button class="chip" data-task="{tid}">{_esc(title)}</button>')

    sections = []
    for tid, title, question in TASKS:
        items = sorted(by_task.get(tid, []), key=lambda x: str(x["name"]).lower())
        if not items:
            continue
        cards = "".join(_card(b) for b in items)
        sections.append(
            f'<section class="task" data-task="{tid}">'
            f'<h2>{_esc(title)}</h2><p class="task-q">{_esc(question)}</p>'
            f'<div class="grid">{cards}</div></section>'
        )

    updated = datetime.now().strftime("%B %Y")
    script = """
<script>
(function(){
  var bar=document.querySelector('.topbar');
  var ctl=document.querySelector('.controls');
  function chrome(){
    // The topbar wraps to two rows at some widths, so the sticky offset and the
    // scroll target have to be measured rather than assumed.
    document.documentElement.style.setProperty('--topbar-h', bar.offsetHeight+'px');
    return bar.offsetHeight + ctl.offsetHeight + 12;
  }
  chrome();
  window.addEventListener('resize', chrome);
  var q=document.getElementById('q');
  var chips=[].slice.call(document.querySelectorAll('.chip'));
  var cards=[].slice.call(document.querySelectorAll('.card'));
  var secs=[].slice.call(document.querySelectorAll('.task'));
  var spine=document.querySelector('.spine');
  var none=document.getElementById('none');
  var task='all';
  function apply(){
    var t=q.value.trim().toLowerCase(), shown=0;
    cards.forEach(function(c){
      var v=(task==='all'||c.dataset.task===task)&&(!t||c.dataset.search.indexOf(t)!==-1);
      c.style.display=v?'':'none';
      if(v)shown++;
    });
    secs.forEach(function(s){
      var any=[].slice.call(s.querySelectorAll('.card')).some(function(c){return c.style.display!=='none';});
      s.style.display=any?'':'none';
    });
    if(spine)spine.style.display=(task==='all'&&!t)?'':'none';
    none.style.display=shown?'none':'block';
  }
  q.addEventListener('input',apply);
  chips.forEach(function(ch){
    ch.addEventListener('click',function(){
      chips.forEach(function(x){x.classList.remove('active');});
      ch.classList.add('active');
      task=ch.dataset.task;
      apply();
      if(task!=='all'){
        var s=document.querySelector('.task[data-task="'+task+'"]');
        if(s)window.scrollTo({top:s.getBoundingClientRect().top+window.pageYOffset-chrome(),behavior:'smooth'});
      }else{window.scrollTo({top:0,behavior:'smooth'});}
    });
  });
})();
</script>"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex">
<title>One Data Platform · tools for data work</title>
<meta name="description" content="Small, exact tools for the parts of data work that quietly go wrong.">
<style>{HOME_CSS}</style></head><body>
{topnav("home", "")}
<div class="wrap"><header class="hero">
<div class="kicker">One Data Platform</div>
<h1>Small, exact tools for the parts of data work that quietly go wrong.</h1>
<p class="tag">Each one started with something breaking in a real pipeline: a metric that
disagreed with itself, a parser that read the same bytes two ways, a model that looked fine
until it shipped. Source code and a runnable notebook on every one. Browse by what you are
trying to do.</p>
</header></div>
<div class="controls"><div class="wrap">
<div class="search">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>
<input id="q" type="search" aria-label="Search the tools" placeholder="Search a symptom, a format, a technique: stale, CRLF, RAG, chi-square, lineage…"
autocomplete="off"></div>
<div class="chips">{''.join(chips)}</div>
</div></div>
<div class="wrap"><main>
{''.join(sections)}
{render_shell_section()}
<div class="noresults" id="none">Nothing matches that. Try a symptom rather than a tool name, or clear the filter.</div>
</main>
<footer>One Data Platform · updated {updated} ·
<a href="wiki/02-architecture.html" style="color:var(--accent)">architecture</a> ·
<a href="wiki/01-why-a-platform.html" style="color:var(--accent)">platform notes</a> ·
<a href="wiki/03-build-log.html" style="color:var(--accent)">build log</a></footer>
</div>{script}</body></html>"""


# ── Repo README ───────────────────────────────────────────────────────────────
# Generated from the same catalog as the homepage. It used to be maintained by
# hand, which is why it sat frozen at 75 builds while the repo passed 150 - a
# second source of truth always loses to the one that is generated.

README_PATH = ROOT.parent / "README.md"
HEADER_OPEN, HEADER_CLOSE = "<!-- phoebe header -->", "<!-- /phoebe header -->"


def _existing_managed_header() -> str:
    """Preserve the header block another tool owns, verbatim."""
    if not README_PATH.exists():
        return ""
    text = README_PATH.read_text(encoding="utf-8")
    if HEADER_OPEN in text and HEADER_CLOSE in text:
        start = text.index(HEADER_OPEN)
        end = text.index(HEADER_CLOSE) + len(HEADER_CLOSE)
        return text[start:end] + "\n"
    return ""


def render_readme(builds: List[Dict[str, object]]) -> str:
    done = [b for b in builds if b["status"] == "done"]
    by_task: Dict[str, List[Dict[str, object]]] = {}
    for b in done:
        by_task.setdefault(str(b["task"]), []).append(b)

    out = [_existing_managed_header()]
    out.append("# phoebe-the-builder\n")
    out.append(
        "Small, exact tools for the parts of data work that quietly go wrong.\n\n"
        "Each one started with something breaking in a real pipeline: a metric that disagreed "
        "with itself, a parser that read the same bytes two ways, a model that looked fine "
        "until it shipped. Every tool ships with source, a runnable Colab/Binder notebook, "
        "a working app, a Dockerfile and CI.\n\n"
        "**[Browse them by what you are trying to do →]"
        "(https://phoebefu6.github.io/phoebe-the-builder/)**\n"
    )

    for tid, title, question in TASKS:
        items = sorted(by_task.get(tid, []), key=lambda x: str(x["name"]).lower())
        if not items:
            continue
        out.append(f"\n---\n\n## {title}\n\n*{question}*\n")
        out.append("| Tool | The problem it was built for |")
        out.append("|------|------------------------------|")
        for b in items:
            problem = problem_line(str(b["product_slug"]), str(b["slug"]), limit=118)
            problem = problem.replace("|", "\\|")
            out.append(f"| [{b['name']}]({b['product_slug']}/{b['slug']}/) | {problem} |")
        out.append("")

    out.append(
        "\n---\n\n"
        "### The spine underneath\n\n"
        "The tools plug into a control plane built alongside them: identity, permissions, "
        "an audit log, a connector layer and orchestration. The "
        "[architecture notes](https://phoebefu6.github.io/phoebe-the-builder/wiki/02-architecture.html) "
        "walk the design decisions.\n\n"
        "*Built in public by Phoebe Fu.*\n"
    )
    return "\n".join(out)


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
    done = [b for b in builds if b["status"] == "done"]

    # A tool with no task has no home on the page and would vanish silently.
    # Fail here instead: the daily build must classify what it ships.
    orphans = sorted(str(b["slug"]) for b in done if not b.get("task"))
    if orphans:
        raise SystemExit(
            "Unclassified tools - add each to TASK_OF in build_site.py:\n  "
            + "\n  ".join(orphans)
        )

    (HERE / "catalog.json").write_text(json.dumps(builds, indent=2))
    (HERE / "index.html").write_text(render_home(builds))
    README_PATH.write_text(render_readme(builds))
    n_docs = render_wiki()
    filled = sum(1 for tid, _t, _q in TASKS if any(b["task"] == tid for b in done))
    print(f"Site built: {filled} task sections + {n_docs} wiki pages -> homepage/, "
          f"README.md regenerated")


if __name__ == "__main__":
    main()
