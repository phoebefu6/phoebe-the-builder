# One Data Platform

> One governed home for the whole data team - connect to any source, process, explore,
> and build & share dashboards, models, and AI products, with access control over all
> of it. Built one day at a time; a new build is added to the catalog every day.

This is the umbrella project: the **shell** (governed gateway), the **wiki** (all design
thinking), the **registry** (app catalog), and the **homepage** (a live index of every
build shipped so far).

## 🌐 Live site

**[Open the live platform →](https://phoebefu6.github.io/phoebe-the-builder/)**
(`phoebefu6.github.io/phoebe-the-builder` - hosted on GitHub Pages: the homepage + full
wiki, shareable with anyone).

## 🏠 Open the homepage

Open [`homepage/index.html`](homepage/index.html) in any browser - it lists all
completed builds as cards (with links to code + notebook) plus the roadmap, and the top
bar links to the **fully rendered HTML wiki** (no more raw markdown). It is
**auto-generated from `../TRACKER.md`** + the `docs/` markdown, so it's always current:
every day a build is finished, re-running the generator adds exactly one card.

```bash
# regenerate the whole site - homepage + wiki HTML (run after each daily build)
python homepage/build_site.py
# then open homepage/index.html
```

The wiki markdown in `docs/` stays the editable source; `build_site.py` renders it to
styled pages under `homepage/wiki/`.

## 📚 The wiki (read in order)

| # | Doc | What it gives you |
|---|-----|-------------------|
| 00 | [Glossary](docs/00-glossary.md) | Every platform term in plain language |
| 01 | [Why a platform](docs/01-why-a-platform.md) | The vision + the reasoning |
| 02 | [Architecture](docs/02-architecture.md) | The picture of how pieces fit |
| 03 | [Build log](docs/03-build-log.md) | Running design-thinking journal |
| 10 | [Gateway + login](docs/10-gateway-login.md) | Step 1: authentication |
| 11 | [RBAC + registry](docs/11-rbac-registry.md) | Step 2: authorization |
| - | [Decisions (ADRs)](docs/decisions/) | Every big choice, recorded |

## 🧱 The shell (what's built)

| Step | Component | Status |
|------|-----------|--------|
| 1 | Gateway + login (authentication) | ✅ done |
| 2 | RBAC + app registry | ✅ done |
| 3 | Audit log | ⬜ next |
| 4 | Connector layer | ⬜ |
| 5 | Mount a real app | ⬜ |
| 6 | Plug in Airflow (orchestration) | ⬜ |

Run the shell:
```bash
cd gateway
pip install -r requirements.txt
uvicorn app:app --reload      # http://localhost:8000
```
Demo logins: `phoebe@team.io` / `admin123` (admin) · `ana@team.io` / `analyst123` (analyst).

## 📂 Folders

```
one-data-platform/
├── homepage/      the live homepage (index.html) + its generator (build_catalog.py)
├── docs/          the wiki  ── docs/decisions/ holds the ADRs
├── gateway/       the shell: login, RBAC, routing
├── registry/      apps.yaml - the app catalog
└── connectors/    one safe place for data-source credentials (Step 4)
```

## 🔁 How "one build per day" stays automatic
The homepage reads `TRACKER.md`. The daily build skill (`daily-github-fde-build`) marks
a day `[x]` and then runs `homepage/build_catalog.py`, so the new build appears on the
homepage the same day - no manual editing. See
[ADR-0001](docs/decisions/adr-0001-governed-shell-not-databricks.md) for the platform
strategy.
