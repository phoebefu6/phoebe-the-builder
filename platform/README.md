# Platform - the governed orchestration shell

> One governed workspace for the whole data team (analysts, engineers, data
> scientists, AI engineers): connect to any source, process, explore, and build &
> share dashboards, models, and AI products - with access control over all of it.

This is the **spine** the 60 portfolio builds plug into. We build only the thin
governance layer ourselves; we mount open source (Airflow, DuckDB, Trino) for the
heavy compute. See [ADR-0001](docs/decisions/adr-0001-governed-shell-not-databricks.md)
for the reasoning.

## 📚 Start here (the wiki)

Read in order - each builds on the last. Written for a data person, no platform
background assumed.

| # | Doc | What it gives you |
|---|-----|-------------------|
| 00 | [Glossary](docs/00-glossary.md) | Every scary term, in plain language |
| 01 | [Why a platform](docs/01-why-a-platform.md) | The vision + the reasoning |
| 02 | [Architecture](docs/02-architecture.md) | The picture of how pieces fit |
| 03 | [Build log](docs/03-build-log.md) | Running design-thinking journal |
| - | [Decisions (ADRs)](docs/decisions/) | Every big choice, recorded |

## 🧭 How we work (the rules)

1. **Understand before we use** - every concept gets a glossary entry + explainer first.
2. **Smallest real version first** - hand-write a readable version, then swap in OSS.
3. **Log every decision** - each meaningful choice becomes an ADR.
4. **Never commit secrets** - credentials live in env/vault, never in git.
5. **One component at a time** - check understanding, then move on.

## 🏗️ Build order

| Step | Component | Status |
|------|-----------|--------|
| 1 | Gateway + login (authentication) | ✅ done |
| 2 | RBAC + app registry | ⬜ |
| 3 | Audit log | ⬜ |
| 4 | Connector layer | ⬜ |
| 5 | Mount a real app (log-parser / db-health) | ⬜ |
| 6 | Plug in Airflow (orchestration) | ⬜ |

## 📂 Folders

```
platform/
├── docs/          the wiki (read this first)
│   └── decisions/ architecture decision records (ADRs)
├── gateway/       the front door: login, RBAC, routing (Step 1+)
├── connectors/    the one safe place for data-source credentials (Step 4)
└── registry/      apps.yaml - the list of mounted apps (Step 2)
```

*Status: foundation + wiki laid 2026-06-20. Gateway is next.*
