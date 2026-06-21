# ADR-0001: Build a governed shell, not a Databricks clone

- **Status:** Accepted
- **Date:** 2026-06-20
- **Deciders:** Phoebe (+ Claude, + 12-mentor roundtable)

## Context
Phoebe has 16 of 60 daily data/AI mini-tools built. She wants them to converge into
ONE enterprise product she can sell to a client - a unified place where her team
(data analysts, data engineers, data scientists, AI engineers) can connect to data,
process it, do EDA, and build & share dashboards, models, and AI products, with
governed access. The instinct was "combine all 60 into one Databricks-style platform."
We needed to decide what to actually build before writing code.

## Options we considered
1. **Clone Databricks/Snowflake** - build our own unified compute + storage + ML
   platform from scratch. Pro: grand vision. Con: thousands of engineer-years; the
   core is a commodity open source already solved; we'd never win on those axes.
2. **Leave 60 tools separate** - just polish each standalone. Pro: simple. Con: 60
   logins, no governance, not a product, not sellable as "a platform."
3. **Build a thin governed shell + mount OSS** - write only the missing governance
   layer (login, roles, connectors, audit, app registry); plug in open source
   (Airflow, DuckDB, Trino) for the heavy compute; mount the 60 tools as governed
   apps. Pro: small, defensible, sellable, leverages existing work. Con: requires
   discipline to not over-build.

## Decision
**Option 3.** Build a thin **governance shell (control plane)** ourselves and
**leverage open source** for all heavy compute and orchestration. The 60 tools become
the shell's app catalog.

## Why (the reasoning)
- The distributed-compute core is a **commodity** - rebuilding it is wasted effort
  with no payoff (LeCun, Jensen).
- The part enterprises actually pay for and that's hard to copy is **governance**:
  one login, role-based access, and an audit trail over data/dashboards/models/LLMs
  (Sigal, Zhamak).
- Building only the thin layer is **achievable by one person + Claude**, and turns
  every existing and future daily build into a platform feature (Ng, Karpathy).
- It supports a **land-and-expand** sale: solve one client's one problem on the shell,
  then grow (Kai-Fu Lee, Satya, Chrissie).

## Consequences
- **Good:** small surface area to build; leverages 16 tools already done; sellable
  governance story; each new daily build plugs in for ~20 min.
- **Cost / risk:** we depend on open-source tools (Airflow etc.) - we must learn to
  govern and integrate them, not control them. We must resist scope creep into
  rebuilding compute.
- **Revisit when:** a client's needs genuinely can't be met by mounting OSS, or the
  shell itself needs to become a distributed system (a good problem to have).

## Glossary terms introduced
control plane, governance shell, RBAC, audit log, app registry, orchestration,
open source (OSS), Airflow, DuckDB, Trino - all defined in `00-glossary.md`.
