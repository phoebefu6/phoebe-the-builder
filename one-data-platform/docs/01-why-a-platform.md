# 01 - Why a platform at all?

> The vision in plain words, and the reasoning behind every big choice.
> Read this when you forget *why* we're doing this.

---

## The one-sentence vision

**A single, governed workspace where my whole team - data analysts, data engineers,
data scientists, and AI engineers - can connect to any data source, process it,
explore it, and build & share dashboards, models, and AI products, with me
controlling who can access what.**

## The problem we're actually solving

Today the team's work is scattered:
- The analyst has SQL in one tool, dashboards in another.
- The engineer's pipelines run who-knows-where.
- The data scientist trains models on their laptop; nobody else can find them.
- The AI engineer's LLM app has API keys pasted in a notebook.
- **Nobody can answer "who has access to the customer data?" with confidence.**

Each of our 60 mini-tools solves a *slice* of this. But 60 separate tools with 60
separate logins is not a solution - it's 60 new problems. The platform's job is to
make them **one coherent, governed place.**

## What we are NOT building (and why)

We are **not** building Databricks or Snowflake. Here's the honest reasoning:

- Those took **thousands of engineers and 5-10 years.** We are one person plus Claude.
- Their core (a distributed compute engine) is a commodity now - **open source already
  does it** (Airflow, DuckDB, Trino, Spark). Rebuilding it would be wasted effort with
  no payoff.
- Competing on "we have a SQL engine too" is a losing game we'd never win.

> **Decision:** leverage open source for the heavy lifting; build only the thin layer
> that's missing. See [ADR-0001](decisions/adr-0001-governed-shell-not-databricks.md).

## What we ARE building (the missing thin layer)

The piece that *isn't* solved by open source, and that every enterprise actually pays
for: **the governance shell** (a.k.a. the control plane). It provides:

1. **One front door** - a single login for the whole team (the *gateway*).
2. **Role-based access** - the analyst sees analyst things; only you see admin things
   (*RBAC*).
3. **One connector layer** - data-source credentials live in one guarded place, not
   pasted across 60 notebooks.
4. **An audit log** - a permanent record of who accessed which dashboard, dataset,
   model, or LLM.
5. **An app registry** - a simple list that mounts every tool in the catalog as a governed
   "app."

That's it. Five things. Everything else, we plug in.

## Why this is sellable (not just a portfolio)

A buyer doesn't pay for "another SQL tool." They pay for **"my team's data work, in one
place, and I can prove to auditors exactly who touched what."** The governance shell IS
that. The catalog is the proof it works and the range of what's possible.

Go-to-market plan (from the mentor roundtable):
- Pick **one** real client problem (likely "governed analytics" for the analyst persona).
- Deliver it on the shell, with a one-page ROI case ("saves your team N hours/week,
  here's the audit trail").
- **Land, then expand** into ML and AI apps as the client is ready.

## How the tools fit

Each daily build stops being a standalone toy and becomes a **registered app** on the
shell. Going forward we write each tool's core logic UI-free and reusable, so mounting
it later is a 20-minute job, not a rewrite. (We already did this with `log-parser` -
its `parser.py` is pure logic, ready to become an "Observability" app.)

## Our build principles

1. **Understand before we use.** Every concept gets a glossary entry and an explainer
   before it appears in code.
2. **Smallest real version first.** We hand-write a 40-line version we can see inside,
   then swap in the industrial OSS version once we get it.
3. **Log every decision.** Each meaningful choice becomes an ADR in `decisions/`.
4. **Never commit secrets.** Credentials live in env/vault, never in git.
5. **Slow is smooth.** One component at a time, checked for understanding, then next.

---

*Next: [02 - Architecture](02-architecture.md) - the picture of how the pieces fit.*
