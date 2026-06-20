# 02 - Architecture (the picture)

> How the pieces fit, drawn for a data person. No prior platform knowledge needed.

---

## The building analogy

Picture an **office building** for your data team:

```
                          ┌─────────────────────────────────────────┐
   You / your team  ───▶  │            THE GATEWAY (front door)       │
   (browser)              │   1. Who are you?      (authentication)   │
                          │   2. What's your role? (authorization)    │
                          │   3. Write it down.    (audit log)        │
                          └───────────────┬───────────────────────────┘
                                          │  (only after the badge check)
                  ┌───────────────────────┼───────────────────────────┐
                  ▼                        ▼                           ▼
          ┌───────────────┐       ┌───────────────┐          ┌────────────────┐
          │   APP: SQL    │       │ APP: Dashboard│   ...    │  APP: LLM tool │
          │  (analyst+)   │       │  (analyst+)   │          │ (ai_engineer+) │
          └───────┬───────┘       └───────┬───────┘          └────────┬───────┘
                  │                        │                          │
                  └────────────┬───────────┴──────────────┬──────────┘
                               ▼                           ▼
                   ┌───────────────────────┐   ┌───────────────────────┐
                   │  CONNECTOR LAYER      │   │  ORCHESTRATION (OSS)  │
                   │  (one safe place for  │   │   Apache Airflow       │
                   │   data-source creds)  │   │   runs the pipelines   │
                   └───────────┬───────────┘   └───────────────────────┘
                               ▼
            Postgres · S3 · BigQuery · Snowflake · files · APIs
```

- **The gateway** is the front desk. Everyone enters here. It does the badge check
  (authn), the permission check (authz), and writes the visit into the diary (audit).
- **The apps** are the offices - our 60 tools. Each office has a sign saying which
  role may enter ("Data Scientists only").
- **The connector layer** is the locked key cabinet - the one place that holds the
  keys to each data source, so no office keeps its own copy.
- **Orchestration (Airflow)** is the building's automated night crew - it runs the
  scheduled pipelines. We don't build it; we plug it in and govern it.

## The three layers (this is the whole platform)

### Layer 1 - The Shell (we build this)
The gateway + login + RBAC + audit log + app registry. This is the **only** part
that's genuinely ours and hard to copy. Small on purpose: a few hundred lines.

### Layer 2 - The Connector Layer (we build a thin version)
One module that, given a name like `"orders_db"`, returns a ready-to-use connection,
reading the secret from a guarded place (env var / vault). Tools never hold their own
credentials. We'll start with a tiny hand-written version, then can grow it.

### Layer 3 - The Apps + Engines (we mount, mostly)
Our 60 tools register as apps. The heavy compute engines (SQL via DuckDB/Trino,
orchestration via Airflow, model serving) are **open source we plug in**, not code we
write.

> Memory hook: **Shell = ours. Connectors = thin & ours. Apps/engines = mostly rented.**

## What a single request looks like (step by step)

1. You open the platform in your browser and **log in** → gateway checks email +
   password (**authentication**).
2. Gateway gives your browser a **token** (the wristband) so you stay logged in.
3. You click **"Churn Model"** → browser sends the request + token to the gateway.
4. Gateway reads the token (you're Phoebe), checks the **app registry** (Churn Model
   needs `data_scientist`), checks your **role** (you're `admin`, which includes it) →
   **allowed** (**authorization**).
5. Gateway writes **"Phoebe opened Churn Model at 10:04"** to the **audit log**.
6. Gateway forwards you to the app. If the app needs the orders database, it asks the
   **connector layer**, which hands back a connection using a secret you never see.

Every box in that flow is a glossary term you now know. That's the whole system.

## Build order (so we never bite off too much)

| Step | Component | What you'll learn | Status |
|------|-----------|-------------------|--------|
| 1 | **Gateway + login (authn)** | sessions, tokens, password checks | ⬜ next |
| 2 | **RBAC + app registry** | roles, permissions, `apps.yaml` | ⬜ |
| 3 | **Audit log** | append-only records, why immutability matters | ⬜ |
| 4 | **Connector layer** | secrets, one source of credentials | ⬜ |
| 5 | **Mount a real app** | wrap `log-parser` / `db-health` behind the shell | ⬜ |
| 6 | **Plug in Airflow** | orchestration, DAGs, governing OSS | ⬜ |

We do them **in order**, one at a time, each with its own explainer doc and a check
that you understand it before moving on.

---

*Next up: Step 1 - the gateway. We'll write the smallest real login you can read
top-to-bottom. See the [build log](03-build-log.md) for the running journal.*
