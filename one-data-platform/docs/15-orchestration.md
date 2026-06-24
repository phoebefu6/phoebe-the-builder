# 15 - Step 6: Orchestration (Airflow)

> The last spine piece. Govern *scheduled* work - pipelines that run on a clock -
> under the same roof, without rebuilding a scheduler.

---

## The decision that shapes this step

ADR-0001 said: **leverage open source, don't rebuild commodities.** Orchestration is
the textbook case. Writing your own reliable cron/DAG engine (retries, backfills,
dependencies, schedules) is years of work that **Apache Airflow** already does, for free,
battle-tested at huge scale. So we do **not** build a scheduler. We plug Airflow in and
make the platform **govern and surface** it.

## The same pattern you've seen twice now

Just like the connector layer (real DB if configured, else sqlite) and mounting
(real app behind the shell), orchestration has two backends behind one interface:

```
get_orchestrator()
   ├─ AIRFLOW_URL set?  ──▶  AirflowOrchestrator  (talks to Airflow's REST API)
   └─ otherwise         ──▶  LocalOrchestrator    (simulation from pipelines.yaml)
```

Apps and the gateway only call the interface - `list_dags()`, `get_dag()`, `trigger()` -
so the platform runs anywhere today and points at a real Airflow the moment one exists.

## The files
- `orchestration/pipelines.yaml` - the pipeline catalog (DAG id, cron schedule, owner,
  tasks). In production these *are* Airflow DAGs.
- `orchestration/orchestrator.py` - the abstraction + both backends.
- `apps/pipelines.py` - a **mounted app** (Step 5 contract) that surfaces the DAGs inside
  the shell: schedule, owner, last-run status.

## Governed, like everything else
The Pipelines app is registered with `required_role: data_scientist`. So:
- a **data scientist / ai engineer / admin** can open it,
- an **analyst** gets a **403** (and it's audited).

Triggering a run goes through the orchestrator interface as a governed, auditable action -
the platform decides *who* may run *what*, Airflow does the running.

## See it yourself
```bash
cd one-data-platform/gateway
uvicorn app:app --reload
```
1. Log in as `sam@team.io` / `scientist123` (data scientist).
2. Open **Pipelines (Orchestration)** - you'll see the DAGs, their cron schedules,
   owners, and last-run status, with the backend shown as `local-sim`.
3. Log in as `ana@team.io` / `analyst123` and try the same - **403**.
4. To wire a real Airflow later: set `AIRFLOW_URL` (+ `AIRFLOW_USER`/`AIRFLOW_PASSWORD`)
   and the exact same app talks to it via the REST API.

## What you learned
- **don't rebuild commodities** - govern Airflow, don't reimplement it
- the **configured-or-fallback** pattern, applied a third time (connectors, mounting, now orchestration)
- scheduled work is governed by the *same* identity/RBAC/audit spine as interactive apps

## 🎉 The spine is complete (6/6)
You now have the whole governance shell:

| # | Module | Gives you |
|---|--------|-----------|
| 1 | Gateway + login | who you are (authentication) |
| 2 | RBAC + registry | what you can open (authorization) |
| 3 | Audit log | provable who-did-what-when |
| 4 | Connector layer | one safe home for credentials |
| 5 | Mounted apps | real builds running behind the shell |
| 6 | Orchestration | governed scheduled pipelines (Airflow) |

That's the sellable product from ADR-0001: a thin governed control plane that hosts the
app catalog. From here it's **breadth** (mount more of the 60 builds - one line each) and
**hardening** (real DB for users, a secrets manager, deploy the shell to Render).
