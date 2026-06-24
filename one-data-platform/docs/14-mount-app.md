# 14 - Step 5: Mount a real app

> The payoff. Everything so far - identity, access, audit, data wiring - now pays
> off: a real build runs *inside* the shell as a governed app.

---

## What "mounting" means here

The Day-10 `db-health` dashboard used to be a standalone Streamlit app on its own
port, with its own (none) access control. Now it runs **inside the platform**: you
log in once, the shell checks your role, logs the access, hands the app a governed
data connection, and renders it - all behind one front door.

```
click "Database Health" ─▶ gateway: authn ✓  authz ✓  audit ✓
                                   │
                                   ▼
                         apps/db_health.render(ctx)  ──ctx.get_connection──▶ connector layer
                                   │
                                   ▼
                         dashboard HTML, served inside the shell
```

## The mount contract (how any build plugs in)

A mounted app is just a Python module that exposes one function:

```python
def render(ctx: dict) -> str:
    # ctx = {"user": {...}, "get_connection": <connector-layer function>}
    return "<html fragment>"
```

The shell does everything around it - login, the role gate, the audit entry, the
page chrome. The app only produces content. To mount a new build, you add **one line**
to the gateway's registry:

```python
MOUNTED_APPS = {
    "db-health": db_health.render,
    # "log-parser": log_parser.render,   # <- next build mounts here
}
```

That one line is the whole point of the last four steps: because the shell already
handles auth/RBAC/audit/connections, mounting build #16..#60 is trivial.

## What we mounted
`apps/db_health.py` - the Day-10 health dashboard, core logic vendored in (tiny,
UI-free). Its `render(ctx)`:
1. scores simulated DB metrics into a RAG table + health score,
2. **uses the connector layer** - asks `ctx["get_connection"]("demo_warehouse")` and
   runs a query, proving an app gets data *without holding any credential*,
3. returns an HTML fragment the shell wraps in chrome.

## See it yourself
```bash
cd one-data-platform/gateway
uvicorn app:app --reload
```
1. Log in as `phoebe@team.io` / `admin123`.
2. Click **Database Health Dashboard** (Observability) - the real dashboard renders
   inside the shell, with a "🔌 Connected to `demo_warehouse`" note.
3. Check `/audit` - your `open_app / db-health / granted` is recorded.
4. The whole chain ran: authentication → authorization → audit → connector → app.

## Why this is the moment that proves the thesis
Back in ADR-0001 we decided to build a thin governance shell and let it host the apps.
Step 5 shows it works: a real build, governed, with one line to mount it. The remaining
59 builds become app-catalog entries, not rewrites.

## What you learned
- the **mount contract** (`render(ctx)`) that decouples an app from the shell
- how auth + RBAC + audit + connectors compose into one governed request
- why a thin shell + a clear contract beats a monolith

## What's next - Step 6: Orchestration (Airflow)
The last piece: govern *scheduled* work, not just interactive apps. Step 6 plugs in
Apache Airflow (open source) so pipelines run on a schedule under the same roof.
Explainer: `15-orchestration.md`.
