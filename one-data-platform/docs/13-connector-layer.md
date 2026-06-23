# 13 - Step 4: The Connector Layer

> Apps need data. This step builds the one safe place that holds *how to reach*
> each source - so no app ever keeps its own copy of a password.

---

## The problem in one picture

Without a connector layer:
```
app A  ──has its own copy of──▶  ORDERS_DB_PASSWORD
app B  ──has its own copy of──▶  ORDERS_DB_PASSWORD   (in a .env)
app C  ──has its own copy of──▶  ORDERS_DB_PASSWORD   (hardcoded, oops)
```
60 apps, 60 scattered copies of every credential. Rotate a password and you chase
it through every repo. One leak and it's everywhere.

With a connector layer:
```
app A ─┐
app B ─┼─ ask "orders_db" ─▶  CONNECTOR LAYER ─▶  reads secret from env ─▶ connection
app C ─┘                       (one registry)
```
One place knows the wiring. Secrets live in environment variables. Apps ask by
**name** and get a ready connection - they never see or store the credential.

## The two-part split (the key idea)

- **Wiring** (host, port, database, bucket, region) lives in `connections.yaml`.
  Safe to commit - there's nothing secret about a hostname.
- **Secrets** (passwords, keys) live in **environment variables**. The YAML only
  names *which env var* holds each secret (`password_env: ORDERS_DB_PASSWORD`).
  The secret value is never in the file, the code, or any log.

> Rule we never break: a credential never appears in source, in YAML, or in output.

## The files
- `connectors/connections.yaml` - the registry: each source's type + wiring +
  the *names* of the env vars that hold its secrets.
- `connectors/connections.py`:
  - `load_connections()` - parse the registry.
  - `get_connection(name)` - the front door. For `sqlite` it returns a **real, live
    connection** (the demo path, no secret). For remote types it resolves the secret
    from env and returns a ready config for the driver. **Missing a required secret?
    It raises immediately** - fail fast and loud, never connect half-configured.
  - `secret_status(spec)` - `n/a` / `configured` / `missing`, without revealing values.
  - `redact(spec)` / `list_status()` - safe-to-show status (wiring + secret *status*,
    never secret *values*).

## Three secrets it protects (in the demo registry)
| Source | Type | Secret comes from |
|--------|------|-------------------|
| `demo_warehouse` | sqlite (in-memory) | none - runs anywhere |
| `orders_db` | postgres | `$ORDERS_DB_PASSWORD` |
| `data_lake` | s3 | `$AWS_ACCESS_KEY_ID`, `$AWS_SECRET_ACCESS_KEY` |

## In the gateway
`GET /connections` (admin-only, and audited) shows each source's status:
```json
{"name":"orders_db","type":"postgres","secret":"missing","needs_env":["ORDERS_DB_PASSWORD"]}
```
Notice: it tells you the secret is *missing* and *which env var* to set - but never
prints a value. An analyst hitting this endpoint gets a 403 (and it's logged).

## See it yourself
```bash
cd one-data-platform/connectors
python -c "
from connections import get_connection
c = get_connection('demo_warehouse')        # real sqlite, no secret
c.execute('create table t(x int)'); c.execute('insert into t values (42)')
print(c.execute('select * from t').fetchone())
"
```
Then try a source that needs a secret you haven't set - it fails fast with the exact
env var to set, instead of silently connecting wrong.

## What you learned
- separating **wiring** (commit-safe) from **secrets** (env-only)
- **ask-by-name** access so apps never hold credentials
- **redaction** - status is shown, values never are
- **fail fast** when a required secret is missing

## What's next - Step 5: Mount a real app
We've got identity, access, audit, and now data wiring. Step 5 wires an existing
build (like `db-health` or `log-parser`) in behind the shell as a real governed app.
Explainer: `14-mount-app.md`.
