# 12 - Step 3: The Audit Log

> Steps 1-2 decide *who* you are and *what* you can open. This step makes it
> **provable** - every login and every access is recorded, permanently.

---

## Why this is the most valuable piece for a buyer

A regulated enterprise doesn't ask "can you do access control?" They ask **"can you
prove who accessed the customer data last Tuesday?"** The audit log is that proof. It's
often the single feature that gets a platform approved or rejected. We build it third,
right after access control, on purpose.

## The one idea: append-only

The audit log is **append-only** (a.k.a. immutable): we only ever *add* lines, never
edit or delete them. That's what makes it trustworthy - if entries could be changed,
they'd be worthless as evidence. Think of it like an **immutable query-history table**:
write-once, read-many.

We store one JSON object per line (JSONL) in `gateway/audit_log.jsonl`:
```json
{"ts":"2026-06-21T02:55:47+00:00","actor":"ana@team.io","role":"analyst","action":"open_app","target":"llm-playground","status":"denied"}
```
Simple, greppable, and easy to ship to a real log store (S3, Datadog, a SIEM) later -
the callers only use `log_event` and `read_events`, so the storage can change underneath.

## What gets recorded

| When | action | status |
|------|--------|--------|
| Login succeeds | `login` | `success` |
| Login fails | `login` | `denied` ← failed logins matter most to auditors |
| App opened (allowed) | `open_app` | `granted` |
| App blocked by role | `open_app` | `denied` |
| Someone views the audit log | `view_audit` | `denied` if not admin |
| Logout | `logout` | `ok` |

Every record has **who** (actor + role), **what** (action), **on what** (target app),
**when** (UTC timestamp), and the **outcome** (status).

## The files
- `gateway/audit.py` - `log_event(...)` appends one record; `read_events(...)` reads them
  back (newest first, optional filters); `summary()` gives quick counts.
- `gateway/app.py` - calls `log_event` on login, open, logout; adds `GET /audit`
  (**admin-only** - reading the trail is itself a governed, logged action).

## See it yourself

```bash
cd one-data-platform/gateway
uvicorn app:app --reload
```
1. Log in as `ana@team.io` / `analyst123` (analyst).
2. Open Log Parser (allowed), then try `/open/llm-playground` (denied → 403).
3. Log out. Log back in as `phoebe@team.io` / `admin123` (admin).
4. Click **🛡️ View audit log** in the workspace, or hit `GET /audit`.

You'll see every step you just took - including the denied attempts.

## Security choices worth noticing
- **Failed logins are logged**, not just successes - that's how you spot brute-force.
- **Reading the audit log is admin-only and itself audited** - a non-admin's attempt to
  peek is recorded as `view_audit / denied`.
- The log file is **gitignored** - it's runtime evidence, never committed.

## What you learned
- **append-only / immutability** and why it equals trust
- **JSONL** as a simple, portable log format
- recording **failed** actions, not just successful ones
- treating "who can read the log" as a governed action too

## What's next - Step 4: Connector layer
Apps will need data. Step 4 builds the one safe place that holds data-source credentials,
so no app keeps its own copy of a password. Explainer: `13-connector-layer.md`.
