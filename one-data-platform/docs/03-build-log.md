# 03 - Build Log (design-thinking journal)

> A running diary of what we built, what we learned, and what's next. Newest entry
> at the top. This is where Phoebe's questions and the "aha" moments get captured -
> not just the answers.

---

## 2026-06-20 - Step 2: RBAC + app registry (authorization)

**What we did:** Added the "what can you open?" layer. `registry/apps.yaml` (directory
board: 8 apps, each with a required_role + status), `gateway/rbac.py` (role ladder
analyst<data_scientist<ai_engineer<admin + one `can_access()`), `gateway/registry.py`
(loads YAML, `visible_apps(role)`). Gateway gained `/apps` and `/open/{slug}` with a
3-check governance gate (logged in? exists? role allowed?). Workspace now renders a
role-filtered app grid: openable / planned / 🔒 locked. Tested all 4 roles - analyst
4/8, DS 6/8, ai_eng 7/8, admin 8/8; analyst→LLM = 403, admin→LLM = 200. Live boot
confirmed the grid + locked cards.

**What Phoebe was learning:** authorization vs authentication; RBAC as role levels;
why the access check must live at the *route* (`/open/...` returns 403 even if typed
directly), not just hidden buttons; a registry as single source of truth for apps.
Explainer: `11-rbac-registry.md`.

**Key decisions logged:**
- Roles as a linear ladder (simplification) behind one `can_access()` - can graduate
  to explicit permission sets later without touching the gateway.
- App registry is YAML so mounting a new daily build = a few lines, no code change.
- Enforce at the gate, not the menu (URL-level 403).

**Mentor input:** Zhamak (registry as source of truth, domain apps declare their own
required role), Sigal (governance enforced at the gate, provable "analyst cannot open
the LLM tool").

**Open questions to revisit:**
- Linear roles vs permission sets - revisit when a real app needs a permission that
  doesn't fit the ladder.
- Per-app data-source scoping (which datasets, not just which app) - later.

**Next step:** Step 3 - **Audit log**: every login and every `/open` writes an
append-only "who did what, when" record. Explainer `12-audit-log.md`.

---

## 2026-06-20 - Step 1: Gateway + login (authentication)

**What we did:** Built the front door. Three files in `gateway/`: `auth.py`
(hand-written password hashing + signed tokens, stdlib only), `users.py` (JSON user
store, seeds 4 demo accounts, stores hashes only), `app.py` (FastAPI: `/login`,
`/me`, `/logout`, login + workspace pages). Tested end-to-end: wrong password → 401,
good login → token cookie → `/me` knows who you are, logout → 401. Live uvicorn boot
confirmed.

**What Phoebe was learning:** authentication vs authorization; password *hashing*
(hash + salt + why); *tokens/JWT* as signed, readable, expiring wristbands; HttpOnly
cookies; not leaking which credential was wrong. Explainer: `10-gateway-login.md`.

**Key decisions logged:**
- Hand-write the security primitives first (Karpathy principle) rather than import
  bcrypt/PyJWT - swap to those later once understood.
- User store is a JSON file behind `authenticate()` so we can swap to SQLite/Postgres
  later without touching the gateway.
- `users.json` is gitignored (self-seeds; never commit hashes). `PLATFORM_SECRET`
  comes from env; app warns loudly on the dev default.

**Mentor input:** Karpathy (smallest readable version), Sigal (don't leak which
credential was wrong; warn on default secret).

**Open questions to revisit:**
- Move from JSON user store to SQLite - when? (After Step 5, when we mount a real app.)
- Real token library (PyJWT) swap - after the concept is solid.

**Next step:** Step 2 - **RBAC + app registry** (`apps.yaml`): the gateway shows you
only the apps your role can open. Explainer `11-rbac-registry.md`.

---

## 2026-06-20 - Foundation: the wiki itself

**What we did:** Set up `one-data-platform/` and the docs/wiki. Wrote the glossary, the "why"
doc, the architecture picture, and ADR-0001 (governed shell, not Databricks clone).
No platform code yet - on purpose. We're laying the map before driving.

**What Phoebe was learning:** the vocabulary. Coming from a data background, terms
like *control plane*, *gateway*, *RBAC*, *audit log*, *connector layer* were
unfamiliar. The glossary translates each into a data analogy (RBAC = database roles,
token = warehouse session, audit log = immutable query history).

**Key decisions logged:**
- [ADR-0001](decisions/adr-0001-governed-shell-not-databricks.md) - build the thin
  governance shell, mount open source for compute.

**Mentor input:** Cassie (log every decision as an ADR), Karpathy (write the smallest
readable version ourselves before importing frameworks), Ng (learn in sequence, check
understanding at each step), Brené ("I'm a data expert learning platform engineering"
is a strength).

**Open questions to revisit:**
- Which client / wedge problem do we target first? (Leaning: governed analytics for
  the analyst persona - to be decided.)
- Local file-based store for the MVP, or a real database from the start? (Leaning:
  start file-based / SQLite so it's readable, swap later.)

**Next step:** Step 1 - the **gateway with login** (authentication). We'll write the
smallest real login Phoebe can read top to bottom, with a companion explainer doc
`10-gateway-login.md`.

---

## Template for future entries

```
## YYYY-MM-DD - <component / step>

**What we did:**
**What Phoebe was learning:**
**Key decisions logged:** (link ADRs)
**Mentor input:**
**Open questions to revisit:**
**Next step:**
```
