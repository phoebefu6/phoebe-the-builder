# 11 - Step 2: RBAC + App Registry (authorization)

> Now that the gateway knows *who* you are (Step 1), this step decides *what you can
> open*. This is the governance you're actually selling.

---

## The two new ideas

### 1. RBAC - Role-Based Access Control
"What are you allowed to do?" We answer it with **roles as levels**:

```
analyst (10)  <  data_scientist (20)  <  ai_engineer (30)  <  admin (99)
```

Each higher role includes everything below it. So a data scientist can open analyst
apps (a superset), and admin can open everything (admin governs all). An app declares
the **minimum** role it needs; you pass if your level is at least that high.

> This is a teaching simplification. Real systems often use explicit permission *sets*
> ("can_train_models", "can_deploy_llm") instead of one ladder. We can graduate to that
> later **without changing the gateway**, because every check goes through one function:
> `can_access(user_role, required_role)` in `gateway/rbac.py`.

### 2. The App Registry
A single file, `registry/apps.yaml` - the "directory board in the lobby." Each app
lists its `slug`, `name`, `required_role`, `status` (live/planned), and `category`.
The gateway reads this to decide what each user sees. **Mounting a new daily build =
adding a few lines here.** No gateway code changes.

## The files (read in order)

| File | Job |
|------|-----|
| `registry/apps.yaml` | The directory board - every app + the role it needs |
| `gateway/rbac.py` | The role ladder + the one `can_access()` check |
| `gateway/registry.py` | Loads `apps.yaml`; `visible_apps(role)` tags each app allowed/locked |
| `gateway/app.py` | New routes `/apps` and `/open/{slug}`; workspace now renders the grid |

## The governance gate (the important bit)

When you click an app, `GET /open/{slug}` runs **three checks** before letting you in:

1. **Logged in?** No → 401.
2. **Does the app exist?** No → 404.
3. **Does your role meet the app's `required_role`?** No → **403 Access denied.**

Only if all three pass do you get in. That 403 *is* the product - it's how you prove to
a client "an analyst physically cannot open the LLM tool or the model registry."

## What each role sees (tested)

| Role | Apps openable | Example locked |
|------|---------------|----------------|
| analyst | 4 / 8 | 🔒 Model Trainer (needs data_scientist) |
| data_scientist | 6 / 8 | 🔒 LLM Playground (needs ai_engineer) |
| ai_engineer | 7 / 8 | 🔒 User & Access Admin (needs admin) |
| admin (you) | 8 / 8 | - governs everything |

In the workspace, openable apps are blue cards; planned ones are greyed; apps your role
can't reach show as 🔒 locked with the role they'd need.

## Run it yourself

```bash
cd one-data-platform/gateway
pip install -r requirements.txt
uvicorn app:app --reload
```
- Log in as `ana@team.io` / `analyst123` → you see Observability + Analytics apps;
  ML and AI apps are locked.
- Visit `/open/llm-playground` directly → **403** (you can't sneak past the menu).
- Log in as `phoebe@team.io` / `admin123` → everything is open.

> Note: the URL-level check matters. Even if a sneaky user *types* `/open/llm-playground`
> instead of clicking, the gate still says 403. Governance lives at the gate, not the menu.

## What you learned
- **authorization** (vs authentication from Step 1)
- **RBAC** as role levels, behind one `can_access()` function
- a **registry** as the single source of truth for "what apps exist + who may open them"
- enforcing access at the **route**, not just hiding buttons in the UI

## What's next - Step 3: Audit log
Right now access checks happen but vanish. Step 3 adds the **audit log**: every login
and every `/open` writes an append-only record - "who did what, when." That permanent
trail is the #1 thing a regulated enterprise buyer asks for. Explainer: `12-audit-log.md`.
