# 10 - Step 1: The Gateway + Login (authentication)

> What we built, why each piece exists, and how to run it yourself.
> Prereqs: skim [00-glossary](00-glossary.md) and [02-architecture](02-architecture.md).

---

## What this step does (and doesn't)

**Does:** lets a user log in with email + password, hands them a tamper-proof token,
and can tell who they are on every later request.

**Doesn't yet:** decide what they're *allowed* to do (that's RBAC, Step 2) or route
them to apps (Step 2). We're only answering **"who are you?"** - authentication.

> One step, one concept. We resist doing more.

## The three files (read in this order)

### 1. `gateway/auth.py` - the security primitives
Hand-written so nothing is a black box. Two halves:

**Passwords.** We never store your real password. We store a *hash* - run the password
through a one-way "meat grinder" (`pbkdf2`, 100,000 rounds) mixed with a random *salt*.
- *Why a hash?* If the file leaks, attackers still don't have passwords.
- *Why a salt?* So two people with the same password get different hashes, and
  attackers can't use precomputed tables.
- To check a login we grind the typed password the same way and compare. We compare
  with `hmac.compare_digest` so the *timing* of the comparison can't leak hints.

**Tokens (the wristband).** After login we issue `payload.signature`:
- `payload` = readable JSON `{"email","role","exp"}` (base64).
- `signature` = HMAC of the payload using a secret `PLATFORM_SECRET`.
- Anyone can *read* the payload, but **can't change it** without the secret - the
  signature wouldn't match. Change one character and `verify_token` returns `None`.
- `exp` is an expiry timestamp - the wristband dies after 8 hours.

> This IS what a real JWT does. We built the 30-line version so the magic is gone.

### 2. `gateway/users.py` - the user store
A JSON file (`users.json`) mapping email → `{password_hash, role}`. Seeds four demo
accounts on first run (admin / analyst / data_scientist / ai_engineer). It stores
**only hashes**, never plain passwords. Later we swap the JSON for SQLite/Postgres
without touching the gateway - that's why it hides behind `authenticate()`.

### 3. `gateway/app.py` - the gateway (front door)
A small FastAPI service:
- `GET /` - workspace if logged in, else the login page.
- `POST /login` - check credentials → put a signed token in an **HttpOnly cookie**.
- `GET /me` - "who am I?" (401 if not logged in). Proves the token round-trip.
- `POST /logout` - clears the cookie.

Two security choices worth noticing:
- **Same error for bad email and bad password** ("invalid email or password") - never
  tell an attacker which half was right.
- **HttpOnly cookie** - JavaScript can't read the token, blocking a common theft trick.

## How a login flows (tie it to the architecture doc)

```
browser ──login(email,pw)──▶ /login ──authenticate()──▶ users.json
                                  │  (password verified)
                                  ▼
                          create_token(email,role)
                                  │
   browser ◀── Set-Cookie: platform_token=<payload.signature> ──┘

later:
browser ──(cookie)──▶ /me ──verify_token()──▶ {email, role}   ✅ who you are
```

## Run it yourself

```bash
cd one-data-platform/gateway
pip install -r requirements.txt
uvicorn app:app --reload
# open http://localhost:8000
```
Log in with `phoebe@team.io` / `admin123` (or any demo account on the login page).
Try a wrong password. Open `/me` in another tab. Log out and watch `/me` go 401.

> Heads-up: it prints `[WARN] PLATFORM_SECRET is the dev default`. That's intentional -
> before any real use, set a real secret: `export PLATFORM_SECRET="something-long-random"`.

## What you learned this step
- **authentication** vs the permission check that comes next
- **password hashing** (hash, salt, why)
- **tokens / JWT** - signed, readable, expiring wristbands
- **HttpOnly cookies** and not leaking which credential was wrong

## What's next - Step 2: RBAC + app registry
Now that we know *who* you are, we decide *what you can open*. We'll add `apps.yaml`
(the directory board) listing each app and the role it needs, and the gateway will
show you only the apps your role allows. Explainer will be `11-rbac-registry.md`.
