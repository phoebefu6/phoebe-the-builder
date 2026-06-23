from __future__ import annotations

"""The Gateway - the single front door to the platform.

Step 1 scope = AUTHENTICATION only ("who are you?"). You can:
  - see a login page
  - log in (email + password)  -> get a token in a cookie
  - see who you are at /me      -> proves the token works
  - log out                     -> cookie cleared

Roles (RBAC) and routing to apps come in Step 2. We keep this tiny on purpose.

Run:  cd platform/gateway && uvicorn app:app --reload
Then open http://localhost:8000
"""

from typing import Dict, Optional

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import sys
from pathlib import Path as _Path

from audit import log_event, read_events
from audit import summary as audit_summary
from auth import create_token, secret_is_default, verify_token
from rbac import can_access
from registry import get_app, visible_apps
from users import authenticate, seed_if_empty

# The connector layer lives in ../connectors; make it importable.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "connectors"))
from connections import list_status as connection_status  # noqa: E402

COOKIE = "platform_token"

app = FastAPI(title="Platform Gateway", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    seed_if_empty()  # create demo users on first run
    is_default, msg = secret_is_default()
    if is_default:
        print(f"[WARN] {msg}")


# ── The "who is calling?" helper ──────────────────────────────────────────────
# Many routes need to know the current user. This reads the cookie, verifies the
# token, and returns the payload (or None). In FastAPI we express it as a
# dependency so any route can just ask for `user`.
def current_user(request: Request) -> Optional[Dict[str, object]]:
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    return verify_token(token)


class LoginRequest(BaseModel):
    email: str
    password: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/login")
def login(req: LoginRequest, response: Response) -> JSONResponse:
    """Check credentials. On success, put a signed token in an HttpOnly cookie."""
    email = req.email.strip().lower()
    role = authenticate(req.email, req.password)
    if role is None:
        # Record the failed attempt - failed logins are exactly what auditors want.
        log_event(email, "login", status="denied")
        # Same message for "no such user" and "wrong password" - never tell an
        # attacker which half they got right.
        return JSONResponse({"ok": False, "error": "invalid email or password"}, status_code=401)

    log_event(email, "login", status="success", role=role)
    token = create_token(email=email, role=role)
    resp = JSONResponse({"ok": True, "email": email, "role": role})
    # HttpOnly = JavaScript can't read the cookie, which blocks a whole class of
    # token-theft attacks. SameSite=Lax limits cross-site sending.
    resp.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=8 * 3600)
    return resp


@app.post("/logout")
def logout(user: Optional[Dict[str, object]] = Depends(current_user)) -> JSONResponse:
    if user is not None:
        log_event(str(user["email"]), "logout", role=str(user["role"]))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE)
    return resp


@app.get("/audit")
def audit(limit: int = 50, user: Optional[Dict[str, object]] = Depends(current_user)) -> JSONResponse:
    """Admin-only: the audit trail. Who did what, when. Append-only, read here."""
    if user is None:
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    if user["role"] != "admin":
        # Reading the audit log is itself a governed action.
        log_event(str(user["email"]), "view_audit", status="denied", role=str(user["role"]))
        return JSONResponse({"error": "audit log is admin-only"}, status_code=403)
    return JSONResponse({"summary": audit_summary(), "events": read_events(limit=limit)})


@app.get("/connections")
def connections(user: Optional[Dict[str, object]] = Depends(current_user)) -> JSONResponse:
    """Admin-only: data-source connection status. Shows wiring + secret status,
    NEVER secret values (the connector layer redacts them)."""
    if user is None:
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    if user["role"] != "admin":
        log_event(str(user["email"]), "view_connections", status="denied", role=str(user["role"]))
        return JSONResponse({"error": "connections are admin-only"}, status_code=403)
    log_event(str(user["email"]), "view_connections", status="ok", role="admin")
    return JSONResponse({"connections": connection_status()})


@app.get("/me")
def me(user: Optional[Dict[str, object]] = Depends(current_user)) -> JSONResponse:
    """Whoami. 401 if not logged in - proves the token round-trip works."""
    if user is None:
        return JSONResponse({"authenticated": False}, status_code=401)
    return JSONResponse({"authenticated": True, "email": user["email"], "role": user["role"]})


@app.get("/apps")
def list_apps(user: Optional[Dict[str, object]] = Depends(current_user)) -> JSONResponse:
    """JSON list of apps this user's role can open (each tagged allowed/locked)."""
    if user is None:
        return JSONResponse({"authenticated": False}, status_code=401)
    return JSONResponse({"role": user["role"], "apps": visible_apps(str(user["role"]))})


@app.get("/open/{slug}", response_class=HTMLResponse)
def open_app(slug: str, user: Optional[Dict[str, object]] = Depends(current_user)):
    """Open an app - the governance gate. Checks login, existence, and role."""
    if user is None:
        return HTMLResponse("Please log in first.", status_code=401)

    app_entry = get_app(slug)
    if app_entry is None:
        return HTMLResponse(f"No app named '{slug}'.", status_code=404)

    # THE governance check: does this role meet the app's required role?
    if not can_access(str(user["role"]), str(app_entry.get("required_role", "admin"))):
        log_event(str(user["email"]), "open_app", target=slug, status="denied", role=str(user["role"]))
        return HTMLResponse(
            f"<h1>🚫 403 - Access denied</h1><p>'{app_entry['name']}' requires role "
            f"<b>{app_entry['required_role']}</b>. You are <b>{user['role']}</b>.</p>"
            f"<p><a href='/'>← back to workspace</a></p>",
            status_code=403,
        )

    # Allowed - record the access, then (Step 5) proxy to the real app. For now, a stub.
    log_event(str(user["email"]), "open_app", target=slug, status="granted", role=str(user["role"]))
    if app_entry.get("status") != "live":
        body = f"<p>✅ You're allowed in, but <b>{app_entry['name']}</b> is still on the roadmap (planned).</p>"
    else:
        body = (f"<p>✅ Access granted to <b>{app_entry['name']}</b>.</p>"
                f"<p><i>{app_entry['description']}</i></p>"
                f"<p>(Step 5 will mount the real app here.)</p>")
    return HTMLResponse(
        f"<!doctype html><html><body style='font-family:system-ui;max-width:640px;margin:3rem auto'>"
        f"<h1>{app_entry['name']}</h1>{body}<p><a href='/'>← back to workspace</a></p></body></html>"
    )


@app.get("/", response_class=HTMLResponse)
def home(user: Optional[Dict[str, object]] = Depends(current_user)) -> str:
    """Show the workspace if logged in, otherwise the login page."""
    if user is not None:
        return _workspace_html(str(user["email"]), str(user["role"]))
    return _login_html()


# ── Tiny inline UI (no framework, easy to read) ───────────────────────────────
def _login_html() -> str:
    return """<!doctype html><html><head><title>Platform - Login</title>
<style>body{font-family:system-ui;max-width:420px;margin:4rem auto;padding:0 1rem}
input{width:100%;padding:.6rem;margin:.3rem 0;font-size:1rem;box-sizing:border-box}
button{width:100%;padding:.6rem;font-size:1rem;cursor:pointer;margin-top:.5rem}
.err{color:#b00;min-height:1.2rem}.hint{color:#666;font-size:.85rem;margin-top:1rem}</style>
</head><body>
<h1>🔐 Platform Gateway</h1><p>Log in to your governed workspace.</p>
<input id="email" placeholder="email" value="phoebe@team.io">
<input id="password" type="password" placeholder="password" value="admin123">
<button onclick="login()">Log in</button>
<div class="err" id="err"></div>
<div class="hint">Demo accounts: phoebe@team.io / admin123 (admin) ·
ana@team.io / analyst123 · sam@team.io / scientist123 · ria@team.io / aieng123</div>
<script>
async function login(){
  const r=await fetch('/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({email:email.value,password:password.value})});
  if(r.ok){location.reload();} else {const d=await r.json();err.textContent=d.error||'login failed';}
}
</script></body></html>"""


def _workspace_html(email: str, role: str) -> str:
    apps = visible_apps(role)

    # Group by category, preserving first-seen order.
    categories: Dict[str, list] = {}
    for a in apps:
        categories.setdefault(str(a.get("category", "Other")), []).append(a)

    sections = []
    for cat, items in categories.items():
        cards = []
        for a in items:
            allowed = a.get("allowed")
            live = a.get("status") == "live"
            if allowed and live:
                card = (f"<a class='card open' href='/open/{a['slug']}'>"
                        f"<b>{a['name']}</b><span>{a['description']}</span>"
                        f"<em>open →</em></a>")
            elif allowed and not live:
                card = (f"<div class='card planned'><b>{a['name']}</b>"
                        f"<span>{a['description']}</span><em>planned</em></div>")
            else:  # locked - role too low
                card = (f"<div class='card locked'><b>🔒 {a['name']}</b>"
                        f"<span>needs role: {a['required_role']}</span></div>")
            cards.append(card)
        sections.append(f"<h2>{cat}</h2><div class='grid'>{''.join(cards)}</div>")

    open_count = sum(1 for a in apps if a.get("allowed") and a.get("status") == "live")
    admin_panel = ('<p style="margin-top:1.2rem">'
                   '<a href="/audit">🛡️ Audit log</a> &nbsp;·&nbsp; '
                   '<a href="/connections">🔌 Data connections</a> '
                   '<span style="color:#888;font-size:.85rem">(admin only)</span></p>'
                   if role == "admin" else "")
    return f"""<!doctype html><html><head><title>Platform - Workspace</title>
<style>body{{font-family:system-ui;max-width:880px;margin:2.5rem auto;padding:0 1rem;color:#1a1a2e}}
.badge{{display:inline-block;background:#eef;border-radius:6px;padding:.2rem .6rem;font-size:.9rem}}
h2{{margin-top:1.6rem;font-size:1.05rem;color:#555;border-bottom:1px solid #eee;padding-bottom:.3rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:.8rem;margin-top:.6rem}}
.card{{display:flex;flex-direction:column;gap:.3rem;padding:.9rem;border-radius:10px;border:1px solid #e6e6ef;
  text-decoration:none;color:inherit}}
.card span{{font-size:.82rem;color:#666}}.card em{{font-size:.8rem;font-style:normal;margin-top:.2rem}}
.card.open{{background:#f5f8ff;border-color:#cdd9ff}}.card.open em{{color:#2952cc;font-weight:600}}
.card.planned{{background:#fafafa;opacity:.8}}.card.planned em{{color:#999}}
.card.locked{{background:#fbf6f6;border-color:#f0dede}}.card.locked span{{color:#b07}}
button{{padding:.5rem 1rem;cursor:pointer;margin-top:1.5rem}}</style></head><body>
<h1>🏠 Your Workspace</h1>
<p>Signed in as <b>{email}</b> · role <span class="badge">{role}</span> ·
{open_count} app(s) you can open</p>
{admin_panel}
{''.join(sections)}
<button onclick="logout()">Log out</button>
<script>async function logout(){{await fetch('/logout',{{method:'POST'}});location.reload();}}</script>
</body></html>"""
