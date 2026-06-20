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

from auth import create_token, secret_is_default, verify_token
from users import authenticate, seed_if_empty

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
    role = authenticate(req.email, req.password)
    if role is None:
        # Same message for "no such user" and "wrong password" - never tell an
        # attacker which half they got right.
        return JSONResponse({"ok": False, "error": "invalid email or password"}, status_code=401)

    token = create_token(email=req.email.strip().lower(), role=role)
    resp = JSONResponse({"ok": True, "email": req.email.strip().lower(), "role": role})
    # HttpOnly = JavaScript can't read the cookie, which blocks a whole class of
    # token-theft attacks. SameSite=Lax limits cross-site sending.
    resp.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=8 * 3600)
    return resp


@app.post("/logout")
def logout() -> JSONResponse:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE)
    return resp


@app.get("/me")
def me(user: Optional[Dict[str, object]] = Depends(current_user)) -> JSONResponse:
    """Whoami. 401 if not logged in - proves the token round-trip works."""
    if user is None:
        return JSONResponse({"authenticated": False}, status_code=401)
    return JSONResponse({"authenticated": True, "email": user["email"], "role": user["role"]})


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
    return f"""<!doctype html><html><head><title>Platform - Workspace</title>
<style>body{{font-family:system-ui;max-width:640px;margin:4rem auto;padding:0 1rem}}
.badge{{display:inline-block;background:#eef;border-radius:6px;padding:.2rem .6rem;font-size:.9rem}}
button{{padding:.5rem 1rem;cursor:pointer}}</style></head><body>
<h1>🏠 Your Workspace</h1>
<p>Signed in as <b>{email}</b> · role <span class="badge">{role}</span></p>
<p>This is the shell. In Step 2 it will list only the apps your role can open.</p>
<button onclick="logout()">Log out</button>
<script>async function logout(){{await fetch('/logout',{{method:'POST'}});location.reload();}}</script>
</body></html>"""
