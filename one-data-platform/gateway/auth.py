from __future__ import annotations

"""The security primitives, hand-written so you can SEE inside them.

Two jobs:
  1. Store passwords safely (never in plain text)  -> hash_password / verify_password
  2. Hand out a tamper-proof "wristband" after login -> create_token / verify_token

Everything here uses only the Python standard library. Later we can swap these for
industrial libraries (bcrypt for hashing, PyJWT for tokens) - but writing the tiny
version first means you understand exactly what those libraries do.

Read the comments top to bottom; this is a teaching file as much as a code file.
"""

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Dict, Optional, Tuple

# ── Where the signing secret comes from ───────────────────────────────────────
# The secret is the "pen" we sign wristbands with. Anyone with it could forge a
# token, so it must NEVER be committed to git. We read it from an environment
# variable; if missing, we use an obvious dev-only default and the app warns.
SECRET = os.environ.get("PLATFORM_SECRET", "dev-only-insecure-secret-change-me").encode()

TOKEN_TTL_SECONDS = 8 * 3600  # a wristband is valid for 8 hours, then you re-login


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 - PASSWORDS
# ══════════════════════════════════════════════════════════════════════════════
# We never store the real password. We store a one-way "hash" of it. A hash is
# like a meat grinder: easy to turn password -> hash, practically impossible to
# turn hash -> password. To check a login we grind the typed password the same
# way and compare hashes.
#
# A "salt" is random bytes mixed in before grinding, unique per user. It stops an
# attacker from precomputing hashes of common passwords (a "rainbow table").
# pbkdf2 runs the grinder 100k times so guessing is slow for attackers.

def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Return 'salt$hash' (both hex). Store THIS, never the password."""
    if salt is None:
        salt = os.urandom(16)  # 16 random bytes, unique to this password
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check a typed password against a stored 'salt$hash'."""
    try:
        salt_hex, _ = stored.split("$", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    # Grind the candidate the same way, then compare in constant time.
    # hmac.compare_digest avoids leaking timing info about where a mismatch is.
    return hmac.compare_digest(hash_password(password, salt), stored)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 - TOKENS (the wristband)
# ══════════════════════════════════════════════════════════════════════════════
# After login we give the browser a token. On every later request the browser
# sends it back, so the user doesn't re-type their password each click.
#
# Our token has two parts joined by a dot:   <payload>.<signature>
#   payload   = base64 of JSON like {"email": "...", "role": "...", "exp": 1750}
#   signature = HMAC of the payload using our SECRET
# Anyone can READ the payload (it's not secret), but they CANNOT change it without
# knowing SECRET, because the signature would no longer match. That's the trick a
# real JWT uses too - we've just built the 30-line version.

def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(payload_b64: str) -> str:
    return _b64(hmac.new(SECRET, payload_b64.encode(), hashlib.sha256).digest())


def create_token(email: str, role: str) -> str:
    """Issue a signed token that expires in TOKEN_TTL_SECONDS."""
    payload = {"email": email, "role": role, "exp": int(time.time()) + TOKEN_TTL_SECONDS}
    payload_b64 = _b64(json.dumps(payload).encode())
    return f"{payload_b64}.{_sign(payload_b64)}"


def verify_token(token: str) -> Optional[Dict[str, object]]:
    """Return the payload if the token is genuine and unexpired, else None."""
    try:
        payload_b64, signature = token.split(".", 1)
    except (ValueError, AttributeError):
        return None
    # 1. Is the signature real? (constant-time compare)
    if not hmac.compare_digest(_sign(payload_b64), signature):
        return None
    # 2. Decode the payload and check it hasn't expired.
    try:
        payload = json.loads(_unb64(payload_b64))
    except (ValueError, json.JSONDecodeError):
        return None
    if payload.get("exp", 0) < time.time():
        return None  # wristband expired - back to the login desk
    return payload


def secret_is_default() -> Tuple[bool, str]:
    """Warn loudly if we're running on the insecure dev secret."""
    if SECRET == b"dev-only-insecure-secret-change-me":
        return True, "PLATFORM_SECRET is the dev default - set a real one before any real use."
    return False, ""
