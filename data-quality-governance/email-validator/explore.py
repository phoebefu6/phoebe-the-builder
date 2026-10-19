"""Ten email validators against an RFC 5321 / RFC 6531 reference, on a hand-labelled corpus.

The reference parser implements the SMTP Mailbox grammar (RFC 5321 s4.1.2), its length limits (s4.5.3.1),
the SMTPUTF8 extension (RFC 6531) and the 63-octet DNS label limit (RFC 1035). It sorts every address into:

- ok    : valid and deliverable on the public internet
- grey  : valid by the grammar, but not deliverable on the public internet (dotless host, IP literal, numeric TLD)
- bad   : not a valid mailbox

Every corpus row also carries a HAND label; tests assert the parser and the hand label agree, so the truth column
is two independent readings of the RFC, not one. No randomness anywhere: every number is exact.

Regexes are applied the way they are pasted into code: `re.match` on a `^...$` pattern. In Python `$` also matches
before a final newline, which is part of what this build measures.
"""

from __future__ import annotations

import ipaddress
import re
import string
from email.utils import parseaddr
from typing import Callable, Dict, List, Tuple

from email_validator import EmailNotValidError, validate_email

ATEXT = set(string.ascii_letters + string.digits + "!#$%&'*+-/=?^_`{|}~")
LDH = set(string.ascii_letters + string.digits + "-")


def _atext(ch: str) -> bool:
    return ch in ATEXT or ord(ch) > 127  # RFC 6531 adds UTF8-non-ascii to atext


def _dot_string(s: str) -> bool:
    return all(p and all(_atext(c) for c in p) for p in s.split("."))


def _quoted_string(s: str) -> bool:
    if len(s) < 2 or s[0] != '"' or s[-1] != '"':
        return False
    body, i = s[1:-1], 0
    while i < len(body):
        o = ord(body[i])
        if body[i] == "\\":  # quoted-pairSMTP = "\" %d32-126
            if i + 1 >= len(body) or not 32 <= ord(body[i + 1]) <= 126:
                return False
            i += 2
            continue
        if not (o in (32, 33) or 35 <= o <= 91 or 93 <= o <= 126 or o > 127):
            return False
        i += 1
    return True


def _label(lab: str) -> bool:
    if any(ord(c) > 127 for c in lab):  # U-label: must survive IDNA encoding to a valid A-label
        try:
            lab = lab.encode("idna").decode("ascii")
        except UnicodeError:
            return False
    return 0 < len(lab) <= 63 and all(c in LDH for c in lab) and lab[0] != "-" and lab[-1] != "-"


def _literal(d: str) -> bool:
    inner = d[1:-1]
    if inner.startswith("IPv6:"):
        try:
            ipaddress.IPv6Address(inner[5:])
            return True
        except ValueError:
            return False
    parts = inner.split(".")
    return len(parts) == 4 and all(p.isdigit() and len(p) <= 3 and int(p) <= 255 for p in parts)


def reference(addr: str) -> Tuple[str, str]:
    """Return (tier, reason). tier is 'ok', 'grey' or 'bad'."""
    if "@" not in addr:
        return "bad", "no @"
    local, domain = addr.rsplit("@", 1)  # a quoted local part may itself contain @
    if not local:
        return "bad", "empty local part"
    if not (_dot_string(local) or _quoted_string(local)):
        return "bad", "local part not a dot-string or quoted-string"
    if len(local.encode()) > 64:
        return "bad", "local part over 64 octets"
    if len(addr.encode()) > 254:
        return "bad", "address over 254 octets"
    if domain.startswith("[") and domain.endswith("]"):
        return ("grey", "address literal") if _literal(domain) else ("bad", "malformed address literal")
    labels = domain.split(".")
    if not domain or not all(_label(x) for x in labels):
        return "bad", "domain label invalid"
    if len(labels) == 1:
        return "grey", "dotless domain"
    if labels[-1].isdigit():
        return "grey", "all-numeric TLD"
    return "ok", "valid"


# ------------------------------------------------------------------ the corpus: (address, group, hand label)
L64 = "a" * 64
CORPUS: List[Tuple[str, str, str]] = [
    ("john.smith@example.com", "plain", "ok"),
    ("jane@example.org", "plain", "ok"),
    ("a.b.c@company.co.uk", "plain", "ok"),
    ("j@e.co", "plain", "ok"),
    ("first_last@example.com", "plain", "ok"),
    ("info@3m.com", "plain", "ok"),
    ("john+news@gmail.com", "plus tag", "ok"),
    ("j.smith+billing@company.com", "plus tag", "ok"),
    ("anna@mail.eng.example.com", "subdomain", "ok"),
    ("ops@eu-west.corp.example.net", "subdomain", "ok"),
    ("kim@my-company.com", "hyphen in domain", "ok"),
    ("kim@a-b-c.co.uk", "hyphen in domain", "ok"),
    ("lee@studio.photography", "long TLD", "ok"),
    ("lee@firm.technology", "long TLD", "ok"),
    ("nina@art.museum", "long TLD", "ok"),
    ("John.Smith@Example.COM", "uppercase", "ok"),
    ("ANNA@EXAMPLE.ORG", "uppercase", "ok"),
    ("o'brien@example.ie", "apostrophe", "ok"),
    ("d'angelo@example.it", "apostrophe", "ok"),
    ("user!tag@example.com", "other legal symbol", "ok"),
    ("a/b@example.com", "other legal symbol", "ok"),
    ("x=y@example.com", "other legal symbol", "ok"),
    ("user#1@example.com", "other legal symbol", "ok"),
    ("{a}@example.com", "other legal symbol", "ok"),
    ("~home@example.com", "other legal symbol", "ok"),
    ("user&co@example.com", "other legal symbol", "ok"),
    ("-x@example.com", "other legal symbol", "ok"),
    (L64 + "@example.com", "at the length limit", "ok"),
    ("x@" + ".".join(["a" * 63] * 3) + "." + "b" * 56 + ".com", "at the length limit", "ok"),
    ('"john smith"@example.com', "quoted local part", "ok"),
    ('"john..doe"@example.com', "quoted local part", "ok"),
    ('"a@b"@example.com', "quoted local part", "ok"),
    ("josé@example.com", "international (SMTPUTF8)", "ok"),
    ("user@münchen.de", "international (SMTPUTF8)", "ok"),
    ("用户@例子.广告", "international (SMTPUTF8)", "ok"),
    ("user@xn--mnchen-3ya.de", "international (SMTPUTF8)", "ok"),
    # grey: valid by the grammar, not deliverable on the public internet
    ("user@localhost", "dotless host", "grey"),
    ("admin@mailserver1", "dotless host", "grey"),
    ("user@[192.168.1.1]", "IP literal", "grey"),
    ("user@[IPv6:2001:db8::1]", "IP literal", "grey"),
    ("user@example.123", "numeric TLD", "grey"),
    # bad: not a valid mailbox
    ("john.example.com", "no @ / two @", "bad"),
    ("a@b@example.com", "no @ / two @", "bad"),
    ("@example.com", "empty part", "bad"),
    ("john@", "empty part", "bad"),
    (".john@example.com", "dot misuse", "bad"),
    ("john.@example.com", "dot misuse", "bad"),
    ("john..doe@example.com", "dot misuse", "bad"),
    ("john@example..com", "dot misuse", "bad"),
    ("john@example.com.", "dot misuse", "bad"),
    ("john doe@example.com", "space / control char", "bad"),
    ("john@example.com\n", "space / control char", "bad"),
    ("john\t@example.com", "space / control char", "bad"),
    ("john@-example.com", "bad domain label", "bad"),
    ("john@example-.com", "bad domain label", "bad"),
    ("john@ex_ample.com", "bad domain label", "bad"),
    ("john@" + "a" * 64 + ".com", "over a length limit", "bad"),
    ("a" + L64 + "@example.com", "over a length limit", "bad"),
    ("x@" + ".".join(["a" * 63] * 3) + "." + "b" * 57 + ".com", "over a length limit", "bad"),
    ('"john@example.com', "broken quoting", "bad"),
    ('"a"b"@example.com', "broken quoting", "bad"),
    ('john."doe"@example.com', "broken quoting", "bad"),
    ("john,doe@example.com", "illegal symbol", "bad"),
    ("john[1]@example.com", "illegal symbol", "bad"),
    ("john(work)@example.com", "RFC 5322 header syntax", "bad"),
    ("<john@example.com>", "RFC 5322 header syntax", "bad"),
    ("John <john@example.com>", "RFC 5322 header syntax", "bad"),
    ("user@[300.1.1.1]", "malformed IP literal", "bad"),
]

# Declared assumption, NOT data: the share of real sign-ups in each deliverable group. Edit it in the app.
SHARE: Dict[str, float] = {
    "plain": 0.880,
    "plus tag": 0.030,
    "subdomain": 0.020,
    "hyphen in domain": 0.020,
    "long TLD": 0.015,
    "uppercase": 0.020,
    "apostrophe": 0.003,
    "other legal symbol": 0.001,
    "at the length limit": 0.0005,
    "quoted local part": 0.0001,
    "international (SMTPUTF8)": 0.0104,
}

# ------------------------------------------------------------------ the ten validators, as commonly pasted
P_SIMPLE = r"^\S+@\S+\.\S+$"
P_TUTORIAL = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
P_STRICT = r"^[a-zA-Z0-9_.]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,4}$"
P_OWASP = r"^[a-zA-Z0-9_+&*-]+(?:\.[a-zA-Z0-9_+&*-]+)*@(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,7}$"
P_HTML5 = (
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
)
P_5322 = (
    r"^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21"
    r"\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9]"
    r"(?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?"
    r"[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-"
    r"\x7f])+)\])$"
)


def _rx(p: str, flags: int = 0) -> Callable[[str], bool]:
    rx = re.compile(p, flags)
    return lambda s: rx.match(s) is not None


def _handrolled(s: str) -> bool:
    if s.count("@") != 1 or " " in s:
        return False
    local, domain = s.split("@")
    return bool(local) and "." in domain and not domain.startswith(".") and not domain.endswith(".")


def _library(s: str) -> bool:
    try:
        validate_email(s, check_deliverability=False)
        return True
    except EmailNotValidError:
        return False


VALIDATORS: Dict[str, Callable[[str], bool]] = {
    "contains @": lambda s: "@" in s,
    "hand-rolled split": _handrolled,
    "\\S+@\\S+\\.\\S+": _rx(P_SIMPLE),
    "tutorial regex": _rx(P_TUTORIAL),
    "strict TLD{2,4}": _rx(P_STRICT),
    "OWASP regex": _rx(P_OWASP),
    "HTML5 type=email": _rx(P_HTML5),
    "RFC 5322 regex": _rx(P_5322, re.I),
    "stdlib parseaddr": lambda s: "@" in parseaddr(s)[1],
    "email-validator 2.3": _library,
}


def score(name: str, share: Dict[str, float] = SHARE) -> Dict:
    """False rejects (ok rejected), false accepts (bad accepted), grey accepted, and sign-ups lost under `share`."""
    f = VALIDATORS[name]
    rows = [(a, g, reference(a)[0], f(a)) for a, g, _ in CORPUS]
    per_group = {g: sum(1 for _, gg, t, _ in rows if gg == g) for g in share}
    lost = sum(share[g] / per_group[g] for _, g, t, acc in rows if t == "ok" and not acc)
    return {
        "false_reject": [a for a, _, t, acc in rows if t == "ok" and not acc],
        "false_accept": [a for a, _, t, acc in rows if t == "bad" and acc],
        "grey_accept": [a for a, _, t, acc in rows if t == "grey" and acc],
        "groups_rejected": sorted({g for _, g, t, acc in rows if t == "ok" and not acc}),
        "groups_accepted_bad": sorted({g for _, g, t, acc in rows if t == "bad" and acc}),
        "signups_lost": lost / sum(share.values()),
    }


def chain(names: List[str]) -> Callable[[str], bool]:
    """A sign-up form usually stacks validators (browser, API, database). An address must pass every one."""
    return lambda s: all(VALIDATORS[n](s) for n in names)


def study(share: Dict[str, float] = SHARE) -> Dict:
    tiers = {t: sum(1 for a, _, _ in CORPUS if reference(a)[0] == t) for t in ("ok", "grey", "bad")}
    per = {n: score(n, share) for n in VALIDATORS}
    ok = [a for a, _, _ in CORPUS if reference(a)[0] == "ok"]
    bad = [a for a, _, _ in CORPUS if reference(a)[0] == "bad"]
    names = list(VALIDATORS)
    stack = ["HTML5 type=email", "tutorial regex", "email-validator 2.3"]
    together = chain(stack)
    return {
        "tiers": tiers,
        "per": per,
        "rejected_by_none": [a for a in ok if all(VALIDATORS[n](a) for n in names)],
        "ok_rejected_by_k": {a: sum(not VALIDATORS[n](a) for n in names) for a in ok},
        "bad_accepted_by_k": {a: sum(VALIDATORS[n](a) for n in names) for a in bad},
        "perfect": [n for n in names if not per[n]["false_reject"] and not per[n]["false_accept"]],
        "stack": stack,
        "stack_false_reject": [a for a in ok if not together(a)],
        "stack_signups_lost": sum(
            share[g] / sum(1 for _, gg, _ in CORPUS if gg == g)
            for a, g, _ in CORPUS
            if reference(a)[0] == "ok" and not together(a)
        )
        / sum(share.values()),
        "newline_accepted_by": [n for n in names if VALIDATORS[n]("john@example.com\n")],
    }
