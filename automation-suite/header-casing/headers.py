"""HTTP field names, and what every hop on the path does to them.

`headers["Content-Type"]` is the wrong shape, and the reason is not style. A field
name is not a string: it is a case-insensitive token whose *wire form* is dictated
by a protocol version the dictionary has never heard of.

    RFC 9110 5.1: field names are case-insensitive.
    RFC 9113 8.2.1: in HTTP/2 field names MUST be lowercase, and a message
                    containing an uppercase field name MUST be treated as
                    malformed.

So `Content-Type` and `content-type` are one field with two spellings, and which
spelling arrives is a property of the *path*, not of your code. A dict keyed by
the string as typed disagrees with HTTP twice over: it holds two entries for one
field, and it holds one entry for two field lines that HTTP keeps separate.

That second one is where data actually goes missing. `Set-Cookie` may legitimately
appear many times in one response and must not be combined into a comma-separated
line - and a `Dict[str, str]` has no way to hold two of anything, so a library
picks first or last and nobody is told which.

Casing is only the entry point. Every hop rewrites something:

* an HTTP/2 gateway lowercases every name, and rejects the message outright if a
  name was uppercase or if a connection-specific field is present
* nginx drops field names containing an underscore, by default, silently
  (`underscores_in_headers off`), so `X_Request_Id` never reaches the app
* CGI and PHP map the name to `HTTP_` + uppercase with `-` replaced by `_`, so
  `X-Request-Id` and `X_Request_Id` become the *same variable* - two distinct
  legal fields collapsing into one is a spoofing primitive, not a formatting
  detail
* a lowercasing implementation that uses the platform locale turns
  `If-Modified-Since` into `ıf-modıfıed-sınce` in a Turkish locale, and the
  lookup misses

Hence the return value is a verdict over the whole path:

    preserved    - every field arrives, with its name spelled as it was sent
    renormalized - every field arrives and means the same thing; the spelling,
                   the order, or the folding changed
    lossy        - a field, or a duplicate of a field, did not arrive - or two
                   distinct fields arrived as one. Nothing errored
    rejected     - a hop must treat the message as malformed

Only `preserved` and `renormalized` are safe, and code that compares names with
`==` is only correct under `preserved`, which nothing on the public internet
guarantees.

Standard library only: `re`, `dataclasses`, `enum`, `typing`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# What a field is
# ---------------------------------------------------------------------------

# RFC 9110 5.6.2: token = 1*tchar
TCHAR = set("!#$%&'*+-.^_`|~0123456789"
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# RFC 9113 8.2.2 / 8.2.1: connection-specific fields must not appear in HTTP/2.
# TE is allowed, and only with the value "trailers".
H2_FORBIDDEN = frozenset({"connection", "proxy-connection", "keep-alive",
                          "transfer-encoding", "upgrade", "te"})

# The HPACK static table (RFC 7541 Appendix A) holds 61 entries and every name in
# it is lowercase. A name in the table costs about one byte on the wire; a name
# that is not costs its own length. Casing therefore has a size, not just a
# spelling.
HPACK_STATIC_NAMES = frozenset({
    ":authority", ":method", ":path", ":scheme", ":status",
    "accept-charset", "accept-encoding", "accept-language", "accept-ranges",
    "accept", "access-control-allow-origin", "age", "allow", "authorization",
    "cache-control", "content-disposition", "content-encoding",
    "content-language", "content-length", "content-location", "content-range",
    "content-type", "cookie", "date", "etag", "expect", "expires", "from",
    "host", "if-match", "if-modified-since", "if-none-match", "if-range",
    "if-unmodified-since", "last-modified", "link", "location", "max-forwards",
    "proxy-authenticate", "proxy-authorization", "range", "referer", "refresh",
    "retry-after", "server", "set-cookie", "strict-transport-security",
    "transfer-encoding", "user-agent", "vary", "via", "www-authenticate",
})

# RFC 9110 5.3 permits combining same-name field lines with commas only where the
# field value is a comma-separated list. Set-Cookie is the field everybody knows
# breaks, because its own value contains commas (in the Expires date).
NEVER_COMBINE = frozenset({"set-cookie", "www-authenticate", "proxy-authenticate"})


@dataclass(frozen=True)
class Field:
    """One field *line*: a name as spelled on the wire, and its value.

    Deliberately not a dict entry. Order and duplication are part of the message,
    and a mapping cannot hold either.
    """

    name: str
    value: str

    @property
    def key(self) -> str:
        """The field's identity: ASCII-lowercased, never locale-lowercased."""
        return ascii_lower(self.name)

    def h1_bytes(self) -> int:
        """Bytes on an HTTP/1.1 wire: name, colon, space, value, CRLF."""
        return len(self.name.encode("utf-8")) + 2 + len(self.value.encode("utf-8")) + 2

    def h2_bytes(self) -> int:
        """Approximate HPACK cost, literal with incremental indexing, no Huffman.

        A name in the static table is one index byte; anything else pays a length
        prefix plus its own bytes. Huffman coding would shrink both sides, so this
        is a model of the *difference*, not an exact frame size.
        """
        name = ascii_lower(self.name)
        name_cost = 1 if name in HPACK_STATIC_NAMES else 1 + len(name.encode("utf-8"))
        return 1 + name_cost + 1 + len(self.value.encode("utf-8"))


def ascii_lower(s: str) -> str:
    """Lowercase the ASCII letters and nothing else - the only correct rule here.

    `str.lower()` happens to agree for ASCII input in Python. The point of naming
    it is that `toLowerCase()` in Java, `tolower()` in C and `LOWER()` in some
    databases consult the platform locale, and one of those locales is Turkish.
    """
    return s.translate(_ASCII_LOWER)


_ASCII_LOWER = {ord(c): ord(c) + 32 for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}

# Turkish and Azeri lowercase I to a dotless i. Any implementation that
# lowercases a field name with the default locale breaks every name with an I.
_TURKISH_LOWER = dict(_ASCII_LOWER)
_TURKISH_LOWER[ord("I")] = 0x131  # LATIN SMALL LETTER DOTLESS I


def turkish_lower(s: str) -> str:
    return s.translate(_TURKISH_LOWER)


@dataclass(frozen=True)
class Message:
    """An ordered list of field lines, plus how it is being sent."""

    name: str
    fields: Tuple[Field, ...]
    version: str = "1.1"  # "1.1" | "2"
    note: str = ""

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(f.key for f in self.fields)

    def identity(self) -> Tuple[Tuple[str, str], ...]:
        """What the message *means*: identities and values, order preserved.

        Two messages with the same identity differ only in spelling. This is the
        thing a path is supposed to preserve, and `Field.name` is not.
        """
        return tuple((f.key, f.value.strip()) for f in self.fields)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class Verdict(Enum):
    PRESERVED = "preserved"
    RENORMALIZED = "renormalized"
    LOSSY = "lossy"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Event:
    """One thing a hop did, with the field it did it to."""

    hop: str
    code: str
    detail: str
    field_name: Optional[str] = None


@dataclass(frozen=True)
class Delivery:
    """The message as it arrived, plus everything that happened on the way."""

    sent: Message
    arrived: Optional[Message]
    events: Tuple[Event, ...]
    path: Tuple[str, ...]

    @property
    def rejected(self) -> bool:
        return self.arrived is None

    def verdict(self) -> Verdict:
        if self.arrived is None:
            return Verdict.REJECTED
        if len(self.arrived.fields) != len(self.sent.fields):
            return Verdict.LOSSY
        if self.arrived.identity() != self.sent.identity():
            return Verdict.LOSSY
        if self.arrived.keys != self.sent.keys or any(
            a.name != b.name for a, b in zip(self.arrived.fields, self.sent.fields)
        ):
            return Verdict.RENORMALIZED
        return Verdict.PRESERVED

    def lost(self) -> Tuple[str, ...]:
        """Field identities that went in and did not come out (with multiplicity)."""
        if self.arrived is None:
            return self.sent.keys
        out: List[str] = []
        remaining = list(self.arrived.keys)
        for k in self.sent.keys:
            if k in remaining:
                remaining.remove(k)
            else:
                out.append(k)
        return tuple(out)

    def renamed(self) -> Tuple[Tuple[str, str], ...]:
        if self.arrived is None:
            return ()
        pairs = []
        for a, b in zip(self.sent.fields, self.arrived.fields):
            if a.name != b.name:
                pairs.append((a.name, b.name))
        return tuple(pairs)


@dataclass(frozen=True)
class Finding:
    """Something the reader has to decide about, with how loud it is."""

    severity: str  # "blocking" | "silent" | "advisory"
    code: str
    text: str


SEVERITY_ORDER = {"blocking": 0, "silent": 1, "advisory": 2}


# ---------------------------------------------------------------------------
# Spelling functions - the four rules that ship, and the one that is a bug
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def is_token(name: str) -> bool:
    """RFC 9110 5.6.2. Note what is legal: `_`, `.`, `~`, `$` are all tchar."""
    return bool(TOKEN_RE.match(name))


def go_canonical(name: str) -> str:
    """Go's `textproto.CanonicalMIMEHeaderKey`, and Python's `email` package.

    Uppercase the first letter and every letter directly after a hyphen;
    lowercase everything else. Go returns the name *unchanged* if it contains a
    byte that is not a tchar, which is why the guard is here and not at the call
    site.
    """
    if not is_token(name):
        return name
    out: List[str] = []
    upper = True
    for ch in name:
        if upper and "a" <= ch <= "z":
            out.append(chr(ord(ch) - 32))
        elif not upper and "A" <= ch <= "Z":
            out.append(chr(ord(ch) + 32))
        else:
            out.append(ch)
        upper = ch == "-"
    return "".join(out)


def py_title(name: str) -> str:
    """`str.title()`, which people reach for expecting `go_canonical`.

    It capitalises after *any* non-letter, digits included, so `x-2fa-token`
    becomes `X-2Fa-Token` where Go gives `X-2fa-Token`.
    """
    return name.title()


def wsgi_key(name: str) -> str:
    """PEP 3333 / RFC 3875 4.1.18: `HTTP_` + uppercase, `-` replaced by `_`.

    The replacement is one-way. `X-Request-Id` and `X_Request_Id` are distinct
    legal field names and produce one identical variable.
    """
    return "HTTP_" + name.upper().replace("-", "_")


SPELLERS: Dict[str, Callable[[str], str]] = {
    "as_written": lambda s: s,
    "ascii_lower": ascii_lower,
    "go_canonical": go_canonical,
    "py_title": py_title,
    "turkish_lower": turkish_lower,
}


# ---------------------------------------------------------------------------
# The registry: a curated subset of real field names, with their real spelling
# ---------------------------------------------------------------------------

# Source of the spelling is the IANA HTTP Field Name Registry entry (or, for the
# `False` rows, the spelling the deploying vendor uses). The flag is honest: not
# every name people send is registered, and the unregistered ones are exactly
# the ones with inventive casing.
REGISTRY: Tuple[Tuple[str, bool], ...] = (
    ("Accept", True), ("Accept-Charset", True), ("Accept-Encoding", True),
    ("Accept-Language", True), ("Accept-Ranges", True), ("Age", True),
    ("Allow", True), ("ALPN", True), ("Alt-Svc", True), ("Alt-Used", True),
    ("Authentication-Info", True), ("Authorization", True),
    ("Cache-Control", True), ("Cache-Status", True), ("CDN-Cache-Control", True),
    ("Connection", True), ("Content-Disposition", True),
    ("Content-Encoding", True), ("Content-Language", True),
    ("Content-Length", True), ("Content-Location", True), ("Content-MD5", True),
    ("Content-Range", True), ("Content-Security-Policy", True),
    ("Content-Type", True), ("Cookie", True), ("Date", True), ("ETag", True),
    ("Expect", True), ("Expires", True), ("Forwarded", True), ("From", True),
    ("Host", True), ("If-Match", True), ("If-Modified-Since", True),
    ("If-None-Match", True), ("If-Range", True), ("If-Unmodified-Since", True),
    ("IM", True), ("Last-Modified", True), ("Link", True), ("Location", True),
    ("Max-Forwards", True), ("MIME-Version", True), ("Origin", True),
    ("P3P", True), ("Permissions-Policy", True), ("Proxy-Authenticate", True),
    ("Proxy-Authorization", True), ("Range", True), ("Referer", True),
    ("Retry-After", True), ("Sec-Fetch-Dest", True), ("Sec-Fetch-Mode", True),
    ("Sec-Fetch-Site", True), ("Sec-Fetch-User", True),
    ("Sec-WebSocket-Accept", True), ("Sec-WebSocket-Extensions", True),
    ("Sec-WebSocket-Key", True), ("Sec-WebSocket-Protocol", True),
    ("Sec-WebSocket-Version", True), ("Server", True), ("Set-Cookie", True),
    ("Strict-Transport-Security", True), ("TE", True), ("Trailer", True),
    ("Transfer-Encoding", True), ("Upgrade", True),
    ("Upgrade-Insecure-Requests", True), ("User-Agent", True), ("Vary", True),
    ("Via", True), ("WWW-Authenticate", True), ("X-Content-Type-Options", True),
    ("X-Frame-Options", True),
    # Not registered. Deployed everywhere anyway.
    ("DNT", False), ("X-XSS-Protection", False), ("X-Forwarded-For", False),
    ("X-Forwarded-Proto", False), ("X-Real-IP", False), ("X-Request-ID", False),
    ("X-Correlation-ID", False), ("X-CSRF-Token", False), ("X-UA-Compatible", False),
    ("X_Request_Id", False), ("X-2FA-Token", False),
)

REGISTRY_NAMES: Tuple[str, ...] = tuple(n for n, _ in REGISTRY)
REGISTERED_ONLY: Tuple[str, ...] = tuple(n for n, r in REGISTRY if r)


# ---------------------------------------------------------------------------
# Hops
# ---------------------------------------------------------------------------

HopFn = Callable[[Message], Tuple[Optional[Message], List[Event]]]


@dataclass(frozen=True)
class Hop:
    name: str
    kind: str
    doc: str
    fn: HopFn


def _msg(base: Message, fields: Sequence[Field], version: Optional[str] = None) -> Message:
    return Message(base.name, tuple(fields), version or base.version, base.note)


def hop_h2_encode(m: Message) -> Tuple[Optional[Message], List[Event]]:
    """An HTTP/2 client or gateway serialising the message.

    RFC 9113 8.2.1 - field names MUST be lowercase. 8.2.2 - connection-specific
    fields MUST NOT appear, and a receiver MUST treat the message as malformed.
    """
    ev: List[Event] = []
    for f in m.fields:
        if f.key in H2_FORBIDDEN and not (f.key == "te" and f.value.strip() == "trailers"):
            ev.append(Event("h2-encode", "h2-forbidden-field",
                            f"{f.name!r} is connection-specific; RFC 9113 8.2.2 "
                            f"makes the whole message malformed", f.name))
            return None, ev
        if not is_token(f.name):
            ev.append(Event("h2-encode", "not-a-token",
                            f"{f.name!r} contains a byte that is not a tchar", f.name))
            return None, ev
    out = []
    for f in m.fields:
        low = ascii_lower(f.name)
        if low != f.name:
            ev.append(Event("h2-encode", "lowercased",
                            f"{f.name!r} goes on the wire as {low!r}", f.name))
        out.append(Field(low, f.value))
    return _msg(m, out, "2"), ev


def hop_h2_strict_receiver(m: Message) -> Tuple[Optional[Message], List[Event]]:
    """A conforming HTTP/2 receiver reading a frame that was built by hand."""
    ev: List[Event] = []
    for f in m.fields:
        if any("A" <= c <= "Z" for c in f.name):
            ev.append(Event("h2-receiver", "malformed-uppercase",
                            f"{f.name!r} has an uppercase byte; RFC 9113 8.2.1 says "
                            f"MUST be treated as malformed", f.name))
            return None, ev
    return m, ev


def hop_nginx_underscores(m: Message) -> Tuple[Optional[Message], List[Event]]:
    """nginx with the default `underscores_in_headers off`.

    The field is dropped. The client is not told, the log does not say so, and
    the application sees a request that simply never had the field.
    """
    ev: List[Event] = []
    out = []
    for f in m.fields:
        if "_" in f.name:
            ev.append(Event("nginx", "dropped-underscore",
                            f"{f.name!r} contains `_`; nginx drops it silently "
                            f"(underscores_in_headers off)", f.name))
        else:
            out.append(f)
    return _msg(m, out), ev


def hop_cgi_environ(m: Message) -> Tuple[Optional[Message], List[Event]]:
    """CGI / WSGI / PHP `$_SERVER`: name becomes `HTTP_` + upper, `-` to `_`.

    Two things happen at once. Duplicates of one field are joined with `", "`,
    which is legal for list-valued fields and wrong for the rest. And two field
    names that differ only by `-` versus `_` land on one variable, last write
    winning - which is how a client forges a header the proxy thought it owned.
    """
    ev: List[Event] = []
    order: List[str] = []
    bucket: Dict[str, List[Field]] = {}
    for f in m.fields:
        k = wsgi_key(f.name)
        if k not in bucket:
            bucket[k] = []
            order.append(k)
        bucket[k].append(f)
    out = []
    for k in order:
        group = bucket[k]
        spellings = {f.name for f in group}
        if len({ascii_lower(f.name) for f in group}) > 1:
            ev.append(Event("cgi", "environ-collision",
                            f"{sorted(spellings)} are distinct field names and share "
                            f"the variable {k}", group[0].name))
        elif len(group) > 1:
            ev.append(Event("cgi", "joined-duplicates",
                            f"{len(group)} lines of {group[0].key!r} joined with ', '",
                            group[0].name))
        value = ", ".join(f.value for f in group)
        out.append(Field(k, value))
    return _msg(m, out), ev


def hop_go_net_http(m: Message) -> Tuple[Optional[Message], List[Event]]:
    """Go's `net/http`: canonical spelling, and writes fields in sorted order.

    `http.Header` is a map, so the order the peer sent is not recoverable, and
    `Header.Write` emits keys sorted. Duplicates survive - Go stores `[]string`.
    """
    ev: List[Event] = []
    out = []
    for f in m.fields:
        c = go_canonical(f.name)
        if c != f.name:
            ev.append(Event("go", "canonicalised",
                            f"{f.name!r} stored and re-emitted as {c!r}", f.name))
        out.append(Field(c, f.value))
    ordered = sorted(out, key=lambda f: f.name)
    if [f.name for f in ordered] != [f.name for f in out]:
        ev.append(Event("go", "reordered",
                        "field order is the map's, not the sender's; Header.Write sorts"))
    return _msg(m, ordered), ev


# Node discards duplicates of these instead of joining them (first line wins).
NODE_DISCARD_DUPES = frozenset({
    "age", "authorization", "content-length", "content-type", "etag", "expires",
    "from", "host", "if-modified-since", "if-unmodified-since", "last-modified",
    "location", "max-forwards", "proxy-authorization", "referer", "retry-after",
    "server", "user-agent",
})


def hop_node_req_headers(m: Message) -> Tuple[Optional[Message], List[Event]]:
    """Node's `req.headers`: lowercase keys, and three different duplicate rules.

    `set-cookie` becomes an array, `cookie` is joined with `"; "`, the list in
    `NODE_DISCARD_DUPES` keeps the first line and throws the rest away, and
    everything else joins with `", "`. Only `req.rawHeaders` still has the wire
    spelling and the wire order.
    """
    ev: List[Event] = []
    order: List[str] = []
    bucket: Dict[str, List[Field]] = {}
    for f in m.fields:
        k = f.key
        if k not in bucket:
            bucket[k] = []
            order.append(k)
        bucket[k].append(f)
        if k != f.name:
            ev.append(Event("node", "lowercased",
                            f"req.headers key is {k!r}; the wire said {f.name!r} "
                            f"(still in req.rawHeaders)", f.name))
    out = []
    for k in order:
        group = bucket[k]
        if k == "set-cookie":
            for f in group:
                out.append(Field(k, f.value))
            continue
        if len(group) > 1:
            if k in NODE_DISCARD_DUPES:
                ev.append(Event("node", "discarded-duplicate",
                                f"{len(group) - 1} further {k!r} line(s) discarded; "
                                f"first wins", k))
                out.append(Field(k, group[0].value))
                continue
            sep = "; " if k == "cookie" else ", "
            ev.append(Event("node", "joined-duplicates",
                            f"{len(group)} lines of {k!r} joined with {sep!r}", k))
            out.append(Field(k, sep.join(f.value for f in group)))
        else:
            out.append(Field(k, group[0].value))
    return _msg(m, out), ev


def hop_str_dict(m: Message) -> Tuple[Optional[Message], List[Event]]:
    """`Dict[str, str]` in the application, keyed by the name as received.

    Two failures in one hop. Duplicates cannot be held, so the last line wins
    silently. And the key is case-*sensitive*, so two spellings of one field sit
    side by side as two entries and every lookup finds at most one of them.
    """
    ev: List[Event] = []
    seen: Dict[str, Field] = {}
    for f in m.fields:
        if f.name in seen:
            ev.append(Event("app-dict", "last-write-wins",
                            f"a second {f.name!r} overwrote {seen[f.name].value!r}",
                            f.name))
        seen[f.name] = f
    by_key: Dict[str, List[str]] = {}
    for name in seen:
        by_key.setdefault(ascii_lower(name), []).append(name)
    for key, names in by_key.items():
        if len(names) > 1:
            ev.append(Event("app-dict", "split-identity",
                            f"one field {key!r} occupies {len(names)} dict entries: "
                            f"{sorted(names)}", names[0]))
    return _msg(m, tuple(seen.values())), ev


def hop_locale_lower(m: Message) -> Tuple[Optional[Message], List[Event]]:
    """A stack that lowercases with the platform locale, running in `tr_TR`.

    Java's `String.toLowerCase()` with no `Locale.ROOT`, or C `tolower()` after
    `setlocale`. `I` becomes a dotless `ı`, which is not a tchar, so the name is
    no longer a legal field name and no ASCII-lowercased lookup can match it.
    """
    ev: List[Event] = []
    out = []
    for f in m.fields:
        low = turkish_lower(f.name)
        if low != ascii_lower(f.name):
            ev.append(Event("locale", "turkish-dotless-i",
                            f"{f.name!r} lowercased to {low!r}, not {ascii_lower(f.name)!r}",
                            f.name))
        out.append(Field(low, f.value))
    return _msg(m, out), ev


HOPS: Tuple[Hop, ...] = (
    Hop("h2-encode", "protocol", "HTTP/2 gateway serialises the message", hop_h2_encode),
    Hop("h2-receiver", "protocol", "conforming HTTP/2 receiver checks the frame",
        hop_h2_strict_receiver),
    Hop("nginx", "proxy", "nginx with underscores_in_headers off", hop_nginx_underscores),
    Hop("cgi", "runtime", "CGI/WSGI/PHP environment mapping", hop_cgi_environ),
    Hop("go", "runtime", "Go net/http canonical keys, sorted on write", hop_go_net_http),
    Hop("node", "runtime", "Node req.headers", hop_node_req_headers),
    Hop("app-dict", "application", "Dict[str, str] keyed by the received name",
        hop_str_dict),
    Hop("locale", "runtime", "locale-sensitive lowercasing in tr_TR", hop_locale_lower),
)

HOP_BY_NAME: Dict[str, Hop] = {h.name: h for h in HOPS}

PATHS: Dict[str, Tuple[str, ...]] = {
    "direct-h1": (),
    "h2-gateway": ("h2-encode",),
    "nginx-h1": ("nginx",),
    "h1-to-dict": ("app-dict",),
    "h2-then-go": ("h2-encode", "go"),
    "nginx-to-cgi": ("nginx", "cgi"),
    "h2-to-node": ("h2-encode", "node"),
    "h2-to-node-to-dict": ("h2-encode", "node", "app-dict"),
    "handwritten-h2-frame": ("h2-receiver",),
    "java-tr-locale": ("locale", "app-dict"),
}


def deliver(m: Message, path_name: str) -> Delivery:
    """Run one message down one named path."""
    hops = PATHS[path_name]
    cur: Optional[Message] = m
    events: List[Event] = []
    for hop_name in hops:
        assert cur is not None
        cur, ev = HOP_BY_NAME[hop_name].fn(cur)
        events.extend(ev)
        if cur is None:
            break
    return Delivery(m, cur, tuple(events), hops)


def deliver_all(m: Message) -> Dict[str, Delivery]:
    return {p: deliver(m, p) for p in PATHS}


# ---------------------------------------------------------------------------
# Lookups: the line of application code that reads the field back out
# ---------------------------------------------------------------------------

LookupFn = Callable[[Message, str], List[str]]


def _lk_exact(m: Message, wanted: str) -> List[str]:
    return [f.value for f in m.fields if f.name == wanted]


def _lk_lower(m: Message, wanted: str) -> List[str]:
    w = ascii_lower(wanted)
    return [f.value for f in m.fields if f.name == w]


def _lk_insensitive(m: Message, wanted: str) -> List[str]:
    w = ascii_lower(wanted)
    return [f.value for f in m.fields if ascii_lower(f.name) == w]


def _lk_canonical(m: Message, wanted: str) -> List[str]:
    w = go_canonical(wanted)
    return [f.value for f in m.fields if f.name == w]


def _lk_environ(m: Message, wanted: str) -> List[str]:
    w = wsgi_key(wanted)
    return [f.value for f in m.fields if f.name == w]


LOOKUPS: Tuple[Tuple[str, str, LookupFn], ...] = (
    ("h[name]", "exact match on the spelling the developer typed", _lk_exact),
    ("h[name.lower()]", "exact match on the lowercased spelling", _lk_lower),
    ("CaseInsensitiveDict", "ASCII-case-insensitive match - the only correct rule",
     _lk_insensitive),
    ("h[Canonical(name)]", "Go/`email` canonical spelling", _lk_canonical),
    ("environ[HTTP_NAME]", "CGI variable name", _lk_environ),
)


def lookup_audit(d: Delivery, wanted: str) -> Dict[str, List[str]]:
    """What each style of lookup returns from what actually arrived."""
    if d.arrived is None:
        return {name: [] for name, _, _ in LOOKUPS}
    return {name: fn(d.arrived, wanted) for name, _, fn in LOOKUPS}


def lookup_matrix(m: Message, wanted: str) -> Dict[str, Dict[str, int]]:
    """Rows are paths, columns are lookup styles, cells are values found."""
    out: Dict[str, Dict[str, int]] = {}
    for path in PATHS:
        d = deliver(m, path)
        out[path] = {k: len(v) for k, v in lookup_audit(d, wanted).items()}
    return out


# ---------------------------------------------------------------------------
# Exhaustive searches over the registry
# ---------------------------------------------------------------------------


def environ_collisions(names: Sequence[str] = REGISTRY_NAMES) -> List[Tuple[str, str, str]]:
    """Every pair of *legal, distinct* field names sharing one CGI variable.

    The search is exhaustive rather than illustrative: for each name, its
    underscore twin is generated (`-` to `_`, still a legal token per RFC 9110
    5.6.2) and the pair is checked. Every hyphenated field name has one.
    """
    out: List[Tuple[str, str, str]] = []
    seen = set()
    pool = list(dict.fromkeys(list(names) + [n.replace("-", "_") for n in names]))
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if ascii_lower(a) == ascii_lower(b):
                continue  # same field, two spellings - not a collision
            if wsgi_key(a) == wsgi_key(b):
                # Dedupe on the *fields* involved, not on the spellings, so
                # `X_Request_Id` and `X_Request_ID` are one collision, not two.
                pair = tuple(sorted((ascii_lower(a), ascii_lower(b))))
                if pair not in seen:
                    seen.add(pair)
                    out.append((a, b, wsgi_key(a)))
    return out


def canonical_mismatches(names: Sequence[str] = REGISTRY_NAMES) -> List[Tuple[str, str]]:
    """Names whose registered spelling a canonicalising stack will not reproduce."""
    return [(n, go_canonical(n)) for n in names if go_canonical(n) != n]


def title_mismatches(names: Sequence[str] = REGISTRY_NAMES) -> List[Tuple[str, str, str]]:
    """Names where `str.title()` and the canonical rule disagree.

    Both are wrong against the registry in places; the point is that they are
    wrong *differently*, so two services that both look reasonable disagree.
    """
    return [(n, py_title(n), go_canonical(n))
            for n in names if py_title(n) != go_canonical(n)]


def turkish_breakage(names: Sequence[str] = REGISTRY_NAMES) -> List[Tuple[str, str]]:
    """Names a locale-sensitive lowercase mangles in `tr_TR`."""
    return [(n, turkish_lower(n)) for n in names
            if turkish_lower(n) != ascii_lower(n)]


def hpack_names(names: Sequence[str] = REGISTRY_NAMES) -> Tuple[int, int]:
    """(in static table, not in static table) once lowercased."""
    inside = sum(1 for n in names if ascii_lower(n) in HPACK_STATIC_NAMES)
    return inside, len(names) - inside


def wire_cost(m: Message) -> Tuple[int, int]:
    """(HTTP/1.1 bytes, modelled HPACK bytes) for the whole field block."""
    return sum(f.h1_bytes() for f in m.fields), sum(f.h2_bytes() for f in m.fields)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

SEVERITY_OF_CODE: Dict[str, str] = {
    "h2-forbidden-field": "blocking",
    "not-a-token": "blocking",
    "malformed-uppercase": "blocking",
    "environ-collision": "silent",
    "dropped-underscore": "silent",
    "discarded-duplicate": "silent",
    "last-write-wins": "silent",
    "split-identity": "silent",
    "turkish-dotless-i": "silent",
    "joined-duplicates": "advisory",
    "lowercased": "advisory",
    "canonicalised": "advisory",
    "reordered": "advisory",
}


def findings(d: Delivery) -> List[Finding]:
    """One finding per event, plus the ones only visible from the whole path."""
    out: List[Finding] = []
    for e in d.events:
        out.append(Finding(SEVERITY_OF_CODE.get(e.code, "advisory"), e.code,
                           f"[{e.hop}] {e.detail}"))
    if d.arrived is not None:
        for f in d.arrived.fields:
            if f.key in NEVER_COMBINE and ", " in f.value and f.key != "set-cookie":
                out.append(Finding("silent", "uncombinable-joined",
                                   f"{f.key!r} must not be sent as one comma-joined "
                                   f"line (RFC 9110 5.3)"))
        sent_cookies = sum(1 for f in d.sent.fields if f.key == "set-cookie")
        got_cookies = sum(1 for f in d.arrived.fields if f.key == "set-cookie")
        if got_cookies < sent_cookies:
            out.append(Finding("silent", "set-cookie-lost",
                               f"{sent_cookies - got_cookies} of {sent_cookies} "
                               f"Set-Cookie lines did not survive"))
    out.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.code))
    return out


def safe_form(name: str) -> str:
    """What to actually do with a field name, in one line."""
    if not is_token(name):
        return (f"{name!r} is not a legal field name (RFC 9110 5.6.2); it cannot be "
                f"sent at all")
    key = ascii_lower(name)
    bits = [f"identity is {key!r}: compare with ASCII case-folding, never `==`"]
    if "_" in name:
        bits.append("contains `_`: nginx drops it by default and CGI merges it with "
                    "the `-` spelling - rename it")
    if key not in HPACK_STATIC_NAMES:
        bits.append("not in the HPACK static table, so it costs its own bytes on "
                    "every HTTP/2 request")
    if turkish_lower(name) != key:
        bits.append("has an `I`: any locale-sensitive lowercase mangles it")
    if go_canonical(name) != name:
        bits.append(f"a canonicalising stack re-spells it {go_canonical(name)!r}")
    return "; ".join(bits)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def _m(name: str, version: str, note: str, *pairs: Tuple[str, str]) -> Message:
    return Message(name, tuple(Field(n, v) for n, v in pairs), version, note)


CORPUS: Tuple[Message, ...] = (
    _m("browser-get", "1.1", "an ordinary Chrome request, casing as Chrome sends it",
       ("Host", "example.com"),
       ("User-Agent", "Mozilla/5.0"),
       ("Accept", "text/html"),
       ("Accept-Encoding", "gzip, br"),
       ("DNT", "1"),
       ("Upgrade-Insecure-Requests", "1")),
    _m("tracing-pair", "1.1",
       "a trace id the proxy sets, and the same id underscored by a client",
       ("X-Request-ID", "from-proxy"),
       ("X_Request_Id", "from-client"),
       ("X-Forwarded-For", "203.0.113.9")),
    _m("login-response", "1.1", "two cookies and two auth challenges",
       ("Set-Cookie", "session=abc; Path=/; Expires=Wed, 21 Oct 2026 07:28:00 GMT"),
       ("Set-Cookie", "csrf=xyz; Path=/"),
       ("WWW-Authenticate", "Basic realm=\"api\""),
       ("WWW-Authenticate", "Bearer realm=\"api\"")),
    _m("h1-keepalive", "1.1", "an HTTP/1.1 message shoved onto an HTTP/2 connection",
       ("Host", "example.com"),
       ("Connection", "keep-alive"),
       ("Keep-Alive", "timeout=5")),
    _m("conditional-get", "1.1", "the field names that contain an `I`",
       ("If-None-Match", "\"v3\""),
       ("If-Modified-Since", "Tue, 18 Aug 2026 12:00:00 GMT"),
       ("IM", "feed")),
    _m("signed-request", "1.1", "a request whose signature covers the field names",
       ("Authorization", "AWS4-HMAC-SHA256 SignedHeaders=content-md5;host"),
       ("Content-MD5", "Q2h1Y2sgSW51ZwdyBGb3IN"),
       ("Host", "s3.example.com")),
    _m("duplicate-content-type", "1.1", "the same field sent twice, one field only",
       ("Content-Type", "application/json"),
       ("Content-Type", "text/plain"),
       ("Content-Length", "17")),
    _m("mixed-spelling", "1.1", "one field, two spellings, in one message",
       ("X-Api-Key", "k1"),
       ("X-API-KEY", "k2"),
       ("Accept", "*/*")),
    _m("cookie-split", "1.1", "three Cookie lines, which HTTP/2 encourages",
       ("Cookie", "a=1"),
       ("Cookie", "b=2"),
       ("Cookie", "c=3")),
    _m("te-trailers", "1.1", "the one connection-specific field HTTP/2 still allows",
       ("Host", "example.com"),
       ("TE", "trailers")),
    _m("te-gzip", "1.1", "the same field with the value that makes it malformed",
       ("Host", "example.com"),
       ("TE", "gzip")),
    _m("handwritten-frame", "2", "a field block built by hand with the wrong case",
       ("Content-Type", "application/grpc"),
       ("grpc-timeout", "10S")),
)

CORPUS_BY_NAME: Dict[str, Message] = {m.name: m for m in CORPUS}


def audit_corpus() -> Dict[Tuple[str, str], Delivery]:
    """Every message down every path."""
    return {(m.name, p): deliver(m, p) for m in CORPUS for p in PATHS}


def verdict_counts() -> Dict[Verdict, int]:
    counts = {v: 0 for v in Verdict}
    for d in audit_corpus().values():
        counts[d.verdict()] += 1
    return counts


def total_findings() -> Dict[str, int]:
    out = {"blocking": 0, "silent": 0, "advisory": 0}
    for d in audit_corpus().values():
        for f in findings(d):
            out[f.severity] += 1
    return out
