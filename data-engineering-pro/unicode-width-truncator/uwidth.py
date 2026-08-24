"""Truncating a string to N.

"Truncate to 20" does not name an operation.  A truncator is three
decisions - a *unit* of length, a *boundary* rule, and a *policy* for the
piece it removes - and the integer 20 carries none of them.  Every layer
that sees `20` picks its own unit: bytes in Go and Oracle, UTF-16 code
units in Java and JavaScript, code points in Python and Postgres,
grapheme clusters in a text renderer, terminal columns in a table.

This module runs one string through ten real truncators at one N and
reports the strings that come out, plus what each output *is*: valid
text or not, the same entity or a different one, inside the limit that
motivated the cut or still over it.

Nothing here is modelled.  The UTF-16 truncators are a real `node`
subprocess, grapheme clusters come from `regex`'s UAX #29 `\\X` on the
Python side and `Intl.Segmenter` on the ICU side, and column widths come
from `wcwidth`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import regex
from wcwidth import wcswidth, wcwidth

# --------------------------------------------------------------------------
# Unicode helpers
# --------------------------------------------------------------------------

ZWJ = "‍"
VS15 = "︎"
VS16 = "️"
REPLACEMENT = "�"

#: Bidi controls that open a scope, and the ones that close it.
BIDI_OPEN = {"‪", "‫", "‭", "‮", "⁦", "⁧", "⁨"}
BIDI_CLOSE = {"‬", "⁩"}


def graphemes(text: str) -> List[str]:
    """UAX #29 extended grapheme clusters, via `regex`'s `\\X`."""
    return regex.findall(r"\X", text)


def columns(text: str) -> int:
    """Terminal columns per `wcwidth`.

    `wcswidth` returns -1 for a string containing a non-printable
    character; we fall back to a per-character sum that charges 0 for
    those so a control character does not erase the whole measurement.
    """
    total = wcswidth(text)
    if total >= 0:
        return total
    return sum(max(wcwidth(ch), 0) for ch in text)


def is_lone_regional_indicator(cluster: str) -> bool:
    ris = [ch for ch in cluster if 0x1F1E6 <= ord(ch) <= 0x1F1FF]
    return len(ris) == 1 and len(cluster.strip(VS16)) == 1


def dangling(text: str) -> Optional[str]:
    """Name the trailing code point that will fuse with whatever is appended.

    A truncator that stops inside a cluster can leave a joiner, a
    combining mark, a variation selector or a single regional indicator
    at the end of the string.  None of these is an error.  Each one
    changes the *next* thing concatenated onto it.
    """
    if not text:
        return None
    last = text[-1]
    if last == ZWJ:
        return "ZERO WIDTH JOINER"
    if last in (VS15, VS16):
        return "VARIATION SELECTOR"
    if unicodedata.combining(last):
        return f"COMBINING MARK U+{ord(last):04X}"
    tail = graphemes(text)[-1]
    if is_lone_regional_indicator(tail):
        return "LONE REGIONAL INDICATOR"
    return None


def bidi_balance(text: str) -> int:
    """Open bidi scopes minus closed ones.  Non-zero leaks into the page."""
    depth = 0
    for ch in text:
        if ch in BIDI_OPEN:
            depth += 1
        elif ch in BIDI_CLOSE:
            depth -= 1
    return depth


def has_replacement(text: str) -> bool:
    return REPLACEMENT in text


def has_lone_surrogate(text: str) -> bool:
    """True when the string holds an unpaired surrogate.

    A JavaScript string is a sequence of UTF-16 code units, not of
    characters, so `s.slice(0, n)` can end mid-pair and leave a lone
    surrogate behind.  The result is a perfectly ordinary JS string that
    **cannot be encoded as UTF-8 at all** - not mojibake, not a
    replacement character, but a value with no byte representation.  It
    fails at the first `encode()`, `JSON.parse` on the far side, or
    database write.
    """
    return any(0xD800 <= ord(ch) <= 0xDFFF for ch in text)


def encode_utf8(text: str) -> bytes:
    """UTF-8 bytes, tolerating lone surrogates so they can be measured.

    `surrogatepass` is what makes the ill-formed output *countable*
    here.  It is not what a database or an HTTP client does - those
    raise, which is the point of section 5.
    """
    return text.encode("utf-8", "surrogatepass")


# --------------------------------------------------------------------------
# The node bridge - real JavaScript string semantics, not an imitation
# --------------------------------------------------------------------------

NODE = shutil.which("node")

_NODE_SCRIPT = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const seg = new Intl.Segmenter('en', {granularity: 'grapheme'});
const out = input.jobs.map(job => {
  const s = job.text, n = job.n;
  switch (job.op) {
    case 'utf16_units':   return s.slice(0, n);
    case 'utf16_safe_cp': return [...s].slice(0, n).join('');
    case 'js_graphemes':  return [...seg.segment(s)].slice(0, n).map(x => x.segment).join('');
    case 'count_graphemes': return String([...seg.segment(s)].length);
    case 'utf16_length':  return String(s.length);
  }
});
process.stdout.write(JSON.stringify({out, icu: process.versions.icu, unicode: process.versions.unicode}));
"""


@lru_cache(maxsize=1)
def _node_available() -> bool:
    return NODE is not None


def node_batch(jobs: Sequence[Dict[str, object]]) -> Tuple[List[str], Dict[str, str]]:
    """Run every JavaScript-side operation in one subprocess.

    Raises if node is missing rather than silently substituting a Python
    imitation - the point of these truncators is that they are the real
    engine, so a fallback would make the comparison meaningless.
    """
    if not _node_available():
        raise RuntimeError(
            "node is required: utf16_units / utf16_safe_cp / js_graphemes are "
            "real JavaScript, not a Python model of it."
        )
    proc = subprocess.run(
        [NODE, "-e", _NODE_SCRIPT],
        input=json.dumps({"jobs": list(jobs)}),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    return payload["out"], {"icu": payload["icu"], "unicode": payload["unicode"]}


@lru_cache(maxsize=4096)
def _node_one(op: str, text: str, n: int) -> str:
    out, _ = node_batch([{"op": op, "text": text, "n": n}])
    return out[0]


@lru_cache(maxsize=1)
def node_versions() -> Dict[str, str]:
    _, meta = node_batch([{"op": "utf16_length", "text": "x", "n": 1}])
    return meta


# --------------------------------------------------------------------------
# The truncators
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Truncator:
    name: str
    unit: str
    #: Where a working engineer actually meets this behaviour.
    seen_in: str
    fn: Callable[[str, int], str]

    def cut(self, text: str, n: int) -> str:
        return self.fn(text, n)


def _utf8_bytes_replace(text: str, n: int) -> str:
    """`s[:n]` on a Go string, `head -c`, a byte-sliced buffer.

    A cut inside a multi-byte sequence leaves a partial sequence.  The
    bytes are no longer UTF-8; decoding them with `errors='replace'` is
    what the *next* consumer does, and it is where U+FFFD appears.
    """
    return text.encode("utf-8")[:n].decode("utf-8", "replace")


def _utf8_bytes_backoff(text: str, n: int) -> str:
    """The same byte limit, backed off to a code point boundary."""
    return text.encode("utf-8")[:n].decode("utf-8", "ignore")


def _code_points(text: str, n: int) -> str:
    """`s[:n]` in Python, `substr()` in Postgres, `LEFT()` on utf8mb4."""
    return text[:n]


def _py_graphemes(text: str, n: int) -> str:
    """UAX #29 clusters as Python's `regex` module segments them."""
    return "".join(graphemes(text)[:n])


def _js_graphemes(text: str, n: int) -> str:
    """UAX #29 clusters as ICU segments them, through `Intl.Segmenter`."""
    return _node_one("js_graphemes", text, n)


def _utf16_units(text: str, n: int) -> str:
    """`String.prototype.slice` - Java, C#, JavaScript, SQL Server nvarchar."""
    return _node_one("utf16_units", text, n)


def _utf16_safe_cp(text: str, n: int) -> str:
    """`[...s].slice(0, n)` - the fix applied after the first mojibake bug."""
    return _node_one("utf16_safe_cp", text, n)


def _term_columns(text: str, n: int) -> str:
    """Fill to at most n terminal columns, code point by code point."""
    used, out = 0, []
    for ch in text:
        w = max(wcwidth(ch), 0)
        if used + w > n:
            break
        used += w
        out.append(ch)
    return "".join(out)


def _grapheme_columns(text: str, n: int) -> str:
    """Fill to at most n columns without ever splitting a cluster."""
    used, out = 0, []
    for cluster in graphemes(text):
        w = columns(cluster)
        if used + w > n:
            break
        used += w
        out.append(cluster)
    return "".join(out)


def _word_boundary(text: str, n: int) -> str:
    """Cut at the last space at or before n code points."""
    if len(text) <= n:
        return text
    head = text[:n]
    idx = head.rfind(" ")
    return head[:idx] if idx > 0 else head


TRUNCATORS: List[Truncator] = [
    Truncator("utf8_bytes_replace", "bytes", "Go s[:n], head -c, byte buffers", _utf8_bytes_replace),
    Truncator("utf8_bytes_backoff", "bytes", "MySQL column overflow, ICU byte trim", _utf8_bytes_backoff),
    Truncator("code_points", "code points", "Python s[:n], Postgres substr, MySQL LEFT", _code_points),
    Truncator("utf16_units", "UTF-16 units", "Java/C#/JS substring, SQL Server nvarchar", _utf16_units),
    Truncator("utf16_safe_cp", "code points", "[...s].slice(0,n) - the first fix", _utf16_safe_cp),
    Truncator("py_graphemes", "graphemes (regex)", "Python text pipeline, UAX #29", _py_graphemes),
    Truncator("js_graphemes", "graphemes (ICU)", "Intl.Segmenter, Swift Character", _js_graphemes),
    Truncator("term_columns", "columns", "wcwidth budget, naive CLI table", _term_columns),
    Truncator("grapheme_columns", "columns", "cluster-safe CLI table, the careful one", _grapheme_columns),
    Truncator("word_boundary", "code points", "teaser/preview text, textwrap.shorten", _word_boundary),
]

BYTE_TRUNCATORS = {"utf8_bytes_replace", "utf8_bytes_backoff"}

TRUNCATOR_BY_NAME = {t.name: t for t in TRUNCATORS}


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    name: str
    text: str
    n: int
    source: str
    #: What the string *is*, when a cut can silently change it into
    #: something else that is equally valid.
    entity: str = ""


def _nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


CORPUS: List[Case] = [
    Case("ascii-short", "Data engineer", 20, "a bio field that fits"),
    Case("ascii-long", "Data engineer, Sydney. Ex-consulting.", 20, "a bio field that does not"),
    Case("accent-nfc", "José Muñoz, Madrid", 12, "a name typed on a Mac", entity="José"),
    Case("accent-nfd", _nfd("José Muñoz, Madrid"), 12, "the same name from an iOS upload", entity="José"),
    Case("emoji-family", "Family: \U0001F468‍\U0001F469‍\U0001F467‍\U0001F466 in Perth", 12,
         "a profile bio", entity="family of four"),
    Case("emoji-skin", "Nice work \U0001F44D\U0001F3FD keep going", 12, "a chat message", entity="thumbs up, medium skin"),
    Case("emoji-flags", "Route \U0001F1FA\U0001F1F8\U0001F1EC\U0001F1E7\U0001F1EB\U0001F1F7 today", 9,
         "a shipping label", entity="US, GB, FR"),
    Case("emoji-keycap", "Press 1️⃣ then 2️⃣", 10, "an instruction line", entity="keycap 1"),
    Case("emoji-vs16", "Coffee ☕️ break", 9, "a calendar title", entity="emoji-presentation coffee"),
    Case("cjk-bio", "数据工程师，您好", 12, "a CJK display name"),
    Case("cjk-mixed", "Tokyo 東京 support desk", 14, "a mixed-script queue name"),
    Case("hangul-nfc", "한국어 지원", 6, "a Korean label"),
    Case("hangul-nfd", _nfd("한국어 지원"), 6, "the same label, NFD from macOS"),
    Case("devanagari", "क्षितिज नाम", 6, "a Hindi display name"),
    Case("thai", "กำหนดการ", 6, "Thai - no spaces to break on"),
    Case("tamil", "நிலா கணக்கு", 6, "a Tamil name"),
    Case("arabic", "مرحبا بالعالم", 8, "an RTL greeting"),
    Case("bidi-override", "file ‮exe.doc‬ ok", 12, "a filename with an RTL override"),
    Case("astral-math", "Set \U0001D400\U0001D401\U0001D402\U0001D403 done", 8, "astral maths letters"),
    Case("combining-stack", "á̂̃̄̅b́̂ tail", 6, "stacked combining marks"),
    Case("zwsp", "long​word​wrap​here", 12, "zero-width spaces from a CMS"),
    Case("tab-control", "col1\tcol2\x07bell", 8, "a control character in a data cell"),
    Case("emoji-run", "\U0001F600\U0001F601\U0001F602\U0001F603\U0001F604\U0001F605", 8, "an emoji-only reaction"),
    Case("url", "https://example.com/a/very/long/path", 20, "a link in a preview card"),
    Case("mixed-width", "ID 日本-2024 \U0001F600 ok", 12, "a mixed CJK / emoji / ASCII cell"),
    Case("ri-odd", "\U0001F1FA\U0001F1F8\U0001F1EC done", 5, "an odd count of regional indicators"),
]

CASE_BY_NAME = {c.name: c for c in CORPUS}


# --------------------------------------------------------------------------
# One cut, fully measured
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cut:
    truncator: str
    unit: str
    text: str
    n: int

    @property
    def bytes_out(self) -> int:
        return len(encode_utf8(self.text))

    @property
    def code_points(self) -> int:
        return len(self.text)

    @property
    def grapheme_count(self) -> int:
        return len(graphemes(self.text))

    @property
    def columns_out(self) -> int:
        return columns(self.text)

    @property
    def well_formed(self) -> bool:
        """False when the cut produced something that is not text any more.

        Two distinct failures: a U+FFFD where a byte cut landed inside a
        multi-byte sequence, and a lone surrogate where a UTF-16 cut
        landed inside a pair.  The first is visible mojibake; the second
        has no UTF-8 encoding at all.
        """
        return not has_replacement(self.text) and not has_lone_surrogate(self.text)

    @property
    def lone_surrogate(self) -> bool:
        return has_lone_surrogate(self.text)

    @property
    def dangling(self) -> Optional[str]:
        return dangling(self.text)

    @property
    def bidi_leak(self) -> int:
        return bidi_balance(self.text)

    @property
    def overflows_own_limit(self) -> bool:
        """A byte truncator whose output does not fit in the byte budget.

        `s.encode()[:n]` cuts a 4-byte emoji after its first byte; the
        consumer decodes that partial byte to U+FFFD, which re-encodes to
        *three* bytes.  The result is longer than the limit it enforced.
        """
        return self.unit == "bytes" and self.bytes_out > self.n


def cut_all(case: Case) -> Dict[str, Cut]:
    """Every truncator's answer for one case, at that case's single N."""
    return {
        t.name: Cut(t.name, t.unit, t.cut(case.text, case.n), case.n)
        for t in TRUNCATORS
    }


def read_corpus(cases: Optional[Sequence[Case]] = None) -> Dict[str, Dict[str, Cut]]:
    return {c.name: cut_all(c) for c in (cases or CORPUS)}


# --------------------------------------------------------------------------
# Identity: a cut that yields a different valid thing
# --------------------------------------------------------------------------

#: A cut is an *identity change* when the output is perfectly well-formed
#: text that names something else.  These are the ones no validator
#: catches, because there is nothing wrong with the result.
IDENTITY_PROBES: List[Tuple[str, str, str]] = [
    ("\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466", "\U0001F468‍\U0001F469",
     "family of four -> couple, no children"),
    ("\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466", "\U0001F468",
     "family of four -> one man"),
    ("\U0001F44D\U0001F3FD", "\U0001F44D", "medium skin tone -> default yellow"),
    ("\U0001F1FA\U0001F1F8", "\U0001F1FA", "flag of the US -> a lone letter U"),
    ("1️⃣", "1", "keycap 1 -> the digit 1"),
    ("☕️", "☕", "emoji coffee -> text-presentation coffee"),
]


def identity_change(original: str, truncated: str) -> Optional[str]:
    """Name the entity swap a cut performed, if it performed one."""
    if not truncated or truncated == original:
        return None
    for whole, part, description in IDENTITY_PROBES:
        if whole in original and truncated.endswith(part) and whole not in truncated:
            if part and part != whole:
                return description
    # An accent silently dropped by an NFD cut is the same class of bug.
    nfc_original = unicodedata.normalize("NFC", original)
    nfc_cut = unicodedata.normalize("NFC", truncated)
    for base in ("é", "ñ", "ü", "á", "ç"):
        stripped = unicodedata.normalize("NFD", base)[0]
        if base in nfc_original and stripped in nfc_cut and base not in nfc_cut:
            return f"'{base}' -> '{stripped}' (combining mark cut off)"
    return None


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

VERDICTS = [
    "agreed",
    "unit-drift",
    "boundary-split",
    "identity-change",
    "dangling-joiner",
    "byte-overflow",
    "width-blowout",
    "bidi-leak",
]


@dataclass(frozen=True)
class Verdict:
    case: str
    verdict: str
    distinct_outputs: int
    detail: str
    flags: Tuple[str, ...] = field(default=())


def verdict_for(case: Case, cuts: Optional[Dict[str, Cut]] = None) -> Verdict:
    """Classify one case by the worst thing any truncator did to it.

    Order matters: a cut that produces invalid text is a worse finding
    than one that produces a different valid entity, which is worse than
    one that merely disagrees about length.
    """
    cuts = cuts or cut_all(case)
    outputs = {c.text for c in cuts.values()}
    flags: List[str] = []
    detail = ""

    split = [c for c in cuts.values() if not c.well_formed]
    ident = [(c, identity_change(case.text, c.text)) for c in cuts.values()]
    ident = [(c, d) for c, d in ident if d]
    dang = [c for c in cuts.values() if c.dangling]
    over = [c for c in cuts.values() if c.overflows_own_limit]
    leak = [c for c in cuts.values() if c.bidi_leak != 0 and bidi_balance(case.text) == 0]
    blow = [
        c for c in cuts.values()
        if c.unit in ("code points", "UTF-16 units", "bytes") and c.columns_out > case.n
    ]

    if split:
        flags.append("boundary-split")
    if ident:
        flags.append("identity-change")
    if dang:
        flags.append("dangling-joiner")
    if over:
        flags.append("byte-overflow")
    if blow:
        flags.append("width-blowout")
    if leak:
        flags.append("bidi-leak")

    if split:
        fffd = [c.truncator for c in split if has_replacement(c.text)]
        surr = [c.truncator for c in split if c.lone_surrogate]
        parts = []
        if fffd:
            parts.append(f"{len(fffd)} emitted U+FFFD")
        if surr:
            parts.append(f"{len(surr)} left a lone surrogate (no UTF-8 encoding)")
        verdict, detail = "boundary-split", "; ".join(parts)
    elif ident:
        verdict, detail = "identity-change", ident[0][1]
    elif dang:
        verdict, detail = "dangling-joiner", f"{dang[0].truncator} ends in {dang[0].dangling}"
    elif leak:
        verdict, detail = "bidi-leak", f"{leak[0].truncator} leaves an unbalanced bidi scope"
    elif over:
        verdict, detail = "byte-overflow", f"{over[0].truncator} -> {over[0].bytes_out} bytes for n={case.n}"
    elif blow:
        verdict, detail = "width-blowout", f"{blow[0].truncator} -> {blow[0].columns_out} columns for n={case.n}"
    elif len(outputs) == 1:
        verdict, detail = "agreed", "every truncator returned the same string"
    else:
        verdict, detail = "unit-drift", f"{len(outputs)} distinct strings, all well-formed"

    return Verdict(case.name, verdict, len(outputs), detail, tuple(flags))


def verdict_census(cases: Optional[Sequence[Case]] = None) -> Dict[str, int]:
    census = {v: 0 for v in VERDICTS}
    for case in cases or CORPUS:
        census[verdict_for(case).verdict] += 1
    return census


# --------------------------------------------------------------------------
# Sinks: what the limit at the other end actually counts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sink:
    name: str
    unit: str
    note: str
    measure: Callable[[str], int]


SINKS: List[Sink] = [
    Sink("mysql_utf8mb4_varchar", "code points", "VARCHAR(N) counts characters; the row byte cap is separate", len),
    Sink("postgres_varchar", "code points", "varchar(N) is N code points, encoding-independent", len),
    Sink("oracle_varchar2_byte", "bytes", "VARCHAR2(N) is BYTE unless NLS_LENGTH_SEMANTICS=CHAR",
         lambda s: len(encode_utf8(s))),
    Sink("sqlserver_nvarchar", "UTF-16 units", "nvarchar(N) counts UTF-16 code units, so astral costs 2",
         lambda s: len(s.encode("utf-16-le", "surrogatepass")) // 2),
    Sink("http_header_bytes", "bytes", "a header value budget is bytes on the wire",
         lambda s: len(encode_utf8(s))),
    Sink("fixed_width_column", "columns", "a CLI table or a monospaced report cell", columns),
]

SINK_BY_NAME = {s.name: s for s in SINKS}


def fits(sink: Sink, text: str, n: int) -> bool:
    return sink.measure(text) <= n


def sink_matrix(case: Case) -> Dict[Tuple[str, str], bool]:
    """Does truncating to N actually make the value fit a limit of N?

    Ten truncators against six sinks.  The answer is not "yes" - it is
    "which unit did each of them mean".
    """
    cuts = cut_all(case)
    return {
        (t_name, s.name): fits(s, cut.text, case.n)
        for t_name, cut in cuts.items()
        for s in SINKS
    }


def sink_failures(cases: Optional[Sequence[Case]] = None) -> List[Tuple[str, str, str, int, int]]:
    """Every (case, truncator, sink) where the value still overflows."""
    out = []
    for case in cases or CORPUS:
        for (t_name, s_name), ok in sink_matrix(case).items():
            if not ok:
                sink = SINK_BY_NAME[s_name]
                text = cut_all(case)[t_name].text
                out.append((case.name, t_name, s_name, sink.measure(text), case.n))
    return out


def choose_truncator(sink_name: str) -> str:
    """The only defensible way to pick a truncator: match the sink's unit.

    A truncator is correct relative to the limit it is protecting, and
    nothing else.  Cutting to grapheme clusters is the humane choice for
    a *display*, and it is the wrong choice for an Oracle `VARCHAR2(20)`,
    which will still reject it.
    """
    unit = SINK_BY_NAME[sink_name].unit
    for t in TRUNCATORS:
        if t.unit == unit and t.name not in ("utf8_bytes_replace", "word_boundary"):
            return t.name
    return "grapheme_columns"


def safe_truncate(text: str, n: int, sink_name: str, ellipsis: str = "…") -> str:
    """Cut for a named sink, reserving room for the ellipsis, never mid-cluster.

    Three properties the ten roster truncators do not all have: the
    result fits the sink's own measure, the ellipsis is inside the
    budget rather than pushing past it, and no grapheme cluster is
    split - so nothing changes identity and nothing dangles.
    """
    sink = SINK_BY_NAME[sink_name]
    if sink.measure(text) <= n:
        return text
    budget = n - sink.measure(ellipsis)
    if budget <= 0:
        return ""
    used, out = 0, []
    for cluster in graphemes(text):
        cost = sink.measure(cluster)
        if used + cost > budget:
            break
        used += cost
        out.append(cluster)
    result = "".join(out)
    while result and (dangling(result) or bidi_balance(result) != 0):
        result = "".join(graphemes(result)[:-1])
    return result + ellipsis


# --------------------------------------------------------------------------
# The ellipsis budget
# --------------------------------------------------------------------------


def ellipsis_cost(ellipsis: str = "…") -> Dict[str, int]:
    """One ellipsis, measured in every unit a limit might be written in."""
    return {
        "bytes": len(encode_utf8(ellipsis)),
        "code points": len(ellipsis),
        "UTF-16 units": len(ellipsis.encode("utf-16-le", "surrogatepass")) // 2,
        "columns": columns(ellipsis),
    }


def naive_ellipsis_overflow(cases: Optional[Sequence[Case]] = None) -> List[Tuple[str, str, int, int]]:
    """Cut to exactly N, then append "..." - how far over N does that land?"""
    out = []
    for case in cases or CORPUS:
        for t_name, cut in cut_all(case).items():
            if cut.text == case.text:
                continue  # nothing was removed, so no ellipsis is appended
            with_ell = cut.text + "…"
            sink_unit = TRUNCATOR_BY_NAME[t_name].unit
            measured = {
                "bytes": len(encode_utf8(with_ell)),
                "code points": len(with_ell),
                "UTF-16 units": len(with_ell.encode("utf-16-le", "surrogatepass")) // 2,
                "graphemes (regex)": len(graphemes(with_ell)),
                "graphemes (ICU)": int(_node_one("count_graphemes", with_ell, 0)),
                "columns": columns(with_ell),
            }[sink_unit]
            if measured > case.n:
                out.append((case.name, t_name, measured, case.n))
    return out


# --------------------------------------------------------------------------
# Idempotence
# --------------------------------------------------------------------------


def idempotent(truncator: str, text: str, n: int) -> bool:
    """truncate(truncate(s, n), n) == truncate(s, n)?

    A truncator that is not idempotent has no fixed point: passing the
    same value through the same limit twice - two services, one retry -
    keeps removing characters.
    """
    t = TRUNCATOR_BY_NAME[truncator]
    once = t.cut(text, n)
    return t.cut(once, n) == once


def idempotence_failures(cases: Optional[Sequence[Case]] = None) -> List[Tuple[str, str]]:
    return [
        (c.name, t.name)
        for c in (cases or CORPUS)
        for t in TRUNCATORS
        if not idempotent(t.name, c.text, c.n)
    ]


# --------------------------------------------------------------------------
# Where the two segmenters in one pipeline disagree
# --------------------------------------------------------------------------


def segmenter_disagreements(cases: Optional[Sequence[Case]] = None) -> List[Tuple[str, int, int]]:
    """Cases where Python's `regex` and ICU count different clusters.

    Both implement UAX #29.  They ship different UCD versions, and the
    rules changed - so an API written in Node and a worker written in
    Python disagree about how many characters a Hindi name has, with no
    error anywhere.
    """
    out = []
    for case in cases or CORPUS:
        py = len(graphemes(case.text))
        js = int(_node_one("count_graphemes", case.text, 0))
        if py != js:
            out.append((case.name, py, js))
    return out


def unit_spread(case: Case) -> Dict[str, int]:
    """The same string, measured in every unit a limit could be written in."""
    return {
        "bytes": len(encode_utf8(case.text)),
        "code points": len(case.text),
        "UTF-16 units": len(case.text.encode("utf-16-le", "surrogatepass")) // 2,
        "graphemes (regex)": len(graphemes(case.text)),
        "graphemes (ICU)": int(_node_one("count_graphemes", case.text, 0)),
        "columns": columns(case.text),
    }


def flag_census(cases: Optional[Sequence[Case]] = None) -> Dict[str, int]:
    """How many cases carry each finding.

    Distinct from `verdict_census`: a verdict is the single worst thing
    that happened to a case, a flag is every thing that happened.  One
    string can split a boundary *and* change identity *and* blow the
    column budget.
    """
    census = {v: 0 for v in VERDICTS if v not in ("agreed", "unit-drift")}
    for case in cases or CORPUS:
        for flag in verdict_for(case).flags:
            census[flag] += 1
    return census


def distinct_output_count(cases: Optional[Sequence[Case]] = None) -> Tuple[int, int]:
    """(distinct strings produced, total cuts performed) across the corpus."""
    total = distinct = 0
    for case in cases or CORPUS:
        cuts = cut_all(case)
        total += len(cuts)
        distinct += len({c.text for c in cuts.values()})
    return distinct, total


def sink_failure_rate(cases: Optional[Sequence[Case]] = None) -> Tuple[int, int]:
    """(runs where the truncated value still overflows, total runs).

    The question this answers: does `truncate(value, 20)` make the value
    fit a limit of 20?  Only when the truncator's unit is the sink's
    unit - which is a coincidence unless somebody arranged it.
    """
    cases = list(cases or CORPUS)
    total = len(cases) * len(TRUNCATORS) * len(SINKS)
    return len(sink_failures(cases)), total


def safe_truncate_audit(cases: Optional[Sequence[Case]] = None) -> List[Tuple[str, str, bool, bool, bool]]:
    """`safe_truncate` against every sink: fits, no dangle, no split."""
    out = []
    for case in cases or CORPUS:
        for sink in SINKS:
            result = safe_truncate(case.text, case.n, sink.name)
            out.append((
                case.name,
                sink.name,
                fits(sink, result, case.n),
                dangling(result) is None,
                not has_replacement(result) and not has_lone_surrogate(result),
            ))
    return out
