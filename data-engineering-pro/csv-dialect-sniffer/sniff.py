"""CSV dialect detection that reports the cases the bytes do not determine.

`csv.Sniffer().sniff()` returns a Dialect or raises. It has no third answer, so a
file with two internally-consistent parses gets one of them and no warning. This
module enumerates every viable parse instead of returning one, and separates
three verdicts: unambiguous (exactly one candidate parses the file cleanly),
contested (several do, and the bytes cannot choose), undetermined (none do).

Standard library only: csv, io, re, codecs, collections, dataclasses.
"""

from __future__ import annotations

import codecs
import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

DELIMITERS: Tuple[str, ...] = (",", ";", "\t", "|", ":", " ")
QUOTECHARS: Tuple[Optional[str], ...] = ('"', "'", None)

ENCODINGS: Tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin-1",
    "utf-16",
    "shift_jis",
)

# Encodings that map every possible byte to some character, so decoding them
# never raises. A successful decode is therefore not evidence of anything.
NEVER_FAIL: frozenset = frozenset({"latin-1", "iso-8859-1", "cp437", "macroman"})

BOMS: Tuple[Tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)

_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
_EURO_FLOAT = re.compile(r"^[+-]?\d+,\d+$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def parse(
    text: str,
    delimiter: str,
    quotechar: Optional[str] = '"',
    doublequote: bool = True,
) -> List[List[str]]:
    """Parse text under one explicit dialect. quotechar=None means no quoting."""
    kwargs: Dict[str, object] = {"delimiter": delimiter, "doublequote": doublequote}
    if quotechar is None:
        kwargs["quoting"] = csv.QUOTE_NONE
        kwargs["quotechar"] = None
    else:
        kwargs["quotechar"] = quotechar
    stream = io.StringIO(text, newline="")
    return list(csv.reader(stream, **kwargs))  # type: ignore[arg-type]


@dataclass
class Shape:
    """What one candidate dialect makes of the file."""

    delimiter: str
    quotechar: Optional[str]
    records: int
    counts: Counter
    modal: int
    consistency: float
    ragged: List[int]
    blanks: int
    unbalanced_quotes: int
    viable: bool
    reason: str
    truncated_tail: bool = False
    fields_with_newline: int = 0

    @property
    def label(self) -> str:
        d = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe", ":": "colon", " ": "space"}
        q = "none" if self.quotechar is None else repr(self.quotechar)
        return "{0} / quote={1}".format(d.get(self.delimiter, repr(self.delimiter)), q)


def shape_of(
    text: str,
    delimiter: str,
    quotechar: Optional[str] = '"',
    min_fields: int = 2,
) -> Shape:
    """Measure one candidate dialect. Never raises; a failed parse is a Shape."""
    try:
        rows = parse(text, delimiter, quotechar)
    except csv.Error as exc:
        return Shape(delimiter, quotechar, 0, Counter(), 0, 0.0, [], 0, 0, False, str(exc))

    blanks = sum(1 for r in rows if not r or (len(r) == 1 and r[0] == ""))
    body = [r for r in rows if r and not (len(r) == 1 and r[0] == "")]
    counts = Counter(len(r) for r in body)
    if not body:
        return Shape(delimiter, quotechar, 0, counts, 0, 0.0, [], blanks, 0, False, "no records")

    modal, modal_n = counts.most_common(1)[0]
    # A file sniffed by prefix ends mid-record. That trailing fragment is an
    # artefact of the sample, not of the file, so it is excluded and reported -
    # only when it is the final record, only when the text has no terminator
    # after it, and only when it is the sole ragged row.
    truncated = False
    if (len(body) > 1 and not text.endswith(("\n", "\r"))
            and len(body[-1]) != modal
            and sum(1 for r in body if len(r) != modal) == 1):
        body = body[:-1]
        counts = Counter(len(r) for r in body)
        modal, modal_n = counts.most_common(1)[0]
        truncated = True

    consistency = modal_n / len(body)
    ragged = [i for i, r in enumerate(body) if len(r) != modal]

    # A field that opens with the quotechar but does not close with it means the
    # quotechar guess is wrong, even when the field counts happen to line up.
    unbalanced = 0
    if quotechar is not None:
        for row in body:
            for f in row:
                s = f.strip()
                if len(s) > 1 and s.startswith(quotechar) != s.endswith(quotechar):
                    unbalanced += 1

    if modal < min_fields:
        viable, reason = False, "delimiter yields {0} field(s) per record".format(modal)
    elif consistency < 1.0:
        viable, reason = False, "{0} of {1} records are ragged".format(len(ragged), len(body))
    elif unbalanced:
        viable, reason = False, "{0} field(s) with an unbalanced quote".format(unbalanced)
    else:
        viable = True
        reason = "{0} records x {1} fields, no ragged rows".format(len(body), modal)
        if truncated:
            reason += " (trailing partial record excluded)"

    swallowed = sum(1 for r in body for f in r if "\n" in f or "\r" in f)

    return Shape(
        delimiter, quotechar, len(body), counts, modal, consistency,
        ragged, blanks, unbalanced, viable, reason, truncated, swallowed,
    )


@dataclass
class DelimiterVerdict:
    status: str  # "unambiguous" | "contested" | "undetermined"
    viable: List[Shape]
    all_shapes: List[Shape]
    preferred: Optional[Shape]
    reason: str
    untested: List[str] = field(default_factory=list)

    @property
    def column_counts(self) -> List[int]:
        return sorted({s.modal for s in self.viable})


def classify_delimiter(
    text: str,
    delimiters: Sequence[str] = DELIMITERS,
    quotechars: Sequence[Optional[str]] = QUOTECHARS,
) -> DelimiterVerdict:
    """Enumerate every dialect that parses the file cleanly, then judge.

    Two candidates that differ only in quotechar but produce an identical parse
    are one candidate: the file contains no quotes, so the setting is untested by
    it. Distinct field counts are what make a file genuinely contested.
    """
    shapes = [shape_of(text, d, q) for d in delimiters for q in quotechars]
    viable = [s for s in shapes if s.viable]

    # Collapse candidates whose parse is indistinguishable on this input. Two
    # quotechars that produce an identical parse are not two answers - they are
    # one answer plus a setting this file does not exercise.
    seen: Dict[Tuple[str, int, int], Shape] = {}
    collapsed: Dict[Tuple[str, int, int], List[Shape]] = {}
    for s in sorted(viable, key=lambda s: (s.delimiter, s.quotechar is None)):
        key = (s.delimiter, s.modal, s.records)
        seen.setdefault(key, s)
        collapsed.setdefault(key, []).append(s)
    distinct = list(seen.values())

    if not distinct:
        return DelimiterVerdict("undetermined", [], shapes, None,
                                "no candidate delimiter parses every record to the same width")

    # Tie-break heuristics, in order: more columns, then more records (a parse
    # that merges records has lost rows), then the earlier-listed delimiter.
    # These are preferences, not evidence, which is why a contested verdict
    # still reports every candidate.
    preferred = max(distinct, key=lambda s: (
        s.modal, s.records,
        -DELIMITERS.index(s.delimiter) if s.delimiter in DELIMITERS else 0))
    pkey = (preferred.delimiter, preferred.modal, preferred.records)
    untested = ["quote={0}".format("none" if s.quotechar is None else repr(s.quotechar))
                for s in collapsed[pkey][1:]]

    if len(distinct) == 1:
        return DelimiterVerdict("unambiguous", distinct, shapes, preferred,
                                "exactly one dialect parses the file cleanly", untested)

    widths = sorted({s.modal for s in distinct})
    counts_seen = sorted({s.records for s in distinct})
    if len(widths) > 1:
        shape_note = "implying {0} column(s)".format(" or ".join(str(w) for w in widths))
    else:
        shape_note = "all {0} columns wide but {1} record(s) long".format(
            widths[0], " or ".join(str(c) for c in counts_seen))
    return DelimiterVerdict(
        "contested", distinct, shapes, preferred,
        "{0} dialects parse the file cleanly, {1} - the bytes do not choose".format(
            len(distinct), shape_note),
        untested,
    )


def sniffer_says(text: str) -> Optional[str]:
    """What csv.Sniffer picks, or None if it raises. For comparison only."""
    try:
        return csv.Sniffer().sniff(text).delimiter
    except csv.Error:
        return None


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #


@dataclass
class EncodingReport:
    bom: Optional[str]
    decodes: Dict[str, bool]
    texts: Dict[str, str]
    ruled_out: List[str]
    not_evidence: List[Tuple[str, str]]
    plausible: List[str]
    distinct_texts: int
    verdict: str
    reason: str

    @property
    def survived(self) -> List[str]:
        return [e for e, ok in self.decodes.items() if ok]


def _not_evidence(enc: str, bom: Optional[str]) -> Optional[Tuple[str, bool]]:
    """Why a successful decode by this encoding proves little, and whether that
    is disqualifying (the decode is not credible) or merely a caveat (the decode
    is credible but its success carries almost no information)."""
    if enc in NEVER_FAIL:
        return "maps all 256 byte values - cannot fail on any input", True
    if enc.startswith(("utf-16", "utf-32")) and bom is None:
        return "any byte string of the right length decodes, usually to noise", True
    if enc == "cp1252":
        return "only 5 of 256 byte values are undefined (81 8d 8f 90 9d)", False
    return None


def probe_encoding(raw: bytes, encodings: Sequence[str] = ENCODINGS) -> EncodingReport:
    """Report which encodings *can* decode these bytes, and which agree.

    A decode that succeeds is weak evidence at best and no evidence at all for
    some encodings. The useful signals are which encodings are *ruled out* (a
    UnicodeDecodeError is a fact about the bytes) and how many distinct strings
    the survivors produce.
    """
    bom: Optional[str] = None
    for marker, name in BOMS:
        if raw.startswith(marker):
            bom = name
            break

    decodes: Dict[str, bool] = {}
    texts: Dict[str, str] = {}
    for enc in encodings:
        try:
            texts[enc] = raw.decode(enc)
            decodes[enc] = True
        except (UnicodeDecodeError, UnicodeError):
            decodes[enc] = False

    ok = [e for e in encodings if decodes[e]]
    ruled_out = [e for e in encodings if not decodes[e]]
    distinct = len({texts[e] for e in ok})

    weak: List[Tuple[str, str]] = []
    disqualified: set = set()
    for e in ok:
        got = _not_evidence(e, bom)
        if got:
            why, hard = got
            weak.append((e, why))
            if hard:
                disqualified.add(e)

    # A decode that yields C1 control characters (U+0080-U+009F) is not credible:
    # real text does not contain them. This is what rules latin-1 out for a
    # Windows export, where cp1252 puts curly quotes in exactly that byte range.
    for e in ok:
        n_c1 = sum(1 for ch in texts[e] if "\x80" <= ch <= "\x9f")
        if n_c1:
            weak.append((e, "decode contains {0} C1 control character(s) - not text".format(n_c1)))
            disqualified.add(e)

    plausible = [e for e in ok if e not in disqualified]

    if bom:
        verdict, reason = bom, "byte order mark present - this one is not a guess"
    elif "utf-8" in ok and distinct == 1:
        verdict, reason = "utf-8", "utf-8 decodes and every surviving encoding agrees"
    elif "utf-8" in ok:
        verdict = "utf-8"
        reason = "utf-8 decodes; {0} encodings decode to {1} different strings".format(
            len(ok), distinct)
    elif len(plausible) == 1:
        verdict = plausible[0]
        reason = "utf-8 ruled out; {0} is the only survivor left after the C1 test".format(
            plausible[0])
    elif ok:
        verdict = "undetermined"
        reason = "utf-8 ruled out; {0} candidates decode to {1} different strings".format(
            len(ok), distinct)
    else:
        verdict, reason = "undecodable", "no candidate encoding decodes these bytes"

    return EncodingReport(bom, decodes, texts, ruled_out, weak, plausible,
                          distinct, verdict, reason)


def mojibake_pairs(raw: bytes, a: str = "utf-8", b: str = "latin-1") -> List[Tuple[str, str]]:
    """Characters where two encodings disagree, aligned by position where possible."""
    try:
        ta, tb = raw.decode(a), raw.decode(b)
    except (UnicodeDecodeError, UnicodeError):
        return []
    out: List[Tuple[str, str]] = []
    for wa, wb in zip(re.findall(r"\S+", ta), re.findall(r"\S+", tb)):
        if wa != wb:
            out.append((wa, wb))
    return out


# --------------------------------------------------------------------------- #
# line terminators
# --------------------------------------------------------------------------- #


@dataclass
class TerminatorReport:
    outside: Counter
    inside: Counter
    naive_lines: int
    records: int
    embedded_fields: int
    verdict: str
    reason: str


def scan_terminators(text: str, quotechar: Optional[str] = '"') -> Tuple[Counter, Counter]:
    """Split newline runs into record terminators and newlines inside a field.

    This needs the quotechar, which is part of the dialect being detected. Line
    counting and dialect detection are therefore mutually dependent: you cannot
    count the records to validate a dialect without already having one.
    """
    outside: Counter = Counter()
    inside: Counter = Counter()
    in_quote = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quotechar is not None and ch == quotechar:
            in_quote = not in_quote
            i += 1
            continue
        if ch == "\r":
            kind = "\\r\\n" if text[i:i + 2] == "\r\n" else "\\r"
            (inside if in_quote else outside)[kind] += 1
            i += 2 if kind == "\\r\\n" else 1
            continue
        if ch == "\n":
            (inside if in_quote else outside)["\\n"] += 1
            i += 1
            continue
        i += 1
    return outside, inside


def probe_terminator(
    text: str, delimiter: str = ",", quotechar: Optional[str] = '"'
) -> TerminatorReport:
    outside, inside = scan_terminators(text, quotechar)
    naive = len([ln for ln in text.split("\n") if ln.strip("\r")])
    try:
        rows = [r for r in parse(text, delimiter, quotechar)
                if r and not (len(r) == 1 and r[0] == "")]
    except csv.Error:
        rows = []
    records = len(rows)
    embedded = sum(1 for r in rows for f in r if "\n" in f or "\r" in f)

    present = [k for k in ("\\r\\n", "\\n", "\\r") if outside[k]]
    if not present:
        verdict, reason = "none", "no record terminator outside a quoted field"
    elif len(present) == 1:
        verdict = present[0]
        reason = "consistent {0} between records".format(present[0])
        if verdict == "\\r":
            reason += " - str.split('\\n') sees {0} line(s); csv sees {1} records".format(
                naive, records)
    else:
        verdict = "mixed"
        reason = "mixed record terminators: " + ", ".join(
            "{0}x{1}".format(outside[k], k) for k in present)
    if inside:
        reason += "; {0} newline(s) inside quoted fields".format(sum(inside.values()))
    return TerminatorReport(outside, inside, naive, records, embedded, verdict, reason)


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #


def cell_type(s: str) -> str:
    s = s.strip()
    if s == "":
        return "empty"
    if _INT.match(s):
        return "int"
    if _FLOAT.match(s):
        return "float"
    if _DATE.match(s):
        return "date"
    if _EURO_FLOAT.match(s):
        return "float,"
    return "text"


@dataclass
class HeaderVerdict:
    status: str  # "header" | "undetermined"
    basis: str
    reason: str
    sniffer: Optional[bool]
    first_types: List[str]
    body_types: List[str]
    differing: List[int]


def classify_header(rows: Sequence[Sequence[str]], text: Optional[str] = None) -> HeaderVerdict:
    """Decide whether row 0 is a header, and refuse when the file does not say.

    There is exactly one decidable direction. A first row that is text where the
    body is numeric or dated cannot be a data row for that column, so `header` is
    provable. The converse is not: a first row whose types match the body is
    consistent with a data row *and* with a header whose labels happen to be
    numbers - `2019,2020,2021` is the common case. `csv.Sniffer.has_header()`
    returns a bool either way, so it answers the undecidable case too.
    """
    sniffer: Optional[bool] = None
    if text is not None:
        try:
            sniffer = csv.Sniffer().has_header(text)
        except csv.Error:
            sniffer = None

    if len(rows) < 2:
        return HeaderVerdict("undetermined", "too_short", "fewer than two records",
                             sniffer, [], [], [])

    first = [cell_type(c) for c in rows[0]]
    body: List[str] = []
    for col in range(len(rows[0])):
        col_types = Counter(cell_type(r[col]) for r in rows[1:] if col < len(r))
        body.append(col_types.most_common(1)[0][0] if col_types else "empty")

    differing = [i for i, (a, b) in enumerate(zip(first, body)) if a != b]
    header_like = [i for i in differing if first[i] == "text" and body[i] != "text"]

    if header_like:
        status, basis = "header", "text_over_nontext"
        reason = "{0} column(s) are text in row 0 and non-text below: {1}".format(
            len(header_like), ", ".join(str(i) for i in header_like))
    elif not differing and set(first) == {"text"}:
        status, basis = "undetermined", "all_text"
        reason = "every column is text in row 0 and below - nothing distinguishes them"
    elif not differing:
        status, basis = "undetermined", "row0_matches_body"
        reason = ("row 0 parses as {0} in every column, same as the body - consistent with a "
                  "data row and with numeric column labels".format("/".join(sorted(set(first)))))
    else:
        status, basis = "undetermined", "mixed"
        reason = "{0} column(s) differ but none are text-over-numeric".format(len(differing))
    return HeaderVerdict(status, basis, reason, sniffer, first, body, differing)


# --------------------------------------------------------------------------- #
# sample-size sensitivity
# --------------------------------------------------------------------------- #


def sample_sensitivity(
    text: str, sizes: Sequence[int] = (128, 1024, 4096)
) -> List[Tuple[str, Optional[str], str]]:
    """Sniff prefixes of increasing size. Returns (size label, sniffer pick, our status).

    Sniffing a prefix is the normal thing to do on a large file, and it is why a
    row 6,000 lines in that contains an embedded delimiter never enters the
    decision.
    """
    out: List[Tuple[str, Optional[str], str]] = []
    for n in list(sizes) + [len(text)]:
        chunk = text[:n]
        label = "all ({0} B)".format(len(text)) if n >= len(text) else "{0} B".format(n)
        out.append((label, sniffer_says(chunk), classify_delimiter(chunk).status))
    return out


# --------------------------------------------------------------------------- #
# full audit
# --------------------------------------------------------------------------- #


@dataclass
class Audit:
    name: str
    size: int
    encoding: EncodingReport
    delimiter: DelimiterVerdict
    terminator: Optional[TerminatorReport]
    header: Optional[HeaderVerdict]
    sensitivity: List[Tuple[str, Optional[str], str]]
    sniffer: Optional[str]
    notes: List[str] = field(default_factory=list)

    @property
    def decided(self) -> bool:
        return (self.delimiter.status == "unambiguous"
                and self.encoding.verdict not in ("undetermined", "undecodable")
                and (self.header is None or self.header.status != "undetermined"))


def audit(raw: bytes, name: str = "<bytes>") -> Audit:
    """Everything the bytes determine, and everything they do not, in one object."""
    enc = probe_encoding(raw)
    notes: List[str] = []

    text = enc.texts.get(enc.verdict) or enc.texts.get("utf-8") or enc.texts.get("latin-1")
    if text is None:
        return Audit(name, len(raw), enc, DelimiterVerdict("undetermined", [], [], None,
                     "undecodable"), None, None, [], None, ["no encoding decodes these bytes"])

    if enc.bom == "utf-8-sig" and (enc.texts.get("utf-8") or "").startswith("﻿"):
        first = (enc.texts["utf-8"].split("\r")[0].split(",")[0].split(";")[0])
        notes.append("read as plain 'utf-8' the BOM stays in the first field: {0!r} - which "
                     "prints as {1}".format(first, first.lstrip("﻿")))
    if enc.not_evidence:
        notes.append("success proves nothing for: " + "; ".join(
            "{0} ({1})".format(e, why) for e, why in enc.not_evidence))
    if enc.distinct_texts > 1:
        notes.append("{0} encodings decode these bytes to {1} different strings, none raising".format(
            len(enc.survived), enc.distinct_texts))

    dv = classify_delimiter(text)
    d = dv.preferred.delimiter if dv.preferred else ","
    q = dv.preferred.quotechar if dv.preferred else '"'
    term = probe_terminator(text, d, q)
    if term.verdict == "\\r":
        notes.append("bare \\r terminator: naive str.split('\\n') yields one row")
    if term.embedded_fields:
        notes.append("{0} field(s) contain a newline: str.split('\\n') counts {1} lines, "
                     "csv counts {2} records".format(
                         term.embedded_fields, term.naive_lines, term.records))

    rows = parse(text, d, q) if dv.preferred else []
    rows = [r for r in rows if r and not (len(r) == 1 and r[0] == "")]
    hdr = classify_header(rows, text) if rows else None
    if hdr and hdr.status == "undetermined" and hdr.sniffer is not None:
        notes.append("csv.Sniffer.has_header() answers {0} where the file does not say".format(
            hdr.sniffer))

    sens = sample_sensitivity(text)
    picks = {p for _, p, _ in sens}
    if len(picks) > 1:
        notes.append("delimiter pick depends on how much of the file is sampled: "
                     + " -> ".join(repr(p) for _, p, _ in sens))

    if dv.untested:
        notes.append("no field in this file is quoted, so {0} is untested here - it parses "
                     "identically and may not on the next export".format(", ".join(dv.untested)))
    if dv.status == "contested":
        notes.append("CONTESTED: " + dv.reason)

    return Audit(name, len(raw), enc, dv, term, hdr, sens, sniffer_says(text), notes)


# --------------------------------------------------------------------------- #
# sample corpus - ten files, each isolating one mechanism
# --------------------------------------------------------------------------- #


def sample_files() -> Dict[str, bytes]:
    """Bytes, not strings: the encoding cases are only real as bytes."""
    files: Dict[str, bytes] = {}

    # 1. Two internally-consistent parses, different widths, no header to break the tie.
    files["sensor.csv"] = (
        "2024-01-01;12;1,50;18,00\r\n"
        "2024-01-02;8;2,25;18,00\r\n"
        "2024-01-03;15;1,20;18,00\r\n"
        "2024-01-04;9;2,00;18,00\r\n"
    ).encode("utf-8")

    # 2. The same export with a header: the header breaks the tie.
    files["sales_eu.csv"] = (
        "day;units;price;total\r\n"
        "2024-01-01;12;1,50;18,00\r\n"
        "2024-01-02;8;2,25;18,00\r\n"
        "2024-01-03;15;1,20;18,00\r\n"
    ).encode("utf-8")

    # 3. cp1252 smart quotes + umlaut: utf-8 is provably ruled out.
    files["cp1252.csv"] = (
        "id,note\r\n"
        "1,\x93Ausf\xfchrung\x94 abgeschlossen\r\n"
        "2,Gr\xf6\xdfe ge\xe4ndert \x96 ok\r\n"
    )
    files["cp1252.csv"] = files["cp1252.csv"].encode("latin-1")

    # 4. utf-8 bytes that latin-1 will happily mis-decode. Nothing raises.
    files["utf8_umlaut.csv"] = "id,note\r\n1,Ausführung\r\n2,Größe\r\n".encode("utf-8")

    # 5. UTF-8 BOM: the first column name gains an invisible character.
    files["bom.csv"] = "﻿id,name,amount\r\n1,Widget,10\r\n2,Bolt,20\r\n".encode("utf-8")

    # 6. Numeric header names - Sniffer's type heuristic has nothing to hold.
    files["years.csv"] = (
        "2019,2020,2021\r\n" "120,140,150\r\n" "90,110,130\r\n"
    ).encode("utf-8")

    # 7. All text, no header: undecidable by construction.
    files["alltext.csv"] = (
        "north,widget,blue\r\n" "south,bolt,red\r\n" "east,nut,green\r\n"
    ).encode("utf-8")

    # 8. Bare \r terminator (classic Mac / some Excel exports).
    files["mac.csv"] = ("id,name\r" "1,Widget\r" "2,Bolt\r").encode("utf-8")

    # 9. Quoted delimiter and quoted newline: physical lines != records.
    files["quoted.csv"] = (
        'id,address,note\r\n'
        '1,"12 High St, Apt 4","line one\nline two"\r\n'
        '2,"9 Mill Rd",ok\r\n'
    ).encode("utf-8")

    # 10. Dutch surnames begin with an apostrophe. Read as a quotechar, the
    #     apostrophe opens a field that runs to the next one - two records
    #     become one, and the column count does not change.
    files["dutch.csv"] = (
        "id,name,town\r\n"
        "1,'t Hooft,Delft\r\n"
        "2,'s Gravesande,Leiden\r\n"
        "3,de Vries,Utrecht\r\n"
    ).encode("utf-8")

    # 11. Clean for the first kilobyte, then a row with an embedded delimiter.
    head = "id,name,amount\r\n" + "".join(
        "{0},Widget {0},{1}\r\n".format(i, i * 10) for i in range(1, 61))
    files["late.csv"] = (head + '61,"Bolt, hex",610\r\n62,Nut,620\r\n').encode("utf-8")

    return files


def tab_or_space() -> bytes:
    """A column-aligned report: tab and space both parse, to different widths."""
    return (
        "region\tunits\tprice\r\n"
        "north\t12\t1.50\r\n"
        "south\t8\t2.25\r\n"
    ).encode("utf-8")
