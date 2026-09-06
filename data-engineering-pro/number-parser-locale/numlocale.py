"""Read one numeric string with every reader that has a claim on it.

A numeric string does not contain a number.  It contains characters.  A
*reader* -- a locale's symbol table, a grouping rule, a strictness setting and
whatever scanner sits underneath -- assigns a number to it.  Change any one of
those four and the same bytes become a different quantity, usually with no
error raised.

The canonical case: ``1.234`` is one-point-two-three-four to a US reader and
one thousand two hundred thirty four to a German one.  Both readings conform
to their own locale.  Neither is a bug.  There is no way to tell from the
string which was meant.

This module ships fifteen readers over one corpus and reports every reading.
Everything is measured, not modelled: the C reader is libc's own ``strtod``
through ctypes, the JavaScript readers are a real ``node`` process, and the
locale readers are Babel's CLDR tables.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------

REJECTED = "rejected"          # reader raised / refused
UNAVAILABLE = "unavailable"    # reader not installed on this machine


@dataclass(frozen=True)
class Reading:
    """What one reader returned for one string."""

    reader: str
    raw: str
    status: str                       # "ok" | "rejected" | "unavailable"
    value: Optional[Decimal] = None   # exact value when representable
    float_value: Optional[float] = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def is_finite(self) -> bool:
        return self.ok and self.value is not None and self.value.is_finite()

    def display(self, width: Optional[int] = None) -> str:
        """Render the reading.  Never silently truncates a digit."""
        if self.status == UNAVAILABLE:
            return "n/a"
        if self.status == REJECTED:
            return "--"
        assert self.value is not None
        v = self.value
        if not v.is_finite():
            return str(v)
        n = v.normalize()
        _, digits, exp = n.as_tuple()
        # Anything that would print as a wall of zeros goes to exponent form.
        if isinstance(exp, int) and (exp > 6 or exp < -12):
            s = "{:e}".format(n)
        else:
            s = format(n, "f")
        if width is not None and len(s) > width:
            # Truncation is marked, so a matrix cell never hides a digit
            # that changes the number (see the int53 case).
            return s[: width - 1] + "~"
        return s


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

NNBSP = " "   # NARROW NO-BREAK SPACE -- CLDR's French group separator
NBSP = " "    # NO-BREAK SPACE        -- CLDR's Russian group separator
RSQUO = "’"   # RIGHT SINGLE QUOTATION MARK -- CLDR's Swiss group separator
MINUS = "−"   # MINUS SIGN -- the typographic one, not HYPHEN-MINUS


@dataclass(frozen=True)
class Case:
    """One numeric string plus where such a string comes from."""

    name: str
    raw: str
    provenance: str
    intended: Optional[str] = None   # what the producer meant, when knowable

    def escaped(self) -> str:
        out = []
        for ch in self.raw:
            if ord(ch) > 126 or ch in "\t\n\r":
                out.append("\\u%04x" % ord(ch))
            else:
                out.append(ch)
        if not self.raw:
            return "(empty)"
        if not self.raw.strip():
            return "(%d spaces)" % len(self.raw)
        return "".join(out)


def corpus() -> List[Case]:
    """Thirty-two strings a data pipeline actually receives."""
    return [
        Case("plain-int", "1234", "any system", "1234"),
        Case("dot-3dp", "1.234", "US decimal OR German thousands", None),
        Case("comma-3dp", "1,234", "US thousands OR German decimal", None),
        Case("us-money", "1,234.56", "US/UK export", "1234.56"),
        Case("de-money", "1.234,56", "German/Spanish export", "1234.56"),
        Case("us-money-trailing0", "1,234.50", "US accounting, fixed 2dp", "1234.50"),
        Case("de-money-trailing0", "1.234,50", "German accounting, fixed 2dp", "1234.50"),
        Case("bare-money", "1234.50", "unformatted 2dp amount", "1234.50"),
        Case("de-grouped", "1.234.567", "German grouped integer", "1234567"),
        Case("us-grouped", "1,234,567", "US grouped integer", "1234567"),
        Case("in-lakh", "12,34,567", "Indian lakh grouping (en_IN)", "1234567"),
        Case("fr-nnbsp", "1" + NNBSP + "234" + NNBSP + "567,89", "French, CLDR-correct U+202F", "1234567.89"),
        Case("fr-nbsp", "1" + NBSP + "234" + NBSP + "567,89", "French, NBSP -- what most tools emit", "1234567.89"),
        Case("fr-space", "1 234 567,89", "French, ASCII space -- what humans type", "1234567.89"),
        Case("ch-rsquo", "1" + RSQUO + "234" + RSQUO + "567.89", "Swiss, CLDR-correct U+2019", "1234567.89"),
        Case("ch-apostrophe", "1'234'567.89", "Swiss, ASCII apostrophe -- what keyboards give", "1234567.89"),
        Case("de-comma-2dp", "1,23", "German 2dp decimal", "1.23"),
        Case("pep515", "1_000", "Python literal pasted into a CSV", "1000"),
        Case("arabic-indic", "١٢٣٤", "Arabic-Indic digits (U+0661..)", "1234"),
        Case("accounting-neg", "(1,234)", "accounting / Excel negative", "-1234"),
        Case("trailing-minus", "1234-", "SAP / COBOL trailing sign", "-1234"),
        Case("true-minus", MINUS + "1234", "copy-paste from a rendered page", "-1234"),
        Case("plain-neg", "-1,234.00", "US negative money", "-1234.00"),
        Case("currency-prefix", "$1,234.00", "currency glued to the amount", "1234.00"),
        Case("percent-suffix", "12.5%", "percentage as text", "0.125 or 12.5"),
        Case("sci-notation", "1e3", "scientific notation", "1000"),
        Case("sci-overflow", "1e309", "beyond IEEE-754 double range", "1e309"),
        Case("int53-plus1", "9007199254740993", "id beyond exact double range", "9007199254740993"),
        Case("binary-inexact", "0.1", "not representable in binary", "0.1"),
        Case("hex", "0x1A", "hex literal in a text column", "26"),
        Case("empty", "", "empty cell", None),
        Case("blank", "   ", "whitespace-only cell", None),
        Case("trailing-junk", "12abc", "unit glued on / dirty cell", None),
        Case("nan-word", "NaN", "missing marker written as a word", None),
        Case("inf-word", "Infinity", "overflow marker", None),
    ]


# --------------------------------------------------------------------------
# Reader 1-2: Python's own scanners
# --------------------------------------------------------------------------

def read_py_float(s: str) -> Reading:
    try:
        f = float(s)
    except (ValueError, TypeError):
        return Reading("py_float", s, REJECTED)
    note = ""
    if math.isinf(f):
        d = Decimal("Infinity") if f > 0 else Decimal("-Infinity")
        note = "overflowed to inf"
    elif math.isnan(f):
        d = Decimal("NaN")
    else:
        d = Decimal(repr(f))
        if "_" in s:
            note = "PEP 515 underscore accepted"
    return Reading("py_float", s, "ok", d, f, note)


def read_py_decimal(s: str) -> Reading:
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError, TypeError):
        return Reading("py_decimal", s, REJECTED)
    note = ""
    if "_" in s:
        note = "PEP 515 underscore accepted"
    if any(ord(c) > 127 for c in s):
        note = "Unicode Nd digits accepted"
    f: Optional[float]
    try:
        f = float(d)
    except (OverflowError, ValueError):
        f = None
    return Reading("py_decimal", s, "ok", d, f, note)


# --------------------------------------------------------------------------
# Reader 3: libc strtod, through ctypes -- the real thing, not a model
# --------------------------------------------------------------------------

_libc: Optional[ctypes.CDLL]
try:
    _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
    _libc.strtod.restype = ctypes.c_double
    _libc.strtod.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
except OSError:  # pragma: no cover - platform without a loadable libc
    _libc = None


def read_c_strtod(s: str) -> Reading:
    """C's ``strtod`` -- a *prefix* parser.  It consumes what it can and stops.

    This is what ``awk``, most C/C++ importers and a lot of embedded ETL use.
    It never fails on trailing junk; it just returns a shorter number.  A
    caller who ignores ``endptr`` cannot tell ``1234`` from ``1234-``.
    """
    if _libc is None:  # pragma: no cover
        return Reading("c_strtod", s, UNAVAILABLE)
    b = s.encode("utf-8")
    end = ctypes.c_char_p()
    val = _libc.strtod(b, ctypes.byref(end))
    remaining = end.value if end.value is not None else b
    consumed = len(b) - len(remaining)
    if consumed == 0:
        # strtod sets endptr == nptr and returns 0.0.  A caller checking only
        # the return value reads this as a legitimate zero.
        return Reading("c_strtod", s, "ok", Decimal(0), 0.0,
                       "consumed 0 bytes -> silent 0")
    note = "" if consumed == len(b) else "prefix only (%d/%d bytes)" % (consumed, len(b))
    if math.isinf(val):
        d = Decimal("Infinity") if val > 0 else Decimal("-Infinity")
    elif math.isnan(val):
        d = Decimal("NaN")
    else:
        d = Decimal(repr(val))
    return Reading("c_strtod", s, "ok", d, val, note)


# --------------------------------------------------------------------------
# Readers 4-5: real JavaScript, via a node subprocess
# --------------------------------------------------------------------------

_NODE = shutil.which("node")

_JS_PROGRAM = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = input.map(s => {
  const n = Number(s), p = parseFloat(s);
  const enc = v => (Number.isNaN(v) ? {k:'nan'}
    : !Number.isFinite(v) ? {k:'inf', sign: v > 0 ? 1 : -1}
    : {k:'num', v: v.toString()});
  return {number: enc(n), parseFloat: enc(p)};
});
process.stdout.write(JSON.stringify(out));
"""


def _js_batch(strings: Sequence[str]) -> Optional[List[Dict[str, Any]]]:
    if _NODE is None:
        return None
    try:
        proc = subprocess.run(
            [_NODE, "-e", _JS_PROGRAM],
            input=json.dumps(list(strings)),
            capture_output=True, text=True, timeout=60, check=True,
        )
        return json.loads(proc.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):  # pragma: no cover
        return None


def _js_reading(reader: str, s: str, enc: Optional[Dict[str, Any]]) -> Reading:
    if enc is None:
        return Reading(reader, s, UNAVAILABLE)
    kind = enc["k"]
    if kind == "nan":
        # NaN is JavaScript's *only* failure channel for Number().  It is a
        # value, not an exception: `Number(x) + 1` is NaN, and NaN < 1 is
        # false, so a range check on it passes silently in the wrong direction.
        return Reading(reader, s, REJECTED, note="NaN (a value, not a raised error)")
    if kind == "inf":
        d = Decimal("Infinity") if enc["sign"] > 0 else Decimal("-Infinity")
        return Reading(reader, s, "ok", d, math.inf * enc["sign"], "overflowed to Infinity")
    d = Decimal(enc["v"])
    note = ""
    if d == 0 and s.strip() == "" and reader == "js_number":
        note = "empty/blank -> 0 (StringNumericLiteral of whitespace is 0)"
    elif reader == "js_parsefloat" and not _consumes_all(s):
        note = "prefix only"
    return Reading(reader, s, "ok", d, float(d) if d.is_finite() else None, note)


def _consumes_all(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    try:
        float(t)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# Readers 6-15: locale readers, from Babel's CLDR tables
# --------------------------------------------------------------------------

LOCALES: List[Tuple[str, str]] = [
    ("en_US", "US / UK: , group  . decimal"),
    ("de_DE", "German: . group  , decimal"),
    ("fr_FR", "French: U+202F group  , decimal"),
    ("en_IN", "Indian: , group at 2,2,3  . decimal"),
    ("de_CH", "Swiss: U+2019 group  . decimal"),
]

try:
    from babel.numbers import (
        NumberFormatError,
        format_decimal,
        get_decimal_symbol,
        get_group_symbol,
        parse_decimal,
    )
    _HAVE_BABEL = True
except ImportError:  # pragma: no cover
    _HAVE_BABEL = False


def read_locale(s: str, locale: str, strict: bool) -> Reading:
    """A CLDR locale reader.

    ``strict=True`` additionally validates the *grouping* -- that separators
    fall where the locale's pattern puts them.  It is the setting you are
    told to use.  It is also the setting that rejects a correctly formatted
    fixed-2dp amount (see :func:`own_output_roundtrip`).
    """
    reader = "%s_%s" % (locale, "strict" if strict else "loose")
    if not _HAVE_BABEL:  # pragma: no cover
        return Reading(reader, s, UNAVAILABLE)
    try:
        d = parse_decimal(s, locale=locale, strict=strict)
    except NumberFormatError:
        return Reading(reader, s, REJECTED)
    except (ValueError, TypeError, IndexError):
        # Babel leaks the underlying Decimal scanner's exceptions on some
        # inputs; treat any of them as a refusal rather than crashing a run.
        return Reading(reader, s, REJECTED)
    note = ""
    group = get_group_symbol(locale)
    if group in s and not strict:
        note = "group symbol stripped without checking position"
    if "_" in s:
        note = "PEP 515 underscore leaked through to Decimal"
    f: Optional[float]
    try:
        f = float(d)
    except (OverflowError, ValueError):
        f = None
    return Reading(reader, s, "ok", d, f, note)


def locale_symbols() -> List[Dict[str, str]]:
    """The actual CLDR symbols on this machine, with code points."""
    rows = []
    for loc, desc in LOCALES:
        g = get_group_symbol(loc)
        d = get_decimal_symbol(loc)
        rows.append({
            "locale": loc,
            "group": g,
            "group_cp": " ".join("U+%04X" % ord(c) for c in g),
            "decimal": d,
            "decimal_cp": " ".join("U+%04X" % ord(c) for c in d),
            "sample": format_decimal(Decimal("1234567.89"), locale=loc),
            "desc": desc,
        })
    return rows


# --------------------------------------------------------------------------
# The reader roster
# --------------------------------------------------------------------------

def reader_names() -> List[str]:
    names = ["py_float", "py_decimal", "c_strtod", "js_number", "js_parsefloat"]
    for loc, _ in LOCALES:
        names.append("%s_strict" % loc)
        names.append("%s_loose" % loc)
    return names


READER_KIND = {
    "py_float": "engine", "py_decimal": "engine", "c_strtod": "engine",
    "js_number": "engine", "js_parsefloat": "engine",
}


def read_all(cases: Optional[Sequence[Case]] = None) -> Dict[str, Dict[str, Reading]]:
    """Every reader x every string.  Returns ``{case_name: {reader: Reading}}``."""
    cases = list(cases if cases is not None else corpus())
    raws = [c.raw for c in cases]
    js = _js_batch(raws)
    table: Dict[str, Dict[str, Reading]] = {}
    for i, case in enumerate(cases):
        s = case.raw
        row: Dict[str, Reading] = {
            "py_float": read_py_float(s),
            "py_decimal": read_py_decimal(s),
            "c_strtod": read_c_strtod(s),
            "js_number": _js_reading("js_number", s, js[i]["number"] if js else None),
            "js_parsefloat": _js_reading("js_parsefloat", s, js[i]["parseFloat"] if js else None),
        }
        for loc, _ in LOCALES:
            row["%s_strict" % loc] = read_locale(s, loc, True)
            row["%s_loose" % loc] = read_locale(s, loc, False)
        table[case.name] = row
    return table


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

VERDICTS = [
    "sign-drift",       # readers disagree about whether it is positive
    "sign-loss",        # the producer meant a negative; no reader returns one
    "magnitude-drift",  # >= 10x apart, silently
    "value-drift",      # different numbers, under 10x apart
    "silent-zero",      # a reader returns 0 where others see a real quantity
    "accept-drift",     # same number, but only some readers will take it
    "agreed",           # every reader that accepts returns the same number
    "rejected-by-all",  # nobody takes it
]


# Notations that mean "negative" and that no reader in the roster decodes.
NEGATIVE_NOTATIONS = [
    ("accounting parentheses", lambda s: s.startswith("(") and s.endswith(")")),
    ("trailing sign (SAP/COBOL)", lambda s: s.endswith("-") and any(c.isdigit() for c in s)),
    ("U+2212 MINUS SIGN", lambda s: s.lstrip().startswith("\u2212")),
    ("U+FF0D FULLWIDTH HYPHEN-MINUS", lambda s: s.lstrip().startswith("\uff0d")),
    ("CR/DB suffix", lambda s: s.strip().upper().endswith(("CR", "DB"))),
]


def negative_notation(raw: str) -> Optional[str]:
    """Name the negative notation a string uses, if it uses one.

    ASCII ``-1234`` is excluded: every reader here decodes a leading
    HYPHEN-MINUS.  These are the notations that do not survive a parse.
    """
    s = raw.strip()
    if not s:
        return None
    for label, test in NEGATIVE_NOTATIONS:
        if test(s):
            return label
    return None


@dataclass
class CaseVerdict:
    case: Case
    verdict: str
    distinct: List[Decimal]
    accepted: List[str]
    rejected: List[str]
    ratio: Optional[Decimal] = None
    zero_readers: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    @property
    def n_distinct(self) -> int:
        return len(self.distinct)


def _distinct_finite(row: Dict[str, Reading]) -> List[Decimal]:
    seen: List[Decimal] = []
    for r in row.values():
        if r.is_finite:
            assert r.value is not None
            if not any(r.value == s for s in seen):
                seen.append(r.value)
    return sorted(seen)


def verdict_for(case: Case, row: Dict[str, Reading]) -> CaseVerdict:
    accepted = [n for n, r in row.items() if r.ok]
    rejected = [n for n, r in row.items() if r.status == REJECTED]
    distinct = _distinct_finite(row)
    zero_readers = [n for n, r in row.items()
                    if r.is_finite and r.value == 0 and "silent 0" in r.note]
    flags: List[str] = []

    nonzero = [d for d in distinct if d != 0]
    signs = {(1 if d > 0 else -1) for d in nonzero}
    ratio: Optional[Decimal] = None
    if len(nonzero) >= 2:
        mags = sorted(abs(d) for d in nonzero)
        if mags[0] != 0:
            ratio = mags[-1] / mags[0]

    notation = negative_notation(case.raw)
    intended_negative = notation is not None or (
        case.intended is not None and case.intended.lstrip().startswith("-"))
    all_positive = bool(nonzero) and all(d > 0 for d in nonzero)

    if not accepted:
        verdict = "rejected-by-all"
    elif len(signs) > 1:
        verdict = "sign-drift"
        flags.append("sign flips between conforming readers")
    elif intended_negative and all_positive:
        verdict = "sign-loss"
        flags.append(
            "%s means negative; every reader that accepts it returns a positive number"
            % (notation or "the producer's notation"))
    elif ratio is not None and ratio >= 10:
        verdict = "magnitude-drift"
        flags.append("%s x apart" % _fmt_ratio(ratio))
    elif len(distinct) > 1:
        verdict = "value-drift"
    elif zero_readers:
        verdict = "silent-zero"
    elif rejected:
        verdict = "accept-drift"
    else:
        verdict = "agreed"

    if notation is not None and verdict == "silent-zero":
        flags.append("%s: no reader decodes it, and it lands as 0" % notation)
    if zero_readers and verdict not in ("silent-zero",):
        flags.append("%d reader(s) return a silent 0" % len(zero_readers))
    if any("NaN" in r.note for r in row.values()):
        flags.append("JS signals failure as the value NaN")
    return CaseVerdict(case, verdict, distinct, accepted, rejected,
                       ratio, zero_readers, flags)


def _fmt_ratio(r: Decimal) -> str:
    """Format a factor.  Sub-unit factors print as 1/N, never as 0.00."""
    f = float(r)
    if f < 1 and f > 0:
        return "1/" + _fmt_ratio(Decimal(1) / r)
    if f >= 1000:
        return "{:,.0f}".format(f)
    if f >= 10:
        return "{:.0f}".format(f)
    return "{:.2f}".format(f)


def all_verdicts(cases: Optional[Sequence[Case]] = None,
                 table: Optional[Dict[str, Dict[str, Reading]]] = None) -> List[CaseVerdict]:
    cases = list(cases if cases is not None else corpus())
    table = table if table is not None else read_all(cases)
    return [verdict_for(c, table[c.name]) for c in cases]


# --------------------------------------------------------------------------
# The border crossing: format in one locale, read in another
# --------------------------------------------------------------------------

MONEY_PATTERN = "#,##0.00"   # what an accounting export emits: fixed 2 dp

ROUNDTRIP_VALUES = [Decimal("1234567.89"), Decimal("1234.50"),
                    Decimal("1000"), Decimal("0.50"), Decimal("1.234")]


@dataclass(frozen=True)
class Crossing:
    value: Decimal
    wrote: str          # locale that formatted it
    read: str           # locale that parsed it
    strict: bool
    rendered: str
    status: str         # "ok" | "error" | "wrong"
    got: Optional[Decimal]
    expected: Optional[Decimal] = None   # value after the pattern's own rounding

    @property
    def target(self) -> Decimal:
        """What a correct reader should return -- the value as *rendered*.

        A ``#,##0.00`` pattern rounds 1.234 to 1.23 on the way out.  That is
        the writer's loss, not the reader's, so the reader is scored against
        the rendered quantity rather than the original.
        """
        return self.expected if self.expected is not None else self.value

    @property
    def ratio(self) -> Optional[Decimal]:
        if self.status != "wrong" or self.got is None:
            return None
        t = self.target
        if t == 0 or not self.got.is_finite():
            return None
        return abs(self.got) / abs(t)


def crossings(values: Optional[Sequence[Decimal]] = None,
              pattern: str = MONEY_PATTERN) -> List[Crossing]:
    """Write a number the way locale A does, read it the way locale B does.

    This is what happens when a CSV crosses a border: nothing about the file
    records which locale wrote it, so the reader supplies its own.  ``wrong``
    means the reader returned a number and it was not the number written --
    no exception, no warning, a value that flows straight into a sum.
    """
    values = list(values if values is not None else ROUNDTRIP_VALUES)
    places = _pattern_places(pattern)
    out: List[Crossing] = []
    for v in values:
        expected = v.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)
        for wrote, _ in LOCALES:
            rendered = format_decimal(v, format=pattern, locale=wrote)
            for read, _ in LOCALES:
                for strict in (True, False):
                    r = read_locale(rendered, read, strict)
                    if not r.ok:
                        status, got = "error", None
                    else:
                        got = r.value
                        assert got is not None
                        status = "ok" if got == expected else "wrong"
                    out.append(Crossing(v, wrote, read, strict, rendered,
                                        status, got, expected))
    return out


def _pattern_places(pattern: str) -> int:
    """Decimal places a CLDR number pattern pins."""
    if "." not in pattern:
        return 0
    return pattern.split(".", 1)[1].count("0")


def own_output_roundtrip(pattern: str = MONEY_PATTERN,
                        values: Optional[Sequence[Decimal]] = None) -> List[Crossing]:
    """The diagonal of :func:`crossings`: same locale wrote it and read it.

    Every cell here *should* be ``ok``.  The ones that are not are the
    interesting finding: a reader refusing a string its own locale produced.
    """
    return [c for c in crossings(values, pattern) if c.wrote == c.read]


# --------------------------------------------------------------------------
# Column-level audit -- the callable a pipeline would actually use
# --------------------------------------------------------------------------

@dataclass
class ColumnAudit:
    column: str
    n: int
    readers_that_take_every_row: List[str]
    candidate_readings: Dict[str, Optional[Decimal]]   # reader -> sum, None if it rejected a row
    disagreement: Optional[Decimal]                    # max/min of the sums
    findings: List[str]
    per_row: List[Dict[str, Reading]] = field(default_factory=list)
    decision: Optional["Decision"] = None              # the locale-hypothesis answer

    @property
    def decidable(self) -> bool:
        return len({str(v) for v in self.candidate_readings.values() if v is not None}) <= 1


def audit_column(values: Sequence[str], column: str = "amount") -> ColumnAudit:
    """Audit a real column: which readers can read every row, and do their totals agree?

    The output is deliberately not a number.  If two readers both read every
    row and produce different totals, the column has two defensible totals and
    the file does not say which is meant.  That is a fact about the file, and
    the honest return value is both totals plus the ratio between them.
    """
    cases = [Case("row%d" % i, v, "column %s" % column) for i, v in enumerate(values)]
    table = read_all(cases)
    names = reader_names()

    sums: Dict[str, Optional[Decimal]] = {}
    complete: List[str] = []
    for reader in names:
        total = Decimal(0)
        good = True
        for c in cases:
            r = table[c.name][reader]
            if not r.is_finite:
                good = False
                break
            assert r.value is not None
            total += r.value
        if good:
            complete.append(reader)
            sums[reader] = total
        else:
            sums[reader] = None

    totals = [t for t in sums.values() if t is not None and t != 0]
    disagreement = None
    if len(totals) >= 2:
        mags = sorted(abs(t) for t in totals)
        if mags[0] != 0:
            disagreement = mags[-1] / mags[0]

    findings: List[str] = []
    distinct_totals = sorted({str(t) for t in sums.values() if t is not None})
    if len(distinct_totals) > 1:
        findings.append(
            "%d readers read every row; they produce %d different totals"
            % (len(complete), len(distinct_totals)))
    if disagreement is not None and disagreement >= 10:
        findings.append("the two extreme totals are %s x apart" % _fmt_ratio(disagreement))
    row_flags: List[str] = []
    for i, c in enumerate(cases):
        v = verdict_for(c, table[c.name])
        if v.verdict in ("sign-drift", "sign-loss"):
            row_flags.append("row %d (%r): %s" % (i, c.raw, v.verdict))
        elif v.verdict == "magnitude-drift":
            row_flags.append("row %d (%r) is read %s x apart" % (i, c.raw, _fmt_ratio(v.ratio)))
    findings.extend(row_flags[:2])
    if len(row_flags) > 2:
        findings.append("... and %d more rows with the same problem" % (len(row_flags) - 2))
    if not complete:
        findings.append("no reader accepts every row -- the column is not uniformly numeric")

    decision = decide_column(values)
    if decision.verdict == "decided" and len(decision.surviving) == 1:
        findings.append("exactly one locale reads every row: %s -> total %s"
                        % (decision.surviving[0],
                           decision.totals[decision.surviving[0]]))
    elif decision.verdict == "ambiguous":
        findings.append("%d locales read every row and disagree: %s"
                        % (len(decision.surviving),
                           ", ".join("%s=%s" % (k, v) for k, v in decision.totals.items())))
    elif decision.verdict == "no-locale-fits":
        findings.append("no locale reads every row under a strict grouping check")

    return ColumnAudit(column, len(values), complete, sums, disagreement, findings,
                       [table[c.name] for c in cases], decision)


# --------------------------------------------------------------------------
# Locale hypotheses -- the only lens that can actually decide a column
# --------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """One candidate answer to "which locale wrote this column?"."""

    locale: str
    survives: bool
    total: Optional[Decimal]
    readings: List[Optional[Decimal]]
    killed_by: Optional[str] = None   # the row that eliminated it


def locale_hypotheses(values: Sequence[str], strict: bool = True) -> List[Hypothesis]:
    """Which locales could have written this column?

    Scanners like ``strtod`` cannot answer this: a prefix parser accepts
    every string, so it never eliminates anything.  Only a reader that
    *refuses* carries information.  A locale survives if it reads every row;
    the surviving set is what the column tells you about its own provenance.

    Two structural facts do most of the eliminating:

    * a group of four digits is not a group, so ``1.2345`` rules out any
      locale that groups with ``.``;
    * two different separators in one value pin which is which, so
      ``1.234,56`` rules out every locale whose decimal symbol is ``.``.

    A column of single three-digit groups -- ``1.234``, ``2.500`` -- offers
    neither, which is exactly the shape a money column has.
    """
    out: List[Hypothesis] = []
    for loc, _ in LOCALES:
        readings: List[Optional[Decimal]] = []
        killed: Optional[str] = None
        for v in values:
            r = read_locale(v, loc, strict)
            if r.is_finite:
                assert r.value is not None
                readings.append(r.value)
            else:
                readings.append(None)
                if killed is None:
                    killed = v
        survives = killed is None and len(readings) == len(values)
        total = sum(readings, Decimal(0)) if survives else None  # type: ignore[arg-type]
        out.append(Hypothesis(loc, survives, total, readings, killed))
    return out


@dataclass
class Decision:
    surviving: List[str]
    totals: Dict[str, Decimal]
    verdict: str        # "decided" | "ambiguous" | "no-locale-fits"
    spread: Optional[Decimal] = None


def decide_column(values: Sequence[str], strict: bool = True) -> Decision:
    """Reduce the hypotheses to a decision, or refuse to make one.

    ``decided`` means one locale survives, or several survive and agree on
    every value.  ``ambiguous`` means two survivors disagree -- the honest
    output there is both totals, not a coin flip.
    """
    hyps = [h for h in locale_hypotheses(values, strict) if h.survives]
    totals = {h.locale: h.total for h in hyps if h.total is not None}
    distinct = {str(t) for t in totals.values()}
    if not hyps:
        return Decision([], {}, "no-locale-fits")
    spread = None
    mags = sorted(abs(t) for t in totals.values() if t != 0)
    if len(mags) >= 2 and mags[0] != 0:
        spread = mags[-1] / mags[0]
    verdict = "decided" if len(distinct) <= 1 else "ambiguous"
    return Decision([h.locale for h in hyps], totals, verdict, spread)


# --------------------------------------------------------------------------
# Summary numbers, so the README and the tests read the same source
# --------------------------------------------------------------------------

def summary() -> Dict[str, Any]:
    cases = corpus()
    table = read_all(cases)
    verds = all_verdicts(cases, table)
    names = reader_names()

    counts = {v: 0 for v in VERDICTS}
    for v in verds:
        counts[v.verdict] += 1

    multi = [v for v in verds if v.n_distinct >= 2]
    worst = max((v for v in verds if v.ratio is not None),
                key=lambda v: v.ratio, default=None)

    accept_counts = {n: sum(1 for c in cases if table[c.name][n].ok) for n in names}
    silent_zero = {n: sum(1 for c in cases
                          if table[c.name][n].is_finite
                          and table[c.name][n].value == 0
                          and "silent 0" in table[c.name][n].note)
                   for n in names}
    js_zero = [c.name for c in cases
               if table[c.name]["js_number"].is_finite
               and table[c.name]["js_number"].value == 0
               and c.raw.strip() == ""]

    cross = crossings()
    diag = [c for c in cross if c.wrote == c.read]
    off = [c for c in cross if c.wrote != c.read]

    return {
        "n_cases": len(cases),
        "n_readers": len(names),
        "n_readings": len(cases) * len(names),
        "verdict_counts": counts,
        "n_multi_valued": len(multi),
        "worst_ratio": worst.ratio if worst else None,
        "worst_case": worst.case.name if worst else None,
        "accept_counts": accept_counts,
        "silent_zero_counts": silent_zero,
        "js_blank_to_zero": js_zero,
        "n_crossings": len(cross),
        "crossing_wrong": sum(1 for c in cross if c.status == "wrong"),
        "crossing_error": sum(1 for c in cross if c.status == "error"),
        "crossing_ok": sum(1 for c in cross if c.status == "ok"),
        "offdiag_wrong": sum(1 for c in off if c.status == "wrong"),
        "diag_not_ok": sum(1 for c in diag if c.status != "ok"),
        "diag_total": len(diag),
        "diag_strict_fail": sum(1 for c in diag if c.strict and c.status != "ok"),
        "diag_loose_wrong": sum(1 for c in diag if not c.strict and c.status == "wrong"),
        "diag_strict_total": sum(1 for c in diag if c.strict),
        "n_sign_loss": counts["sign-loss"],
        "worst_crossing_ratio": max((c.ratio for c in cross if c.ratio is not None),
                                    default=None),
        "ambiguous_money_column": decide_column(["1.234", "2.500", "3.000", "1.750"]).verdict,
        "decidable_by_group_count": decide_column(["1.234.567", "89.012", "3.456"]).verdict,
        "decidable_by_4digit": decide_column(["1.2345", "2.500"]).verdict,
        "decidable_by_mixed": decide_column(["1.234,56", "7.890,12"]).verdict,
    }


if __name__ == "__main__":  # pragma: no cover
    import pprint
    pprint.pprint(summary())
