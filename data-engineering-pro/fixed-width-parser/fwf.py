"""Byte-accurate fixed-width record parsing.

A fixed-width file is defined in *bytes*, not characters. Every mainstream reader
(pandas.read_fwf included) decodes the record to `str` first and then slices by
character offset. On pure-ASCII data the two agree, which is why the sample file
passes and production does not.

This module keeps the record as `bytes` until the last possible moment: fields are
sliced by byte offset and each field is decoded independently. That makes three
things possible that character slicing cannot do at all:

  * multi-byte text anywhere in the record without shifting later fields
  * COMP-3 packed decimal, which is not text and has no valid decoding
  * signed zoned decimal (overpunch), where the sign lives inside the last digit

Standard library only. No pandas, no numpy.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field as _dcfield
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "FieldError",
    "SpecError",
    "Text",
    "Int",
    "Implied",
    "Overpunch",
    "Packed",
    "Date",
    "Field",
    "RecordSpec",
    "ParseResult",
    "frame_records",
    "parse",
    "parse_naive",
    "audit",
    "AuditReport",
    "CUSTOMER_SPEC",
    "BALANCE_SPEC",
    "build_customer_file",
    "build_balance_file",
]


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class FieldError(ValueError):
    """A single field could not be decoded. Carries enough to locate it."""


class SpecError(ValueError):
    """The layout itself is inconsistent - overlaps, negative offsets, bad length."""


# --------------------------------------------------------------------------
# field kinds
# --------------------------------------------------------------------------


class Kind:
    """Decodes one field's raw bytes. Subclasses must not look outside their slice."""

    #: True when the field's bytes are not text under any encoding.
    binary: bool = False

    def decode(self, raw: bytes, encoding: str) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover - trivial
        return type(self).__name__.lower()


class Text(Kind):
    """Character data. Padding is stripped; the pad byte is part of the layout."""

    def __init__(self, strip: bool = True, pad: bytes = b" \x00") -> None:
        self.strip = strip
        self.pad = pad

    def decode(self, raw: bytes, encoding: str) -> Optional[str]:
        if self.strip:
            raw = raw.strip(self.pad)
        if not raw:
            return None
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            raise FieldError(
                f"bytes {raw!r} are not valid {encoding} (offset {exc.start} in field)"
            ) from None

    def describe(self) -> str:
        return "text"


class Int(Kind):
    """Unsigned display numeric, zero- or space-padded. No sign carrier."""

    def decode(self, raw: bytes, encoding: str) -> Optional[int]:
        s = raw.strip(b" \x00")
        if not s:
            return None
        if not s.isdigit():
            # The single most common cause is an overpunch sign the layout forgot
            # to declare, so say so rather than emitting a generic parse error.
            last = s[-1:].decode("latin-1")
            if last in _OVERPUNCH_POS or last in _OVERPUNCH_NEG:
                raise FieldError(
                    f"{s!r} ends in overpunch sign character {last!r} - "
                    f"this field is signed zoned decimal, declare Overpunch()"
                )
            raise FieldError(f"{s!r} is not a display integer")
        return int(s)

    def describe(self) -> str:
        return "int"


class Implied(Kind):
    """PIC 9(n)V9(s) - an integer with an implied decimal point and no sign."""

    def __init__(self, scale: int) -> None:
        if scale < 0:
            raise SpecError("scale must be >= 0")
        self.scale = scale

    def decode(self, raw: bytes, encoding: str) -> Optional[Decimal]:
        s = raw.strip(b" \x00")
        if not s:
            return None
        if not s.isdigit():
            raise FieldError(f"{s!r} is not a display integer")
        return Decimal(int(s)).scaleb(-self.scale)

    def describe(self) -> str:
        return f"implied(scale={self.scale})"


_OVERPUNCH_POS = {"{": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9}
_OVERPUNCH_NEG = {"}": 0, "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "O": 6, "P": 7, "Q": 8, "R": 9}


class Overpunch(Kind):
    """PIC S9(n)V9(s) DISPLAY - trailing sign punched into the final digit.

    ``0001234{`` is +12.34 at scale 2; ``0001234M`` is -12.34. The magnitude is
    identical and only one byte differs, which is why dropping the sign is easy
    to do and almost impossible to see in a spot check.
    """

    def __init__(self, scale: int = 0, leading: bool = False) -> None:
        self.scale = scale
        self.leading = leading

    def decode(self, raw: bytes, encoding: str) -> Optional[Decimal]:
        s = raw.strip(b" \x00")
        if not s:
            return None
        txt = s.decode("latin-1")
        sign_char = txt[0] if self.leading else txt[-1]
        rest = txt[1:] if self.leading else txt[:-1]
        if sign_char in _OVERPUNCH_POS:
            sign, last = 1, _OVERPUNCH_POS[sign_char]
        elif sign_char in _OVERPUNCH_NEG:
            sign, last = -1, _OVERPUNCH_NEG[sign_char]
        elif sign_char.isdigit():
            # Unpunched positive - common when the producer is not a COBOL program.
            sign, last = 1, int(sign_char)
        else:
            raise FieldError(f"{sign_char!r} is not a sign carrier in {txt!r}")
        if rest and not rest.isdigit():
            raise FieldError(f"{rest!r} contains non-digits")
        digits = (rest + str(last)) if not self.leading else (str(last) + rest)
        return Decimal(sign * int(digits)).scaleb(-self.scale)

    def describe(self) -> str:
        return f"overpunch(scale={self.scale})"


_SIGN_POS = {0xC, 0xF, 0xA, 0xE}
_SIGN_NEG = {0xD, 0xB}


class Packed(Kind):
    """COMP-3 / packed decimal: two digits per byte, sign in the final nibble.

    Not text. A five-byte COMP-3 field holds nine digits, and one of its byte
    values is whatever the digits happen to make - including 0x0D and 0x1A.
    """

    binary = True

    def __init__(self, scale: int = 0) -> None:
        self.scale = scale

    def decode(self, raw: bytes, encoding: str) -> Optional[Decimal]:
        if not raw or set(raw) == {0x00}:
            return None
        digits: List[str] = []
        for i, byte in enumerate(raw):
            hi, lo = byte >> 4, byte & 0x0F
            if hi > 9:
                raise FieldError(f"nibble {hi:X} at byte {i} is not a digit ({raw.hex()})")
            digits.append(str(hi))
            if i == len(raw) - 1:
                if lo in _SIGN_NEG:
                    sign = -1
                elif lo in _SIGN_POS:
                    sign = 1
                else:
                    raise FieldError(f"final nibble {lo:X} is not a sign ({raw.hex()})")
            else:
                if lo > 9:
                    raise FieldError(f"nibble {lo:X} at byte {i} is not a digit ({raw.hex()})")
                digits.append(str(lo))
        return Decimal(sign * int("".join(digits))).scaleb(-self.scale)

    def describe(self) -> str:
        return f"packed(scale={self.scale})"


class Date(Kind):
    """Display date. Mainframe null dates are all zeros or all spaces, not empty."""

    def __init__(self, fmt: str = "%Y%m%d", nulls: Sequence[str] = ("00000000", "")) -> None:
        self.fmt = fmt
        self.nulls = tuple(nulls)

    def decode(self, raw: bytes, encoding: str) -> Optional[_dt.date]:
        s = raw.strip(b" \x00").decode("latin-1")
        if s in self.nulls:
            return None
        try:
            return _dt.datetime.strptime(s, self.fmt).date()
        except ValueError:
            raise FieldError(f"{s!r} does not match {self.fmt}") from None

    def describe(self) -> str:
        return f"date({self.fmt})"


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    """One field. ``start`` is expressed in the spec's own index base."""

    name: str
    start: int
    length: int
    kind: Kind

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise SpecError(f"{self.name}: length must be positive, got {self.length}")


@dataclass
class RecordSpec:
    """A record layout plus the two conventions that decide what it means.

    ``index_base`` is the killer. Copybooks, data dictionaries and every
    hand-written column spec are 1-indexed and inclusive. ``pandas.read_fwf``
    colspecs are 0-indexed and half-open. Pasting one into the other shifts every
    field by exactly one byte, which truncates the last digit of every amount and
    still parses cleanly.
    """

    fields: Sequence[Field]
    index_base: int = 1
    encoding: str = "utf-8"
    length: Optional[int] = None
    name: str = "record"
    _slices: Dict[str, Tuple[int, int]] = _dcfield(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._slices = {}
        for f in self.fields:
            off = f.start - self.index_base
            if off < 0:
                raise SpecError(
                    f"{f.name}: start {f.start} is before the record under "
                    f"index_base={self.index_base}"
                )
            self._slices[f.name] = (off, off + f.length)

    # -- geometry ---------------------------------------------------------

    @property
    def span(self) -> int:
        """Bytes from record start to the end of the last declared field."""
        return max((end for _, end in self._slices.values()), default=0)

    @property
    def record_length(self) -> int:
        return self.length if self.length is not None else self.span

    def slice_of(self, name: str) -> Tuple[int, int]:
        return self._slices[name]

    def rebase(self, index_base: int, keep_length: bool = False) -> "RecordSpec":
        """The same declared numbers read under a different convention.

        ``keep_length`` defaults to False because the interesting case is the one
        where nobody wrote the record length down - with it declared, the shift is
        caught for free by the span check.
        """
        return RecordSpec(
            list(self.fields),
            index_base=index_base,
            encoding=self.encoding,
            length=self.length if keep_length else None,
            name=f"{self.name}@base{index_base}",
        )

    # -- validation -------------------------------------------------------

    def structural_issues(self) -> List[str]:
        issues: List[str] = []
        ordered = sorted(self.fields, key=lambda f: self._slices[f.name][0])
        cursor = 0
        for f in ordered:
            start, end = self._slices[f.name]
            if start < cursor:
                prev = [g.name for g in ordered if self._slices[g.name][1] > start and g is not f]
                issues.append(
                    f"ERROR overlap: {f.name} starts at byte {start} but "
                    f"{prev[0] if prev else 'a prior field'} runs to {cursor}"
                )
            elif start > cursor:
                issues.append(f"note  gap: {start - cursor} undeclared byte(s) before {f.name}")
            cursor = max(cursor, end)
        if self.length is not None and cursor != self.length:
            issues.append(
                f"ERROR length: fields span {cursor} bytes, declared record length is {self.length}"
            )
        return issues

    def validate(self) -> None:
        errs = [i for i in self.structural_issues() if i.startswith("ERROR")]
        if errs:
            raise SpecError("; ".join(errs))

    def describe(self) -> str:
        rows = [f"{'field':<14}{'bytes':>12}  {'len':>4}  kind"]
        rows.append("-" * 52)
        for f in sorted(self.fields, key=lambda f: self._slices[f.name][0]):
            s, e = self._slices[f.name]
            rows.append(f"{f.name:<14}{f'[{s}:{e})':>12}  {f.length:>4}  {f.kind.describe()}")
        rows.append("-" * 52)
        rows.append(f"{'record':<14}{self.record_length:>12} bytes  index_base={self.index_base}")
        return "\n".join(rows)


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------


def frame_records(
    data: bytes, spec: RecordSpec, framing: str = "auto"
) -> Tuple[List[bytes], str, List[str]]:
    """Split a byte stream into records.

    ``lines``  - newline delimited, the modern default
    ``block``  - RECFM=F: no delimiter at all, records are cut at a fixed length
    ``auto``   - block when the stream divides evenly and has no bare newline

    Returns (records, framing_used, notes). Choosing wrong is not a subtle error
    on text files and is a completely silent one on files containing COMP-3,
    because packed bytes contain 0x0D and 0x0A as ordinary data.
    """
    notes: List[str] = []
    n = spec.record_length
    has_binary = any(f.kind.binary for f in spec.fields)

    if framing == "auto":
        divides = n > 0 and len(data) % n == 0
        if has_binary and divides:
            framing = "block"
            notes.append(
                "auto-framing chose block: the layout contains packed fields, whose "
                "bytes include 0x0A/0x0D as data"
            )
        elif b"\n" not in data and b"\r" not in data:
            framing = "block"
        else:
            framing = "lines"

    if framing == "block":
        if n <= 0:
            raise SpecError("block framing needs a record length")
        if len(data) % n:
            notes.append(
                f"WARNING stream is {len(data)} bytes, not a multiple of {n} - "
                f"{len(data) % n} trailing byte(s) form a partial record"
            )
        return [data[i : i + n] for i in range(0, len(data), n)], "block", notes

    recs = [r[:-1] if r.endswith(b"\r") else r for r in data.split(b"\n")]
    if recs and recs[-1] == b"":
        recs.pop()
    if has_binary:
        notes.append(
            "WARNING line framing on a layout with packed fields - any packed value "
            "whose bytes include 0x0A or 0x0D will be split mid-record"
        )
    return recs, "lines", notes


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


@dataclass
class ParseResult:
    rows: List[Dict[str, Any]]
    errors: List[Tuple[int, str, str]]
    framing: str
    notes: List[str]
    lengths: List[int]

    @property
    def ok(self) -> bool:
        return not self.errors

    def column(self, name: str) -> List[Any]:
        return [r[name] for r in self.rows]

    def total(self, name: str) -> Decimal:
        return sum((v for v in self.column(name) if v is not None), Decimal(0))


def parse(
    data: bytes,
    spec: RecordSpec,
    framing: str = "auto",
    on_error: str = "collect",
) -> ParseResult:
    """Parse by byte offset. ``on_error`` is 'collect' (row kept, field None) or 'raise'."""
    spec.validate()
    records, used, notes = frame_records(data, spec, framing)
    rows: List[Dict[str, Any]] = []
    errors: List[Tuple[int, str, str]] = []
    lengths = [len(r) for r in records]

    for i, rec in enumerate(records):
        row: Dict[str, Any] = {}
        if len(rec) < spec.span:
            errors.append(
                (i, "<record>", f"short record: {len(rec)} bytes, layout needs {spec.span}")
            )
        for f in spec.fields:
            s, e = spec.slice_of(f.name)
            raw = rec[s:e]
            if len(raw) < f.length:
                row[f.name] = None
                if len(rec) >= spec.span:  # already reported as a short record otherwise
                    errors.append((i, f.name, f"truncated: {len(raw)} of {f.length} bytes"))
                continue
            try:
                row[f.name] = f.kind.decode(raw, spec.encoding)
            except FieldError as exc:
                if on_error == "raise":
                    raise FieldError(f"record {i}, field {f.name}: {exc}") from None
                row[f.name] = None
                errors.append((i, f.name, str(exc)))
        rows.append(row)

    return ParseResult(rows, errors, used, notes, lengths)


def parse_naive(data: bytes, spec: RecordSpec, encoding: Optional[str] = None) -> List[Dict[str, Any]]:
    """What every character-slicing reader does, reproduced faithfully.

    Decode the whole stream, split lines, slice by *character* offset, strip, and
    coerce with int() then float() then leave as text. This is ``read_fwf`` with
    the pandas removed, and it is the baseline every comparison here is against.
    """
    enc = encoding or spec.encoding
    text = data.decode(enc, errors="replace")
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        row: Dict[str, Any] = {}
        for f in spec.fields:
            s, e = spec.slice_of(f.name)
            cell = line[s:e].strip()
            if not cell:
                row[f.name] = None
                continue
            try:
                row[f.name] = int(cell)
            except ValueError:
                try:
                    row[f.name] = float(cell)
                except ValueError:
                    row[f.name] = cell
        out.append(row)
    return out


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


@dataclass
class AuditReport:
    verdict: str
    findings: List[str]
    stats: Dict[str, Any]

    def text(self) -> str:
        return "\n".join([self.verdict] + [f"  {f}" for f in self.findings])

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text()


def audit(data: bytes, spec: RecordSpec, framing: str = "auto") -> AuditReport:
    """Everything that decides whether this file loads correctly, before it loads.

    Runs on bytes. Every finding is something that produces a *plausible* wrong
    answer rather than an exception, which is the only class of defect worth a
    pre-flight check.
    """
    findings: List[str] = list(spec.structural_issues())
    stats: Dict[str, Any] = {}
    fatal = any(f.startswith("ERROR") for f in findings)

    records, used, notes = frame_records(data, spec, framing)
    findings.extend(n if n.startswith(("WARNING", "note")) else f"note  {n}" for n in notes)
    stats["framing"] = used
    stats["records"] = len(records)

    lengths = [len(r) for r in records]
    stats["lengths"] = lengths
    short = [i for i, ln in enumerate(lengths) if ln < spec.span]
    long_ = [i for i, ln in enumerate(lengths) if ln > spec.record_length]
    if short:
        findings.append(
            f"WARNING {len(short)} record(s) shorter than the {spec.span}-byte layout "
            f"(first: #{short[0]}, {lengths[short[0]]} bytes) - trailing fields read as null, "
            f"not as an error. Common cause: an editor or FTP transfer stripping trailing blanks"
        )
    if long_:
        findings.append(f"WARNING {len(long_)} record(s) longer than {spec.record_length} bytes")

    # byte length vs decoded character length: exactly the records a character
    # slicer misaligns, and only those.
    divergent = 0
    undecodable = 0
    for rec in records:
        try:
            if len(rec.decode(spec.encoding)) != len(rec):
                divergent += 1
        except UnicodeDecodeError:
            undecodable += 1
    stats["divergent_records"] = divergent
    stats["undecodable_records"] = undecodable
    if divergent:
        findings.append(
            f"WARNING {divergent} of {len(records)} record(s) contain multi-byte characters. "
            f"Byte offsets and character offsets differ on those records only, so a "
            f"character-slicing reader misaligns every field after the first wide character - "
            f"in {divergent} rows out of {len(records)}, data-dependent and invisible in a head()"
        )
    if undecodable:
        findings.append(
            f"note  {undecodable} record(s) are not decodable as {spec.encoding} at all "
            f"(expected when the layout has packed fields)"
        )

    # sign carriers sitting in fields declared unsigned
    sign_chars = set((_OVERPUNCH_POS.keys() | _OVERPUNCH_NEG.keys()))
    for f in spec.fields:
        if not isinstance(f.kind, Int):
            continue
        s, e = spec.slice_of(f.name)
        carriers = [
            rec[s:e].strip(b" \x00")[-1:].decode("latin-1")
            for rec in records
            if len(rec) >= e and rec[s:e].strip(b" \x00")[-1:].decode("latin-1") in sign_chars
        ]
        if carriers:
            neg = sum(1 for c in carriers if c in _OVERPUNCH_NEG)
            findings.append(
                f"WARNING field {f.name} is declared Int but {len(carriers)} value(s) carry an "
                f"overpunch sign, {neg} of them negative. Declared unsigned, those {neg} row(s) "
                f"flip - and the sign character is also the final digit, so removing it as "
                f"punctuation loses a significant figure too"
            )

    # packed fields whose bytes collide with record separators
    for f in spec.fields:
        if not f.kind.binary:
            continue
        s, e = spec.slice_of(f.name)
        collide = sum(1 for rec in records if len(rec) >= e and set(rec[s:e]) & {0x0A, 0x0D})
        if collide:
            findings.append(
                f"WARNING field {f.name} is packed decimal and {collide} value(s) contain the "
                f"byte 0x0A or 0x0D. A negative amount whose last digit is 0 encodes as 0x?D. "
                f"Any line-based read, and any FTP transfer in ASCII mode, corrupts those records"
            )

    if fatal:
        verdict = "NOT SAFE TO LOAD - the layout itself is inconsistent"
    elif any(x.startswith("WARNING") for x in findings):
        verdict = (
            f"LOADS, BUT NOT AS WRITTEN - {sum(1 for x in findings if x.startswith('WARNING'))} "
            f"warning(s) across {len(records)} records"
        )
    else:
        verdict = f"CLEAN - {len(records)} records, {spec.record_length} bytes each"

    return AuditReport(verdict, findings, stats)


# --------------------------------------------------------------------------
# sample files
# --------------------------------------------------------------------------


def _pad(s: str, n: int, encoding: str = "utf-8") -> bytes:
    b = s.encode(encoding)
    if len(b) > n:
        raise SpecError(f"{s!r} is {len(b)} bytes, field is {n}")
    return b + b" " * (n - len(b))


def _overpunch(value: Decimal, width: int, scale: int) -> bytes:
    unscaled = int((value * (10 ** scale)).to_integral_value())
    neg = unscaled < 0
    digits = str(abs(unscaled)).rjust(width, "0")
    if len(digits) > width:
        raise SpecError(f"{value} does not fit in {width} digits")
    table = _OVERPUNCH_NEG if neg else _OVERPUNCH_POS
    inv = {v: k for k, v in table.items()}
    return (digits[:-1] + inv[int(digits[-1])]).encode("ascii")


def _packed(value: Decimal, nbytes: int, scale: int) -> bytes:
    unscaled = int((value * (10 ** scale)).to_integral_value())
    ndigits = nbytes * 2 - 1
    digits = str(abs(unscaled)).rjust(ndigits, "0")
    if len(digits) > ndigits:
        raise SpecError(f"{value} needs more than {ndigits} digits")
    nibbles = [int(c) for c in digits] + [0x0D if unscaled < 0 else 0x0C]
    return bytes(nibbles[i] << 4 | nibbles[i + 1] for i in range(0, len(nibbles), 2))


CUSTOMER_SPEC = RecordSpec(
    [
        Field("cust_id", 1, 8, Text()),
        Field("name", 9, 20, Text()),
        Field("country", 29, 2, Text()),
        Field("status", 31, 1, Text()),
        Field("qty", 32, 5, Int()),
        Field("net_amount", 37, 9, Overpunch(scale=2)),
        Field("list_price", 46, 9, Implied(scale=2)),
        Field("open_date", 55, 8, Date()),
    ],
    index_base=1,
    encoding="utf-8",
    length=62,
    name="CUSTMAST",
)


#: (cust_id, name, country, status, qty, net_amount, list_price, open_date)
_CUSTOMERS: List[Tuple[str, str, str, str, int, str, str, str]] = [
    ("C0001001", "Acme Industrial", "US", "A", 120, "24500.00", "26000.00", "20210304"),
    ("C0001002", "Zoe Ahlstrom AB", "SE", "A", 40, "8100.50", "8100.50", "20220117"),
    ("C0001003", "Zoë Ahlström AB", "SE", "A", 40, "8100.50", "8100.50", "20220117"),
    ("C0001004", "Muller Werke", "DE", "A", 305, "61250.00", "62000.00", "20190822"),
    ("C0001005", "Müller Werke", "DE", "H", 305, "-1425.30", "62000.00", "20190822"),
    ("C0001006", "陈晓贸易", "CN", "A", 900, "182300.00", "185000.00", "20230405"),
    ("C0001007", "Nordvik AS", "NO", "A", 15, "-980.00", "3200.00", "20240211"),
    ("C0001008", "Ferreira Ltda", "BR", "A", 60, "12750.25", "12750.25", "20201130"),
    ("C0001009", "Sakura KK", "JP", "A", 210, "45900.00", "46000.00", "20220906"),
    ("C0001010", "さくら商事", "JP", "A", 210, "45900.00", "46000.00", "20220906"),
    ("C0001011", "Delta Logistics", "US", "C", 0, "-15600.75", "0.00", "20180719"),
    ("C0001012", "Orion Freight", "GB", "A", 88, "19420.00", "19800.00", "20231001"),
]


def build_customer_file(short_record_at: Optional[int] = 11) -> bytes:
    """Newline-framed UTF-8 customer master.

    Two pairs of rows are deliberately identical apart from an accent (rows 2/3 and
    4/5) and one row is CJK, so the byte-versus-character question has a control.
    ``short_record_at`` simulates trailing blanks stripped in transit.
    """
    out = bytearray()
    for i, (cid, nm, cc, st, qty, net, lp, dt) in enumerate(_CUSTOMERS):
        rec = bytearray()
        rec += _pad(cid, 8)
        rec += _pad(nm, 20)
        rec += _pad(cc, 2)
        rec += _pad(st, 1)
        rec += str(qty).rjust(5, "0").encode("ascii")
        rec += _overpunch(Decimal(net), 9, 2)
        rec += str(int(Decimal(lp) * 100)).rjust(9, "0").encode("ascii")
        rec += dt.encode("ascii")
        if short_record_at is not None and i == short_record_at:
            rec = bytearray(bytes(rec).rstrip(b" ")[:-8])  # blanks stripped, date lost
        out += rec + b"\n"
    return bytes(out)


BALANCE_SPEC = RecordSpec(
    [
        Field("acct", 1, 10, Text()),
        Field("branch", 11, 3, Text()),
        Field("balance", 14, 5, Packed(scale=2)),
        Field("adjustment", 19, 5, Packed(scale=2)),
        Field("asof", 24, 8, Date()),
    ],
    index_base=1,
    encoding="latin-1",
    length=31,
    name="ACCTBAL",
)

_BALANCES = [
    ("ACC0000101", "SGP", "18420.55", "0.00", "20240131"),
    ("ACC0000102", "SGP", "902.10", "-1234.50", "20240131"),  # adjustment ends 0x0D
    ("ACC0000103", "LDN", "-45.99", "12.00", "20240131"),
    ("ACC0000104", "NYC", "1250000.00", "-300.10", "20240131"),
    ("ACC0000105", "LDN", "0.00", "0.00", "20240131"),
    ("ACC0000106", "NYC", "77310.40", "-9.90", "20240131"),
]


def build_balance_file() -> bytes:
    """RECFM=F block-framed file with two COMP-3 fields and no record separator."""
    out = bytearray()
    for acct, br, bal, adj, dt in _BALANCES:
        out += _pad(acct, 10, "latin-1")
        out += _pad(br, 3, "latin-1")
        out += _packed(Decimal(bal), 5, 2)
        out += _packed(Decimal(adj), 5, 2)
        out += dt.encode("ascii")
    return bytes(out)
