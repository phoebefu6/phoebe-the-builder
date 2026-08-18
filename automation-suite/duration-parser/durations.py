"""Duration string parsing, and the strings that mean two things at once.

`parse(text) -> timedelta` is the shape every library ships. It is the wrong
shape, and not because any particular implementation is buggy - eight of the
implementations modelled here are correct, each against its own rule set, and
they still return different numbers for the same characters.

Three separate reasons, none of which a single return value can express.

1. The grammar is not in the string. `90` is 90 seconds to ffmpeg, 90 minutes
   to Jira, and 90 *days* to Excel - whose time unit is the day, so a bare
   number in a time-formatted cell is a day count. Nothing in the text says
   which parser is reading it. Same characters, three answers, ratio 86400.
2. The unit letter is overloaded, sometimes by position and sometimes by case.
   In ISO 8601 `M` is months before the `T` and minutes after it: `P1M` and
   `PT1M` differ by a factor of ~43800. In systemd the same distinction is
   carried by *case alone*: `1M` is a month, `1m` is a minute. A shift key.
3. A calendar duration has no length until you say *when*. `P1M` from 1 Jan is
   31 days; from 1 Feb 2024 it is 29; from 31 Jan it is 29 and also not
   invertible. `P1D` across a US spring-forward is 82800 seconds, not 86400.
   The length is a property of `(duration, instant, timezone)`, and a function
   of the text alone has two thirds of that missing.

So the interesting output is not a `timedelta`. It is a *verdict over every
reading a conforming parser could return*:

    exact      - every grammar that accepts the text agrees on one number, and
                 that number does not depend on when you start
    anchored   - the accepting grammars agree symbolically, but the elapsed
                 seconds depend on the anchor instant (month length, DST)
    ambiguous  - two or more grammars accept it and return different numbers.
                 Both are right. The string is the problem
    rejected   - no modelled grammar accepts it

Only `exact` is safe to hand to `timedelta`. `ambiguous` is the dangerous one,
because every parser involved returns successfully.

What "fixed" units cost
-----------------------
Prometheus defines `d` as exactly 24h and `y` as exactly 365d; systemd defines
`M` as 30.44 days and `y` as 365.25 days; Jira defines `d` as 8 *working* hours
and `w` as 5 days. These are exact numbers, so they never look anchored - they
have simply pre-committed to an answer that is wrong twice a year (DST) or wrong
in 11 of 12 months (a 30.44-day month). The audit reports that substitution as a
finding rather than silently agreeing with itself.

Grammar kinds are labelled honestly:
  specification - written down and normative (Go, ISO 8601, Prometheus, systemd)
  tool format   - the documented input format of one tool (ffmpeg, Jira)
  convention    - widespread and unwritten (Excel h:mm cells, `1h30` shorthand)

Standard library only: `re`, `datetime`, `zoneinfo`, `calendar`, `dataclasses`,
`enum`.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:  # zoneinfo needs a tz database; every CI image used here has one
    from zoneinfo import ZoneInfo

    _NY = ZoneInfo("America/New_York")
    HAVE_TZDB = True
except Exception:  # pragma: no cover - fallback keeps the module importable
    _NY = timezone(timedelta(hours=-5))
    HAVE_TZDB = False

SECOND = 1.0
MINUTE = 60.0
HOUR = 3600.0
DAY24 = 86400.0

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class Verdict(Enum):
    """Four-valued, and only the first is safe to convert to a timedelta."""

    EXACT = "exact"
    ANCHORED = "anchored"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Nominal:
    """The part of a duration that has no length until an instant is named.

    Months and days are kept separate and *not* reduced to each other: adding
    one month then one day is not adding 31 days, and neither is commutative
    with the other when the month end clamps.
    """

    months: int = 0
    days: int = 0

    def is_zero(self) -> bool:
        return self.months == 0 and self.days == 0

    def __str__(self) -> str:
        bits = []
        if self.months:
            bits.append(f"{self.months}mo")
        if self.days:
            bits.append(f"{self.days}d")
        return "+".join(bits) if bits else "0"


@dataclass(frozen=True)
class Reading:
    """What one grammar makes of one string.

    `exact_s` is the part that is a fixed number of seconds under that grammar
    (which includes any *fixed substitution* it makes for a calendar unit).
    `nominal` is the part that still needs an anchor. A reading can have both:
    ISO `P1MT30M` is one nominal month plus 1800 exact seconds.
    """

    grammar: str
    ok: bool
    exact_s: float = 0.0
    nominal: Nominal = field(default_factory=Nominal)
    error: Optional[str] = None
    notes: Tuple[str, ...] = ()

    @property
    def anchored(self) -> bool:
        return self.ok and not self.nominal.is_zero()

    def resolve(self, anchor: datetime) -> float:
        """Elapsed seconds from `anchor`, wall-clock semantics.

        Calendar months are added on the local calendar, then days, then the
        exact part. Elapsed time is measured in UTC afterwards, which is the
        only way DST shows up at all.
        """
        if not self.ok:
            raise ValueError(f"{self.grammar} did not accept this text")
        end = add_nominal(anchor, self.nominal.months, self.nominal.days)
        wall = (end.astimezone(timezone.utc) - anchor.astimezone(timezone.utc)).total_seconds()
        return wall + self.exact_s


@dataclass(frozen=True)
class Finding:
    """One named mechanism, with a severity that says who finds out.

    blocking - some parser refuses the string outright
    silent   - every parser involved returns successfully and they disagree,
               or the value quietly depends on something not in the string
    advisory - true and worth knowing, no divergence in this corpus
    """

    code: str
    severity: str
    message: str
    grammar: Optional[str] = None


@dataclass(frozen=True)
class Audit:
    """Everything known about one duration string."""

    text: str
    verdict: Verdict
    readings: Tuple[Reading, ...]
    findings: Tuple[Finding, ...]
    min_s: Optional[float] = None
    max_s: Optional[float] = None
    anchor_min_s: Optional[float] = None
    anchor_max_s: Optional[float] = None

    @property
    def accepted(self) -> Tuple[Reading, ...]:
        return tuple(r for r in self.readings if r.ok)

    @property
    def rejected(self) -> Tuple[Reading, ...]:
        return tuple(r for r in self.readings if not r.ok)

    @property
    def spread_ratio(self) -> Optional[float]:
        """max/min across grammars. 1.0 means unanimous, None means no reading."""
        if self.min_s is None or self.max_s is None:
            return None
        if self.min_s == 0:
            return float("inf") if self.max_s > 0 else 1.0
        return self.max_s / self.min_s

    def distinct_values(self, anchor: Optional[datetime] = None) -> Tuple[float, ...]:
        a = anchor or REFERENCE_ANCHOR
        vals = sorted({round(r.resolve(a), 9) for r in self.accepted})
        return tuple(vals)


# ---------------------------------------------------------------------------
# Calendar arithmetic - the part a pure text parser cannot do
# ---------------------------------------------------------------------------


def add_nominal(anchor: datetime, months: int, days: int) -> datetime:
    """Add calendar months then wall-clock days, the way every date library does.

    Two documented distortions live here and both are reported as findings by
    `audit()` rather than hidden:

    * month-end clamping: 31 Jan + 1 month = 29 Feb (2024), so the operation is
      not injective and not invertible - 29 Feb - 1 month = 29 Jan.
    * a nonexistent local result: 9 Mar 02:30 New York + 1 day lands on
      10 Mar 02:30, which did not happen. `zoneinfo` still returns an instant
      (PEP 495 fold semantics), so the call succeeds with a value that is one
      hour away from the naive reading.
    """
    if months:
        total = (anchor.year * 12 + (anchor.month - 1)) + months
        y, m = divmod(total, 12)
        m += 1
        day = min(anchor.day, calendar.monthrange(y, m)[1])
        naive = anchor.replace(tzinfo=None).replace(year=y, month=m, day=day)
    else:
        naive = anchor.replace(tzinfo=None)
    if days:
        naive = naive + timedelta(days=days)
    return naive.replace(tzinfo=anchor.tzinfo)


def clamped(anchor: datetime, months: int) -> bool:
    """True when month-end clamping moved the day number."""
    if not months:
        return False
    total = (anchor.year * 12 + (anchor.month - 1)) + months
    y, m = divmod(total, 12)
    m += 1
    return anchor.day > calendar.monthrange(y, m)[1]


def nonexistent_local(anchor: datetime, months: int, days: int) -> bool:
    """True when the wall-clock result names a local time that never occurred."""
    if not HAVE_TZDB:
        return False
    end = add_nominal(anchor, months, days)
    round_trip = end.astimezone(timezone.utc).astimezone(end.tzinfo)
    return round_trip.replace(tzinfo=None) != end.replace(tzinfo=None)


def _ny(y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=_NY)


# One anchor per month, plus the three dates that break things: a 31st (clamping),
# the day before each US DST transition (23h and 25h days), and 28 Feb of a
# non-leap year.
DEFAULT_ANCHORS: Tuple[datetime, ...] = tuple(
    [_ny(2024, m, 1) for m in range(1, 13)]
    + [
        _ny(2024, 1, 31),
        _ny(2024, 3, 9, 12),
        _ny(2024, 11, 2, 12),
        _ny(2024, 3, 9, 2, 30),
        _ny(2023, 2, 28),
    ]
)

# The anchor used whenever a single number has to be printed. Deliberately dull:
# a Monday, mid-month, nowhere near a DST edge or a month end.
REFERENCE_ANCHOR: datetime = _ny(2024, 6, 10, 12)


# ---------------------------------------------------------------------------
# Grammar metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grammar:
    """One duration dialect, modelled from its own rule set.

    `kind` is the honest provenance of the rules:
      specification - normative and written down
      tool format   - the documented input format of one tool
      convention    - widespread, unwritten, and implemented by many UIs
    """

    name: str
    kind: str
    unitless: str  # what a bare number means: "reject" | unit name
    note: str


GRAMMARS: Tuple[Grammar, ...] = (
    Grammar(
        "go",
        "specification",
        "reject",
        "time.ParseDuration: ns/us/µs/μs/ms/s/m/h only. Fractions anywhere, any "
        "order, repeats summed, one leading sign for the whole string. No unit "
        "larger than an hour exists, because a day is not a fixed length.",
    ),
    Grammar(
        "iso8601",
        "specification",
        "reject",
        "PnYnMnWnDTnHnMnS. M is months before the T and minutes after it. "
        "Y/M/W/D are nominal - they have no length without an anchor.",
    ),
    Grammar(
        "prometheus",
        "specification",
        "reject",
        "y/w/d/h/m/s/ms, integers only, strictly descending order, each unit at "
        "most once. d is exactly 24h and y exactly 365d - fixed, not calendar.",
    ),
    Grammar(
        "systemd",
        "specification",
        "s",
        "systemd.time: long and short unit names, whitespace optional, bare "
        "numbers take the default unit (seconds). M is a month of 30.44 days, "
        "m is a minute - the difference is the shift key.",
    ),
    Grammar(
        "ffmpeg",
        "tool format",
        "s",
        "[-][HH:]MM:SS[.m] or [-]S+[.m][s|ms|us]. Colon fields are "
        "right-aligned to seconds, so 1:30 is ninety seconds.",
    ),
    Grammar(
        "jira",
        "tool format",
        "m",
        "w/d/h/m where a day is 8 working hours and a week is 5 days "
        "(the defaults). A bare number is minutes.",
    ),
    Grammar(
        "excel",
        "convention",
        "d",
        "A time-formatted cell: h:mm[:ss] left-aligned to hours, and the "
        "underlying unit is the *day*, so a bare 1 is 24 hours.",
    ),
    Grammar(
        "shorthand",
        "convention",
        "reject",
        "What human typists mean: a trailing number with no unit takes the next "
        "unit down, so 1h30 is an hour and a half.",
    ),
)

GRAMMAR_BY_NAME: Dict[str, Grammar] = {g.name: g for g in GRAMMARS}

# ---------------------------------------------------------------------------
# Grammar 1: Go time.ParseDuration
# ---------------------------------------------------------------------------

GO_UNITS: Dict[str, float] = {
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,  # MICRO SIGN
    "μs": 1e-6,  # GREEK SMALL LETTER MU - Go accepts both, they are not equal
    "ms": 1e-3,
    "s": 1.0,
    "m": MINUTE,
    "h": HOUR,
}
_GO_TOKEN = re.compile(r"(\d*\.?\d+)(ns|us|µs|μs|ms|s|m|h)")
# int64 nanoseconds, so the representable range stops just under 292.47 years
GO_MAX_S = (2 ** 63 - 1) / 1e9


def parse_go(text: str) -> Reading:
    s = text.strip()
    notes: List[str] = []
    if not s:
        return Reading("go", False, error="empty duration")
    if s in ("0", "+0", "-0"):
        return Reading("go", True, 0.0, notes=("bare 0 is the one unitless string Go accepts",))
    sign = 1.0
    if s[:1] in "+-":
        sign = -1.0 if s[0] == "-" else 1.0
        s = s[1:]
    if not s:
        return Reading("go", False, error="sign with no magnitude")
    pos, total = 0, 0.0
    seen: Dict[str, int] = {}
    while pos < len(s):
        m = _GO_TOKEN.match(s, pos)
        if not m:
            rest = s[pos:]
            # Distinguish Go's two error strings, because they say different
            # things about the input: a number with no unit at all, versus a
            # unit Go does not have (every unit above the hour).
            num = re.match(r"\d*\.?\d+", rest)
            if num is None:
                return Reading("go", False, error=f"invalid syntax at {rest!r}")
            tail = rest[num.end():]
            if not tail:
                return Reading("go", False, error=f"missing unit in duration {text!r}")
            unit = re.match(r"[^\d.]+", tail)
            if unit is None:
                return Reading("go", False, error=f"invalid syntax at {tail!r}")
            return Reading("go", False, error=f"unknown unit {unit.group(0)!r} in duration {text!r}")
        num, unit = m.group(1), m.group(2)
        seen[unit] = seen.get(unit, 0) + 1
        total += float(num) * GO_UNITS[unit]
        pos = m.end()
    for unit, n in seen.items():
        if n > 1:
            notes.append(f"unit {unit!r} appears {n} times and Go sums them")
    if abs(total) > GO_MAX_S:
        return Reading("go", False, error="out of range for int64 nanoseconds")
    return Reading("go", True, sign * total, notes=tuple(notes))


# ---------------------------------------------------------------------------
# Grammar 2: ISO 8601 durations
# ---------------------------------------------------------------------------

_ISO = re.compile(
    r"^(?P<sign>[+-])?P"
    r"(?:(?P<y>\d+(?:[.,]\d+)?)Y)?"
    r"(?:(?P<mo>\d+(?:[.,]\d+)?)M)?"
    r"(?:(?P<w>\d+(?:[.,]\d+)?)W)?"
    r"(?:(?P<d>\d+(?:[.,]\d+)?)D)?"
    r"(?:T"
    r"(?:(?P<h>\d+(?:[.,]\d+)?)H)?"
    r"(?:(?P<mi>\d+(?:[.,]\d+)?)M)?"
    r"(?:(?P<sec>\d+(?:[.,]\d+)?)S)?"
    r")?$"
)
_ISO_ORDER = ("y", "mo", "w", "d", "h", "mi", "sec")


def parse_iso(text: str) -> Reading:
    s = text.strip()
    m = _ISO.match(s)
    if not m or s.rstrip("+-") in ("P", ""):
        return Reading("iso8601", False, error="not a PnYnMnWnDTnHnMnS duration")
    parts = {k: m.group(k) for k in _ISO_ORDER}
    if all(v is None for v in parts.values()):
        return Reading("iso8601", False, error="P with no components")
    if "T" in s and not any(parts[k] for k in ("h", "mi", "sec")):
        return Reading("iso8601", False, error="T designator with no time components")
    present = [k for k in _ISO_ORDER if parts[k] is not None]
    notes: List[str] = []
    # "The smallest value used may have a decimal fraction" - anything else is
    # a fraction of a unit whose length is not yet decided.
    for k in present[:-1]:
        if "." in parts[k] or "," in parts[k]:
            return Reading(
                "iso8601",
                False,
                error=f"fraction on {k!r}, which is not the smallest component present",
            )
    def num(k: str) -> float:
        v = parts[k]
        return 0.0 if v is None else float(v.replace(",", "."))

    for k in ("y", "mo", "w", "d"):
        v = num(k)
        if v != int(v):
            return Reading(
                "iso8601",
                False,
                error=f"fractional {k!r}: a fraction of a calendar unit has no defined length",
            )
    sign = -1.0 if m.group("sign") == "-" else 1.0
    if sign < 0 and len(present) > 1:
        notes.append("the leading minus applies to the whole duration, not the first component")
    months = int(num("y")) * 12 + int(num("mo"))
    days = int(num("w")) * 7 + int(num("d"))
    exact = num("h") * HOUR + num("mi") * MINUTE + num("sec")
    if parts["mo"] is not None:
        notes.append("M before the T is months; the same letter after the T is minutes")
    if sign < 0:
        months, days, exact = -months, -days, -exact
    return Reading("iso8601", True, exact, Nominal(months, days), notes=tuple(notes))


# ---------------------------------------------------------------------------
# Grammar 3: Prometheus
# ---------------------------------------------------------------------------

_PROM = re.compile(
    r"^(?:(\d+)y)?(?:(\d+)w)?(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?(?:(\d+)ms)?$"
)
PROM_FACTORS = (365 * DAY24, 7 * DAY24, DAY24, HOUR, MINUTE, SECOND, 1e-3)
PROM_UNITS = ("y", "w", "d", "h", "m", "s", "ms")


def parse_prometheus(text: str) -> Reading:
    s = text.strip()
    if s == "0":
        return Reading("prometheus", True, 0.0, notes=("0 is accepted without a unit",))
    if not s:
        return Reading("prometheus", False, error="empty")
    m = _PROM.match(s)
    if not m or not any(m.groups()):
        if re.search(r"[.,]", s):
            return Reading("prometheus", False, error="fractions are not permitted")
        return Reading(
            "prometheus",
            False,
            error="units must be y,w,d,h,m,s,ms, each at most once, in descending order",
        )
    total, notes = 0.0, []
    for unit, factor, g in zip(PROM_UNITS, PROM_FACTORS, m.groups()):
        if g is None:
            continue
        total += int(g) * factor
        if unit in ("y", "w", "d"):
            notes.append(
                {"y": "y is a fixed 365 days here, never 366",
                 "w": "w is a fixed 7x24h, so it is an hour out either side of a DST change",
                 "d": "d is a fixed 24h, so it is an hour out either side of a DST change"}[unit]
            )
    return Reading("prometheus", True, total, notes=tuple(notes))


# ---------------------------------------------------------------------------
# Grammar 4: systemd.time
# ---------------------------------------------------------------------------
#
# The unit table is case-sensitive in exactly one place, and it is the place
# that matters: `M` is a month and `m` is a minute. systemd defines the month as
# 30.44 days and the year as 365.25 days - fixed averages, so they are never
# right for a particular month and never anchored either.

SYSTEMD_UNITS: Dict[str, float] = {
    "usec": 1e-6, "us": 1e-6, "µs": 1e-6,
    "msec": 1e-3, "ms": 1e-3,
    "seconds": SECOND, "second": SECOND, "sec": SECOND, "s": SECOND,
    "minutes": MINUTE, "minute": MINUTE, "min": MINUTE, "m": MINUTE,
    "hours": HOUR, "hour": HOUR, "hr": HOUR, "h": HOUR,
    "days": DAY24, "day": DAY24, "d": DAY24,
    "weeks": 7 * DAY24, "week": 7 * DAY24, "w": 7 * DAY24,
    "months": 30.44 * DAY24, "month": 30.44 * DAY24, "M": 30.44 * DAY24,
    "years": 365.25 * DAY24, "year": 365.25 * DAY24, "y": 365.25 * DAY24,
}
# Longest first so "min" wins over "m" and "msec" over "ms".
_SYSTEMD_UNIT_RE = "|".join(sorted(SYSTEMD_UNITS, key=len, reverse=True))
_SYSTEMD_TOKEN = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(" + _SYSTEMD_UNIT_RE + r")?")


def parse_systemd(text: str) -> Reading:
    s = text.strip()
    if not s or s[:1] == "-":
        return Reading("systemd", False, error="no sign is accepted in a time span")
    pos, total, notes, unitless = 0, 0.0, [], 0
    while pos < len(s):
        m = _SYSTEMD_TOKEN.match(s, pos)
        if not m or m.end() == pos:
            return Reading("systemd", False, error=f"not a time span at {s[pos:]!r}")
        num, unit = float(m.group(1)), m.group(2)
        if unit is None:
            unitless += 1
            total += num  # the default unit, which callers can change per setting
        else:
            total += num * SYSTEMD_UNITS[unit]
            if unit in ("M", "month", "months"):
                notes.append("a month here is a fixed 30.44 days, not this month")
            if unit in ("y", "year", "years"):
                notes.append("a year here is a fixed 365.25 days")
        pos = m.end()
    if unitless:
        notes.append(
            f"{unitless} component(s) had no unit and took the default (seconds); "
            "in a unit file the default depends on the setting being parsed"
        )
    return Reading("systemd", True, total, notes=tuple(notes))


# ---------------------------------------------------------------------------
# Grammar 5: ffmpeg -t / -ss duration form
# ---------------------------------------------------------------------------
#
# Colon fields are RIGHT-aligned to seconds, which is the opposite of every
# timesheet UI: `1:30` is a minute and a half, not an hour and a half.

_FF_CLOCK = re.compile(r"^(?P<sign>-)?(?:(?:(?P<h>\d+):)?(?P<m>\d+):)?(?P<s>\d+(?:\.\d+)?)$")
_FF_PLAIN = re.compile(r"^(?P<sign>-)?(?P<n>\d+(?:\.\d+)?)(?P<u>s|ms|us)?$")


def parse_ffmpeg(text: str) -> Reading:
    s = text.strip()
    m = _FF_PLAIN.match(s)
    if m and ":" not in s:
        f = {"s": 1.0, "ms": 1e-3, "us": 1e-6, None: 1.0}[m.group("u")]
        v = float(m.group("n")) * f
        note = () if m.group("u") else ("a bare number is seconds here",)
        return Reading("ffmpeg", True, -v if m.group("sign") else v, notes=note)
    m = _FF_CLOCK.match(s)
    if not m:
        return Reading("ffmpeg", False, error="not [-][HH:]MM:SS[.m] or [-]S+[.m][s|ms|us]")
    h = float(m.group("h") or 0)
    mi = float(m.group("m") or 0)
    sec = float(m.group("s"))
    notes: List[str] = []
    if m.group("m") is not None and m.group("h") is None:
        notes.append("two colon fields are MM:SS here, not HH:MM - the fields fill from the right")
    if m.group("m") is not None and mi >= 60:
        notes.append(f"minute field is {mi:g}, above 60, and is accepted as-is")
    v = h * HOUR + mi * MINUTE + sec
    return Reading("ffmpeg", True, -v if m.group("sign") else v, notes=tuple(notes))


# ---------------------------------------------------------------------------
# Grammar 6: Jira time tracking
# ---------------------------------------------------------------------------
#
# The only grammar here whose day is not 24 hours: a working day of 8 hours and
# a working week of 5 days are the defaults, and both are per-instance settings.
# A bare number is minutes.

JIRA_UNITS: Dict[str, float] = {"w": 5 * 8 * HOUR, "d": 8 * HOUR, "h": HOUR, "m": MINUTE}
_JIRA_TOKEN = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([wdhm])?")


def parse_jira(text: str) -> Reading:
    s = text.strip()
    if not s or s[:1] == "-":
        return Reading("jira", False, error="no negative work log")
    pos, total, notes, unitless = 0, 0.0, [], 0
    while pos < len(s):
        m = _JIRA_TOKEN.match(s, pos)
        if not m or m.end() == pos:
            return Reading("jira", False, error=f"units are w,d,h,m only; failed at {s[pos:]!r}")
        num, unit = float(m.group(1)), m.group(2)
        if unit is None:
            unitless += 1
            total += num * MINUTE
        else:
            total += num * JIRA_UNITS[unit]
            if unit == "d":
                notes.append("a day is 8 working hours here, a third of a calendar day")
            if unit == "w":
                notes.append("a week is 5 working days of 8 hours, a quarter of a calendar week")
        pos = m.end()
    if unitless:
        notes.append(f"{unitless} component(s) had no unit and were read as minutes")
    return Reading("jira", True, total, notes=tuple(notes))


# ---------------------------------------------------------------------------
# Grammar 7: a time-formatted spreadsheet cell
# ---------------------------------------------------------------------------
#
# Colon fields are LEFT-aligned from hours, the exact opposite of ffmpeg, and
# the underlying unit of the cell is the DAY - so a bare 1 is 24 hours and a
# bare 90 is three months.

_XL_CLOCK = re.compile(r"^(?P<sign>-)?(?P<h>\d+):(?P<m>\d{1,2})(?::(?P<s>\d{1,2}(?:\.\d+)?))?$")
_XL_PLAIN = re.compile(r"^(?P<sign>-)?(?P<n>\d+(?:\.\d+)?)$")


def parse_excel(text: str) -> Reading:
    s = text.strip()
    m = _XL_PLAIN.match(s)
    if m:
        v = float(m.group("n")) * DAY24
        return Reading(
            "excel",
            True,
            -v if m.group("sign") else v,
            notes=("the serial unit of a time cell is the day, so a bare number is days",),
        )
    m = _XL_CLOCK.match(s)
    if not m:
        return Reading("excel", False, error="not [-]h:mm[:ss] and not a serial number")
    notes = ["two colon fields are h:mm here - the fields fill from the left"]
    if m.group("s") is None and int(m.group("m")) > 59:
        notes.append("minute field above 59 rolls over into hours in a real cell")
    v = float(m.group("h")) * HOUR + float(m.group("m")) * MINUTE + float(m.group("s") or 0)
    return Reading("excel", True, -v if m.group("sign") else v, notes=tuple(notes))


# ---------------------------------------------------------------------------
# Grammar 8: what a human means by "1h30"
# ---------------------------------------------------------------------------
#
# No specification, widely implemented: a trailing number with no unit takes the
# next unit down from the last one that had one. This is the reading that makes
# `1h30` an hour and a half, and it is why the same string is 3630 seconds to
# systemd and 5400 to the person who typed it.

SHORTHAND_UNITS: Dict[str, float] = {"w": 7 * DAY24, "d": DAY24, "h": HOUR, "m": MINUTE, "s": SECOND}
_NEXT_DOWN: Dict[str, str] = {"w": "d", "d": "h", "h": "m", "m": "s"}
_SH_TOKEN = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([wdhms])?")


def parse_shorthand(text: str) -> Reading:
    s = text.strip()
    sign = 1.0
    if s[:1] == "-":
        sign, s = -1.0, s[1:]
    if not s or ":" in s:
        return Reading("shorthand", False, error="units required, no clock form")
    pos, total, notes, last = 0, 0.0, [], None
    while pos < len(s):
        m = _SH_TOKEN.match(s, pos)
        if not m or m.end() == pos:
            return Reading("shorthand", False, error=f"units are w,d,h,m,s; failed at {s[pos:]!r}")
        num, unit = float(m.group(1)), m.group(2)
        if unit is None:
            if last is None or last not in _NEXT_DOWN:
                return Reading(
                    "shorthand", False, error="a bare number needs a preceding unit to step down from"
                )
            unit = _NEXT_DOWN[last]
            notes.append(f"trailing {num:g} read as {unit!r}, one step below the {last!r} before it")
        total += num * SHORTHAND_UNITS[unit]
        last = unit
        pos = m.end()
    return Reading("shorthand", True, sign * total, notes=tuple(notes))


PARSERS: Dict[str, Callable[[str], Reading]] = {
    "go": parse_go,
    "iso8601": parse_iso,
    "prometheus": parse_prometheus,
    "systemd": parse_systemd,
    "ffmpeg": parse_ffmpeg,
    "jira": parse_jira,
    "excel": parse_excel,
    "shorthand": parse_shorthand,
}

# ---------------------------------------------------------------------------
# Findings - eighteen named mechanisms
# ---------------------------------------------------------------------------
#
# Each detector below fires only on evidence from the readings themselves. None
# of them is a guess about intent: they are all "these two parsers were handed
# this string and returned these two numbers".

FINDING_CODES: Tuple[Tuple[str, str, str], ...] = (
    ("NO_GRAMMAR_ACCEPTS", "blocking", "no modelled grammar accepts the text"),
    ("LEGALITY_DISAGREEMENT", "blocking", "at least one grammar accepts it and at least one refuses"),
    ("FRACTION_REJECTED", "blocking", "a fraction is legal in one grammar and illegal in another"),
    ("AMBIGUOUS_VALUE", "silent", "two grammars accept it and return different numbers"),
    ("UNIT_CASE_SENSITIVE", "silent", "changing the case of a unit letter changes the value"),
    ("POSITIONAL_UNIT", "silent", "the same letter means different units in different positions"),
    ("BARE_NUMBER_DEFAULT_UNIT", "silent", "a unitless number was given a unit by the parser"),
    ("TRAILING_NUMBER_SHORTHAND", "silent", "a trailing unitless number reads two ways"),
    ("COLON_ALIGNMENT", "silent", "colon fields fill from the right in one grammar and the left in another"),
    ("SIGN_SCOPE", "silent", "a leading minus over several components"),
    ("CALENDAR_UNIT_UNANCHORED", "silent", "the length depends on the instant you start from"),
    ("MONTH_END_CLAMP", "silent", "adding a month clamps the day number, so the operation is not invertible"),
    ("NONEXISTENT_LOCAL_TIME", "silent", "the wall-clock result names a local time that never happened"),
    ("FIXED_AVERAGE_SUBSTITUTION", "silent", "a calendar unit was replaced by a fixed average"),
    ("WORKING_TIME_UNIT", "silent", "a day means working hours, not elapsed hours"),
    ("ORDER_OR_REPEAT", "advisory", "component order or repetition is legal in one grammar and not another"),
    ("PRECISION", "advisory", "the value is near a representation limit"),
    ("MU_SIGN_VARIANT", "advisory", "two distinct Unicode characters spell the microsecond unit"),
)
SEVERITY_OF: Dict[str, str] = {c: s for c, s, _ in FINDING_CODES}
DESCRIPTION_OF: Dict[str, str] = {c: d for c, _, d in FINDING_CODES}


def _f(code: str, message: str, grammar: Optional[str] = None) -> Finding:
    return Finding(code, SEVERITY_OF[code], message, grammar)


def _fmt(seconds: float) -> str:
    """Human-sized rendering, used only for messages."""
    a = abs(seconds)
    if a == 0:
        return "0s"
    if a < 1:
        return f"{seconds * 1000:g}ms"
    if a < MINUTE:
        return f"{seconds:g}s"
    if a < HOUR:
        return f"{seconds / MINUTE:g}min"
    if a < 2 * DAY24:
        return f"{seconds / HOUR:.4g}h"
    return f"{seconds / DAY24:.4g}d"


def _ratio(hi: float, lo: float) -> str:
    """Ratios are the point of this tool, so they are never rounded to 1."""
    if lo == 0:
        return "infinite"
    r = abs(hi) / abs(lo)
    return f"{r:,.0f}" if r >= 10 else f"{r:.3g}"


def _swap_unit_case(text: str) -> str:
    return "".join(c.swapcase() if c.isalpha() else c for c in text)


def audit(text: str, anchors: Sequence[datetime] = DEFAULT_ANCHORS) -> Audit:
    """Read one string under every grammar and report what they disagree about."""
    readings = tuple(PARSERS[g.name](text) for g in GRAMMARS)
    accepted = [r for r in readings if r.ok]
    rejected = [r for r in readings if not r.ok]
    findings: List[Finding] = []

    if not accepted:
        return Audit(
            text,
            Verdict.REJECTED,
            readings,
            (_f("NO_GRAMMAR_ACCEPTS", f"{len(rejected)} grammars refused: " +
                "; ".join(f"{r.grammar}: {r.error}" for r in rejected[:3]) + " ..."),),
        )

    ref_vals = {r.grammar: r.resolve(REFERENCE_ANCHOR) for r in accepted}
    per_anchor: List[List[float]] = [[r.resolve(a) for r in accepted] for a in anchors]
    all_vals = [v for row in per_anchor for v in row]
    min_s, max_s = min(ref_vals.values()), max(ref_vals.values())
    a_min, a_max = min(all_vals), max(all_vals)

    distinct_ref = sorted({round(v, 6) for v in ref_vals.values()})
    ambiguous = len(distinct_ref) > 1 or any(len({round(v, 6) for v in row}) > 1 for row in per_anchor)
    # Anchor sensitivity is a property of ONE grammar's reading across anchors.
    # Taking the spread over every grammar at once would report the ambiguity
    # between grammars a second time, which is a different mechanism.
    per_grammar_span = {
        r.grammar: (min(r.resolve(a) for a in anchors), max(r.resolve(a) for a in anchors))
        for r in accepted
    }
    anchored_spread = any(round(hi - lo, 6) != 0 for lo, hi in per_grammar_span.values())

    # --- blocking -----------------------------------------------------------
    if rejected and accepted:
        findings.append(
            _f(
                "LEGALITY_DISAGREEMENT",
                f"{len(accepted)} of {len(readings)} grammars accept it "
                f"({', '.join(r.grammar for r in accepted)}); "
                f"{len(rejected)} refuse ({', '.join(r.grammar for r in rejected)})",
            )
        )
    if re.search(r"\d[.,]\d", text):
        frac_ok = [r.grammar for r in accepted]
        frac_no = [r.grammar for r in rejected if r.error and ("fraction" in r.error or "not permitted" in r.error)]
        if frac_ok and frac_no:
            findings.append(
                _f("FRACTION_REJECTED", f"fraction accepted by {', '.join(frac_ok)} and refused by {', '.join(frac_no)}")
            )

    # --- silent -------------------------------------------------------------
    if ambiguous:
        lo = min(ref_vals, key=lambda k: ref_vals[k])
        hi = max(ref_vals, key=lambda k: ref_vals[k])
        findings.append(
            _f(
                "AMBIGUOUS_VALUE",
                f"{len(distinct_ref)} different values from {len(accepted)} grammars: "
                f"{lo} says {_fmt(min_s)} and {hi} says {_fmt(max_s)}, "
                f"a factor of {_ratio(max_s, min_s)}",
            )
        )

    swapped = _swap_unit_case(text)
    if swapped != text:
        for r in accepted:
            other = PARSERS[r.grammar](swapped)
            if other.ok and round(other.resolve(REFERENCE_ANCHOR), 6) != round(ref_vals[r.grammar], 6):
                findings.append(
                    _f(
                        "UNIT_CASE_SENSITIVE",
                        f"{text!r} is {_fmt(ref_vals[r.grammar])} and {swapped!r} is "
                        f"{_fmt(other.resolve(REFERENCE_ANCHOR))} - a factor of "
                        f"{_ratio(max(other.resolve(REFERENCE_ANCHOR), ref_vals[r.grammar]), min(other.resolve(REFERENCE_ANCHOR), ref_vals[r.grammar]))}",
                        r.grammar,
                    )
                )

    iso = next((r for r in readings if r.grammar == "iso8601" and r.ok), None)
    if iso is not None and "M" in text:
        head, _, tail = text.partition("T")
        if "M" in head and "M" in tail:
            pos_note = "both positions used in one string"
        elif "M" in head:
            pos_note = "before the T, so months"
        else:
            pos_note = "after the T, so minutes"
        alt = text.replace("T", "") if "T" in text else text.replace("P", "PT", 1)
        alt_r = PARSERS["iso8601"](alt)
        if alt_r.ok:
            findings.append(
                _f(
                    "POSITIONAL_UNIT",
                    f"M is {pos_note}: {text!r} is {_fmt(iso.resolve(REFERENCE_ANCHOR))} and "
                    f"{alt!r} is {_fmt(alt_r.resolve(REFERENCE_ANCHOR))}",
                    "iso8601",
                )
            )

    for r in accepted:
        for n in r.notes:
            if "no unit" in n or "bare number" in n or "unitless" in n:
                findings.append(_f("BARE_NUMBER_DEFAULT_UNIT", f"{r.grammar}: {n}", r.grammar))
            elif "one step below" in n:
                findings.append(_f("TRAILING_NUMBER_SHORTHAND", f"{r.grammar}: {n}", r.grammar))
            elif "fixed 30.44" in n or "fixed 365" in n or "fixed 7x24h" in n or "fixed 24h" in n:
                findings.append(_f("FIXED_AVERAGE_SUBSTITUTION", f"{r.grammar}: {n}", r.grammar))
            elif "working" in n:
                findings.append(_f("WORKING_TIME_UNIT", f"{r.grammar}: {n}", r.grammar))
            elif "whole duration" in n:
                findings.append(_f("SIGN_SCOPE", f"{r.grammar}: {n}", r.grammar))
            elif "sums them" in n or "as-is" in n or "rolls over" in n:
                findings.append(_f("ORDER_OR_REPEAT", f"{r.grammar}: {n}", r.grammar))
            elif "from the right" in n or "from the left" in n:
                findings.append(_f("COLON_ALIGNMENT", f"{r.grammar}: {n}", r.grammar))

    ff = next((r for r in readings if r.grammar == "ffmpeg" and r.ok), None)
    xl = next((r for r in readings if r.grammar == "excel" and r.ok), None)
    if ff and xl and ":" in text:
        a, b = ff.resolve(REFERENCE_ANCHOR), xl.resolve(REFERENCE_ANCHOR)
        if round(a, 6) != round(b, 6):
            findings.append(
                _f(
                    "COLON_ALIGNMENT",
                    f"ffmpeg reads {text!r} as {_fmt(a)} (fields from the right), a spreadsheet cell "
                    f"as {_fmt(b)} (from the left) - a factor of {_ratio(max(a, b), min(a, b))}",
                )
            )

    for r in accepted:
        lo_v, hi_v = per_grammar_span[r.grammar]
        if round(hi_v - lo_v, 6) != 0:
            best = min(anchors, key=lambda a: r.resolve(a))
            worst = max(anchors, key=lambda a: r.resolve(a))
            findings.append(
                _f(
                    "CALENDAR_UNIT_UNANCHORED",
                    f"{r.grammar} reads it as {r.nominal} plus {_fmt(r.exact_s)}, which elapses "
                    f"{_fmt(lo_v)} from {best:%Y-%m-%d %H:%M} and {_fmt(hi_v)} from "
                    f"{worst:%Y-%m-%d %H:%M} - a difference of {_fmt(hi_v - lo_v)}",
                    r.grammar,
                )
            )
    if anchored_spread:
        for r in accepted:
            if not r.anchored:
                continue
            if any(clamped(a, r.nominal.months) for a in anchors):
                findings.append(
                    _f("MONTH_END_CLAMP", f"{r.grammar}: from a 31st, +{r.nominal.months} month(s) "
                       "clamps to the month end, and subtracting it again does not return", r.grammar)
                )
            if any(nonexistent_local(a, r.nominal.months, r.nominal.days) for a in anchors):
                findings.append(
                    _f(
                        "NONEXISTENT_LOCAL_TIME",
                        f"{r.grammar}: from at least one anchor the wall-clock result names a local "
                        "time that never happened. Nothing raises: zoneinfo resolves the gap with "
                        "fold=0, the elapsed count comes out as a clean 86400 per day, and the "
                        "instant displays one hour later than the wall time that was asked for",
                        r.grammar,
                    )
                )

    # --- advisory -----------------------------------------------------------
    if any(0 < abs(v) <= 1e-6 for v in ref_vals.values()) or abs(max_s) > GO_MAX_S / 10:
        findings.append(
            _f("PRECISION", f"|value| = {abs(max_s):g}s; Go stores int64 nanoseconds "
               f"(max {GO_MAX_S:,.0f}s) and this module carries floats")
        )
    if "µ" in text or "μ" in text:
        findings.append(
            _f("MU_SIGN_VARIANT", "the microsecond unit is spelled with MICRO SIGN U+00B5 here and "
               "GREEK SMALL LETTER MU U+03BC elsewhere; Go accepts both, most parsers accept one")
        )

    if ambiguous:
        verdict = Verdict.AMBIGUOUS
    elif anchored_spread:
        verdict = Verdict.ANCHORED
    else:
        verdict = Verdict.EXACT

    # De-duplicate while keeping order; the same mechanism can be reported by
    # two grammars and that is one finding, not two.
    seen, uniq = set(), []
    for fi in findings:
        key = (fi.code, fi.message)
        if key not in seen:
            seen.add(key)
            uniq.append(fi)
    return Audit(text, verdict, readings, tuple(uniq), min_s, max_s, a_min, a_max)


# ---------------------------------------------------------------------------
# A corpus, and what a whole configuration looks like at once
# ---------------------------------------------------------------------------
#
# Twenty-six duration strings of the kind that end up in one repository: a
# Prometheus rule file, a systemd timer, an ffmpeg invocation in a Makefile, a
# Jira worklog export, a timesheet CSV, and a JSON API field. Every one of them
# is somebody's correct input. Together they are not readable by any one parser.

CORPUS: Tuple[Tuple[str, str], ...] = (
    ("30s", "prometheus scrape_interval"),
    ("5m", "prometheus rule for/evaluation_interval"),
    ("2h45m", "go flag, cache TTL"),
    ("1h30", "typed into a form by a person"),
    ("1:30", "timesheet CSV cell"),
    ("90", "JSON field named duration, no unit anywhere"),
    ("0.5", "JSON field named timeout"),
    ("1M", "systemd RuntimeMaxSec"),
    ("1m", "the same setting, one shift key away"),
    ("P1M", "ISO field in a subscription record"),
    ("PT1M", "ISO field in a retry policy"),
    ("P1D", "ISO field, retention"),
    ("1d", "prometheus retention flag"),
    ("1w", "log rotation"),
    ("1y", "archive policy"),
    ("P1Y", "contract term"),
    ("P1Y2M3DT4H5M6S", "ISO, every component at once"),
    ("-1.5h", "go, a clock skew allowance"),
    ("1h1h", "a copy-paste accident that parses"),
    ("30m1h", "components out of order"),
    ("500ms", "http client timeout"),
    ("1µs", "benchmark budget, MICRO SIGN"),
    ("1:30:45.5", "media clip length"),
    ("2h 30min", "systemd, long names and a space"),
    ("3w 2d", "a work estimate"),
    ("1 hour", "a person filling in a text box"),
    ("-P1DT1H", "ISO with a leading sign over two components"),
    ("forever", "a retention field somebody filled in honestly"),
)


@dataclass(frozen=True)
class CorpusReport:
    """Corpus-level accounting: the per-grammar and pairwise views."""

    audits: Tuple[Audit, ...]
    accepted_by: Dict[str, int]
    verdicts: Dict[str, int]
    disagreements: Dict[Tuple[str, str], int]
    finding_counts: Dict[str, int]

    @property
    def total(self) -> int:
        return len(self.audits)

    def unanimous(self) -> Tuple[Audit, ...]:
        """Strings where every accepting grammar returns one anchor-free number."""
        return tuple(a for a in self.audits if a.verdict is Verdict.EXACT)

    def lonely(self) -> Tuple[Audit, ...]:
        """Exact only because exactly one grammar could read it at all.

        Worth separating from `unanimous()`: agreement among one parser is not
        agreement, it is a monopoly. These strings are portable nowhere.
        """
        return tuple(a for a in self.unanimous() if len(a.accepted) == 1)

    def contested(self) -> Tuple[Audit, ...]:
        """Two or more grammars accept it and return different numbers."""
        return tuple(a for a in self.audits if a.verdict is Verdict.AMBIGUOUS)


def audit_corpus(
    texts: Sequence[str] = tuple(t for t, _ in CORPUS),
    anchors: Sequence[datetime] = DEFAULT_ANCHORS,
) -> CorpusReport:
    audits = tuple(audit(t, anchors) for t in texts)
    accepted_by = {g.name: 0 for g in GRAMMARS}
    verdicts = {v.value: 0 for v in Verdict}
    disagreements: Dict[Tuple[str, str], int] = {}
    finding_counts: Dict[str, int] = {c: 0 for c, _, _ in FINDING_CODES}

    for a in audits:
        verdicts[a.verdict.value] += 1
        for r in a.accepted:
            accepted_by[r.grammar] += 1
        for fi in a.findings:
            finding_counts[fi.code] += 1
        acc = a.accepted
        for i, x in enumerate(acc):
            for y in acc[i + 1:]:
                # A pair "disagrees" when both accept and either returns a
                # different number at the reference anchor. Same number =
                # agreement even if they got there by different rules.
                if round(x.resolve(REFERENCE_ANCHOR), 6) != round(y.resolve(REFERENCE_ANCHOR), 6):
                    key = tuple(sorted((x.grammar, y.grammar)))
                    disagreements[key] = disagreements.get(key, 0) + 1
    return CorpusReport(audits, accepted_by, verdicts, disagreements, finding_counts)


def best_single_grammar(report: Optional[CorpusReport] = None) -> Tuple[str, int, int]:
    """The best case for `parse(text) -> timedelta`: pick one library and count.

    Returns (grammar, accepted, wrong_number), where `wrong_number` counts the
    strings it accepted and read differently from at least one other grammar -
    the silent failures, not the loud ones.
    """
    rep = report or audit_corpus()
    best: Tuple[str, int, int] = ("", -1, 0)
    for g in GRAMMARS:
        acc = wrong = 0
        for a in rep.audits:
            mine = next((r for r in a.accepted if r.grammar == g.name), None)
            if mine is None:
                continue
            acc += 1
            others = [r for r in a.accepted if r.grammar != g.name]
            if any(
                round(o.resolve(REFERENCE_ANCHOR), 6) != round(mine.resolve(REFERENCE_ANCHOR), 6)
                for o in others
            ):
                wrong += 1
        if acc > best[1]:
            best = (g.name, acc, wrong)
    return best


def safe_form(seconds: float) -> str:
    """The one output shape with no ambiguity left in it: integer seconds.

    Every grammar above reads `<n>s` the same way, and it carries no calendar
    unit, so it needs no anchor. It is also unreadable, which is why nobody
    writes configuration this way and why this whole file exists.
    """
    return f"{int(seconds)}s" if float(seconds).is_integer() else f"{seconds!r}s"


__all__ = [
    "Verdict", "Nominal", "Reading", "Finding", "Audit", "CorpusReport", "Grammar",
    "GRAMMARS", "GRAMMAR_BY_NAME", "PARSERS", "CORPUS", "FINDING_CODES",
    "DEFAULT_ANCHORS", "REFERENCE_ANCHOR", "audit", "audit_corpus",
    "best_single_grammar", "safe_form", "add_nominal", "clamped", "nonexistent_local",
    "parse_go", "parse_iso", "parse_prometheus", "parse_systemd", "parse_ffmpeg",
    "parse_jira", "parse_excel", "parse_shorthand",
]
