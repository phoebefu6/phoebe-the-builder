"""Read a cron expression the way cron reads it, not the way it looks.

Three things in this module do work that a plain-English translator does not:

1.  `match()` implements the POSIX/Vixie day rule, in which a restricted
    day-of-month and a restricted day-of-week are a **union**, not an
    intersection. `0 0 13 * 5` is not Friday the 13th.

2.  `fires()` resolves each wall-clock match against a real IANA time zone, so a
    fire time that lands in a skipped or repeated local hour is reported as such
    instead of being silently coerced. Vixie cron treats those two cases
    differently depending on whether the job has a fixed time or an interval,
    and both behaviours are modelled here.

3.  `audit()` returns the findings - the places where the expression means
    something other than what it reads, fires at an instant you did not intend,
    or changes meaning on a different scheduler.

Every claim the README makes is computed by `evidence.py` from this module.

Run:  python3 -m pytest test_cron.py -q
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

UTC = timezone.utc

# --------------------------------------------------------------------------
# Severities. A finding is not a style note; each of these changes when the
# job runs or what days it runs on.
# --------------------------------------------------------------------------

MISREAD = "MISREAD"  # the expression means something other than it reads
TIMING = "TIMING"  # it fires at an instant you probably did not intend
PORTABILITY = "PORTABILITY"  # it means something else on another scheduler

SEVERITIES = (MISREAD, TIMING, PORTABILITY)

# How far ahead either search will look before giving up on a match. Five years
# covers 29 February and the once-a-year expressions; both searches use it, so
# "no next fire" means the same thing in each.
HORIZON = timedelta(days=366 * 5)

MONTH_NAMES = {
    n: i
    for i, n in enumerate(
        "jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1
    )
}
DOW_NAMES = {
    n: i for i, n in enumerate("sun mon tue wed thu fri sat".split(), start=0)
}
DOW_LABEL = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
MONTH_LABEL = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

MACROS = {
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
    "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}


class CronError(ValueError):
    """The expression cannot be parsed. Raised instead of guessing."""


@dataclass(frozen=True)
class FieldSpec:
    name: str
    lo: int
    hi: int
    names: Optional[Dict[str, int]] = None


SPECS = (
    FieldSpec("minute", 0, 59),
    FieldSpec("hour", 0, 23),
    FieldSpec("day-of-month", 1, 31),
    FieldSpec("month", 1, 12, MONTH_NAMES),
    FieldSpec("day-of-week", 0, 7, DOW_NAMES),
)


@dataclass(frozen=True)
class Field:
    spec: FieldSpec
    raw: str
    values: Tuple[int, ...]
    star: bool  # Vixie sets its STAR flag when the field *begins* with '*'

    def __contains__(self, v: int) -> bool:
        return v in self.values


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_TERM = re.compile(r"^(?P<a>[^-/]+)(?:-(?P<b>[^/]+))?(?:/(?P<step>\d+))?$")


def _atom(text: str, spec: FieldSpec) -> int:
    t = text.strip().lower()
    if spec.names and t in spec.names:
        return spec.names[t]
    if not re.fullmatch(r"\d+", t):
        raise CronError(f"{spec.name}: {text!r} is not a number or a known name")
    v = int(t)
    if v < spec.lo or v > spec.hi:
        raise CronError(f"{spec.name}: {v} is outside {spec.lo}-{spec.hi}")
    return v


def parse_field(raw: str, spec: FieldSpec) -> Field:
    raw = raw.strip()
    if not raw:
        raise CronError(f"{spec.name}: empty field")
    star = raw.startswith("*")
    out: Set[int] = set()
    for term in raw.split(","):
        term = term.strip()
        m = _TERM.match(term)
        if not m:
            raise CronError(f"{spec.name}: cannot parse {term!r}")
        step = int(m.group("step")) if m.group("step") else 1
        if step < 1:
            raise CronError(f"{spec.name}: step must be >= 1")
        a_raw = m.group("a")
        if a_raw == "*":
            lo, hi = spec.lo, spec.hi
        else:
            lo = _atom(a_raw, spec)
            hi = _atom(m.group("b"), spec) if m.group("b") else (
                spec.hi if m.group("step") else lo
            )
        if hi < lo:
            # Vixie rejects a reversed range rather than wrapping it.
            raise CronError(f"{spec.name}: range {term!r} runs backwards")
        out.update(range(lo, hi + 1, step))
    if spec.name == "day-of-week":
        # 7 and 0 are both Sunday. Normalise so matching has one Sunday.
        if 7 in out:
            out.discard(7)
            out.add(0)
    if not out:
        raise CronError(f"{spec.name}: {raw!r} selects nothing")
    return Field(spec=spec, raw=raw, values=tuple(sorted(out)), star=star)


@dataclass
class Cron:
    expr: str
    fields: Tuple[Field, ...]
    macro: Optional[str] = None
    extra_fields: int = 0  # 6th/7th field seen and dropped (seconds or year)

    @property
    def minute(self) -> Field:
        return self.fields[0]

    @property
    def hour(self) -> Field:
        return self.fields[1]

    @property
    def dom(self) -> Field:
        return self.fields[2]

    @property
    def month(self) -> Field:
        return self.fields[3]

    @property
    def dow(self) -> Field:
        return self.fields[4]

    @property
    def union_day_rule(self) -> bool:
        """True when day-of-month OR day-of-week applies (neither is starred)."""
        return not (self.dom.star or self.dow.star)

    @property
    def is_interval(self) -> bool:
        """Vixie's split: a job is 'interval' if minute or hour begins with '*'.

        Interval jobs follow the wall clock through a DST change. Fixed-time
        jobs get compensated. The two behaviours differ, so the class matters.
        """
        return self.minute.star or self.hour.star


def parse(expr: str) -> Cron:
    text = expr.strip()
    if not text:
        raise CronError("empty expression")
    macro = None
    if text.startswith("@"):
        key = text.lower()
        if key == "@reboot":
            raise CronError("@reboot has no schedule - it fires once at boot")
        if key not in MACROS:
            raise CronError(f"unknown macro {text!r}")
        macro, text = key, MACROS[key]
    parts = text.split()
    extra = 0
    if len(parts) > 5:
        # 6 fields is either Quartz/systemd seconds (leading) or a trailing
        # year. We keep the standard five and report the drop as a finding.
        extra = len(parts) - 5
        parts = parts[:5] if _looks_like_year(parts[-1]) else parts[-5:]
    if len(parts) != 5:
        raise CronError(f"expected 5 fields, got {len(parts)}")
    fields = tuple(parse_field(p, s) for p, s in zip(parts, SPECS))
    return Cron(expr=expr.strip(), fields=fields, macro=macro, extra_fields=extra)


def _looks_like_year(tok: str) -> bool:
    return bool(re.fullmatch(r"(19|20)\d\d(-(19|20)\d\d)?(/\d+)?", tok))


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def match(c: Cron, dt: datetime) -> bool:
    """Does this naive local wall-clock minute match?

    The day clause is the whole point:

        if day-of-month is '*' or day-of-week is '*':  DOM and DOW
        else:                                          DOM or  DOW
    """
    if dt.minute not in c.minute or dt.hour not in c.hour:
        return False
    if dt.month not in c.month:
        return False
    dom_hit = dt.day in c.dom
    dow_hit = (dt.weekday() + 1) % 7 in c.dow  # Mon=0 -> cron Sun=0
    if c.union_day_rule:
        return dom_hit or dow_hit
    return dom_hit and dow_hit


def _next_naive(c: Cron, start: datetime) -> Optional[datetime]:
    """Field-jumping search for the next matching wall-clock minute.

    Deliberately a different algorithm from the minute-by-minute scan in
    `brute_naive`, so the two can be diffed against each other in the tests.
    """
    dt = (start + timedelta(minutes=1)).replace(second=0, microsecond=0)
    limit = start + HORIZON
    while dt <= limit:
        if dt.month not in c.month:
            # jump to the first minute of the next month
            y, m = (dt.year + 1, 1) if dt.month == 12 else (dt.year, dt.month + 1)
            dt = datetime(y, m, 1)
            continue
        dom_hit = dt.day in c.dom
        dow_hit = (dt.weekday() + 1) % 7 in c.dow
        day_ok = (dom_hit or dow_hit) if c.union_day_rule else (dom_hit and dow_hit)
        if not day_ok:
            dt = (dt + timedelta(days=1)).replace(hour=0, minute=0)
            continue
        if dt.hour not in c.hour:
            nxt = _next_in(c.hour.values, dt.hour)
            if nxt is None:
                dt = (dt + timedelta(days=1)).replace(hour=0, minute=0)
            else:
                dt = dt.replace(hour=nxt, minute=0)
            continue
        if dt.minute not in c.minute:
            nxt = _next_in(c.minute.values, dt.minute)
            if nxt is None:
                dt = (dt + timedelta(hours=1)).replace(minute=0)
            else:
                dt = dt.replace(minute=nxt)
            continue
        return dt
    return None


def _next_in(values: Sequence[int], current: int) -> Optional[int]:
    for v in values:
        if v >= current:
            return v
    return None


def brute_naive(c: Cron, start: datetime, count: int) -> List[datetime]:
    """Scan every minute. Slow, obviously correct, used to check `_next_naive`."""
    out: List[datetime] = []
    dt = (start + timedelta(minutes=1)).replace(second=0, microsecond=0)
    limit = start + HORIZON
    while dt <= limit and len(out) < count:
        if match(c, dt):
            out.append(dt)
            limit = dt + HORIZON  # same rolling window as the jumping search
        dt += timedelta(minutes=1)
    return out


def next_naive(c: Cron, start: datetime, count: int) -> List[datetime]:
    out: List[datetime] = []
    cur = start
    for _ in range(count):
        nxt = _next_naive(c, cur)
        if nxt is None:
            break
        out.append(nxt)
        cur = nxt
    return out


# --------------------------------------------------------------------------
# Resolving wall-clock minutes onto a real time line
# --------------------------------------------------------------------------

NORMAL = "normal"
SKIPPED = "skipped"  # local time does not exist (spring forward)
REPEATED = "repeated"  # local time happens twice (fall back)


@dataclass
class Fire:
    local: datetime  # wall clock the crontab line names
    instant: Optional[datetime]  # actual UTC instant, None if the job is dropped
    kind: str = NORMAL
    note: str = ""

    @property
    def offset_hours(self) -> Optional[float]:
        if self.instant is None or self.local.tzinfo is None:
            return None
        return (self.local.replace(tzinfo=None) - self.instant.replace(tzinfo=None)).total_seconds() / 3600.0


def classify_local(naive: datetime, tz) -> str:
    """SKIPPED / REPEATED / NORMAL for a naive local time in `tz`.

    Order matters. Under PEP 495 the two folds carry different offsets for a
    *skipped* time as well as a repeated one, so the offset test alone calls
    every gap a fold. The round trip is what separates them: a time that does
    not exist comes back as a different wall clock, a time that happens twice
    comes back as itself. So test existence first.
    """
    a = naive.replace(tzinfo=tz, fold=0)
    b = naive.replace(tzinfo=tz, fold=1)
    if a.astimezone(UTC).astimezone(tz).replace(tzinfo=None) != naive:
        return SKIPPED
    if a.utcoffset() != b.utcoffset():
        return REPEATED
    return NORMAL


def _transition_instant(naive: datetime, tz) -> datetime:
    """The UTC instant at which the clock jumped over `naive`.

    Vixie runs a skipped fixed-time job immediately after the jump, so this is
    the instant it actually runs. Found by bisecting the offset change.
    """
    lo = (naive - timedelta(days=1)).replace(tzinfo=tz, fold=0).astimezone(UTC)
    hi = (naive + timedelta(days=1)).replace(tzinfo=tz, fold=0).astimezone(UTC)
    off_lo = lo.astimezone(tz).utcoffset()
    for _ in range(64):
        mid = lo + (hi - lo) / 2
        if mid.astimezone(tz).utcoffset() == off_lo:
            lo = mid
        else:
            hi = mid
        if (hi - lo) <= timedelta(seconds=1):
            break
    return hi.replace(microsecond=0)


def fires(
    c: Cron,
    start: datetime,
    count: int,
    tz_name: str = "UTC",
    utc_scheduler: bool = False,
) -> List[Fire]:
    """Next `count` fire events, resolved against a real zone.

    `utc_scheduler=True` models GitHub Actions / EventBridge / Kubernetes
    CronJob, which read the expression in UTC. Those never see a DST case at
    all - which is precisely why the local wall time they land on moves.
    """
    tz = _zone(tz_name)
    if utc_scheduler:
        naive = next_naive(c, start, count)
        return [Fire(local=n.replace(tzinfo=UTC), instant=n.replace(tzinfo=UTC)) for n in naive]

    out: List[Fire] = []
    cur = start
    guard = 0
    while len(out) < count and guard < count * 8 + 64:
        guard += 1
        nxt = _next_naive(c, cur)
        if nxt is None:
            break
        cur = nxt
        kind = classify_local(nxt, tz)
        if kind == NORMAL:
            aware = nxt.replace(tzinfo=tz)
            out.append(Fire(local=aware, instant=aware.astimezone(UTC)))
        elif kind == REPEATED:
            first = nxt.replace(tzinfo=tz, fold=0)
            second = nxt.replace(tzinfo=tz, fold=1)
            if c.is_interval:
                # The wall clock genuinely shows this minute twice.
                out.append(
                    Fire(first, first.astimezone(UTC), REPEATED,
                         "interval job: the clock reaches this minute twice, so it runs twice")
                )
                if len(out) < count:
                    out.append(Fire(second, second.astimezone(UTC), REPEATED,
                                    "second pass, one hour later in real time"))
            else:
                out.append(
                    Fire(first, first.astimezone(UTC), REPEATED,
                         "fixed-time job: Vixie runs it once; a naive scheduler runs it twice")
                )
        else:  # SKIPPED
            if c.is_interval:
                out.append(
                    Fire(nxt.replace(tzinfo=tz), None, SKIPPED,
                         "interval job: this wall-clock minute never happens, so it is skipped")
                )
            else:
                inst = _transition_instant(nxt, tz)
                out.append(
                    Fire(nxt.replace(tzinfo=tz), inst, SKIPPED,
                         "fixed-time job: Vixie runs it once at the jump, "
                         f"at {inst.astimezone(tz):%H:%M} local")
                )
    return out[:count]


def _zone(name: str):
    if name.upper() == "UTC" or ZoneInfo is None:
        return UTC
    return ZoneInfo(name)


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    detail: str = ""


def _fmt_list(values: Sequence[int], labels: Optional[Sequence[str]] = None) -> str:
    if labels:
        items = [labels[v] for v in values]
    else:
        items = [str(v) for v in values]
    if len(items) == 1:
        return items[0]
    if len(items) <= 4:
        return ", ".join(items[:-1]) + " and " + items[-1]
    return f"{items[0]}, {items[1]}, ... {items[-1]} ({len(items)} values)"


def step_gaps(f: Field) -> Optional[Tuple[int, int, int]]:
    """(usual step, gap at wrap, last value) when a step does not tile the field.

    `*/7` on minutes gives 0,7,...,56 and then waits 4 minutes, not 7. The
    field wraps but the step does not, so the last interval of every cycle is
    short. Returns None when the step divides the range evenly.
    """
    if "/" not in f.raw or not f.star:
        return None
    vals = f.values
    if len(vals) < 2:
        return None
    steps = {b - a for a, b in zip(vals, vals[1:])}
    if len(steps) != 1:
        return None
    step = steps.pop()
    span = f.spec.hi - f.spec.lo + 1
    wrap = span - (vals[-1] - vals[0])
    if wrap == step:
        return None
    return step, wrap, vals[-1]


def audit(
    c: Cron,
    tz_name: str = "UTC",
    start: Optional[datetime] = None,
    horizon_days: int = 400,
) -> List[Finding]:
    """Every finding is checked against real fire times, not against the text."""
    out: List[Finding] = []
    start = start or datetime(datetime.now().year, 1, 1)

    # 1. The union day rule.
    if c.union_day_rule:
        days = _matching_days(c, start, horizon_days)
        both = _matching_days(c, start, horizon_days, force_intersection=True)
        out.append(
            Finding(
                "DOM_DOW_UNION",
                MISREAD,
                "day-of-month and day-of-week are both restricted, so cron takes "
                "the union: it fires on either, not on both",
                f"over {horizon_days} days this fires on {len(days)} days; read as "
                f"'and' it would be {len(both)}",
            )
        )

    # 2. Steps that do not tile their field.
    for f in (c.minute, c.hour):
        g = step_gaps(f)
        if g:
            step, wrap, last = g
            out.append(
                Finding(
                    "STEP_WRAP_GAP",
                    TIMING,
                    f"{f.spec.name} step /{step} does not divide {f.spec.hi + 1}, so the "
                    f"interval after {f.spec.name} {last} is {wrap}, not {step}",
                    f"values: {_fmt_list(f.values)}",
                )
            )

    # 3. Impossible calendar dates.
    never = _impossible_dates(c)
    if never:
        out.append(
            Finding(
                "NEVER_FIRES",
                MISREAD,
                "some day/month combinations in this expression never occur",
                "; ".join(never),
            )
        )

    # 4/5. DST, checked by generating a year of fire times in the zone.
    if tz_name.upper() != "UTC":
        tz = _zone(tz_name)
        skipped, repeated = _dst_hits(c, tz, start, horizon_days)
        if skipped:
            out.append(
                Finding(
                    "DST_SKIPPED",
                    TIMING,
                    f"{len(skipped)} scheduled wall-clock time(s) do not exist in "
                    f"{tz_name} - the clock jumps over them",
                    "; ".join(f"{d:%Y-%m-%d %H:%M}" for d in skipped[:3]),
                )
            )
        if repeated:
            out.append(
                Finding(
                    "DST_REPEATED",
                    TIMING,
                    f"{len(repeated)} scheduled wall-clock time(s) happen twice in "
                    f"{tz_name} - whether the job runs once or twice is the "
                    "scheduler's choice, not yours",
                    "; ".join(f"{d:%Y-%m-%d %H:%M}" for d in repeated[:3]),
                )
            )
        if not c.hour.star:
            out.append(
                Finding(
                    "UTC_DRIFT",
                    PORTABILITY,
                    "GitHub Actions, EventBridge and Kubernetes CronJob read this "
                    f"in UTC, so the {tz_name} wall-clock time moves by an hour "
                    "across the year",
                    _drift_detail(c, tz, start),
                )
            )

    # 6. Dialect hazards.
    if any(tok in c.dow.raw for tok in ("7", "0")) or re.search(
        r"[a-z]", c.dow.raw, re.I
    ):
        out.append(
            Finding(
                "DOW_DIALECT",
                PORTABILITY,
                "day-of-week numbering is not the same everywhere: Unix cron is "
                "0-6 with 0=Sunday (7 also Sunday), Quartz is 1-7 with 1=Sunday",
                f"{c.dow.raw!r} selects {_fmt_list(c.dow.values, DOW_LABEL)} here; "
                f"under Quartz numbering the same digits shift by one day",
            )
        )
    if c.extra_fields:
        out.append(
            Finding(
                "FIELD_COUNT",
                PORTABILITY,
                f"{5 + c.extra_fields} fields given; standard cron takes 5",
                "a 6th field is seconds in Quartz/systemd and a year in others - "
                "the same string schedules different things",
            )
        )
    return out


def _matching_days(
    c: Cron, start: datetime, horizon_days: int, force_intersection: bool = False
) -> List[datetime]:
    out = []
    for i in range(horizon_days):
        d = start + timedelta(days=i)
        if d.month not in c.month:
            continue
        dom_hit = d.day in c.dom
        dow_hit = (d.weekday() + 1) % 7 in c.dow
        if force_intersection:
            ok = dom_hit and dow_hit
        else:
            ok = (dom_hit or dow_hit) if c.union_day_rule else (dom_hit and dow_hit)
        if ok:
            out.append(d)
    return out


def _impossible_dates(c: Cron) -> List[str]:
    out = []
    for m in c.month.values:
        longest = 29 if m == 2 else calendar.monthrange(2023, m)[1]
        bad = [d for d in c.dom.values if d > longest]
        if bad and not c.union_day_rule and len(bad) == len(c.dom.values):
            out.append(f"{MONTH_LABEL[m]} has no day {_fmt_list(bad)}")
        elif m == 2 and 29 in c.dom.values and len(c.dom.values) == 1 and not c.union_day_rule:
            out.append("29 February exists only in a leap year")
    return out


def _dst_hits(c: Cron, tz, start: datetime, horizon_days: int):
    skipped, repeated = [], []
    seen: Set[datetime] = set()
    cur = start
    end = start + timedelta(days=horizon_days)
    while cur < end:
        nxt = _next_naive(c, cur)
        if nxt is None or nxt >= end:
            break
        cur = nxt
        if nxt in seen:
            continue
        seen.add(nxt)
        k = classify_local(nxt, tz)
        if k == SKIPPED:
            skipped.append(nxt)
        elif k == REPEATED:
            repeated.append(nxt)
    return skipped, repeated


def _drift_detail(c: Cron, tz, start: datetime) -> str:
    hour = c.hour.values[0]
    winter = datetime(start.year, 1, 15, hour).replace(tzinfo=UTC).astimezone(tz)
    summer = datetime(start.year, 7, 15, hour).replace(tzinfo=UTC).astimezone(tz)
    if winter.hour == summer.hour:
        return f"this zone holds {winter:%H:%M} all year"
    return (
        f"UTC {hour:02d}:00 is {winter:%H:%M} local in January and "
        f"{summer:%H:%M} local in July"
    )


# --------------------------------------------------------------------------
# English
# --------------------------------------------------------------------------


def describe(c: Cron) -> str:
    """A sentence that says what cron does, including the union rule."""
    parts = []
    m, h = c.minute, c.hour
    if m.star and h.star:
        parts.append("Every minute")
    elif m.star:
        parts.append(f"Every minute of hour {_fmt_list(h.values)}")
    elif h.star:
        if len(m.values) == 1:
            parts.append(f"At minute {m.values[0]} of every hour")
        else:
            parts.append(f"At minutes {_fmt_list(m.values)} of every hour")
    else:
        times = [f"{hh:02d}:{mm:02d}" for hh in h.values for mm in m.values]
        shown = ", ".join(times[:4])
        if len(times) > 4:
            shown += f" ... ({len(times)} times a day)"
        parts.append(f"At {shown}")

    dom_all = len(c.dom.values) == 31
    dow_all = len(c.dow.values) == 7
    if c.union_day_rule:
        parts.append(
            f"on day-of-month {_fmt_list(c.dom.values)} **or** on "
            f"{_fmt_list(c.dow.values, DOW_LABEL)} (whichever comes first - cron "
            f"takes the union when both are restricted)"
        )
    elif not dom_all:
        parts.append(f"on day-of-month {_fmt_list(c.dom.values)}")
    elif not dow_all:
        parts.append(f"on {_fmt_list(c.dow.values, DOW_LABEL)}")
    else:
        parts.append("every day")

    if len(c.month.values) != 12:
        parts.append(f"in {_fmt_list(c.month.values, MONTH_LABEL)}")
    return ", ".join(parts) + "."
