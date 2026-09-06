"""Normalising local timestamps to UTC, with the ambiguous cases named.

A local wall-clock timestamp is not a point in time. Twice a year it is either
two points in time or none, and for the rest of the year it is one point only if
you also know the zone - not the offset, the *zone*, because an offset cannot
tell you what the clock will do next month.

`datetime` with `zoneinfo` never raises on either case. PEP 495 gave every aware
datetime a `fold` attribute and made both readings representable, which is
correct and complete, and also means the failure is silent by construction: you
get an answer for 01:30 on the fall-back night whether or not you meant the
first one.

This module resolves each timestamp explicitly, records *which* case it hit, and
refuses to guess unless told which way to guess.

Standard library only - `zoneinfo`, no pytz, no pandas.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from dataclasses import field as _dcfield
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

__all__ = [
    "AmbiguousTime",
    "NonExistentTime",
    "UnknownZone",
    "Reading",
    "classify",
    "resolve",
    "normalize",
    "tzdata_version",
    "same_rules",
    "find_alias_groups",
    "etc_zone_is_inverted",
    "local_day",
    "utc_day",
    "audit",
    "AuditReport",
    "build_session_log",
    "ground_truth",
    "SESSION_COLUMNS",
]

UTC = dt.timezone.utc


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class AmbiguousTime(ValueError):
    """The wall clock read this twice. Which one did you mean?"""


class NonExistentTime(ValueError):
    """The wall clock never read this. It was skipped by a forward transition."""


class UnknownZone(ValueError):
    """The zone identifier is not in this machine's tz database."""


# --------------------------------------------------------------------------
# the two hard cases
# --------------------------------------------------------------------------


def classify(naive: dt.datetime, zone: str) -> str:
    """``ok``, ``ambiguous`` or ``nonexistent`` for a wall-clock time in a zone.

    The test is a round trip, which is the only definition that does not depend
    on knowing the transition table. Convert the wall time to UTC and back: an
    ordinary time returns itself, a skipped time cannot and comes back as some
    other wall time, and an ambiguous time returns itself under both folds while
    the two folds carry different offsets.
    """
    tz = _zone(zone)
    aware0 = naive.replace(tzinfo=tz, fold=0)
    aware1 = naive.replace(tzinfo=tz, fold=1)
    round_trip = aware0.astimezone(UTC).astimezone(tz).replace(tzinfo=None, fold=0)
    if round_trip != naive.replace(fold=0):
        return "nonexistent"
    if aware0.utcoffset() != aware1.utcoffset():
        return "ambiguous"
    return "ok"


def _zone(zone: str) -> ZoneInfo:
    try:
        return ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise UnknownZone(f"{zone!r} is not in this machine's tz database ({exc})") from None


# --------------------------------------------------------------------------
# a single reading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """One timestamp, resolved or explicitly not."""

    raw: str
    zone: str
    status: str
    utc: Optional[dt.datetime] = None
    local: Optional[dt.datetime] = None
    offset: Optional[dt.timedelta] = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.utc is not None

    def offset_hours(self) -> Optional[float]:
        return None if self.offset is None else self.offset.total_seconds() / 3600


def _parse(raw: str) -> Optional[dt.datetime]:
    """Parse a wall-clock string. Anything carrying its own offset is returned aware."""
    text = raw.strip().replace(" ", "T", 1) if " " in raw.strip() else raw.strip()
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M"):
            try:
                return dt.datetime.strptime(raw.strip(), fmt)
            except ValueError:
                continue
    return None


def resolve(
    raw: str,
    zone: str,
    ambiguous: str = "raise",
    nonexistent: str = "raise",
) -> Reading:
    """Resolve one local timestamp to UTC.

    ``ambiguous``   - ``raise`` | ``earlier`` (fold=0) | ``later`` (fold=1) | ``flag``
    ``nonexistent`` - ``raise`` | ``shift_forward`` | ``flag``

    The defaults raise, because in both cases the input does not determine the
    answer and a default that quietly picks one is how a duration ends up
    negative in a table nobody re-reads.
    """
    parsed = _parse(raw)
    if parsed is None:
        return Reading(raw, zone, "unparsed", note="no recognised date format")

    if parsed.tzinfo is not None:
        # Already carries an offset. That pins the instant but not the zone - see
        # `Reading.note`, and experiment D.
        return Reading(
            raw,
            zone,
            "offset_only",
            utc=parsed.astimezone(UTC),
            local=parsed.replace(tzinfo=None),
            offset=parsed.utcoffset(),
            note=(
                "the string carried a fixed offset, so the instant is pinned but the "
                "zone is not - this value cannot answer 'same local time next month'"
            ),
        )

    tz = _zone(zone)
    kind = classify(parsed, zone)

    if kind == "ambiguous":
        if ambiguous == "raise":
            raise AmbiguousTime(
                f"{raw} occurs twice in {zone}: "
                f"{parsed.replace(tzinfo=tz, fold=0).astimezone(UTC)} and "
                f"{parsed.replace(tzinfo=tz, fold=1).astimezone(UTC)}"
            )
        if ambiguous == "flag":
            return Reading(raw, zone, "ambiguous", note="occurs twice; unresolved by request")
        fold = 0 if ambiguous == "earlier" else 1
        aware = parsed.replace(tzinfo=tz, fold=fold)
        return Reading(
            raw,
            zone,
            "ambiguous",
            utc=aware.astimezone(UTC),
            local=parsed,
            offset=aware.utcoffset(),
            note=f"occurs twice; took the {'earlier' if fold == 0 else 'later'} reading",
        )

    if kind == "nonexistent":
        if nonexistent == "raise":
            raise NonExistentTime(f"{raw} does not exist in {zone}: skipped by a forward transition")
        if nonexistent == "flag":
            return Reading(raw, zone, "nonexistent", note="clock skipped this time; unresolved")
        # shift_forward: land on the first instant that does exist
        aware = parsed.replace(tzinfo=tz, fold=1)
        shifted = aware.astimezone(UTC)
        return Reading(
            raw,
            zone,
            "nonexistent",
            utc=shifted,
            local=shifted.astimezone(tz).replace(tzinfo=None),
            offset=shifted.astimezone(tz).utcoffset(),
            note="clock skipped this time; shifted forward past the gap",
        )

    aware = parsed.replace(tzinfo=tz)
    return Reading(raw, zone, "ok", utc=aware.astimezone(UTC), local=parsed, offset=aware.utcoffset())


def normalize(
    rows: Sequence[Dict[str, Any]],
    ts_key: str = "local_ts",
    zone_key: str = "zone",
    ambiguous: str = "flag",
    nonexistent: str = "flag",
) -> List[Reading]:
    """Resolve a column of local timestamps. Defaults to flagging, not guessing."""
    out: List[Reading] = []
    for row in rows:
        try:
            out.append(resolve(row[ts_key], row[zone_key], ambiguous, nonexistent))
        except UnknownZone as exc:
            out.append(Reading(row[ts_key], row[zone_key], "unknown_zone", note=str(exc)))
    return out


# --------------------------------------------------------------------------
# the tz database itself
# --------------------------------------------------------------------------


def tzdata_version() -> str:
    """Which rules produced these answers.

    Time zone rules are political and change several times a year. Two runs of
    the same code against the same input give different UTC values if the tz
    database moved underneath them, so the version belongs in the output next to
    the numbers, not in a comment.
    """
    try:
        import tzdata  # noqa: PLC0415 - optional, present only when pip-installed

        return f"tzdata {tzdata.__version__} (Python package)"
    except Exception:  # noqa: BLE001 - any failure means we are on the system db
        pass
    for path in ("/usr/share/zoneinfo/+VERSION", "/usr/lib/zoneinfo/+VERSION"):
        try:
            with open(path) as fh:
                return f"tzdata {fh.read().strip()} (system)"
        except OSError:
            continue
    return "tzdata version unknown"


def same_rules(a: str, b: str, start_year: int = 1990, end_year: int = 2035) -> bool:
    """Do two zone identifiers behave identically over a long window?

    zoneinfo does not expose the tz database's Link records, so identity is
    established behaviourally: sample the offset every six hours across decades
    and compare. Slower than reading a link table and correct without one.
    """
    za, zb = _zone(a), _zone(b)
    cursor = dt.datetime(start_year, 1, 1)
    step = dt.timedelta(hours=6)
    end = dt.datetime(end_year, 1, 1)
    while cursor < end:
        if cursor.replace(tzinfo=za).utcoffset() != cursor.replace(tzinfo=zb).utcoffset():
            return False
        cursor += step
    return True


def find_alias_groups(zones: Iterable[str]) -> List[List[str]]:
    """Group the zone identifiers present in the data by actual behaviour.

    Any group with more than one member is a set of names for the same place -
    a `GROUP BY zone` splits that place's rows across several buckets while every
    per-row conversion stays perfectly correct.
    """
    unique = sorted(set(zones))
    groups: List[List[str]] = []
    for z in unique:
        try:
            _zone(z)
        except UnknownZone:
            continue
        for g in groups:
            if same_rules(g[0], z):
                g.append(z)
                break
        else:
            groups.append([z])
    return [g for g in groups if len(g) > 1]


def etc_zone_is_inverted(zone: str) -> Optional[str]:
    """`Etc/GMT+5` is UTC-05:00. The POSIX sign convention is the other way round.

    Returns a description when the identifier's sign does not mean what a reader
    of the name would assume, and None otherwise.
    """
    if not zone.startswith("Etc/GMT") or zone in ("Etc/GMT", "Etc/GMT0"):
        return None
    try:
        off = dt.datetime(2024, 7, 1, tzinfo=_zone(zone)).utcoffset()
    except UnknownZone:
        return None
    if off is None:
        return None
    hours = off.total_seconds() / 3600
    sign_in_name = "+" if "+" in zone else "-"
    actual = "+" if hours >= 0 else "-"
    if sign_in_name != actual:
        return f"{zone} is UTC{hours:+03.0f}:00 - the POSIX sign convention is inverted"
    return None


# --------------------------------------------------------------------------
# day bucketing
# --------------------------------------------------------------------------


def utc_day(reading: Reading) -> Optional[dt.date]:
    return None if reading.utc is None else reading.utc.date()


def local_day(reading: Reading) -> Optional[dt.date]:
    """The calendar day the event happened on *where it happened*.

    Not the same question as `utc_day`, and for anywhere east of London or west
    of Reykjavik it is frequently a different answer.
    """
    if reading.utc is None:
        return None
    try:
        return reading.utc.astimezone(_zone(reading.zone)).date()
    except UnknownZone:
        return None


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


@dataclass
class AuditReport:
    verdict: str
    findings: List[str]
    stats: Dict[str, Any] = _dcfield(default_factory=dict)

    def text(self) -> str:
        return "\n".join([self.verdict] + [f"  {f}" for f in self.findings])

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text()


def audit(
    rows: Sequence[Dict[str, Any]], ts_key: str = "local_ts", zone_key: str = "zone"
) -> AuditReport:
    """Everything about this column that a UTC conversion will not tell you.

    Runs before the conversion, and reports only failures that produce a
    plausible answer - a timestamp that cannot be parsed is not interesting here,
    because it announces itself.
    """
    findings: List[str] = []
    stats: Dict[str, Any] = {"tzdata": tzdata_version(), "rows": len(rows)}
    readings = normalize(rows, ts_key, zone_key, ambiguous="flag", nonexistent="flag")

    counts: Dict[str, int] = {}
    for r in readings:
        counts[r.status] = counts.get(r.status, 0) + 1
    stats["status_counts"] = counts

    if counts.get("ambiguous"):
        n = counts["ambiguous"]
        findings.append(
            f"WARNING {n} timestamp(s) occur twice in their zone - a clock going back. "
            f"Both readings are real instants an hour apart, the input does not say which, "
            f"and Python will not raise: `datetime` carries a `fold` flag that defaults to 0, "
            f"so the earlier reading is chosen for you"
        )
    if counts.get("nonexistent"):
        n = counts["nonexistent"]
        findings.append(
            f"WARNING {n} timestamp(s) never happened - a clock going forward skipped them. "
            f"zoneinfo still returns a UTC value, and converting it back gives a *different* "
            f"wall time from the one you started with"
        )
    if counts.get("offset_only"):
        n = counts["offset_only"]
        findings.append(
            f"note  {n} value(s) already carried a numeric offset. The instant is pinned, "
            f"the zone is not - an offset cannot answer 'what will this clock read in "
            f"December', because that depends on rules the offset does not carry"
        )
    if counts.get("unknown_zone"):
        findings.append(
            f"ERROR {counts['unknown_zone']} row(s) name a zone this machine's tz database "
            f"does not have"
        )
    if counts.get("unparsed"):
        findings.append(f"note  {counts['unparsed']} value(s) could not be parsed at all")

    zones = [str(r[zone_key]) for r in rows]
    groups = find_alias_groups(zones)
    stats["alias_groups"] = groups
    for g in groups:
        affected = sum(1 for z in zones if z in g)
        findings.append(
            f"WARNING {' and '.join(g)} are the same zone under different names "
            f"({affected} rows). Every conversion is correct; a GROUP BY on the zone column "
            f"splits one place into {len(g)} buckets"
        )

    for z in sorted(set(zones)):
        msg = etc_zone_is_inverted(z)
        if msg:
            findings.append(f"WARNING {msg}")

    offsets = {r.offset for r in readings if r.offset is not None}
    odd = sorted(
        {o for o in offsets if o.total_seconds() % 3600}, key=lambda o: o.total_seconds()
    )
    stats["sub_hour_offsets"] = [str(o) for o in odd]
    if odd:
        findings.append(
            f"note  {len(odd)} distinct offset(s) in this data are not a whole number of hours "
            f"({', '.join(str(o) for o in odd)}). Any bucketing that assumes hour boundaries "
            f"align across zones is wrong for these rows"
        )

    findings.append(f"note  resolved against {stats['tzdata']} - record this next to the results")

    n_bad = counts.get("ambiguous", 0) + counts.get("nonexistent", 0)
    if counts.get("unknown_zone"):
        verdict = "NOT SAFE TO CONVERT - unknown zone identifiers"
    elif n_bad:
        verdict = (
            f"CONVERTS, BUT {n_bad} OF {len(rows)} ROWS ARE UNDETERMINED - the input does not "
            f"say which instant it means"
        )
    elif any(f.startswith("WARNING") for f in findings):
        verdict = f"CONVERTS - every row resolves, but {sum(1 for f in findings if f.startswith('WARNING'))} structural warning(s)"
    else:
        verdict = f"CLEAN - {len(rows)} rows resolve to exactly one instant each"

    return AuditReport(verdict, findings, stats)


# --------------------------------------------------------------------------
# sample data
# --------------------------------------------------------------------------

SESSION_COLUMNS = ("session_id", "office", "zone", "local_ts", "event", "amount")

#: (session, office, zone, true UTC instant or None, event, amount)
#:
#: The sample is generated *from* true instants and the local wall-clock string is
#: rendered from them, with the offset dropped - which is exactly what a system
#: does when it writes `datetime.now()` in local time into a `TIMESTAMP` column.
#: That makes the ground truth intrinsic rather than asserted, so every experiment
#: below measures real recovery error instead of comparing two guesses.
#:
#: `true_utc=None` marks a row whose local timestamp was *derived* rather than
#: observed - computed as "start + 45 minutes" in local terms by an upstream job.
#: Those are the rows that land on wall-clock times which never existed.
_TRUTH: List[Tuple[str, str, str, Optional[str], str, float]] = [
    # New York, the night the clocks go back. 06:00Z is the transition.
    ("S-101", "New York", "America/New_York", "2024-11-03T04:45:00Z", "open", 0.0),
    ("S-101", "New York", "America/New_York", "2024-11-03T04:58:00Z", "close", 120.0),
    # opens before the transition, closes after it: 80 real minutes
    ("S-104", "New York", "America/New_York", "2024-11-03T05:30:00Z", "open", 0.0),
    ("S-104", "New York", "America/New_York", "2024-11-03T06:50:00Z", "close", 340.0),
    # same, and the wall clock runs backwards across it: 20 real minutes
    ("S-105", "New York", "America/New_York", "2024-11-03T05:50:00Z", "open", 0.0),
    ("S-105", "New York", "America/New_York", "2024-11-03T06:10:00Z", "close", 90.0),
    # New York, the night the clocks go forward. 07:00Z is the transition.
    ("S-110", "New York", "America/New_York", "2024-03-10T06:45:00Z", "open", 0.0),
    ("S-110", "New York", "America/New_York", None, "close", 210.0),  # derived: 02:30 never existed
    # London, clocks back
    ("S-201", "London", "Europe/London", "2024-10-27T00:30:00Z", "open", 0.0),
    ("S-201", "London", "Europe/London", "2024-10-27T03:05:00Z", "close", 480.0),
    # Singapore: no DST, but far enough east to change the calendar day
    ("S-301", "Singapore", "Asia/Singapore", "2024-11-03T00:15:00Z", "open", 0.0),
    ("S-301", "Singapore", "Asia/Singapore", "2024-11-03T01:05:00Z", "close", 1250.0),
    ("S-302", "Singapore", "Asia/Singapore", "2024-11-03T23:40:00Z", "open", 0.0),
    ("S-302", "Singapore", "Asia/Singapore", "2024-11-04T00:20:00Z", "close", 640.0),
    # Bengaluru, logged under two different names for the same zone
    ("S-401", "Bengaluru", "Asia/Calcutta", "2024-11-03T04:00:00Z", "open", 0.0),
    ("S-401", "Bengaluru", "Asia/Calcutta", "2024-11-03T04:45:00Z", "close", 410.0),
    ("S-402", "Bengaluru", "Asia/Kolkata", "2024-11-03T18:50:00Z", "open", 0.0),
    ("S-402", "Bengaluru", "Asia/Kolkata", "2024-11-03T19:35:00Z", "close", 380.0),
    # Kathmandu: +05:45
    ("S-501", "Kathmandu", "Asia/Kathmandu", "2024-11-03T08:20:00Z", "open", 0.0),
    ("S-501", "Kathmandu", "Asia/Kathmandu", "2024-11-03T18:40:00Z", "close", 220.0),
    # Lord Howe: a THIRTY minute DST shift, so the spring gap is half an hour wide
    ("S-601", "Lord Howe", "Australia/Lord_Howe", None, "open", 0.0),  # derived: 02:15 never existed
    ("S-601", "Lord Howe", "Australia/Lord_Howe", "2024-10-05T16:40:00Z", "close", 95.0),
    # a vendor feed labelled with a POSIX-style identifier
    ("S-701", "Vendor feed", "Etc/GMT+5", "2024-11-03T17:00:00Z", "open", 0.0),
    ("S-701", "Vendor feed", "Etc/GMT+5", "2024-11-03T17:45:00Z", "close", 700.0),
]

#: The same fall-back sessions from a partner API that transmits the offset.
#: Same instants, same wall-clock times, one extra field - and it is decidable.
_PARTNER = [("S-104", "America/New_York"), ("S-105", "America/New_York")]


def _render_local(utc_iso: str, zone: str, with_offset: bool = False) -> str:
    """Format a true instant as the local wall clock, dropping the offset.

    This is the lossy step the whole module is about, and it is one line of
    ordinary application code in every system that has ever done it.
    """
    inst = dt.datetime.fromisoformat(utc_iso.replace("Z", "+00:00")).astimezone(_zone(zone))
    if with_offset:
        return inst.isoformat(timespec="seconds")
    return inst.strftime("%Y-%m-%d %H:%M")


#: wall-clock strings for the two derived rows, which have no true instant
_DERIVED = {("S-110", "close"): "2024-03-10 02:30", ("S-601", "open"): "2024-10-06 02:15"}


def build_session_log(with_partner_feed: bool = False) -> List[Dict[str, Any]]:
    """Support sessions across six offices, straddling the 2024 transitions.

    Set ``with_partner_feed`` to append the same two contested New York sessions
    as they arrive from a feed that transmits the UTC offset alongside the wall
    clock - the control group for experiment C.
    """
    rows: List[Dict[str, Any]] = []
    for sid, office, zone, utc_iso, event, amount in _TRUTH:
        local = _DERIVED[(sid, event)] if utc_iso is None else _render_local(utc_iso, zone)
        rows.append(
            dict(zip(SESSION_COLUMNS, (sid, office, zone, local, event, amount)))
        )
    if with_partner_feed:
        for sid, zone in _PARTNER:
            for s, office, z, utc_iso, event, amount in _TRUTH:
                if s == sid and utc_iso is not None:
                    rows.append(
                        dict(
                            zip(
                                SESSION_COLUMNS,
                                (
                                    f"{sid}-api",
                                    "Partner API",
                                    zone,
                                    _render_local(utc_iso, zone, with_offset=True),
                                    event,
                                    amount,
                                ),
                            )
                        )
                    )
    return rows


def ground_truth() -> Dict[Tuple[str, str], Optional[dt.datetime]]:
    """(session_id, event) -> the instant the event actually happened, or None."""
    return {
        (sid, event): (
            None
            if utc_iso is None
            else dt.datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
        )
        for sid, _office, _zone, utc_iso, event, _amount in _TRUTH
    }
