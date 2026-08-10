"""Tests. The interesting ones are the two that could actually fail.

`test_jump_agrees_with_scan` runs the field-jumping search and the
minute-by-minute scan over 40 expressions and demands identical output. They
share the field sets and nothing else, so a calendar-rollover or union-rule bug
in one shows up as a diff.

`test_dst_instants_are_real` checks the DST claims against measured UTC offsets
rather than against the strings this module prints.

Run:  python3 -m pytest test_cron.py -q
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import cron as C

UTC = timezone.utc
TZ = "Europe/London"
NY = "America/New_York"

EXPRS = [
    "* * * * *",
    "0 * * * *",
    "30 2 * * *",
    "*/7 * * * *",
    "*/15 * * * *",
    "0 */5 * * *",
    "0 0 13 * 5",
    "0 0 * * 1-5",
    "0 9 1 * *",
    "15,45 8-18 * * mon-fri",
    "0 0 1 1 *",
    "0 0 29 2 *",
    "5 4 * * sun",
    "0 22 * * 1,3,5",
    "@daily",
    "@hourly",
    "@weekly",
    "0 0 31 * *",
    "*/10 9-17 * * 1-5",
    "59 23 28-31 * *",
]


def test_parse_round_trip():
    c = C.parse("15,45 8-18 * * mon-fri")
    assert c.minute.values == (15, 45)
    assert c.hour.values == tuple(range(8, 19))
    assert c.dow.values == (1, 2, 3, 4, 5)
    assert c.month.values == tuple(range(1, 13))


def test_seven_is_sunday():
    assert C.parse("0 0 * * 7").dow.values == (0,)
    assert C.parse("0 0 * * 0,7").dow.values == (0,)


def test_bad_input_raises_rather_than_guesses():
    for bad in ["", "* * * *", "60 * * * *", "0 0 * * 9", "0 0 * * mon-", "5-1 * * * *"]:
        with pytest.raises(C.CronError):
            C.parse(bad)


# ---------------------------------------------------------------- union rule


def test_friday_the_13th_is_not_friday_the_13th():
    """The headline claim, measured."""
    c = C.parse("0 0 13 * 5")
    assert c.union_day_rule
    start = datetime(2026, 1, 1)
    days = C._matching_days(c, start, 365)
    both = C._matching_days(c, start, 365, force_intersection=True)
    # A union of "the 13th" and "every Friday" over a year.
    assert len(days) > 60
    assert len(both) < 4
    assert len(days) != len(both)


def test_star_slash_still_counts_as_star():
    """Vixie sets its STAR flag on any field beginning with '*', including */2."""
    assert C.parse("0 0 */2 * 5").union_day_rule is False
    assert C.parse("0 0 1 * 5").union_day_rule is True


def test_union_off_when_either_field_is_star():
    assert C.parse("0 0 13 * *").union_day_rule is False
    assert C.parse("0 0 * * 5").union_day_rule is False


# ------------------------------------------------- the cross-check that bites


@pytest.mark.parametrize("expr", EXPRS)
def test_jump_agrees_with_scan(expr):
    c = C.parse(expr)
    start = datetime(2026, 2, 26, 23, 51)
    fast = C.next_naive(c, start, 25)
    slow = C.brute_naive(c, start, 25)
    assert fast == slow, f"{expr}: jump and scan disagree"


@pytest.mark.parametrize("expr", ["0 0 29 2 *", "0 0 31 * *", "59 23 28-31 * *"])
def test_jump_agrees_across_year_boundaries(expr):
    """Rare dates force the month/year rollover path."""
    c = C.parse(expr)
    start = datetime(2027, 12, 30, 12, 0)
    assert C.next_naive(c, start, 6) == C.brute_naive(c, start, 6)


# ------------------------------------------------------------------ DST


def test_spring_forward_time_does_not_exist():
    tz = C._zone(TZ)
    # London 2026-03-29: 01:00 -> 02:00. 01:30 never happens.
    assert C.classify_local(datetime(2026, 3, 29, 1, 30), tz) == C.SKIPPED
    assert C.classify_local(datetime(2026, 3, 29, 3, 30), tz) == C.NORMAL


def test_fall_back_time_happens_twice():
    tz = C._zone(TZ)
    # London 2026-10-25: 02:00 -> 01:00. 01:30 happens twice.
    assert C.classify_local(datetime(2026, 10, 25, 1, 30), tz) == C.REPEATED


def test_dst_instants_are_real():
    """Check the offsets, not the labels."""
    tz = C._zone(TZ)
    naive = datetime(2026, 10, 25, 1, 30)
    a = naive.replace(tzinfo=tz, fold=0).astimezone(UTC)
    b = naive.replace(tzinfo=tz, fold=1).astimezone(UTC)
    assert b - a == timedelta(hours=1)  # same wall clock, an hour apart


def test_fixed_time_job_skipped_by_spring_forward_still_runs():
    """Vixie runs it once at the jump. It must land on a real instant."""
    c = C.parse("30 1 * * *")
    out = C.fires(c, datetime(2026, 3, 28, 12, 0), 3, TZ)
    skipped = [f for f in out if f.kind == C.SKIPPED]
    assert skipped, "expected the 01:30 job to hit the gap"
    f = skipped[0]
    assert f.instant is not None
    assert f.instant.tzinfo is not None
    # It runs at the transition: 01:00 GMT == 02:00 BST.
    assert f.instant == datetime(2026, 3, 29, 1, 0, tzinfo=UTC)


def test_interval_job_skipped_by_spring_forward_is_dropped():
    c = C.parse("*/30 * * * *")
    out = C.fires(c, datetime(2026, 3, 29, 0, 45), 6, TZ)
    dropped = [f for f in out if f.kind == C.SKIPPED and f.instant is None]
    assert dropped, "an interval job cannot run at a wall-clock time that never occurs"


def test_interval_job_runs_twice_on_fall_back():
    c = C.parse("*/30 * * * *")
    out = C.fires(c, datetime(2026, 10, 25, 0, 45), 6, TZ)
    rep = [f for f in out if f.kind == C.REPEATED]
    assert len(rep) >= 2
    assert rep[1].instant - rep[0].instant == timedelta(hours=1)


def test_fixed_time_job_runs_once_on_fall_back():
    c = C.parse("30 1 * * *")
    out = C.fires(c, datetime(2026, 10, 24, 12, 0), 3, TZ)
    rep = [f for f in out if f.kind == C.REPEATED]
    assert len(rep) == 1


def test_utc_scheduler_never_sees_dst_but_drifts():
    """The same line on GitHub Actions: no DST case, and an hour of drift."""
    c = C.parse("0 9 * * *")
    tz = C._zone(TZ)
    jan = C.fires(c, datetime(2026, 1, 10), 1, TZ, utc_scheduler=True)[0]
    jul = C.fires(c, datetime(2026, 7, 10), 1, TZ, utc_scheduler=True)[0]
    assert jan.instant.hour == jul.instant.hour == 9  # fixed in UTC
    assert jan.instant.astimezone(tz).hour == 9
    assert jul.instant.astimezone(tz).hour == 10  # moved, locally


# ------------------------------------------------------------- step wrapping


def test_step_that_does_not_tile_its_field():
    f = C.parse("*/7 * * * *").minute
    step, wrap, last = C.step_gaps(f)
    assert (step, wrap, last) == (7, 4, 56)


def test_step_that_tiles_cleanly_has_no_finding():
    assert C.step_gaps(C.parse("*/15 * * * *").minute) is None
    assert C.step_gaps(C.parse("*/12 * * * *").minute) is None


def test_hour_step_gap_is_measured_on_real_fires():
    c = C.parse("0 */5 * * *")
    out = C.next_naive(c, datetime(2026, 5, 1, 0, 0), 6)
    gaps = [(b - a).total_seconds() / 3600 for a, b in zip(out, out[1:])]
    assert 4.0 in gaps, "0,5,10,15,20 then a 4-hour wait"


# ----------------------------------------------------------------- findings


def test_audit_flags_the_union():
    codes = {f.code for f in C.audit(C.parse("0 0 13 * 5"), start=datetime(2026, 1, 1))}
    assert "DOM_DOW_UNION" in codes


def test_audit_is_quiet_on_a_plain_expression():
    codes = {f.code for f in C.audit(C.parse("0 3 * * *"), start=datetime(2026, 1, 1))}
    assert codes == set()


def test_audit_flags_dst_only_when_a_zone_is_given():
    c = C.parse("30 1 * * *")
    assert not any(
        f.code.startswith("DST") for f in C.audit(c, "UTC", datetime(2026, 1, 1))
    )
    codes = {f.code for f in C.audit(c, TZ, datetime(2026, 1, 1))}
    assert {"DST_SKIPPED", "DST_REPEATED"} <= codes


def test_audit_flags_impossible_date():
    codes = {f.code for f in C.audit(C.parse("0 0 31 2 *"), start=datetime(2026, 1, 1))}
    assert "NEVER_FIRES" in codes


def test_describe_says_or_not_and():
    text = C.describe(C.parse("0 0 13 * 5"))
    assert "**or**" in text
    assert "union" in text


def test_reboot_is_refused_rather_than_faked():
    with pytest.raises(C.CronError):
        C.parse("@reboot")
