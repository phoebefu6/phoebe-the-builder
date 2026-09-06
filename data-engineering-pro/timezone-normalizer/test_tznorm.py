"""Tests for tznorm.py. Plain asserts: ``python3 test_tznorm.py``.

Note on determinism: these assert *structural* facts (this time is ambiguous,
these two identifiers agree, this round trip fails) rather than specific UTC
values for arbitrary dates, because the tz database is political and a test that
hard-codes a 2035 offset is a test that will fail for a correct reason. The dates
that are pinned are historical transitions, which do not move.
"""

from __future__ import annotations

import datetime as dt
import sys
import traceback
from typing import Callable, List, Tuple
from zoneinfo import ZoneInfo

from tznorm import (
    UTC,
    AmbiguousTime,
    NonExistentTime,
    UnknownZone,
    audit,
    build_session_log,
    classify,
    etc_zone_is_inverted,
    find_alias_groups,
    ground_truth,
    local_day,
    normalize,
    resolve,
    same_rules,
    tzdata_version,
    utc_day,
)

_TESTS: List[Tuple[str, Callable[[], None]]] = []

NY = "America/New_York"
LHI = "Australia/Lord_Howe"


def test(fn: Callable[[], None]) -> Callable[[], None]:
    _TESTS.append((fn.__name__, fn))
    return fn


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------


@test
def ordinary_time_is_ok() -> None:
    assert classify(dt.datetime(2024, 6, 15, 12, 0), NY) == "ok"
    assert classify(dt.datetime(2024, 11, 3, 0, 45), NY) == "ok"
    assert classify(dt.datetime(2024, 11, 3, 3, 0), NY) == "ok"


@test
def fall_back_hour_is_ambiguous() -> None:
    for minute in (0, 1, 30, 59):
        assert classify(dt.datetime(2024, 11, 3, 1, minute), NY) == "ambiguous", minute
    # the minute either side is not
    assert classify(dt.datetime(2024, 11, 3, 0, 59), NY) == "ok"
    assert classify(dt.datetime(2024, 11, 3, 2, 0), NY) == "ok"


@test
def spring_forward_hour_does_not_exist() -> None:
    for minute in (0, 30, 59):
        assert classify(dt.datetime(2024, 3, 10, 2, minute), NY) == "nonexistent", minute
    assert classify(dt.datetime(2024, 3, 10, 1, 59), NY) == "ok"
    assert classify(dt.datetime(2024, 3, 10, 3, 0), NY) == "ok"


@test
def lord_howe_gap_is_only_thirty_minutes_wide() -> None:
    """The whole reason to test on a zone that is not America/New_York."""
    assert classify(dt.datetime(2024, 10, 6, 2, 15), LHI) == "nonexistent"
    assert classify(dt.datetime(2024, 10, 6, 2, 29), LHI) == "nonexistent"
    assert classify(dt.datetime(2024, 10, 6, 2, 30), LHI) == "ok"  # gap is half an hour
    assert classify(dt.datetime(2024, 10, 6, 1, 59), LHI) == "ok"


@test
def a_zone_without_dst_is_never_ambiguous() -> None:
    for month in range(1, 13):
        for hour in (0, 1, 2, 3):
            assert classify(dt.datetime(2024, month, 15, hour, 30), "Asia/Singapore") == "ok"


@test
def unknown_zone_is_named() -> None:
    err = raises(UnknownZone, classify, dt.datetime(2024, 1, 1), "Mars/Olympus_Mons")
    assert "Mars/Olympus_Mons" in str(err)


# --------------------------------------------------------------------------
# resolve
# --------------------------------------------------------------------------


@test
def default_policy_refuses_to_guess() -> None:
    raises(AmbiguousTime, resolve, "2024-11-03 01:30", NY)
    raises(NonExistentTime, resolve, "2024-03-10 02:30", NY)


@test
def the_two_folds_are_exactly_an_hour_apart_in_new_york() -> None:
    early = resolve("2024-11-03 01:30", NY, ambiguous="earlier")
    late = resolve("2024-11-03 01:30", NY, ambiguous="later")
    assert late.utc - early.utc == dt.timedelta(hours=1)
    assert early.offset == dt.timedelta(hours=-4)
    assert late.offset == dt.timedelta(hours=-5)


@test
def the_two_folds_are_half_an_hour_apart_on_lord_howe() -> None:
    early = resolve("2024-04-07 01:45", LHI, ambiguous="earlier")
    late = resolve("2024-04-07 01:45", LHI, ambiguous="later")
    assert late.utc - early.utc == dt.timedelta(minutes=30)


@test
def flag_policy_returns_no_instant() -> None:
    r = resolve("2024-11-03 01:30", NY, ambiguous="flag")
    assert r.status == "ambiguous" and r.utc is None and not r.ok


@test
def shift_forward_lands_outside_the_gap() -> None:
    r = resolve("2024-03-10 02:30", NY, nonexistent="shift_forward")
    assert r.utc is not None
    assert classify(r.local, NY) == "ok"
    assert r.local != dt.datetime(2024, 3, 10, 2, 30)


@test
def a_string_with_an_offset_is_pinned_and_labelled() -> None:
    r = resolve("2024-11-03T01:30:00-04:00", NY)
    assert r.status == "offset_only"
    assert r.utc == dt.datetime(2024, 11, 3, 5, 30, tzinfo=UTC)
    assert "zone is not" in r.note


@test
def the_same_wall_time_with_two_offsets_is_two_instants() -> None:
    a = resolve("2024-11-03T01:30:00-04:00", NY)
    b = resolve("2024-11-03T01:30:00-05:00", NY)
    assert b.utc - a.utc == dt.timedelta(hours=1)


@test
def unparseable_input_is_reported_not_guessed() -> None:
    r = resolve("last tuesday", NY)
    assert r.status == "unparsed" and r.utc is None


@test
def several_wall_clock_formats_parse() -> None:
    for raw in ("2024-06-15 12:00", "2024-06-15T12:00", "2024-06-15T12:00:00"):
        assert resolve(raw, NY).utc == dt.datetime(2024, 6, 15, 16, 0, tzinfo=UTC), raw


@test
def offset_hours_is_a_float_not_an_int() -> None:
    r = resolve("2024-11-03 14:05", "Asia/Kathmandu")
    assert r.offset_hours() == 5.75


# --------------------------------------------------------------------------
# the round trip that defines the whole thing
# --------------------------------------------------------------------------


@test
def ordinary_times_round_trip_and_skipped_times_do_not() -> None:
    tz = ZoneInfo(NY)
    for naive, expect in (
        (dt.datetime(2024, 6, 1, 12, 0), True),
        (dt.datetime(2024, 11, 3, 1, 30), True),  # ambiguous, but it does round trip
        (dt.datetime(2024, 3, 10, 2, 30), False),
    ):
        back = naive.replace(tzinfo=tz).astimezone(UTC).astimezone(tz).replace(tzinfo=None)
        assert (back == naive) is expect, naive


# --------------------------------------------------------------------------
# identifiers
# --------------------------------------------------------------------------


@test
def known_links_agree() -> None:
    for a, b in (
        ("Asia/Kolkata", "Asia/Calcutta"),
        ("Europe/Kyiv", "Europe/Kiev"),
        ("America/Nuuk", "America/Godthab"),
        ("US/Eastern", "America/New_York"),
    ):
        assert same_rules(a, b), (a, b)


@test
def different_zones_do_not_agree() -> None:
    assert not same_rules("America/New_York", "America/Chicago")
    assert not same_rules("Asia/Kolkata", "Asia/Kathmandu")
    # same offset today, different rules: London keeps DST, Reykjavik does not
    assert not same_rules("Europe/London", "Atlantic/Reykjavik")


@test
def alias_groups_only_contain_real_duplicates() -> None:
    groups = find_alias_groups([r["zone"] for r in build_session_log()])
    assert len(groups) == 1
    assert set(groups[0]) == {"Asia/Calcutta", "Asia/Kolkata"}


@test
def etc_gmt_sign_is_inverted() -> None:
    msg = etc_zone_is_inverted("Etc/GMT+5")
    assert msg and "UTC-05:00" in msg
    assert etc_zone_is_inverted("Etc/GMT-5")  # also inverted, the other way
    assert etc_zone_is_inverted("America/New_York") is None
    assert etc_zone_is_inverted("Etc/GMT") is None


@test
def etc_gmt_plus_five_really_is_utc_minus_five() -> None:
    off = dt.datetime(2024, 7, 1, tzinfo=ZoneInfo("Etc/GMT+5")).utcoffset()
    assert off == dt.timedelta(hours=-5)


# --------------------------------------------------------------------------
# day bucketing
# --------------------------------------------------------------------------


@test
def utc_day_and_local_day_differ_east_of_greenwich() -> None:
    r = resolve("2024-11-04 00:25", "Asia/Kathmandu")
    assert local_day(r) == dt.date(2024, 11, 4)
    assert utc_day(r) == dt.date(2024, 11, 3)


@test
def day_bucketing_moves_revenue_without_changing_the_total() -> None:
    rows = build_session_log()
    readings = normalize(rows, ambiguous="earlier", nonexistent="shift_forward")
    u = sum(row["amount"] for row, r in zip(rows, readings) if utc_day(r))
    loc = sum(row["amount"] for row, r in zip(rows, readings) if local_day(r))
    assert u == loc  # the control total always reconciles - that is why it survives


# --------------------------------------------------------------------------
# the sample, and its ground truth
# --------------------------------------------------------------------------


@test
def the_sample_renders_from_true_instants() -> None:
    rows = build_session_log()
    truth = ground_truth()
    tz = ZoneInfo(NY)
    row = next(r for r in rows if r["session_id"] == "S-104" and r["event"] == "open")
    inst = truth[("S-104", "open")]
    assert inst is not None
    assert row["local_ts"] == inst.astimezone(tz).strftime("%Y-%m-%d %H:%M")


@test
def no_fold_policy_recovers_the_contested_sessions() -> None:
    """The central claim: the information is absent, not merely defaulted badly."""
    rows = build_session_log()
    truth = ground_truth()
    for policy in ("earlier", "later"):
        readings = normalize(rows, ambiguous=policy, nonexistent="flag")
        by = {}
        for row, r in zip(rows, readings):
            by.setdefault(row["session_id"], {})[row["event"]] = r
        for sid in ("S-104", "S-105"):
            got = (by[sid]["close"].utc - by[sid]["open"].utc).total_seconds() / 60
            real = (truth[(sid, "close")] - truth[(sid, "open")]).total_seconds() / 60
            assert got != real, (sid, policy)


@test
def the_offset_carrying_feed_recovers_them_exactly() -> None:
    rows = build_session_log(with_partner_feed=True)
    truth = ground_truth()
    readings = normalize(rows, ambiguous="earlier")
    by = {}
    for row, r in zip(rows, readings):
        by.setdefault(row["session_id"], {})[row["event"]] = r
    for sid in ("S-104", "S-105"):
        got = (by[f"{sid}-api"]["close"].utc - by[f"{sid}-api"]["open"].utc).total_seconds() / 60
        real = (truth[(sid, "close")] - truth[(sid, "open")]).total_seconds() / 60
        assert got == real, sid


@test
def one_session_really_does_close_before_it_opens() -> None:
    rows = build_session_log()
    readings = normalize(rows, ambiguous="earlier", nonexistent="flag")
    by = {}
    for row, r in zip(rows, readings):
        by.setdefault(row["session_id"], {})[row["event"]] = r
    dur = (by["S-105"]["close"].utc - by["S-105"]["open"].utc).total_seconds() / 60
    assert dur == -40


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


@test
def audit_counts_both_hard_cases() -> None:
    rep = audit(build_session_log())
    counts = rep.stats["status_counts"]
    assert counts["ambiguous"] == 5
    assert counts["nonexistent"] == 2


@test
def audit_reports_the_alias_group() -> None:
    rep = audit(build_session_log())
    assert any("Asia/Calcutta and Asia/Kolkata" in f for f in rep.findings)


@test
def audit_reports_the_inverted_etc_zone() -> None:
    rep = audit(build_session_log())
    assert any("POSIX sign convention is inverted" in f for f in rep.findings)


@test
def audit_stamps_the_tzdata_version() -> None:
    rep = audit(build_session_log())
    assert rep.stats["tzdata"] == tzdata_version()
    assert any("record this next to the results" in f for f in rep.findings)


@test
def audit_flags_sub_hour_offsets() -> None:
    rep = audit(build_session_log())
    assert set(rep.stats["sub_hour_offsets"]) == {"5:30:00", "5:45:00"}


@test
def audit_verdict_is_clean_on_unambiguous_data() -> None:
    rows = [
        {"local_ts": "2024-06-15 09:00", "zone": "Asia/Singapore"},
        {"local_ts": "2024-06-16 09:00", "zone": "Asia/Singapore"},
    ]
    assert audit(rows).verdict.startswith("CLEAN"), audit(rows).text()


@test
def audit_refuses_on_an_unknown_zone() -> None:
    rows = [{"local_ts": "2024-06-15 09:00", "zone": "Europe/Atlantis"}]
    assert audit(rows).verdict.startswith("NOT SAFE")


@test
def audit_names_the_count_of_undetermined_rows() -> None:
    rep = audit(build_session_log())
    assert "7 OF 24" in rep.verdict


def main() -> int:
    print(f"tz database: {tzdata_version()}\n")
    failures = 0
    for name, fn in _TESTS:
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
