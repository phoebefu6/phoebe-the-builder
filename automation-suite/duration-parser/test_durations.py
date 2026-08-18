"""Tests for the duration audit.

Two kinds of assertion here, and the second kind is the point of the file:

1. per-grammar behaviour against each grammar's own rules
2. *disagreement* assertions - two grammars are handed one string and the test
   pins the two different numbers they return. If a future edit quietly makes
   them agree, that is a modelling regression, not a fix.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from durations import (
    DAY24,
    DEFAULT_ANCHORS,
    HOUR,
    MINUTE,
    PARSERS,
    REFERENCE_ANCHOR,
    Verdict,
    _NY,
    add_nominal,
    audit,
    audit_corpus,
    best_single_grammar,
    clamped,
    nonexistent_local,
    parse_excel,
    parse_go,
    parse_iso,
    parse_prometheus,
    parse_shorthand,
    safe_form,
)


def val(grammar: str, text: str, anchor: datetime = REFERENCE_ANCHOR) -> float:
    r = PARSERS[grammar](text)
    assert r.ok, f"{grammar} rejected {text!r}: {r.error}"
    return r.resolve(anchor)


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,seconds",
    [("1h30m", 5400), ("2h45m", 9900), ("-1.5h", -5400), ("500ms", 0.5),
     ("1µs", 1e-6), ("1μs", 1e-6), ("1us", 1e-6), ("0", 0), ("1h1h", 7200),
     ("30m1h", 5400), ("100ns", 1e-7)],
)
def test_go_accepts(text, seconds):
    assert val("go", text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["1h30", "1d", "1w", "1y", "90", "", "1x2h", "1.5.5s", "1M"])
def test_go_rejects(text):
    assert not parse_go(text).ok


def test_go_has_no_unit_above_the_hour():
    # Not an omission: a day is not a fixed length, so the library that refuses
    # to guess is the one that cannot express it.
    assert not parse_go("1d").ok
    assert "unknown unit 'd'" in parse_go("1d").error


def test_go_repeats_are_summed_and_reported():
    r = parse_go("1h1h")
    assert r.exact_s == 7200
    assert any("sums them" in n for n in r.notes)


def test_go_range_is_int64_nanoseconds():
    assert parse_go("2562047h").ok  # just inside
    assert not parse_go("2562048h").ok  # just outside


# ---------------------------------------------------------------------------
# ISO 8601
# ---------------------------------------------------------------------------


def test_iso_m_means_months_before_t_and_minutes_after():
    month = parse_iso("P1M")
    minute = parse_iso("PT1M")
    assert month.nominal.months == 1 and month.exact_s == 0
    assert minute.nominal.months == 0 and minute.exact_s == 60
    # The same letter, 43800-odd times apart.
    assert month.resolve(REFERENCE_ANCHOR) / minute.resolve(REFERENCE_ANCHOR) > 40000


def test_iso_full_form():
    r = parse_iso("P1Y2M3DT4H5M6S")
    assert r.nominal.months == 14
    assert r.nominal.days == 3
    assert r.exact_s == 4 * HOUR + 5 * MINUTE + 6


def test_iso_weeks_are_seven_nominal_days():
    assert parse_iso("P1W").nominal.days == 7


@pytest.mark.parametrize("text", ["P", "PT", "1M", "P1.5M", "P1.5YT1H", "P1H", "PT1D"])
def test_iso_rejects(text):
    assert not parse_iso(text).ok


def test_iso_fraction_only_on_the_smallest_component():
    assert parse_iso("PT1H30.5M").ok
    assert not parse_iso("PT1.5H30M").ok


def test_iso_sign_covers_the_whole_duration():
    r = parse_iso("-P1DT1H")
    assert r.nominal.days == -1 and r.exact_s == -HOUR
    assert any("whole duration" in n for n in r.notes)


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,seconds",
    [("30s", 30), ("5m", 300), ("1h30m", 5400), ("1d", DAY24), ("1w", 7 * DAY24),
     ("1y", 365 * DAY24), ("500ms", 0.5), ("1y6w3d12h", 365 * DAY24 + 6 * 7 * DAY24 + 3 * DAY24 + 12 * HOUR)],
)
def test_prometheus_accepts(text, seconds):
    assert val("prometheus", text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["1.5h", "30m1h", "1h1h", "90", "-5m", "1M", "1h 30m"])
def test_prometheus_rejects(text):
    assert not parse_prometheus(text).ok


def test_prometheus_day_is_fixed_and_says_so():
    r = parse_prometheus("1d")
    assert r.exact_s == DAY24
    assert any("fixed 24h" in n for n in r.notes)


# ---------------------------------------------------------------------------
# systemd
# ---------------------------------------------------------------------------


def test_systemd_case_of_the_unit_letter_is_the_whole_difference():
    minute = val("systemd", "1m")
    month = val("systemd", "1M")
    assert minute == 60
    assert month == pytest.approx(30.44 * DAY24)
    assert month / minute > 43000


@pytest.mark.parametrize(
    "text,seconds",
    [("2h 30min", 9000), ("2h30min", 9000), ("1h30", HOUR + 30), ("90", 90),
     ("1 hour", HOUR), ("3w 2d", 3 * 7 * DAY24 + 2 * DAY24), ("500ms", 0.5)],
)
def test_systemd_accepts(text, seconds):
    assert val("systemd", text) == pytest.approx(seconds)


def test_systemd_trailing_bare_number_is_seconds_not_minutes():
    # The heart of the "1h30" question: systemd says 1h and 30 *seconds*.
    assert val("systemd", "1h30") == 3630
    assert val("shorthand", "1h30") == 5400
    assert val("jira", "1h30") == 5400


# ---------------------------------------------------------------------------
# Clock forms: the same colons, aligned from opposite ends
# ---------------------------------------------------------------------------


def test_colon_alignment_disagreement():
    assert val("ffmpeg", "1:30") == 90
    assert val("excel", "1:30") == 5400
    assert val("excel", "1:30") / val("ffmpeg", "1:30") == 60


def test_three_field_clock_agrees():
    assert val("ffmpeg", "1:30:45.5") == pytest.approx(5445.5)
    assert val("excel", "1:30:45") == pytest.approx(5445)


def test_bare_number_spans_five_orders_of_magnitude():
    assert val("ffmpeg", "90") == 90
    assert val("systemd", "90") == 90
    assert val("jira", "90") == 90 * MINUTE
    assert val("excel", "90") == 90 * DAY24
    assert val("excel", "90") / val("ffmpeg", "90") == 86400


def test_jira_day_is_working_hours():
    assert val("jira", "1d") == 8 * HOUR
    assert val("jira", "1w") == 5 * 8 * HOUR
    assert val("prometheus", "1d") / val("jira", "1d") == 3


def test_shorthand_needs_a_unit_to_step_down_from():
    assert not parse_shorthand("90").ok
    assert parse_shorthand("1h30").ok
    assert parse_shorthand("1d6").exact_s == DAY24 + 6 * HOUR


def test_excel_serial_unit_is_the_day():
    assert parse_excel("1").exact_s == DAY24
    assert parse_excel("0.5").exact_s == DAY24 / 2


# ---------------------------------------------------------------------------
# Calendar arithmetic
# ---------------------------------------------------------------------------


def test_month_end_clamps_and_is_not_invertible():
    jan31 = datetime(2024, 1, 31, tzinfo=_NY)
    feb = add_nominal(jan31, 1, 0)
    assert (feb.month, feb.day) == (2, 29)
    assert clamped(jan31, 1)
    back = add_nominal(feb, -1, 0)
    assert back.day == 29 and back != jan31  # +1M then -1M lands somewhere else


def test_a_calendar_day_is_not_always_86400_seconds():
    spring = datetime(2024, 3, 9, 12, tzinfo=_NY)
    autumn = datetime(2024, 11, 2, 12, tzinfo=_NY)
    r = parse_iso("P1D")
    assert r.resolve(spring) == 23 * HOUR
    assert r.resolve(autumn) == 25 * HOUR


def test_nonexistent_local_time_still_returns_a_value():
    # 02:30 on 9 March + 1 day is 02:30 on 10 March, which did not happen in
    # New York. Nothing raises. The elapsed count is a clean 86400 - and the
    # instant it lands on displays as 03:30, an hour past the wall time asked
    # for. So the failure is invisible in the number *and* in the exception.
    gap = datetime(2024, 3, 9, 2, 30, tzinfo=_NY)
    assert nonexistent_local(gap, 0, 1)
    end = add_nominal(gap, 0, 1)
    assert parse_iso("P1D").resolve(gap) == DAY24
    # NY -> NY is a no-op in Python, so the real wall time only appears after a
    # round trip through UTC - which is itself part of why this class of bug hides.
    displayed = end.astimezone(timezone.utc).astimezone(_NY)
    assert (displayed.hour, displayed.minute) == (3, 30)


def test_a_month_spans_28_to_31_days_over_the_anchor_set():
    r = parse_iso("P1M")
    lengths = sorted({round(r.resolve(a) / DAY24, 4) for a in DEFAULT_ANCHORS})
    assert min(lengths) < 29 and max(lengths) == 31


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,verdict",
    [
        ("30s", Verdict.EXACT),
        ("2h45m", Verdict.EXACT),
        ("1h30", Verdict.AMBIGUOUS),
        ("1:30", Verdict.AMBIGUOUS),
        ("90", Verdict.AMBIGUOUS),
        ("1d", Verdict.AMBIGUOUS),
        ("P1M", Verdict.ANCHORED),
        ("P1D", Verdict.ANCHORED),
        ("forever", Verdict.REJECTED),
    ],
)
def test_verdicts(text, verdict):
    assert audit(text).verdict is verdict


def test_rejected_reports_no_reading():
    a = audit("forever")
    assert not a.accepted and a.min_s is None and a.spread_ratio is None


def test_ambiguity_is_reported_with_both_numbers():
    a = audit("90")
    codes = {f.code for f in a.findings}
    assert "AMBIGUOUS_VALUE" in codes
    assert a.spread_ratio == 86400
    assert len(a.distinct_values()) == 3


def test_anchored_verdict_is_not_produced_by_cross_grammar_spread():
    # "1d" has four readings and two values; that is ambiguity, not anchoring.
    a = audit("1d")
    assert a.verdict is Verdict.AMBIGUOUS
    assert all(not r.anchored for r in a.accepted)


def test_case_swap_finding_fires_on_the_shift_key_pair():
    codes = {f.code for f in audit("1M").findings}
    assert "UNIT_CASE_SENSITIVE" in codes


def test_every_finding_code_fires_somewhere_in_the_corpus():
    rep = audit_corpus()
    unused = [c for c, n in rep.finding_counts.items() if n == 0]
    assert unused == [], f"finding codes with no evidence: {unused}"


def test_no_single_grammar_reads_the_whole_corpus():
    rep = audit_corpus()
    assert max(rep.accepted_by.values()) < rep.total
    name, accepted, wrong = best_single_grammar(rep)
    assert accepted < rep.total
    assert wrong > 0  # and the ones it does read, it reads differently from others


def test_corpus_accounting_adds_up():
    rep = audit_corpus()
    assert sum(rep.verdicts.values()) == rep.total == len(rep.audits)
    assert len(rep.unanimous()) >= len(rep.lonely())


def test_lonely_exact_is_not_agreement():
    rep = audit_corpus()
    for a in rep.lonely():
        assert len(a.accepted) == 1
        assert a.verdict is Verdict.EXACT


def test_safe_form_round_trips_through_every_grammar_that_has_seconds():
    for seconds in (30, 5400, 86400):
        text = safe_form(seconds)
        for g in ("go", "prometheus", "systemd", "shorthand"):
            assert val(g, text) == seconds


def test_resolve_refuses_a_rejected_reading():
    with pytest.raises(ValueError):
        parse_go("1d").resolve(REFERENCE_ANCHOR)
