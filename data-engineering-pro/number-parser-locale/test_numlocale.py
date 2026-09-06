"""Every headline number in the README is asserted here.

If a Babel or Node upgrade changes a reading, these fail loudly rather than
letting the README quietly become fiction.
"""

from __future__ import annotations

from decimal import Decimal

import numlocale as N
import pytest

# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------

def test_corpus_and_roster_shape():
    assert len(N.corpus()) == 35
    assert len(N.reader_names()) == 15
    assert len(N.corpus()) * len(N.reader_names()) == 525


def test_every_case_name_unique():
    names = [c.name for c in N.corpus()]
    assert len(set(names)) == len(names)


def test_readers_all_available():
    """The whole point is measurement; a missing reader invalidates the counts."""
    table = N.read_all()
    for row in table.values():
        for name, r in row.items():
            assert r.status != N.UNAVAILABLE, "%s not available on this machine" % name


# --------------------------------------------------------------------------
# The headline: one string, two numbers
# --------------------------------------------------------------------------

def test_dot_three_digits_is_read_two_ways():
    assert N.read_locale("1.234", "en_US", True).value == Decimal("1.234")
    assert N.read_locale("1.234", "de_DE", True).value == Decimal("1234")


def test_comma_three_digits_is_read_two_ways():
    assert N.read_locale("1,234", "en_US", True).value == Decimal("1234")
    assert N.read_locale("1,234", "de_DE", True).value == Decimal("1.234")


def test_dot_three_digits_is_a_thousandfold_disagreement():
    v = next(x for x in N.all_verdicts() if x.case.name == "dot-3dp")
    assert v.verdict == "magnitude-drift"
    assert v.ratio == Decimal(1000)


def test_grouped_integer_is_the_worst_case_in_the_corpus():
    v = next(x for x in N.all_verdicts() if x.case.name == "us-grouped")
    assert v.ratio is not None and v.ratio > Decimal(1_000_000)
    # parseFloat stops at the first comma and returns 1.
    assert N.read_all()["us-grouped"]["js_parsefloat"].value == Decimal(1)
    assert N.read_all()["us-grouped"]["en_US_strict"].value == Decimal(1234567)


# --------------------------------------------------------------------------
# Verdict counts
# --------------------------------------------------------------------------

def test_agreed_is_four_of_thirty_five():
    verds = N.all_verdicts()
    agreed = [v for v in verds if v.verdict == "agreed"]
    assert len(agreed) == 4
    assert {v.case.name for v in agreed} == {
        "plain-int", "sci-notation", "sci-overflow", "inf-word"}


def test_sixteen_strings_are_ten_x_apart_or_more():
    verds = N.all_verdicts()
    assert sum(1 for v in verds if v.verdict == "magnitude-drift") == 16


def test_nineteen_of_thirtyfive_have_more_than_one_reading():
    verds = N.all_verdicts()
    assert sum(1 for v in verds if v.n_distinct >= 2) == 19


def test_nothing_is_rejected_by_every_reader():
    """strtod has no failure channel, so every string gets at least one value."""
    verds = N.all_verdicts()
    assert sum(1 for v in verds if v.verdict == "rejected-by-all") == 0


# --------------------------------------------------------------------------
# CLDR symbols: the two that are not on a keyboard
# --------------------------------------------------------------------------

@pytest.mark.parametrize("locale,group_cp,decimal_sym", [
    ("en_US", 0x002C, "."),
    ("de_DE", 0x002E, ","),
    ("fr_FR", 0x202F, ","),
    ("en_IN", 0x002C, "."),
    ("de_CH", 0x2019, "."),
])
def test_cldr_symbols(locale, group_cp, decimal_sym):
    rows = {r["locale"]: r for r in N.locale_symbols()}
    assert ord(rows[locale]["group"]) == group_cp
    assert rows[locale]["decimal"] == decimal_sym


def test_french_nnbsp_parses_and_its_lookalikes_do_not():
    good = "1" + N.NNBSP + "234" + N.NNBSP + "567,89"
    assert N.read_locale(good, "fr_FR", True).value == Decimal("1234567.89")
    for bad in ("1" + N.NBSP + "234" + N.NBSP + "567,89", "1 234 567,89"):
        assert not N.read_locale(bad, "fr_FR", True).ok
        assert not N.read_locale(bad, "fr_FR", False).ok


def test_swiss_rsquo_parses_and_ascii_apostrophe_does_not():
    assert N.read_locale("1" + N.RSQUO + "234" + N.RSQUO + "567.89", "de_CH", True).value \
        == Decimal("1234567.89")
    assert not N.read_locale("1'234'567.89", "de_CH", True).ok


def test_indian_lakh_grouping_is_mutually_exclusive_with_us_grouping():
    assert N.read_locale("12,34,567", "en_IN", True).value == Decimal(1234567)
    assert not N.read_locale("12,34,567", "en_US", True).ok
    assert N.read_locale("1,234,567", "en_US", True).value == Decimal(1234567)
    assert not N.read_locale("1,234,567", "en_IN", True).ok


# --------------------------------------------------------------------------
# Scanner behaviour
# --------------------------------------------------------------------------

def test_strtod_accepts_everything_in_the_corpus():
    table = N.read_all()
    assert all(table[c.name]["c_strtod"].ok for c in N.corpus())


def test_strtod_returns_a_silent_zero_six_times():
    table = N.read_all()
    z = [c.name for c in N.corpus()
         if table[c.name]["c_strtod"].is_finite
         and table[c.name]["c_strtod"].value == 0
         and "silent 0" in table[c.name]["c_strtod"].note]
    assert len(z) == 6
    assert "accounting-neg" in z and "currency-prefix" in z and "true-minus" in z


def test_js_number_turns_blank_cells_into_zero():
    table = N.read_all()
    for name in ("empty", "blank"):
        r = table[name]["js_number"]
        assert r.is_finite and r.value == 0


def test_js_parsefloat_refuses_blank_cells():
    table = N.read_all()
    for name in ("empty", "blank"):
        assert not table[name]["js_parsefloat"].ok


def test_pep515_underscore_is_accepted_by_python_and_leaks_into_babel():
    assert N.read_py_float("1_000").value == Decimal(1000)
    assert N.read_locale("1_000", "en_US", True).value == Decimal(1000)
    # ... while the C and JS scanners do not accept it
    assert N.read_c_strtod("1_000").value == Decimal(1)
    assert not N.read_all()["pep515"]["js_number"].ok


def test_arabic_indic_digits_are_accepted_by_python_and_babel_not_by_js():
    table = N.read_all()
    row = table["arabic-indic"]
    assert row["py_decimal"].value == Decimal(1234)
    assert row["en_US_strict"].value == Decimal(1234)
    assert not row["js_number"].ok


def test_id_beyond_double_range_is_off_by_one_everywhere_but_decimal():
    table = N.read_all()["int53-plus1"]
    assert table["py_decimal"].value == Decimal("9007199254740993")
    for reader in ("py_float", "js_number", "c_strtod"):
        assert table[reader].value == Decimal("9007199254740992")


# --------------------------------------------------------------------------
# Negatives
# --------------------------------------------------------------------------

def test_no_reader_recovers_an_intended_negative():
    table = N.read_all()
    for name in ("accounting-neg", "trailing-minus", "true-minus"):
        for r in table[name].values():
            if r.is_finite:
                assert r.value >= 0, "%s read %s as negative" % (name, r.value)


def test_trailing_minus_is_read_positive():
    v = next(x for x in N.all_verdicts() if x.case.name == "trailing-minus")
    assert v.verdict == "sign-loss"
    assert N.read_c_strtod("1234-").value == Decimal(1234)


# --------------------------------------------------------------------------
# The border crossing
# --------------------------------------------------------------------------

def test_border_crossing_counts():
    cross = N.crossings()
    assert len(cross) == 250
    assert sum(1 for c in cross if c.status == "wrong") == 22
    assert sum(1 for c in cross if c.status == "error") == 131
    assert sum(1 for c in cross if c.status == "ok") == 97


def test_every_silently_wrong_crossing_is_a_loose_reader():
    for c in N.crossings():
        if c.status == "wrong":
            assert c.strict is False


def test_a_us_half_dollar_read_as_german_becomes_fifty():
    r = N.read_locale("0.50", "de_DE", False)
    assert r.value == Decimal(50)


def test_us_money_read_as_german_loses_three_orders_of_magnitude():
    r = N.read_locale("1,234.50", "de_DE", False)
    assert r.value == Decimal("1.23450")


def test_strict_mode_refuses_its_own_locales_output_six_times():
    diag = N.own_output_roundtrip()
    assert len(diag) == 50
    bad = [c for c in diag if c.status != "ok"]
    assert len(bad) == 6
    assert all(c.strict for c in bad), "all six failures are strict-mode refusals"


def test_strict_refuses_a_trailing_zero_cent_but_accepts_the_same_amount_without_it():
    assert not N.read_locale("1,234.50", "en_US", True).ok
    assert N.read_locale("1,234.5", "en_US", True).value == Decimal("1234.5")
    assert N.read_locale("1,234.56", "en_US", True).value == Decimal("1234.56")


def test_pattern_can_override_a_locales_own_grouping():
    """en_IN groups at 2,2,3; the pattern makes it emit 1,234,567.89 anyway."""
    from babel.numbers import format_decimal
    rendered = format_decimal(Decimal("1234567.89"), format="#,##0.00", locale="en_IN")
    assert rendered == "1,234,567.89"
    assert not N.read_locale(rendered, "en_IN", True).ok
    assert format_decimal(Decimal("1234567.89"), locale="en_IN") == "12,34,567.89"


def test_pattern_rounding_is_not_counted_as_a_reader_error():
    """#,##0.00 rounds 1.234 to 1.23 on the way out; that is the writer's loss."""
    cs = [c for c in N.crossings() if c.value == Decimal("1.234")
          and c.wrote == c.read and c.read == "en_US"]
    assert cs and all(c.expected == Decimal("1.23") for c in cs)
    assert any(c.status == "ok" for c in cs)


# --------------------------------------------------------------------------
# Locale hypotheses
# --------------------------------------------------------------------------

def test_a_money_column_of_three_digit_groups_is_undecidable():
    d = N.decide_column(["1.234", "2.500", "3.000", "1.750"])
    assert d.verdict == "ambiguous"
    assert len(d.surviving) == 5
    assert set(map(str, d.totals.values())) == {"8.484", "8484"}
    assert d.spread == Decimal(1000)


def test_a_group_count_above_one_decides_the_column():
    d = N.decide_column(["1.234.567", "89.012", "3.456"])
    assert d.verdict == "decided"
    assert d.surviving == ["de_DE"]
    assert d.totals["de_DE"] == Decimal(1327035)


def test_a_four_digit_group_eliminates_dot_grouping_locales():
    d = N.decide_column(["1.2345", "2.500"])
    assert d.verdict == "decided"
    assert "de_DE" not in d.surviving
    assert set(map(str, d.totals.values())) == {"3.7345"}


def test_both_separators_present_pins_the_decimal():
    d = N.decide_column(["1.234,56", "7.890,12"])
    assert d.verdict == "decided"
    assert d.surviving == ["de_DE"]


def test_lakh_grouping_identifies_en_in_uniquely():
    d = N.decide_column(["12,34,567", "1,23,456"])
    assert d.surviving == ["en_IN"]


def test_a_column_no_locale_can_read():
    d = N.decide_column(["1.2345,67", "9"])
    assert d.verdict == "no-locale-fits"
    assert d.surviving == []


def test_hypotheses_report_the_row_that_killed_them():
    hyps = {h.locale: h for h in N.locale_hypotheses(["1.2345", "2.500"])}
    assert hyps["de_DE"].survives is False
    assert hyps["de_DE"].killed_by == "1.2345"


def test_prefix_parsers_never_eliminate_anything():
    """Accepting everything means carrying no information about provenance."""
    a = N.audit_column(["1.234", "2.500", "3.000", "1.750"])
    assert "c_strtod" in a.readers_that_take_every_row
    assert a.decision is not None and a.decision.verdict == "ambiguous"
    assert a.decidable is False


def test_audit_column_reports_both_totals_rather_than_picking_one():
    a = N.audit_column(["1.234", "2.500"])
    totals = {str(v) for v in a.candidate_readings.values() if v is not None}
    assert totals == {"3.734", "3734"}
    assert a.disagreement == Decimal(1000)


def test_audit_column_row_findings_are_summarised_not_repeated():
    a = N.audit_column(["1.234", "2.500", "3.000", "1.750"])
    assert any("and 2 more rows" in f for f in a.findings)


# --------------------------------------------------------------------------
# Display never lies
# --------------------------------------------------------------------------

def test_display_marks_truncation():
    r = N.read_py_decimal("9007199254740993")
    assert r.display() == "9007199254740993"
    assert r.display(width=11).endswith("~")


def test_display_does_not_expand_a_huge_exponent():
    r = N.read_py_decimal("1e309")
    assert r.display() == "1e+309"


def test_escaped_shows_invisible_inputs():
    cases = {c.name: c for c in N.corpus()}
    assert cases["empty"].escaped() == "(empty)"
    assert cases["blank"].escaped() == "(3 spaces)"
    assert "\\u202f" in cases["fr-nnbsp"].escaped()


def test_sub_unit_ratios_never_print_as_zero():
    assert N._fmt_ratio(Decimal("0.001")) == "1/1,000"
    assert N._fmt_ratio(Decimal("1000")) == "1,000"


# --------------------------------------------------------------------------
# Summary is self-consistent
# --------------------------------------------------------------------------

def test_summary_matches_the_readme_headlines():
    s = N.summary()
    assert s["n_readings"] == 525
    assert s["verdict_counts"]["agreed"] == 4
    assert s["n_multi_valued"] == 19
    assert s["crossing_wrong"] == 22
    assert s["diag_strict_fail"] == 6
    assert s["diag_loose_wrong"] == 0
    assert s["ambiguous_money_column"] == "ambiguous"
    assert s["decidable_by_group_count"] == "decided"
    assert s["decidable_by_4digit"] == "decided"
    assert s["decidable_by_mixed"] == "decided"


def test_evidence_script_runs_clean():
    import evidence
    assert evidence.main([]) == 0


# --------------------------------------------------------------------------
# Negative notations are named, not guessed at
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,label", [
    ("(1,234)", "accounting parentheses"),
    ("1234-", "trailing sign (SAP/COBOL)"),
    ("−1234", "U+2212 MINUS SIGN"),
    ("－1234", "U+FF0D FULLWIDTH HYPHEN-MINUS"),
    ("1234 CR", "CR/DB suffix"),
])
def test_negative_notation_is_named(raw, label):
    assert N.negative_notation(raw) == label


@pytest.mark.parametrize("raw", ["-1234", "1234", "1.234", "", "   ", "abc"])
def test_ascii_hyphen_and_plain_numbers_are_not_flagged(raw):
    assert N.negative_notation(raw) is None


def test_a_pasted_string_gets_a_sign_loss_verdict_without_declared_intent():
    """The app has no 'intended' field to lean on; the notation carries it."""
    case = N.Case("pasted", "1234-", "pasted into the app")
    v = N.verdict_for(case, N.read_all([case])["pasted"])
    assert v.verdict == "sign-loss"
    assert "trailing sign" in v.flags[0]
