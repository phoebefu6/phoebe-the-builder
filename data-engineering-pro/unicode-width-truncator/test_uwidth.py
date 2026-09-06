"""Every headline number in the README, asserted.

These tests are deliberately exact. If a Unicode data upgrade, a new
`regex` release or a different Node/ICU build changes a count, this
suite fails loudly rather than leaving the README quietly wrong.
"""

from __future__ import annotations

import unicodedata

import pytest
import uwidth as U

# --------------------------------------------------------------------------
# Environment - the claims are only true for these versions
# --------------------------------------------------------------------------


def test_node_is_present():
    assert U._node_available(), "node is required; the UTF-16 truncators are real JS"


def test_recorded_unicode_versions():
    assert unicodedata.unidata_version == "14.0.0"
    assert U.node_versions()["unicode"] == "15.0"


def test_corpus_and_roster_size():
    assert len(U.CORPUS) == 26
    assert len(U.TRUNCATORS) == 10
    assert len(U.SINKS) == 6


# --------------------------------------------------------------------------
# 2 / 16. One string, many strings out
# --------------------------------------------------------------------------


def test_one_string_yields_six_outputs():
    case = U.CASE_BY_NAME["emoji-family"]
    outputs = {c.text for c in U.cut_all(case).values()}
    assert len(outputs) == 6


def test_distinct_outputs_over_corpus():
    distinct, total = U.distinct_output_count()
    assert (distinct, total) == (100, 260)


def test_only_ascii_cases_agree():
    agreed = [c.name for c in U.CORPUS if U.verdict_for(c).verdict == "agreed"]
    assert agreed == ["ascii-short", "url"]
    for name in agreed:
        assert U.CASE_BY_NAME[name].text.isascii()


def test_verdict_census():
    assert U.verdict_census() == {
        "agreed": 2,
        "unit-drift": 6,
        "boundary-split": 8,
        "identity-change": 0,
        "dangling-joiner": 4,
        "byte-overflow": 0,
        "width-blowout": 5,
        "bidi-leak": 1,
    }


def test_flag_census():
    assert U.flag_census() == {
        "boundary-split": 8,
        "identity-change": 3,
        "dangling-joiner": 9,
        "byte-overflow": 5,
        "width-blowout": 7,
        "bidi-leak": 1,
    }


def test_every_verdict_name_is_known():
    for case in U.CORPUS:
        assert U.verdict_for(case).verdict in U.VERDICTS


# --------------------------------------------------------------------------
# 3 / 4. The unit is the whole question
# --------------------------------------------------------------------------


def test_unit_spread_of_the_bio():
    spread = U.unit_spread(U.CASE_BY_NAME["emoji-family"])
    assert spread == {
        "bytes": 42,
        "code points": 24,
        "UTF-16 units": 28,
        "graphemes (regex)": 18,
        "graphemes (ICU)": 18,
        "columns": 25,
    }


def test_one_family_emoji_measured_six_ways():
    fam = "\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466"
    assert len(U.encode_utf8(fam)) == 25
    assert len(fam) == 7
    assert len(fam.encode("utf-16-le")) // 2 == 11
    assert len(U.graphemes(fam)) == 1
    assert int(U._node_one("count_graphemes", fam, 0)) == 1
    assert U.columns(fam) == 8  # wcwidth; a renderer that composes it draws 2


def test_cjk_is_two_columns_per_code_point():
    assert U.columns("数据工程师") == 10
    assert len("数据工程师") == 5


# --------------------------------------------------------------------------
# 5. Output that is not text
# --------------------------------------------------------------------------


def test_utf16_cut_leaves_a_lone_surrogate():
    text = U.CASE_BY_NAME["emoji-family"].text
    out = U._node_one("utf16_units", text, 9)
    assert U.has_lone_surrogate(out)
    with pytest.raises(UnicodeEncodeError):
        out.encode("utf-8")


def test_lone_surrogates_come_only_from_utf16_units():
    producers = {
        name
        for case in U.CORPUS
        for name, cut in U.cut_all(case).items()
        if cut.lone_surrogate
    }
    assert producers == {"utf16_units"}


def test_lone_surrogate_and_replacement_counts():
    surrogate = [
        1 for case in U.CORPUS for cut in U.cut_all(case).values() if cut.lone_surrogate
    ]
    replacement = [
        1
        for case in U.CORPUS
        for cut in U.cut_all(case).values()
        if U.has_replacement(cut.text)
    ]
    assert len(surrogate) == 4
    assert len(replacement) == 6


# --------------------------------------------------------------------------
# 6. The truncator that overflows its own limit
# --------------------------------------------------------------------------


def test_byte_truncation_can_exceed_the_byte_limit():
    text = "aa\U0001F600"
    cut = text.encode("utf-8")[:3].decode("utf-8", "replace")
    assert len(cut.encode("utf-8")) == 5 > 3


def test_overflow_cases_in_corpus():
    rows = [
        (case.name, cut.bytes_out - case.n)
        for case in U.CORPUS
        for cut in U.cut_all(case).values()
        if cut.overflows_own_limit
    ]
    assert len(rows) == 5
    assert max(delta for _, delta in rows) == 2
    assert {name for name, _ in rows} == {
        "emoji-skin", "emoji-vs16", "combining-stack", "zwsp", "ri-odd",
    }


def test_backoff_never_overflows():
    for case in U.CORPUS:
        cut = U.cut_all(case)["utf8_bytes_backoff"]
        assert cut.bytes_out <= case.n


# --------------------------------------------------------------------------
# 7. Identity changes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("whole,part,_desc", U.IDENTITY_PROBES)
def test_identity_probe_parts_are_well_formed(whole, part, _desc):
    """Every probe's truncated form is valid text - that is the problem."""
    assert part.encode("utf-8").decode("utf-8") == part
    assert not U.has_replacement(part)
    assert part != whole


def test_family_cut_to_three_code_points_is_a_couple():
    fam = "\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466"
    assert fam[:3] == "\U0001F468‍\U0001F469"
    assert len(U.graphemes(fam[:3])) == 1  # still one cluster, a different one


def test_skin_tone_is_dropped_not_broken():
    thumb = "\U0001F44D\U0001F3FD"
    assert thumb[:1] == "\U0001F44D"
    assert U.identity_change(thumb, thumb[:1]) is not None


def test_identity_change_is_flagged_in_three_cases():
    flagged = [c.name for c in U.CORPUS if "identity-change" in U.verdict_for(c).flags]
    assert flagged == ["emoji-family", "emoji-skin", "ri-odd"]


# --------------------------------------------------------------------------
# 8. Dangling joiners
# --------------------------------------------------------------------------


def test_dangling_zwj_fuses_with_the_next_chunk():
    fam = "\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466"
    stub = fam[:6]                    # man ZWJ woman ZWJ girl ZWJ
    assert U.dangling(stub) == "ZERO WIDTH JOINER"
    joined = stub + "\U0001F466"
    assert len(U.graphemes(joined)) == 1
    assert joined == fam              # two values fused back into one cluster


def test_lone_regional_indicator_is_named():
    assert U.dangling("\U0001F1FA") == "LONE REGIONAL INDICATOR"
    assert U.dangling("\U0001F1FA\U0001F1F8") is None


def test_dangling_cut_count():
    rows = [
        1
        for case in U.CORPUS
        for cut in U.cut_all(case).values()
        if cut.dangling
    ]
    assert len(rows) == 27


# --------------------------------------------------------------------------
# 9. Normalisation
# --------------------------------------------------------------------------


def test_nfc_and_nfd_truncate_differently():
    nfc = U.CASE_BY_NAME["accent-nfc"]
    nfd = U.CASE_BY_NAME["accent-nfd"]
    assert nfc.text != nfd.text
    assert unicodedata.normalize("NFC", nfd.text) == nfc.text  # same visible name
    a = U.TRUNCATOR_BY_NAME["code_points"].cut(nfc.text, 12)
    b = U.TRUNCATOR_BY_NAME["code_points"].cut(nfd.text, 12)
    assert a != b
    # The same limit of 12 stores 12 visible characters of one and 10 of the other.
    assert len(U.graphemes(a)) == 12
    assert len(U.graphemes(b)) == 10


def test_nfd_cut_drops_the_accent_silently():
    jose = unicodedata.normalize("NFD", "José")
    assert len(jose) == 5
    assert jose[:4] == "Jose"
    assert not U.has_replacement(jose[:4])


def test_nfd_cut_changes_the_name_at_a_mark_boundary():
    """n=9 lands between n and its tilde: Muñ becomes Mun."""
    nfc = U.CASE_BY_NAME["accent-nfc"].text
    nfd = U.CASE_BY_NAME["accent-nfd"].text
    assert nfc[:9] == "José Muño"
    assert unicodedata.normalize("NFC", nfd[:9]) == "José Mun"
    assert "ñ" not in unicodedata.normalize("NFC", nfd[:9])


# --------------------------------------------------------------------------
# 10. Two segmenters
# --------------------------------------------------------------------------


def test_segmenters_disagree_on_devanagari_only():
    dis = U.segmenter_disagreements()
    assert [name for name, _, _ in dis] == ["devanagari"]
    assert dis[0][1:] == (7, 6)


def test_conjunct_is_the_cause():
    assert len(U.graphemes("क्ष")) == 2
    assert int(U._node_one("count_graphemes", "क्ष", 0)) == 1


# --------------------------------------------------------------------------
# 11. Width
# --------------------------------------------------------------------------


def test_code_point_cut_blows_the_column_budget():
    case = U.CASE_BY_NAME["cjk-bio"]
    assert U.cut_all(case)["code_points"].columns_out == 16 > case.n
    assert U.cut_all(case)["grapheme_columns"].columns_out <= case.n


def test_column_truncator_never_exceeds_its_budget():
    for case in U.CORPUS:
        for name in ("term_columns", "grapheme_columns"):
            assert U.cut_all(case)[name].columns_out <= case.n


def test_zero_width_and_control_widths():
    assert U.columns("​") == 0
    assert U.columns("\x07") == 0     # charged 0, not -1, so it does not erase the sum


# --------------------------------------------------------------------------
# 12. Bidi
# --------------------------------------------------------------------------


def test_cut_can_leak_a_bidi_override():
    case = U.CASE_BY_NAME["bidi-override"]
    assert U.bidi_balance(case.text) == 0
    leaks = [n for n, c in U.cut_all(case).items() if c.bidi_leak != 0]
    assert leaks, "expected at least one truncator to leave the override open"


def test_safe_truncate_never_leaks_bidi():
    case = U.CASE_BY_NAME["bidi-override"]
    for sink in U.SINKS:
        assert U.bidi_balance(U.safe_truncate(case.text, case.n, sink.name)) == 0


# --------------------------------------------------------------------------
# 13. The ellipsis budget
# --------------------------------------------------------------------------


def test_ellipsis_costs_something_in_every_unit():
    assert U.ellipsis_cost() == {
        "bytes": 3,
        "code points": 1,
        "UTF-16 units": 1,
        "columns": 1,
    }


def test_appending_an_ellipsis_overflows_almost_every_cut():
    over = U.naive_ellipsis_overflow()
    performed = sum(
        1 for c in U.CORPUS for t in U.TRUNCATORS if t.cut(c.text, c.n) != c.text
    )
    assert (len(over), performed) == (210, 230)


def test_safe_truncate_keeps_the_ellipsis_inside_the_budget():
    for case in U.CORPUS:
        for sink in U.SINKS:
            out = U.safe_truncate(case.text, case.n, sink.name)
            assert sink.measure(out) <= case.n


# --------------------------------------------------------------------------
# 14. Sinks
# --------------------------------------------------------------------------


def test_sink_failure_rate():
    assert U.sink_failure_rate() == (540, 1560)


def test_truncating_to_n_does_not_imply_fitting_n():
    case = U.CASE_BY_NAME["emoji-family"]
    cut = U.cut_all(case)["code_points"].text
    oracle = U.SINK_BY_NAME["oracle_varchar2_byte"]
    assert len(cut) <= case.n                       # 12 code points, as asked
    assert not U.fits(oracle, cut, case.n)          # and 22 bytes, so it is rejected


def test_choose_truncator_matches_the_sink_unit():
    for sink in U.SINKS:
        chosen = U.TRUNCATOR_BY_NAME[U.choose_truncator(sink.name)]
        assert chosen.unit == sink.unit


# --------------------------------------------------------------------------
# 15. safe_truncate
# --------------------------------------------------------------------------


def test_safe_truncate_audit_is_clean():
    audit = U.safe_truncate_audit()
    assert len(audit) == 156
    assert all(fits and no_dangle and well_formed for _, _, fits, no_dangle, well_formed in audit)


def test_safe_truncate_is_idempotent():
    for case in U.CORPUS:
        for sink in U.SINKS:
            once = U.safe_truncate(case.text, case.n, sink.name)
            assert U.safe_truncate(once, case.n, sink.name) == once


def test_safe_truncate_leaves_short_values_alone():
    assert U.safe_truncate("hi", 20, "postgres_varchar") == "hi"


def test_safe_truncate_returns_empty_when_the_marker_will_not_fit():
    assert U.safe_truncate("abcdefgh", 2, "oracle_varchar2_byte") == ""


def test_roster_truncators_are_all_idempotent():
    """The roster is not the problem; the unit is."""
    assert U.idempotence_failures() == []


def test_sink_failures_break_down_by_sink():
    per_sink = {}
    for _, _, sink, _, _ in U.sink_failures():
        per_sink[sink] = per_sink.get(sink, 0) + 1
    assert per_sink == {
        "oracle_varchar2_byte": 173,
        "http_header_bytes": 173,
        "sqlserver_nvarchar": 67,
        "fixed_width_column": 51,
        "mysql_utf8mb4_varchar": 38,
        "postgres_varchar": 38,
    }
    assert sum(per_sink.values()) == 540


def _clean_rate(pick):
    """Runs that fit the sink, do not dangle and are still text."""
    ok = total = 0
    for case in U.CORPUS:
        cuts = U.cut_all(case)
        for sink in U.SINKS:
            cut = cuts[pick(sink)]
            total += 1
            if U.fits(sink, cut.text, case.n) and cut.dangling is None and cut.well_formed:
                ok += 1
    return ok, total


def test_the_three_approaches():
    assert _clean_rate(lambda s: "code_points") == (84, 156)          # 54%
    assert _clean_rate(lambda s: U.choose_truncator(s.name)) == (129, 156)  # 83%
    audit = U.safe_truncate_audit()
    assert sum(1 for _, _, f, d, w in audit if f and d and w) == 156  # 100%


def test_matching_the_unit_is_not_enough_on_its_own():
    """The 83% row: the limit is satisfied and the meaning is still broken."""
    case = U.CASE_BY_NAME["emoji-family"]
    sink = U.SINK_BY_NAME["postgres_varchar"]
    cut = U.cut_all(case)[U.choose_truncator(sink.name)]
    assert U.fits(sink, cut.text, case.n)      # the unit matches, so it fits
    assert cut.dangling == "ZERO WIDTH JOINER"  # and it still ends in a joiner
