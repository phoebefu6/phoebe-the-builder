"""Tests for the collation model.

The interesting assertions are the ones that pin a *disagreement*: two
collations that must order a pair in opposite directions, and a pagination
scheme that must lose rows. A test suite that only checked "sorting works"
would pass on a byte comparison and miss everything this tool is about.
"""

from __future__ import annotations

import unicodedata

import pytest
from collate import (
    COLLATION_BY_NAME,
    COLLATIONS,
    CORPUS,
    CORPUS_BY_ID,
    Verdict,
    bmp_flip,
    distinct_count,
    findings,
    flips,
    in_range,
    keyset_pagination,
    libc_agreement,
    locale_lower,
    normalization_pairs,
    offset_pagination,
    order,
    pagination_totals,
    positions,
    range_counts,
    sort_key,
    turkish_case_breakage,
    unique_violations,
    verdict,
)


def coll(name: str):
    return COLLATION_BY_NAME[name]


def pos(name: str, collation: str) -> int:
    return positions(coll(collation))[name_id(name)]


def name_id(name: str) -> int:
    for r in CORPUS:
        if r.name == name and unicodedata.is_normalized("NFC", r.name):
            return r.id
    for r in CORPUS:
        if r.name == name:
            return r.id
    raise KeyError(name)


# -- the corpus itself ------------------------------------------------------


def test_corpus_ids_are_unique_and_dense():
    ids = [r.id for r in CORPUS]
    assert ids == list(range(1, len(CORPUS) + 1))


def test_two_rows_are_the_same_string_in_different_normal_forms():
    pairs = normalization_pairs()
    assert len(pairs) == 1
    a, b = pairs[0]
    assert a.name != b.name  # different code points
    assert unicodedata.normalize("NFC", a.name) == unicodedata.normalize("NFC", b.name)
    assert len(a.name) != len(b.name)


def test_corpus_holds_one_pair_that_straddles_the_bmp():
    above = [r for r in CORPUS if max(ord(c) for c in r.name) > 0xFFFF]
    inside = [r for r in CORPUS if 0xE000 <= max(ord(c) for c in r.name) <= 0xFFFF]
    assert above and inside


# -- key construction -------------------------------------------------------


def test_byte_order_puts_every_capital_before_every_lowercase():
    c = coll("C")
    assert sort_key("Zoe", c) < sort_key("aaberg", c)


def test_linguistic_order_does_not():
    c = coll("en_US_icu")
    assert sort_key("aaberg", c) < sort_key("Zoe", c)


def test_case_is_the_last_level_not_the_first():
    c = coll("en_US_icu")
    assert sort_key("Aaberg", c) < sort_key("Ahtari", c)
    assert sort_key("aaberg", c) < sort_key("Ahtari", c)


def test_swedish_moves_a_ring_past_z():
    sv, en = coll("sv_SE"), coll("en_US_icu")
    assert sort_key("Zoe", sv) < sort_key("Åberg", sv)
    assert sort_key("Åberg", en) < sort_key("Zoe", en)


def test_phonebook_expands_umlauts_and_din_does_not():
    pb, din = coll("de_phonebook"), coll("de_DIN")
    # Under the phonebook rule Müller *is* Mueller, so Muench sits between
    # them and Muller; under DIN, Müller is Muller with a secondary mark.
    assert sort_key("Müller", pb) == sort_key("Mueller", pb)
    assert sort_key("Müller", din) != sort_key("Mueller", din)
    assert sort_key("Mueller", din) < sort_key("Muench", din) < sort_key("Muller", din)


def test_sharp_s_expands_only_in_the_phonebook_collation():
    assert sort_key("Straße", coll("de_phonebook")) == sort_key("Strasse", coll("de_phonebook"))
    assert sort_key("Straße", coll("en_US_icu")) != sort_key("Strasse", coll("en_US_icu"))


def test_turkish_puts_dotless_i_before_i():
    tr, en = coll("tr_TR"), coll("en_US_icu")
    assert sort_key("ısık", tr) < sort_key("isik", tr)
    assert sort_key("isik", en) < sort_key("ısık", en)


def test_turkish_lowercases_capital_i_to_dotless():
    assert locale_lower("I", "tr") == "ı"
    assert locale_lower("I", "root") == "i"
    assert locale_lower("İ", "tr") == "i"


def test_numeric_collation_reads_a_digit_run_as_a_number():
    num, en = coll("icu_numeric"), coll("en_US_icu")
    assert sort_key("Item 9", num) < sort_key("Item 10", num) < sort_key("Item 100", num)
    assert sort_key("Item 10", en) < sort_key("Item 9", en)


def test_glibc_model_drops_punctuation_entirely():
    g = coll("glibc_en_US")
    assert sort_key("van der Berg", g) == sort_key("vanderBerg", g)
    # ICU shifts it instead of dropping it, so the two stay distinct.
    assert sort_key("van der Berg", coll("en_US_icu")) != sort_key(
        "vanderBerg", coll("en_US_icu")
    )


def test_accent_insensitive_collation_merges_but_case_sensitive_one_does_not():
    ai, en = coll("ai_ci"), coll("en_US_icu")
    assert sort_key("Ähtäri", ai) == sort_key("Ahtari", ai)
    assert sort_key("Ähtäri", en) != sort_key("Ahtari", en)


def test_every_uca_collation_agrees_the_two_jose_rows_are_one_value():
    a, b = normalization_pairs()[0]
    for c in COLLATIONS:
        if c.kind != "uca":
            continue
        assert sort_key(a.name, c) == sort_key(b.name, c), c.key_name


def test_byte_order_does_not_agree():
    a, b = normalization_pairs()[0]
    assert sort_key(a.name, coll("C")) != sort_key(b.name, coll("C"))


def test_fullwidth_a_folds_to_a_letter_not_to_a_symbol():
    en = coll("en_US_icu")
    assert sort_key("Ａ Corp", en) < sort_key("Zoe", en)


# -- verdicts ---------------------------------------------------------------


def test_byte_collations_are_stable_total():
    assert verdict(coll("C")) is Verdict.STABLE_TOTAL
    assert verdict(coll("UTF16_BIN")) is Verdict.STABLE_TOTAL


def test_no_linguistic_collation_is_total_over_this_corpus():
    # Because two rows are the same string, every Unicode-aware collation ties.
    assert all(verdict(c) is not Verdict.TOTAL for c in COLLATIONS if c.kind == "uca")


def test_only_the_nondeterministic_collation_merges():
    merging = [c.key_name for c in COLLATIONS if verdict(c) is Verdict.MERGING]
    assert merging == ["ai_ci"]


def test_deterministic_ties_do_not_change_row_counts():
    for c in COLLATIONS:
        if c.deterministic:
            assert distinct_count(c) == len({r.name for r in CORPUS})
            assert unique_violations(c) == []


def test_nondeterministic_ties_do():
    ai = coll("ai_ci")
    assert distinct_count(ai) < len({r.name for r in CORPUS})
    assert len(unique_violations(ai)) > 0


# -- drift ------------------------------------------------------------------


def test_two_collations_of_the_same_language_disagree():
    assert flips(coll("de_DIN"), coll("de_phonebook"))


def test_byte_orders_agree_inside_the_bmp_and_split_above_it():
    fl = bmp_flip()
    assert len(fl) == 1
    a, b = fl[0]
    tops = {max(ord(ch) for ch in a.name), max(ord(ch) for ch in b.name)}
    assert max(tops) > 0xFFFF and min(tops) <= 0xFFFF


def test_range_predicate_membership_is_collation_dependent():
    counts = range_counts()
    assert min(counts.values()) != max(counts.values())
    assert in_range("Åberg", coll("en_US_icu")) and not in_range("Åberg", coll("sv_SE"))


def test_turkish_lower_changes_rows():
    assert len(turkish_case_breakage()) >= 3


# -- pagination -------------------------------------------------------------


def test_offset_pagination_loses_rows_somewhere():
    tot = pagination_totals()
    assert tot["offset_lost"] > 0
    assert tot["offset_lost"] == tot["offset_dup"]  # every drop is another's repeat


def test_a_unique_tiebreak_makes_offset_pagination_exact():
    tot = pagination_totals()
    assert tot["tiebreak_lost"] == 0
    assert tot["tiebreak_dup"] == 0
    for c in COLLATIONS:
        for n in (4, 6, 8):
            assert offset_pagination(c, n, tiebreak=True).clean


def test_strict_keyset_paging_drops_the_rest_of_a_tie_group():
    ai = coll("ai_ci")
    audit = keyset_pagination(ai, 4, strict=True)
    assert audit.lost
    assert not audit.duplicated


def test_loose_keyset_paging_stalls_even_with_no_ties():
    audit = keyset_pagination(coll("C"), 4, strict=False)
    assert audit.stalled
    assert audit.duplicated


def test_offset_pagination_is_clean_when_the_sort_key_is_unique():
    # C has no ties, so re-planning between pages cannot move anything.
    assert offset_pagination(coll("C"), 6).clean


# -- ordering is a permutation, always -------------------------------------


@pytest.mark.parametrize("c", COLLATIONS, ids=[c.key_name for c in COLLATIONS])
def test_every_collation_returns_every_row_exactly_once(c):
    ids = sorted(r.id for r in order(CORPUS, c))
    assert ids == sorted(CORPUS_BY_ID)


@pytest.mark.parametrize("c", COLLATIONS, ids=[c.key_name for c in COLLATIONS])
def test_keys_are_totally_comparable(c):
    keys = [sort_key(r.name, c) for r in CORPUS]
    for a in keys:
        for b in keys:
            assert (a < b) or (b < a) or (a == b)


# -- the model against the host's libc -------------------------------------


def test_model_mostly_agrees_with_the_hosts_libc_where_available():
    rows = [r for r in libc_agreement() if r[4].startswith("compared") and r[3]]
    if not rows:
        pytest.skip("no matching libc locale installed")
    for key_name, loc, agree, total, _status in rows:
        assert agree / total > 0.9, f"{key_name} vs {loc}: {agree}/{total}"


# -- findings ---------------------------------------------------------------


def test_findings_are_severity_tagged_and_nonempty():
    fs = findings()
    assert fs
    assert {f.severity for f in fs} <= {"blocking", "silent", "advisory"}
    assert any(f.severity == "blocking" for f in fs)
    assert sum(1 for f in fs if f.severity == "silent") > sum(
        1 for f in fs if f.severity == "blocking"
    )
