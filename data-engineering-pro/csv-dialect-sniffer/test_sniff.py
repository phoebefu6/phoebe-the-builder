"""Tests for sniff.py. Standard library only; no pytest required.

    python3 test_sniff.py

The tests assert *structural* facts - this file is contested, this encoding is
ruled out, this setting is untested - rather than the specific dialect a given
Python version's csv.Sniffer happens to return. Sniffer's answers are recorded in
evidence.py, where they are the finding, not the expectation.
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable, List, Tuple

import sniff

PASS = 0
FAIL: List[Tuple[str, str]] = []


def check(name: str, fn: Callable[[], None]) -> None:
    global PASS
    try:
        fn()
    except AssertionError as exc:
        FAIL.append((name, str(exc) or "assertion failed"))
    except Exception:
        FAIL.append((name, traceback.format_exc(limit=1).strip()))
    else:
        PASS += 1


FILES = sniff.sample_files()
T = {name: raw.decode("utf-8", errors="replace") for name, raw in FILES.items()}


# --------------------------------------------------------------------------- #
# parsing and shapes
# --------------------------------------------------------------------------- #


def t_parse_quoted_delimiter() -> None:
    rows = sniff.parse('a,b\n1,"x,y"\n', ",")
    assert rows[1] == ["1", "x,y"], rows[1]


def t_parse_no_quoting() -> None:
    rows = sniff.parse('a,b\n1,"x,y"\n', ",", quotechar=None)
    assert rows[1] == ["1", '"x', 'y"'], rows[1]


def t_shape_ragged_not_viable() -> None:
    s = sniff.shape_of("a,b,c\n1,2\n3,4,5\n", ",")
    assert not s.viable
    assert "ragged" in s.reason


def t_shape_single_field_not_viable() -> None:
    s = sniff.shape_of("a;b;c\n1;2;3\n", ",")
    assert not s.viable
    assert s.modal == 1


def t_shape_consistency_is_a_fraction() -> None:
    s = sniff.shape_of("a,b\n1,2\n3,4,5\n", ",")
    assert 0.0 < s.consistency < 1.0, s.consistency


def t_wrong_quotechar_merges_records_silently() -> None:
    # A leading apostrophe read as a quotechar opens a field that runs to the
    # next apostrophe, absorbing a record terminator. The width is unchanged.
    good = sniff.shape_of(T["dutch.csv"], ",", '"')
    bad = sniff.shape_of(T["dutch.csv"], ",", "'")
    assert good.modal == bad.modal == 3, (good.modal, bad.modal)
    assert bad.records == good.records - 1, (bad.records, good.records)
    assert bad.viable, "and it is still perfectly consistent"
    assert bad.fields_with_newline == 1


def t_record_count_contest_is_reported_as_contested() -> None:
    v = sniff.classify_delimiter(T["dutch.csv"])
    assert v.status == "contested", v.status
    assert v.column_counts == [3], v.column_counts
    assert "record(s) long" in v.reason, v.reason


def t_tie_break_prefers_the_parse_that_loses_no_rows() -> None:
    v = sniff.classify_delimiter(T["dutch.csv"])
    assert v.preferred is not None and v.preferred.quotechar == '"'
    assert v.preferred.records == 4


def t_truncated_tail_excluded() -> None:
    text = "a,b,c\n1,2,3\n4,5"
    s = sniff.shape_of(text, ",")
    assert s.truncated_tail
    assert s.viable and s.modal == 3, (s.viable, s.modal)


def t_truncated_tail_only_without_terminator() -> None:
    # A complete file with no trailing newline must keep its last record.
    s = sniff.shape_of("a,b,c\n1,2,3\n4,5,6", ",")
    assert not s.truncated_tail
    assert s.records == 3, s.records


def t_truncated_tail_not_applied_to_two_ragged() -> None:
    s = sniff.shape_of("a,b,c\n1,2\n4,5", ",")
    assert not s.truncated_tail
    assert not s.viable


# --------------------------------------------------------------------------- #
# the central claim: contested files exist
# --------------------------------------------------------------------------- #


def t_sensor_is_contested() -> None:
    v = sniff.classify_delimiter(T["sensor.csv"])
    assert v.status == "contested", v.status
    assert v.column_counts == [3, 4], v.column_counts


def t_both_contested_parses_are_perfectly_consistent() -> None:
    for d in (",", ";"):
        s = sniff.shape_of(T["sensor.csv"], d)
        assert s.viable and s.consistency == 1.0 and not s.ragged, (d, s.reason)


def t_header_row_resolves_the_contest() -> None:
    v = sniff.classify_delimiter(T["sales_eu.csv"])
    assert v.status == "unambiguous", v.status
    assert v.preferred is not None and v.preferred.delimiter == ";"


def t_contested_is_not_repaired_by_a_wider_candidate_list() -> None:
    # Offering more delimiters cannot make an ambiguous file unambiguous.
    v = sniff.classify_delimiter(T["sensor.csv"], delimiters=(",", ";", "\t", "|", ":", " ", "^"))
    assert v.status == "contested", v.status


def t_undetermined_when_nothing_parses() -> None:
    v = sniff.classify_delimiter("one\ntwo three\nfour\n")
    assert v.status == "undetermined", v.status
    assert v.preferred is None


def t_untested_quotechar_is_reported() -> None:
    v = sniff.classify_delimiter(T["sales_eu.csv"])
    assert v.untested, "a file with no quoted field leaves quotechar untested"


def t_quoted_file_has_no_untested_quotechar() -> None:
    v = sniff.classify_delimiter(T["quoted.csv"])
    assert not v.untested, v.untested


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #


def t_cp1252_rules_out_utf8() -> None:
    r = sniff.probe_encoding(FILES["cp1252.csv"])
    assert "utf-8" in r.ruled_out


def t_latin1_never_fails() -> None:
    for raw in FILES.values():
        r = sniff.probe_encoding(raw)
        assert r.decodes["latin-1"], "latin-1 must decode every byte string"


def t_latin1_success_is_flagged_as_no_evidence() -> None:
    r = sniff.probe_encoding(FILES["cp1252.csv"])
    assert any(e == "latin-1" for e, _ in r.not_evidence)
    assert "latin-1" not in r.plausible


def t_c1_test_disqualifies_latin1_for_cp1252_bytes() -> None:
    r = sniff.probe_encoding(FILES["cp1252.csv"])
    assert r.verdict == "cp1252", r.verdict
    assert r.plausible == ["cp1252"], r.plausible


def t_c1_test_is_half_blind_by_construction() -> None:
    # U+00E0-U+00FF encode to a continuation byte outside the C1 range, so a
    # lower-case-only mojibake file passes the test while still being wrong.
    lower = "id,name\r\n1,ü\r\n".encode("utf-8")
    r = sniff.probe_encoding(lower)
    assert "latin-1" in r.decodes and r.decodes["latin-1"]
    assert not any("C1 control" in why for e, why in r.not_evidence if e == "latin-1")


def t_c1_test_catches_upper_case_mojibake() -> None:
    upper = "id,name\r\n1,Ü\r\n".encode("utf-8")
    r = sniff.probe_encoding(upper)
    assert any("C1 control" in why for e, why in r.not_evidence if e == "latin-1")


def t_bom_is_not_a_guess() -> None:
    r = sniff.probe_encoding(FILES["bom.csv"])
    assert r.bom == "utf-8-sig" and r.verdict == "utf-8-sig"


def t_bom_survives_plain_utf8_decode() -> None:
    rows = sniff.parse(FILES["bom.csv"].decode("utf-8"), ",")
    assert rows[0][0] != "id"
    assert rows[0][0].lstrip("﻿") == "id"
    assert len(rows[0][0]) == 3


def t_utf16_without_bom_is_not_evidence() -> None:
    r = sniff.probe_encoding(FILES["sensor.csv"])
    if r.decodes.get("utf-16"):
        assert "utf-16" not in r.plausible


def t_distinct_texts_counts_disagreement() -> None:
    r = sniff.probe_encoding(FILES["utf8_umlaut.csv"])
    assert r.distinct_texts >= 3, r.distinct_texts
    assert r.verdict == "utf-8"


def t_undecodable_reports_undecodable() -> None:
    r = sniff.probe_encoding(b"\x81\x8d\x8f", encodings=("utf-8", "cp1252"))
    assert r.verdict == "undecodable", r.verdict


# --------------------------------------------------------------------------- #
# terminators
# --------------------------------------------------------------------------- #


def t_bare_cr_is_detected() -> None:
    t = sniff.probe_terminator(T["mac.csv"], ",")
    assert t.verdict == "\\r"
    assert t.naive_lines == 1 and t.records == 3, (t.naive_lines, t.records)


def t_newline_inside_field_is_not_a_terminator() -> None:
    outside, inside = sniff.scan_terminators(T["quoted.csv"], '"')
    assert sum(inside.values()) == 1, inside
    assert outside["\\r\\n"] == 3, outside


def t_terminator_scan_needs_the_quotechar() -> None:
    # With quoting switched off the embedded newline is counted as a record
    # terminator, which is the circularity the module reports rather than hides.
    _, inside_none = sniff.scan_terminators(T["quoted.csv"], None)
    assert sum(inside_none.values()) == 0


def t_naive_line_count_is_wrong_in_both_directions() -> None:
    a = sniff.probe_terminator(T["quoted.csv"], ",")
    b = sniff.probe_terminator(T["mac.csv"], ",")
    assert a.naive_lines > a.records
    assert b.naive_lines < b.records


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #


def t_text_over_numeric_is_a_header() -> None:
    rows = sniff.parse(T["sales_eu.csv"], ";")
    h = sniff.classify_header([r for r in rows if r], T["sales_eu.csv"])
    assert h.status == "header" and h.basis == "text_over_nontext"


def t_numeric_header_is_undetermined_not_absent() -> None:
    rows = sniff.parse(T["years.csv"], ",")
    h = sniff.classify_header([r for r in rows if r], T["years.csv"])
    assert h.status == "undetermined", h.status
    assert h.basis == "row0_matches_body"


def t_all_text_is_undetermined() -> None:
    rows = sniff.parse(T["alltext.csv"], ",")
    h = sniff.classify_header([r for r in rows if r], T["alltext.csv"])
    assert h.status == "undetermined" and h.basis == "all_text"


def t_sniffer_answers_the_undecidable_cases() -> None:
    for name in ("years.csv", "alltext.csv"):
        rows = [r for r in sniff.parse(T[name], ",") if r]
        h = sniff.classify_header(rows, T[name])
        assert h.status == "undetermined"
        assert h.sniffer in (True, False), "Sniffer has no third answer"


def t_no_status_claims_no_header() -> None:
    # There is no decidable evidence for the absence of a header, so the module
    # must never emit that verdict.
    for name, text in T.items():
        v = sniff.classify_delimiter(text)
        if not v.preferred:
            continue
        rows = [r for r in sniff.parse(text, v.preferred.delimiter) if r]
        h = sniff.classify_header(rows, text)
        assert h.status in ("header", "undetermined"), (name, h.status)


def t_cell_type_separates_decimal_conventions() -> None:
    assert sniff.cell_type("1.50") == "float"
    assert sniff.cell_type("1,50") == "float,"
    assert sniff.cell_type("2024-01-01") == "date"
    assert sniff.cell_type("12") == "int"
    assert sniff.cell_type("") == "empty"
    assert sniff.cell_type("north") == "text"


# --------------------------------------------------------------------------- #
# sample size and the whole audit
# --------------------------------------------------------------------------- #


def t_prefix_sniff_disagrees_with_full_file() -> None:
    picks = {p for _, p, _ in sniff.sample_sensitivity(T["late.csv"], sizes=(64, 128, 256))}
    assert len(picks) > 1, picks


def t_prefix_sniff_can_return_a_letter() -> None:
    pick = sniff.sniffer_says(T["late.csv"][:128])
    assert pick not in sniff.DELIMITERS, pick


def t_audit_never_raises_on_any_sample() -> None:
    for name, raw in FILES.items():
        a = sniff.audit(raw, name)
        assert a.name == name and a.size == len(raw)


def t_audit_marks_the_undecidable_files() -> None:
    undecided = {n for n, raw in FILES.items() if not sniff.audit(raw, n).decided}
    assert undecided == {"sensor.csv", "years.csv", "alltext.csv", "dutch.csv"}, undecided


def t_audit_notes_are_never_empty_for_contested() -> None:
    a = sniff.audit(FILES["sensor.csv"], "sensor.csv")
    assert any(n.startswith("CONTESTED") for n in a.notes)


def t_audit_handles_empty_input() -> None:
    a = sniff.audit(b"", "empty")
    assert a.delimiter.status == "undetermined"


def t_audit_handles_one_line() -> None:
    a = sniff.audit(b"a,b,c\n", "oneline")
    assert a.header is not None and a.header.status == "undetermined"


TESTS = [(k[2:], v) for k, v in sorted(globals().items())
         if k.startswith("t_") and callable(v)]


def main() -> int:
    for name, fn in TESTS:
        check(name, fn)
    print("{0} passed, {1} failed, of {2}".format(PASS, len(FAIL), len(TESTS)))
    for name, err in FAIL:
        print("\nFAIL {0}\n  {1}".format(name, err))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
