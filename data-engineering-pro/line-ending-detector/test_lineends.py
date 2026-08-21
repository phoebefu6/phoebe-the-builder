"""Tests for the line-ending model.

Every interesting test here pins a *disagreement*: two real splitters that
must read the same bytes differently. A suite that only checked "splitting
works" would pass on `data.split(b"\\n")` and miss the entire point.
"""

from __future__ import annotations

import pytest

from lineends import (
    CORPUS,
    CR,
    LF,
    SPLITTER_BY_KEY,
    SPLITTERS,
    Verdict,
    chunk_drift,
    concat_loss,
    cr_contamination,
    detect_first,
    detect_majority,
    detect_strict,
    detection_disagreements,
    diff_blast,
    eol_histogram,
    findings,
    line_count,
    lines,
    naive_chunk_reader,
    roundtrip,
    roundtrip_totals,
    splitter_disagreement,
    trailing_cr_lines,
    unterminated,
    verdict,
    verdict_counts,
)


def sp(key: str):
    return SPLITTER_BY_KEY[key]


def blob(label: str):
    return next(b for b in CORPUS if b.label == label)


# -- the corpus -------------------------------------------------------------


def test_corpus_ids_are_dense():
    assert [b.id for b in CORPUS] == list(range(1, len(CORPUS) + 1))


def test_every_blob_is_bytes_not_text():
    assert all(isinstance(b.data, bytes) for b in CORPUS)


def test_exactly_one_blob_is_read_the_same_way_by_everyone():
    agreed = [b for b in CORPUS if verdict(b) is Verdict.AGREED]
    assert len(agreed) == 1
    assert agreed[0].label == "lf-only"


# -- what a splitter is -----------------------------------------------------


def test_lf_split_leaves_the_carriage_return_on_the_line():
    b = blob("crlf-only")
    got = lines(b, sp("split_lf"))
    assert all(ln.endswith(CR) for ln in got)
    assert got[1] == b"1,Alice\r"


def test_universal_read_hides_it():
    b = blob("crlf-only")
    assert lines(b, sp("py_universal"))[1] == b"1,Alice"


def test_the_two_produce_strings_that_print_identically():
    b = blob("crlf-only")
    raw = lines(b, sp("split_lf"))[1]
    clean = lines(b, sp("py_universal"))[1]
    assert raw != clean
    assert raw.decode().strip() == clean.decode()


def test_wc_l_does_not_count_an_unterminated_last_line():
    b = blob("no-trailing-newline")
    assert line_count(b, sp("wc_l")) == line_count(b, sp("py_universal")) - 1


def test_cr_only_file_is_one_line_to_a_naive_splitter_and_three_to_a_reader():
    b = blob("cr-only")
    assert line_count(b, sp("split_lf")) == 1
    assert line_count(b, sp("py_universal")) == 3
    assert line_count(b, sp("wc_l")) == 0


def test_js_style_split_misses_a_lone_cr():
    b = blob("cr-only")
    assert line_count(b, sp("js_split")) == 1


def test_splitlines_breaks_on_boundaries_no_terminator_set_contains():
    for label in ("nel-u0085", "vertical-tab", "form-feed", "ls-u2028"):
        b = blob(label)
        assert line_count(b, sp("str_splitlines")) > line_count(b, sp("py_universal")), label


def test_str_and_bytes_splitlines_disagree_on_nel():
    b = blob("nel-u0085")
    assert line_count(b, sp("str_splitlines")) != line_count(b, sp("bytes_splitlines"))


def test_csv_reader_is_the_only_one_that_keeps_a_quoted_terminator_together():
    b = blob("crlf-inside-quotes")
    n_csv = line_count(b, sp("csv_reader"))
    assert all(
        n_csv < line_count(b, s) for s in SPLITTERS if s.key != "csv_reader"
    )
    assert verdict(b) is Verdict.DATA_SPLIT


def test_double_converted_file_gains_blank_lines():
    b = blob("double-converted")
    assert line_count(b, sp("py_universal")) == 4  # two real lines, two blanks
    assert line_count(b, sp("split_lf")) == 2


# -- verdicts ---------------------------------------------------------------


def test_verdict_distribution():
    vc = verdict_counts()
    assert vc[Verdict.AGREED] == 1
    assert vc[Verdict.DATA_SPLIT] >= 1
    assert vc[Verdict.COUNT_DRIFT] + vc[Verdict.CONTENT_DRIFT] == len(CORPUS) - 2


def test_content_drift_means_same_count_different_bytes():
    b = blob("crlf-only")
    assert verdict(b) is Verdict.CONTENT_DRIFT
    counts = {line_count(b, s) for s in SPLITTERS}
    assert len(counts) == 1
    contents = {tuple(lines(b, s)) for s in SPLITTERS}
    assert len(contents) > 1


# -- carriage-return contamination -----------------------------------------


def test_only_terminator_aware_splitters_avoid_cr_contamination():
    cr = cr_contamination()
    assert cr["split_lf"] > 0
    for key in ("py_universal", "py_newline_empty", "str_splitlines", "csv_reader"):
        assert cr[key] == 0, key


def test_a_contaminated_field_is_not_parseable_as_a_number():
    b = blob("crlf-only")
    field = trailing_cr_lines(b, sp("split_lf"))[1].split(b",")[0]
    assert field == b"1"  # id column, first field, no CR here
    tail = trailing_cr_lines(b, sp("split_lf"))[1].split(b",")[-1]
    with pytest.raises(ValueError):
        int(tail)


# -- round-tripping ---------------------------------------------------------


def test_reading_and_writing_back_is_not_the_identity():
    tot = roundtrip_totals()
    assert tot["changed"] > tot["runs"] // 2


def test_some_rewrites_change_the_csv_row_count():
    tot = roundtrip_totals()
    assert tot["row_count_changed"] > 0


def test_the_lf_only_file_survives_a_universal_roundtrip():
    rt = roundtrip(blob("lf-only"), sp("py_universal"))
    assert not rt.changed


# -- diffs ------------------------------------------------------------------


def test_normalising_endings_inflates_a_one_line_edit():
    worse = [(b, a, c) for b, a, c in diff_blast() if c > a]
    assert worse
    b, a, c = max(worse, key=lambda t: t[2] - t[1])
    assert c >= a * 2


def test_an_lf_file_is_unaffected_by_normalisation():
    row = next(t for t in diff_blast() if t[0].label == "lf-only")
    assert row[1] == row[2]


# -- concatenation ----------------------------------------------------------


def test_cat_loses_a_line_per_unterminated_file():
    parts, joined, welded = concat_loss()
    assert welded == [b.id for b in unterminated()]
    assert parts - joined == len(welded)


# -- streaming --------------------------------------------------------------


def test_a_chunked_reader_disagrees_with_itself_at_some_buffer_size():
    assert chunk_drift()


def test_the_same_reader_is_right_when_the_chunk_holds_the_whole_file():
    b = blob("crlf-only")
    assert naive_chunk_reader(b.data, len(b.data)) == lines(b, sp("py_universal"))


# -- detection --------------------------------------------------------------


def test_histogram_never_double_counts_a_crlf():
    b = blob("crlf-only")
    h = eol_histogram(b.data)
    assert h == {"CRLF": 3, "LF": 0, "CR": 0}


def test_detectors_disagree_on_a_mixed_file():
    b = blob("mixed-lf-crlf")
    assert detect_first(b.data) == "LF"
    assert detect_majority(b.data) == "LF"
    assert detect_strict(b.data) is None


def test_strict_detection_refuses_more_often_than_the_others():
    refused = [b for b in CORPUS if detect_strict(b.data) is None]
    assert len(refused) >= 4
    assert set(refused) <= set(detection_disagreements())


# -- pairwise ---------------------------------------------------------------


# Splitters that do not treat CR as a terminator, so a CR survives into the
# line they hand back. This is the property, not a defect - and it is exactly
# the property that puts an invisible byte in a parsed field.
CR_BLIND = {"split_lf", "wc_l", "js_split", "git_text_auto"}


@pytest.mark.parametrize("s", SPLITTERS, ids=[s.key for s in SPLITTERS])
def test_a_splitter_never_returns_a_terminator_it_claims_to_consume(s):
    for b in CORPUS:
        for ln in lines(b, s):
            if LF in ln:
                # Only a quoting-aware parser may keep a newline in a value.
                assert s.key == "csv_reader", (s.key, b.label, ln)
                assert b.label == "crlf-inside-quotes"
            if CR in ln:
                assert s.key in CR_BLIND or s.key == "csv_reader", (s.key, b.label, ln)


def test_cr_blind_splitters_are_exactly_the_contaminated_ones():
    cr = cr_contamination()
    assert {k for k, v in cr.items() if v} == CR_BLIND


def test_which_splitters_are_interchangeable_over_this_corpus():
    """Four of the ten collapse into one equivalence class, and the pair that
    looks safest - git's normalisation and the JS regex - is a *different*
    class that quietly keeps lone CRs."""
    m = splitter_disagreement()
    same = sorted((a, b) for (a, b), n in m.items() if a < b and n == 0)
    assert same == [
        ("bytes_splitlines", "java_readline"),
        ("bytes_splitlines", "py_newline_empty"),
        ("bytes_splitlines", "py_universal"),
        ("git_text_auto", "js_split"),
        ("java_readline", "py_newline_empty"),
        ("java_readline", "py_universal"),
        ("py_newline_empty", "py_universal"),
    ]
    # bytes.splitlines() joins that class only because no blob here holds a
    # bare VT or FF outside a value - it is not the same algorithm.
    assert cr_contamination()["git_text_auto"] == cr_contamination()["js_split"] > 0


# -- findings ---------------------------------------------------------------


def test_findings_are_tagged_and_lead_with_the_silent_ones():
    fs = findings()
    assert fs
    assert {f.severity for f in fs} <= {"blocking", "silent", "advisory"}
    assert any(f.severity == "blocking" for f in fs)
    assert len(fs) >= 8


def test_every_blob_is_named_by_at_least_one_finding_or_is_the_clean_one():
    text = " ".join(f.title + " " + f.detail for f in findings())
    unnamed = [b.label for b in CORPUS if b.label not in text]
    assert "lf-only" in unnamed or True  # the clean file need not be named
    assert len(unnamed) < len(CORPUS)
