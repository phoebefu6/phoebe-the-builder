"""Every headline number in the README, asserted.

These tests are deliberately exact.  If a node upgrade, a new PyYAML, a
different SQLite build or a git release changes a count, this suite fails
loudly rather than leaving the README quietly wrong.

They also pin the two `spec` readers - Go and Java - against their
published contracts, since neither toolchain is present to answer for
itself on this machine.
"""

from __future__ import annotations

import collections
import json
import sqlite3

import boolparse as B
import pytest

# --------------------------------------------------------------------------
# Environment - the claims hold for these builds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("binary", ["node", "git", "awk", "perl", "ruby", "jq", "bash"])
def test_live_readers_are_actually_present(binary):
    assert B.have(binary), f"`{binary}` is required; this reader is real, not modelled"


def test_only_go_and_java_are_spec():
    spec = sorted(r.name for r in B.READERS if r.source == "spec")
    assert spec == ["go_parsebool", "java_parsebool"]
    assert sum(1 for r in B.READERS if r.source == "live") == 14


def test_shape_of_the_experiment():
    assert len(B.CORPUS) == 45
    assert len(B.READERS) == 16
    assert len(B.CORPUS) * len(B.READERS) == 720


def test_corpus_strings_are_unique_and_single_line():
    texts = [s.text for s in B.CORPUS]
    assert len(set(texts)) == len(texts)
    # awk and the shell readers pass the corpus one line per string.
    for t in texts:
        assert "\n" not in t


# --------------------------------------------------------------------------
# The headline: no string means true
# --------------------------------------------------------------------------


def test_no_string_is_read_the_same_way_by_all_sixteen():
    assert B.unanimous() == []


def test_even_true_is_not_unanimous():
    s = next(x for x in B.CORPUS if x.text == "true")
    dissent = sorted(n for n, r in B.verdicts_for(s).items() if r.verdict != B.TRUE)
    assert dissent == ["js_loose_eq", "sqlite_where"]


def test_every_corpus_string_flips_sign_somewhere():
    assert len(B.sign_flips()) == len(B.CORPUS) == 45


def test_the_title_bug_false_is_true_in_six_readers():
    s = next(x for x in B.CORPUS if x.text == "false")
    trues = sorted(n for n, r in B.verdicts_for(s).items() if r.verdict == B.TRUE)
    assert trues == [
        "awk_field", "jq_truthy", "js_truthy", "perl_truthy", "py_truthy", "ruby_truthy",
    ]
    # Each is a truthiness reader: it never consulted a boolean table.
    assert all(not B.READERS_BY_NAME[n].can_refuse for n in trues)


# --------------------------------------------------------------------------
# Failure policy
# --------------------------------------------------------------------------


def test_ten_of_sixteen_readers_never_refuse_anything():
    never = B.never_refuse()
    assert len(never) == 10
    assert set(never) == {
        "py_truthy", "js_truthy", "js_loose_eq", "sqlite_where", "awk_field",
        "perl_truthy", "ruby_truthy", "jq_truthy", "bash_eq_true", "java_parsebool",
    }


def test_exact_refusal_counts():
    assert B.refusal_counts() == {
        "py_truthy": 0, "py_strtobool": 19, "json_strict": 34, "yaml11": 0,
        "yaml12": 0, "js_truthy": 0, "js_loose_eq": 0, "sqlite_where": 0,
        "git_bool": 22, "awk_field": 0, "perl_truthy": 0, "ruby_truthy": 0,
        "jq_truthy": 0, "bash_eq_true": 0, "go_parsebool": 33, "java_parsebool": 0,
    }


def test_only_four_readers_ever_refuse():
    refusers = sorted(n for n, c in B.refusal_counts().items() if c)
    assert refusers == ["git_bool", "go_parsebool", "json_strict", "py_strtobool"]


def test_yaml_neither_refuses_nor_misleads_it_defers():
    nb = B.notbool_counts()
    assert nb["yaml11"] == 25
    assert nb["yaml12"] == 35
    assert nb["json_strict"] == 6
    assert B.refusal_counts()["yaml11"] == 0
    wrong = collections.Counter(name for _, name, _ in B.silently_wrong())
    assert wrong["yaml11"] == 0 and wrong["yaml12"] == 0


def test_silently_wrong_totals():
    wrong = B.silently_wrong()
    assert len(wrong) == 161
    intent_cells = sum(1 for s in B.CORPUS if s.intent is not None) * len(B.READERS)
    assert intent_cells == 560
    assert sum(B.refusal_counts().values()) == 108
    assert sum(B.notbool_counts().values()) == 66


def test_permissive_readers_carry_the_silent_errors():
    """No reader is both permissive and safe: the two columns are exclusive."""
    wrong = collections.Counter(name for _, name, _ in B.silently_wrong())
    refuse = B.refusal_counts()
    for r in B.READERS:
        assert not (wrong[r.name] and refuse[r.name]), r.name


# --------------------------------------------------------------------------
# Individual readers
# --------------------------------------------------------------------------


def test_sqlite_reads_only_three_of_fortyfive_as_true():
    truthy = [s.text for s in B.CORPUS if B.verdict("sqlite_where", s).verdict == B.TRUE]
    assert truthy == ["1", "2", "-1"]


def test_sqlite_reads_the_string_true_as_false():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE feature (name TEXT, flag TEXT)")
    con.execute("INSERT INTO feature VALUES ('beta', 'true')")
    assert con.execute("SELECT count(*) FROM feature WHERE flag").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM feature WHERE flag = TRUE").fetchone()[0] == 0
    con.close()


def test_yaml_11_and_12_differ_on_exactly_the_norway_ten():
    differ = [
        s.text for s in B.CORPUS
        if (B.verdict("yaml11", s).verdict, B.verdict("yaml11", s).raw)
        != (B.verdict("yaml12", s).verdict, B.verdict("yaml12", s).raw)
    ]
    assert differ == ["yes", "Yes", "YES", "no", "No", "NO", "on", "ON", "off", "OFF"]
    assert len(differ) == 10


def test_pyyaml_does_not_implement_yaml_11_y_and_n():
    """YAML 1.1's type repository lists y/n; PyYAML leaves them as strings."""
    for text in ("y", "n"):
        s = next(x for x in B.CORPUS if x.text == text)
        assert B.verdict("yaml11", s).verdict == B.NOTBOOL


def test_git_reads_any_integer_as_a_boolean():
    for text, expected in (("2", B.TRUE), ("-1", B.TRUE), ("0", B.FALSE), ("", B.FALSE)):
        s = next(x for x in B.CORPUS if x.text == text)
        assert B.verdict("git_bool", s).verdict == expected, text
    # ... and refuses the ones it has no rule for.
    for text in ("t", "f", "y", "0.0", "null"):
        s = next(x for x in B.CORPUS if x.text == text)
        assert B.verdict("git_bool", s).verdict == B.REFUSED, text


def test_js_boolean_and_loose_equality_are_different_readers():
    disagree = sum(
        1 for s in B.CORPUS
        if B.verdict("js_truthy", s).verdict != B.verdict("js_loose_eq", s).verdict
    )
    assert disagree == 43
    one = next(x for x in B.CORPUS if x.text == "1")
    true_ = next(x for x in B.CORPUS if x.text == "true")
    assert B.verdict("js_loose_eq", one).verdict == B.TRUE
    assert B.verdict("js_loose_eq", true_).verdict == B.FALSE


def test_four_strings_are_truthy_and_loosely_equal_to_false():
    both = [
        s.text for s, lf in zip(B.CORPUS, B.js_loose_false(B.CORPUS))
        if lf and B.verdict("js_truthy", s).verdict == B.TRUE
    ]
    assert both == ["0", "0.0", "00", " "]


def test_awk_strnum_gives_two_answers_for_the_same_characters():
    for text in ("0", "00", "0.0", "0e0"):
        assert B.awk_program_literal(text) is True, text
        assert B.awk_input_field(text) is False, text
        assert B.awk_assigned_var(text) is False, text
    # A non-numeric string is true whichever way it arrives.
    assert B.awk_program_literal("false") is B.awk_input_field("false") is True


def test_perl_falsy_set_is_exactly_two_strings():
    falsy = [s.text for s in B.CORPUS if B.verdict("perl_truthy", s).verdict == B.FALSE]
    assert falsy == ["0", ""]


def test_ruby_and_jq_are_true_for_every_string():
    for name in ("ruby_truthy", "jq_truthy"):
        assert all(r.verdict == B.TRUE for r in B.grid()[name]), name


def test_json_accepts_json_whitespace_but_not_a_bom():
    conf = [s.text for s in B.CORPUS if B.verdict("json_strict", s).confident]
    assert conf == ["true", "false", "true ", " true", "true\r"]
    bom = next(x for x in B.CORPUS if x.text == "﻿true")
    assert B.verdict("json_strict", bom).verdict == B.REFUSED


def test_bash_eq_true_accepts_exactly_one_spelling():
    accepted = [s.text for s in B.CORPUS if B.verdict("bash_eq_true", s).verdict == B.TRUE]
    assert accepted == ["true"]


def test_strtobool_returns_ints_not_booleans():
    """A caller writing `if strtobool(v) is True` gets false for every input."""
    fn = B._strtobool()
    assert fn("yes") == 1 and fn("yes") is not True
    assert fn("no") == 0 and fn("no") is not False


# --------------------------------------------------------------------------
# The two spec readers, pinned to their published contracts
# --------------------------------------------------------------------------


def test_go_parsebool_table_is_the_documented_twelve():
    """strconv.ParseBool: "It accepts 1, t, T, TRUE, true, True, 0, f, F,
    FALSE, false, False. Any other value returns an error."
    """
    assert B.GO_TRUE == {"1", "t", "T", "TRUE", "true", "True"}
    assert B.GO_FALSE == {"0", "f", "F", "FALSE", "false", "False"}
    assert len(B.GO_TRUE | B.GO_FALSE) == 12
    # Nothing outside the table is accepted, including every alias git takes.
    for text in ("yes", "no", "on", "off", "y", "n", "tRuE", "enabled"):
        s = next(x for x in B.CORPUS if x.text == text)
        assert B.verdict("go_parsebool", s).verdict == B.REFUSED, text


def test_java_parsebool_is_true_iff_equalsignorecase_true():
    """Boolean.parseBoolean returns true iff the argument equalsIgnoreCase "true"."""
    for text in ("true", "True", "TRUE", "tRuE"):
        s = next(x for x in B.CORPUS if x.text == text)
        assert B.verdict("java_parsebool", s).verdict == B.TRUE, text
    # Everything else is false, and nothing raises - the dangerous part.
    for text in ("yes", "1", "on", "t", "T", "", "undefined"):
        s = next(x for x in B.CORPUS if x.text == text)
        assert B.verdict("java_parsebool", s).verdict == B.FALSE, text
    assert all(r.confident for r in B.grid()["java_parsebool"])


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_casefold_accepts_what_lower_refuses():
    """LATIN SMALL LETTER LONG S casefolds to `s`, and does not lowercase."""
    assert "FALſE".casefold() == "false"
    assert "FALſE".lower() != "false"
    acc = B.accepted_after("FALſE")
    assert acc["casefold"] is True and acc["lower"] is False
    acc = B.accepted_after("yeſ")
    assert acc["casefold"] is True and acc["lower"] is False


def test_fullwidth_needs_nfkc_before_any_casing_helps():
    acc = B.accepted_after("ＴＲＵＥ")
    assert acc["lower"] is False and acc["casefold"] is False
    assert acc["NFKC+casefold"] is True


def test_stripping_is_a_separate_decision_from_casing():
    for text in (" true", "true ", "true\r"):
        acc = B.accepted_after(text)
        assert acc["strip"] is True, text
        assert acc["lower"] is False, text


def test_turkish_lowercase_breaks_exactly_one_literal():
    words = sorted(w.upper() for w in B.EXTENDED_TABLE if w.isalpha())
    assert len(words) == 12
    tr = B.locale_lower(words, "tr")
    broken = [w for w in words if tr[w] not in B.EXTENDED_TABLE]
    assert broken == ["DISABLED"]
    assert tr["DISABLED"] == "dısabled"
    # The twelve core literals are immune only because none contains an I.
    assert all("i" not in w and "I" not in w for w in B.BASE_TABLE)


# --------------------------------------------------------------------------
# Cross-reader structure
# --------------------------------------------------------------------------


def test_exactly_two_reader_pairs_agree_on_everything():
    names = [r.name for r in B.READERS]
    identical = [
        (a, b) for i, a in enumerate(names) for b in names[i + 1:]
        if B.agreement(a, b)[0] == len(B.CORPUS)
    ]
    assert identical == [("py_truthy", "js_truthy"), ("ruby_truthy", "jq_truthy")]


def test_round_trip_is_asymmetric():
    rt = B.round_trip()
    assert rt[("py_truthy", "json_strict")] == 41
    assert rt[("json_strict", "py_truthy")] == 1
    assert rt[("py_truthy", "json_strict")] != rt[("json_strict", "py_truthy")]


def test_only_eight_ordered_pairs_lose_nothing():
    rt = B.round_trip()
    names = [r.name for r in B.READERS]
    clean = [(a, b) for a in names for b in names if a != b and rt[(a, b)] == 0]
    assert len(clean) == 8


def test_migration_towards_strictness_is_the_safe_direction():
    """From strict to permissive loses little; the reverse loses almost all."""
    rt = B.round_trip()
    for strict, loose in (("json_strict", "py_truthy"), ("go_parsebool", "ruby_truthy")):
        assert rt[(strict, loose)] < rt[(loose, strict)]


# --------------------------------------------------------------------------
# The grid itself
# --------------------------------------------------------------------------


def test_every_cell_has_a_legal_verdict():
    for name, readings in B.grid().items():
        assert len(readings) == len(B.CORPUS), name
        for r in readings:
            assert r.verdict in B.VERDICTS, (name, r)


def test_show_makes_every_invisible_character_visible():
    assert B.show("") == "''"
    assert B.show("true\r") == "true<CR>"
    assert B.show("﻿true") == "<BOM>true"
    assert B.show(" ") == "␣"


def test_grid_is_deterministic():
    """Two runs of the subprocess readers give byte-identical answers."""
    first = {k: [r.verdict for r in v] for k, v in B.grid().items()}
    B.grid.cache_clear()
    B._node_batch.cache_clear()
    second = {k: [r.verdict for r in v] for k, v in B.grid().items()}
    assert first == second


def test_readings_serialise_for_the_streamlit_app():
    payload = {
        name: [{"verdict": r.verdict, "raw": r.raw} for r in readings]
        for name, readings in B.grid().items()
    }
    assert json.loads(json.dumps(payload)) == payload
