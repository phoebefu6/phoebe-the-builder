"""Tests for markdown-tabler.

The round-trip tests are the ones that matter: they render a table, hand it to a
real GFM parser, and assert that the audit predicted every cell that came back
different. They skip cleanly when markdown-it-py is absent.

Run:  python3 test_tabler.py     (no pytest required)
      pytest test_tabler.py      (also works)
"""

from __future__ import annotations

import re
import sys
import traceback
from typing import Callable, List, Optional, Tuple

import tabler as T

try:
    from evidence import HAVE_PARSER, parse_back
except Exception:  # pragma: no cover
    HAVE_PARSER = False

    def parse_back(*a: object, **k: object) -> List[List[str]]:  # type: ignore[misc]
        raise RuntimeError("no parser")


# --------------------------------------------------------------------------
# display_width
# --------------------------------------------------------------------------


def test_width_ascii() -> None:
    assert T.display_width("Ana Ruiz") == 8 == len("Ana Ruiz")


def test_width_cjk_is_two_columns_per_glyph() -> None:
    assert len("陈伟") == 2
    assert T.display_width("陈伟") == 4


def test_width_combining_mark_is_zero() -> None:
    combining = "cafe\u0301"  # e + combining acute
    assert len(combining) == 5
    assert T.display_width(combining) == 4


def test_width_zwj_sequence() -> None:
    # family emoji: several code points, one grapheme
    fam = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
    assert len(fam) == 5
    assert T.display_width(fam) == 6  # three wide glyphs, ZWJ free


def test_width_ambiguous_is_a_parameter() -> None:
    assert T.display_width("±") == 1
    assert T.display_width("±", ambiguous_wide=True) == 2


def test_pad_uses_columns_not_codepoints() -> None:
    assert T.pad("陈伟", 8) == "陈伟" + " " * 4
    assert T.pad("x", 5, "right") == "    x"
    assert T.pad("x", 5, "center") == "  x  "


# --------------------------------------------------------------------------
# escaping
# --------------------------------------------------------------------------


def test_pipe_escaped_with_backslash() -> None:
    assert T.escape_cell("a|b") == "a\\|b"


def test_already_escaped_pipe_not_double_escaped() -> None:
    assert T.escape_cell("a\\|b") == "a\\|b"


def test_newline_policies() -> None:
    v = "one\ntwo"
    assert T.escape_cell(v, newline="br") == "one<br>two"
    assert T.escape_cell(v, newline="space") == "one two"
    assert T.escape_cell(v, newline="strip") == "one"


def test_crlf_handled() -> None:
    assert T.escape_cell("a\r\nb", newline="br") == "a<br>b"


def test_bad_newline_policy_raises() -> None:
    try:
        T.escape_cell("x", newline="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_escape_emphasis_covers_whole_delimiter_run() -> None:
    # __dunder__ is strong emphasis; escaping one underscore per side leaves
    # _dunder_, which is still emphasis.
    assert T.escape_cell("__dunder__", escape_emphasis=True) == "\\_\\_dunder\\_\\_"


def test_escape_emphasis_leaves_safe_identifiers_alone() -> None:
    assert T.escape_cell("snake_case_ok", escape_emphasis=True) == "snake_case_ok"
    assert T.escape_cell("a_b", escape_emphasis=True) == "a_b"


def test_emphasis_inside_code_span_ignored() -> None:
    # A star inside backticks is literal; escaping it would show the backslash.
    assert T.escape_cell("`2*3*4`", escape_emphasis=True) == "`2*3*4`"


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


def _codes(text: str, **kw: object) -> List[str]:
    return [f.code for f in T.audit_cell(text, 0, "c", **kw)]  # type: ignore[arg-type]


def test_audit_entity_only_flagged_inside_code_span() -> None:
    assert "ENTITY_IN_CODE" in _codes("`a&#124;b`")
    assert "ENTITY_IN_CODE" not in _codes("a&#124;b")


def test_audit_edge_space_is_loss() -> None:
    f = [x for x in T.audit_cell("  indent", 0, "c") if x.code == "EDGE_SPACE"]
    assert f and f[0].severity == T.LOSS


def test_audit_newline_severity_depends_on_policy() -> None:
    br = [f for f in T.audit_cell("a\nb", 0, "c", newline="br") if f.code == "NEWLINE"]
    sp = [f for f in T.audit_cell("a\nb", 0, "c", newline="space") if f.code == "NEWLINE"]
    assert br[0].severity == T.PORTABILITY
    assert sp[0].severity == T.LOSS


def test_audit_emphasis_downgrades_when_escaped() -> None:
    plain = [f for f in T.audit_cell("_x_", 0, "c") if f.code == "EMPHASIS"]
    esc = [
        f for f in T.audit_cell("_x_", 0, "c", escape_emphasis=True) if f.code == "EMPHASIS"
    ]
    assert plain[0].severity == T.LOSS
    assert esc[0].severity == T.COSMETIC


def test_audit_trailing_backslash() -> None:
    assert "BACKSLASH_END" in _codes("C:\\path\\")
    assert "BACKSLASH_END" not in _codes("C:/path/")


def test_audit_clean_cell_is_silent() -> None:
    assert _codes("set -e") == []


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------


def test_ragged_extra_is_loss_and_names_the_dropped_value() -> None:
    res = T.render([["1", "2", "GONE"]], ["a", "b"])
    hit = [f for f in res.findings if f.code == "RAGGED_EXTRA"]
    assert hit and hit[0].severity == T.LOSS
    assert "GONE" in hit[0].detail
    assert res.loses_data


def test_ragged_short_pads_and_reports() -> None:
    res = T.render([["1"]], ["a", "b"])
    assert [f.code for f in res.findings] == ["RAGGED_SHORT"]
    assert res.markdown.splitlines()[-1].count("|") == 3


def test_alignment_inferred_from_content() -> None:
    res = T.render([["a", "1"], ["b", "2,300"]], ["label", "n"])
    assert res.alignment == ["left", "right"]


def test_alignment_mixed_column_falls_back_to_left() -> None:
    res = T.render([["1"], ["n/a"]], ["n"])
    assert res.alignment == ["left"]


def test_alignment_explicit_wins_and_is_padded() -> None:
    res = T.render([["a", "1"]], ["x", "y"], align=["center"])
    assert res.alignment == ["center", "left"]


def test_delimiter_row_carries_the_colons() -> None:
    res = T.render([["a", "1"]], ["label", "n"])
    delim = res.markdown.splitlines()[1]
    assert ":---" in delim  # left
    assert delim.rstrip().endswith(":|") or "---:" in delim  # right


def test_every_row_has_the_same_pipe_count() -> None:
    res = T.sample_table()
    counts = {line.count("|") - line.count("\\|") for line in res.markdown.splitlines()}
    assert len(counts) == 1, counts


def test_none_and_nan_become_empty() -> None:
    res = T.render([[None, float("nan"), 1.50]], ["a", "b", "c"])
    assert "| 1.5 " in res.markdown
    body = res.markdown.splitlines()[-1]
    assert body.startswith("|     |") or body.startswith("|   |")


def test_bools_lowercase() -> None:
    res = T.render([[True, False]], ["a", "b"])
    assert "true" in res.markdown and "false" in res.markdown


def test_headers_inferred_when_absent() -> None:
    res = T.render([["x", "y"]])
    assert res.markdown.splitlines()[0].startswith("| col0 | col1 |")


def test_zero_rows_without_headers_raises() -> None:
    try:
        T.render([])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_render_never_raises_on_hostile_input() -> None:
    hostile = [
        ["|||", "`", "\n\n\n", "\\", "*"],
        ["", "  ", "---", ":---:", "&#124;"],
        ["\u200d", "\U0001f6a6" * 3, "a" * 200, "_", "__"],
    ]
    res = T.render(hostile, ["a", "b", "c", "d", "e"])
    assert res.markdown.count("\n") == 4  # header + delimiter + 3 rows


def test_pad_cells_off_is_still_valid() -> None:
    res = T.render([["a", "1"]], ["x", "y"], pad_cells=False)
    lines = res.markdown.splitlines()
    assert lines[0] == "| x | y |"
    assert set(lines[1].replace(" ", "").replace("|", "")) <= set(":-")


# --------------------------------------------------------------------------
# round trip against a real parser
# --------------------------------------------------------------------------


def test_roundtrip_audit_predicts_every_changed_cell() -> None:
    if not HAVE_PARSER:
        return
    res = T.sample_table()
    back = parse_back(res.markdown)
    head, body = back[0], back[1:]
    predicted = {(f.row, f.column) for f in res.findings}
    for i, row in enumerate(body):
        for j, got in enumerate(row):
            want = T._stringify(T.SAMPLE_ROWS[i][j], "{:g}", "")
            if got != want:
                assert (i, head[j]) in predicted, (i, head[j], want, got)


def test_roundtrip_clean_table_is_byte_identical() -> None:
    if not HAVE_PARSER:
        return
    rows = [["etl/load.py", "12"], ["api/auth.py", "3"]]
    res = T.render(rows, ["file", "hits"])
    assert not res.findings
    body = parse_back(res.markdown)[1:]
    assert body == rows


def test_roundtrip_escaped_pipe_survives_a_code_span() -> None:
    if not HAVE_PARSER:
        return
    res = T.render([[T.escape_cell("`a|b`")]], ["c"])
    assert parse_back(res.markdown)[1][0] == "a|b"


def test_roundtrip_escape_emphasis_recovers_the_identifier() -> None:
    if not HAVE_PARSER:
        return
    for value in ("_id_field_", "2*3*4", "__dunder__", "*star*"):
        res = T.render([[value]], ["c"], escape_emphasis=True)
        assert parse_back(res.markdown)[1][0] == value, value


def test_roundtrip_column_count_is_the_header_count() -> None:
    if not HAVE_PARSER:
        return
    res = T.sample_table()
    for row in parse_back(res.markdown):
        assert len(row) == len(T.SAMPLE_HEADERS)


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


def main() -> int:
    tests: List[Tuple[str, Callable[[], None]]] = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS  %s" % name)
        except Exception:
            failed += 1
            print("FAIL  %s" % name)
            traceback.print_exc()
    print("\n%d passed, %d failed, %d total%s" % (
        len(tests) - failed,
        failed,
        len(tests),
        "" if HAVE_PARSER else "  (round-trip tests skipped: no markdown-it-py)",
    ))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
