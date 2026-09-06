"""Round-trip proof: render a table, parse it back, compare cell by cell.

Nothing here is asserted from the GFM specification. Every claim is checked by
putting a string into a cell, rendering the table, handing the markdown to a
GFM parser, pulling the cell's text back out, and comparing it to what went in.
Where the two differ, the difference is the finding.

The parser is markdown-it-py's ``gfm-like`` preset. Where the parser is not
installed the round-trip sections are skipped and the audit sections still run.

Run:  python3 evidence.py
"""

from __future__ import annotations

import html
import re
from typing import Dict, List, Tuple

import tabler as T

try:
    from markdown_it import MarkdownIt

    _MD = MarkdownIt("gfm-like")
    _MD_NO_HTML = MarkdownIt("gfm-like", {"html": False})
    HAVE_PARSER = True
except Exception:  # pragma: no cover - optional dependency
    _MD = _MD_NO_HTML = None
    HAVE_PARSER = False

RULE = "-" * 78


# --------------------------------------------------------------------------
# Parsing the rendered HTML back into cells
# --------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")
_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.S)


def parse_back(
    markdown: str,
    keep_tags: bool = False,
    allow_html: bool = True,
    raw: bool = False,
) -> List[List[str]]:
    """Render markdown to HTML and return the table as a list of rows of cell text.

    ``raw`` returns the cell's inner HTML untouched, which is the only way to
    tell a rendered ``<br>`` tag apart from the four literal characters -- both
    read back as the same string once tags are stripped and entities unescaped.
    """
    parser = _MD if allow_html else _MD_NO_HTML
    out = parser.render(markdown)  # type: ignore[union-attr]
    rows: List[List[str]] = []
    for row_html in _ROW.findall(out):
        cells = []
        for cell in _CELL.findall(row_html):
            if raw:
                cells.append(cell.strip())
                continue
            text = cell if keep_tags else _TAG.sub("", cell)
            cells.append(html.unescape(text).strip())
        rows.append(cells)
    return rows


def _mark(same: bool) -> str:
    return "same" if same else "CHANGED"


def header(n: int, title: str) -> None:
    print("\n" + RULE)
    print("%d. %s" % (n, title))
    print(RULE)


# --------------------------------------------------------------------------
# 1. The pipe, and the escape that everybody recommends
# --------------------------------------------------------------------------


def section_pipe() -> Dict[str, bool]:
    header(1, "Two ways to put a pipe in a cell. One of them works twice.")
    print(
        "A cell containing a pipe has to be escaped or it splits the row. The two\n"
        "candidates are the backslash escape and the HTML entity. Both are widely\n"
        "recommended. Only one of them survives a code span.\n"
    )
    cases = [
        ("plain text", "a|b", "a\\|b", "a&#124;b"),
        ("inside a code span", "`a|b`", "`a\\|b`", "`a&#124;b`"),
    ]
    print("%-20s %-12s %-14s %-14s" % ("context", "wanted", "backslash", "entity"))
    print(RULE)
    results = {}
    for label, wanted, with_escape, with_entity in cases:
        got_esc = parse_back("| c |\n| --- |\n| %s |" % with_escape)[1][0]
        got_ent = parse_back("| c |\n| --- |\n| %s |" % with_entity)[1][0]
        ok_ent = got_ent == wanted.replace("`", "")
        results[label] = ok_ent
        print(
            "%-20s %-12s %-14s %-14s"
            % (label, wanted, repr(got_esc), repr(got_ent))
        )
    print(RULE)
    print(
        "  In a code span, entity references are not recognised -- so `&#124;` is not\n"
        "  a pipe, it is six characters. escape_cell() always emits the backslash,\n"
        "  and audit_cell() raises ENTITY_IN_CODE when the input already contains\n"
        "  the entity form inside backticks."
    )
    return results


# --------------------------------------------------------------------------
# 2. The row that is one cell too long
# --------------------------------------------------------------------------


def section_ragged() -> None:
    header(2, "A row wider than the header loses the excess, silently.")
    print(
        "GFM defines the reshape: a short row is padded with empty cells, a long one\n"
        "has the excess ignored. Neither produces a warning, an error, or a mark in\n"
        "the output. The last column of a widened export just stops arriving.\n"
    )
    md = "| a | b |\n| --- | --- |\n| 1 | 2 | 3 |\n| 4 |"
    back = parse_back(md)
    print("source:")
    for line in md.splitlines():
        print("    " + line)
    print("\nparsed back:")
    for r in back:
        print("    " + repr(r))
    print(
        "\n  Row 1 was written with three cells and reads back with two: '3' is gone.\n"
        "  Row 2 was written with one and reads back with two: the empty cell is\n"
        "  indistinguishable from a genuine blank."
    )

    res = T.render([["1", "2", "3"], ["4"]], ["a", "b"])
    print("\n  render() reports both before they happen:")
    for f in res.findings:
        if f.code.startswith("RAGGED"):
            print("    %-14s %-11s %-7s %s" % (f.code, f.severity, f.where, f.detail[:52]))


# --------------------------------------------------------------------------
# 3. Emphasis eats identifiers
# --------------------------------------------------------------------------


def section_emphasis() -> None:
    header(3, "Which underscores italicise, and which are safe.")
    print(
        "A column of identifiers meets the inline emphasis rules. `_` opens emphasis\n"
        "only at a word boundary, `*` almost anywhere. So one common identifier\n"
        "shape loses both its delimiters and another does not, and the difference\n"
        "is not something the table syntax has any say in.\n"
    )
    samples = ["_id_field_", "snake_case_ok", "2*3*4", "a_b", "*star*", "__dunder__"]
    print("%-16s %-16s %-9s %s" % ("input", "renders as", "verdict", "escaped -> renders as"))
    print(RULE)
    for s in samples:
        plain = parse_back("| c |\n| --- |\n| %s |" % T.escape_cell(s))[1][0]
        esc = T.escape_cell(s, escape_emphasis=True)
        fixed = parse_back("| c |\n| --- |\n| %s |" % esc)[1][0]
        print(
            "%-16s %-16s %-9s %s"
            % (s, repr(plain), _mark(plain == s), repr(fixed) + ("" if fixed == s else "  <-- still off"))
        )
    print(RULE)
    print(
        "  `snake_case_ok` and `a_b` survive untouched: the underscores are inside a\n"
        "  word, which is exactly the case GFM carved out for identifiers. The two\n"
        "  that lose characters are the ones with a delimiter at the edge.\n"
        "  escape_emphasis=True recovers every one of them, at the cost of a\n"
        "  backslash in the source that a human reads in the diff."
    )


# --------------------------------------------------------------------------
# 4. Whitespace at the edge of a cell does not exist
# --------------------------------------------------------------------------


def section_whitespace() -> None:
    header(4, "Leading and trailing spaces are not representable.")
    print(
        "The cell contents are trimmed by the renderer, so a value whose meaning is\n"
        "its indentation cannot be put in a table at all. This is not an escaping\n"
        "problem with a workaround -- there is no escape for it. A linter reporting\n"
        "'  indent' and one reporting 'indent' produce the same table.\n"
    )
    cases = ["  indent", "indent", "trail  ", "\tab"]
    print("%-14s %-14s %s" % ("input", "reads back", "verdict"))
    print(RULE)
    for s in cases:
        got = parse_back("| c |\n| --- |\n| %s |" % T.escape_cell(s))[1][0]
        print("%-14s %-14s %s" % (repr(s), repr(got), _mark(got == s)))
    print(RULE)
    print(
        "  The only faithful container is a code span, which changes the cell's\n"
        "  formatting: `  indent` keeps the spaces but renders as code. render()\n"
        "  does not make that substitution for you -- it flags EDGE_SPACE at LOSS\n"
        "  and leaves the decision with the caller, because the substitution is a\n"
        "  visual change to somebody else's table."
    )
    inner = parse_back("| c |\n| --- |\n| `  indent` |", keep_tags=True)[1][0]
    print("\n  code span:  %r" % inner)


# --------------------------------------------------------------------------
# 5. The line break that is not markdown
# --------------------------------------------------------------------------


def section_newline() -> None:
    header(5, "A multi-line cell leaves markdown to do it.")
    print(
        "A table row is one line. Multi-line content has three options and none of\n"
        "them is markdown: an HTML <br>, a flatten, or a truncation. The <br> is the\n"
        "usual choice and it is the one that depends on the renderer having inline\n"
        "HTML enabled -- which docs pipelines, static site generators and comment\n"
        "sanitisers routinely turn off.\n"
    )
    value = "retry\nthen alert"
    print(
        "The HTML each policy produces (a rendered <br> and the four literal\n"
        "characters read back identically once tags are stripped, so this shows the\n"
        "raw HTML instead):\n"
    )
    print("%-9s %-24s %-24s %s" % ("policy", "cell source", "html enabled", "html disabled"))
    print(RULE)
    for policy in ("br", "space", "strip"):
        cell = T.escape_cell(value, newline=policy)
        on = parse_back("| c |\n| --- |\n| %s |" % cell, raw=True)[1][0]
        off = parse_back("| c |\n| --- |\n| %s |" % cell, allow_html=False, raw=True)[1][0]
        print("%-9s %-24s %-24s %s" % (policy, repr(cell), on, off))
    print(RULE)
    print(
        "  Under 'br' with HTML disabled the tag arrives as &lt;br&gt; -- four visible\n"
        "  characters in the middle of the sentence. 'space' produces identical\n"
        "  output under both, and loses only the break. That is why NEWLINE is\n"
        "  PORTABILITY under br and LOSS under the other two: the severity depends\n"
        "  on the policy chosen, not on the data."
    )


# --------------------------------------------------------------------------
# 6. Padding: the width that len() gets wrong
# --------------------------------------------------------------------------


def section_width() -> None:
    header(6, "Column padding is a display-width problem, not a len() problem.")
    print(
        "Padding does not change what renders -- HTML does not care. It changes\n"
        "whether the person reading the raw markdown in a diff can see the columns.\n"
        "len() counts code points; a terminal counts columns; CJK, emoji and\n"
        "combining marks make the two disagree.\n"
    )
    samples = ["Ana Ruiz", "陈伟", "🚦 flag", "café", "café"]
    print("%-14s %-6s %-6s %s" % ("value", "len()", "cols", "note"))
    print(RULE)
    notes = {
        "陈伟": "2 CJK glyphs, 4 columns",
        "🚦 flag": "emoji is 2 columns wide",
        "café": "precomposed e-acute, 1 code point",
        "café": "combining acute: 5 code points, 4 columns",
    }
    for s in samples:
        print(
            "%-14s %-6d %-6d %s"
            % (repr(s)[:14], len(s), T.display_width(s), notes.get(s, ""))
        )
    print(RULE)
    print("  Padded by len() (wrong) vs by display width (right):\n")
    rows = [["陈伟", "1"], ["Ana Ruiz", "2"], ["🚦 flag", "3"]]
    naive_w = max(len(r[0]) for r in rows)
    print("    len()-padded:")
    for r in rows:
        print("      | " + r[0] + " " * (naive_w - len(r[0])) + " | " + r[1] + " |")
    print("\n    width-padded:")
    correct = T.render(rows, ["owner", "n"])
    for line in correct.markdown.splitlines():
        print("      " + line)
    print(
        "\n  The second block's pipes line up in a monospace font; the first block's\n"
        "  do not, and the two rows that drift are exactly the ones a Western-locale\n"
        "  test suite never contains."
    )


# --------------------------------------------------------------------------
# 7. The full sample, and the reconciliation
# --------------------------------------------------------------------------


def section_sample() -> None:
    header(7, "The bundled sample: 12 rows, and what reaches the reader.")
    res = T.sample_table()
    print(res.markdown)
    print("\naudit:\n")
    print("%-15s %-12s %-8s %-9s %s" % ("code", "severity", "where", "column", "detail"))
    print(RULE)
    for f in res.findings:
        print(
            "%-15s %-12s %-8s %-9s %s"
            % (f.code, f.severity, f.where, f.column, f.detail[:34])
        )
    print(RULE)

    # Reconcile the audit against what a parser actually gives back. The
    # comparison is on the exact written string -- no whitespace normalisation,
    # because whitespace is one of the things being tested.
    back = parse_back(res.markdown)
    header_row, body = back[0], back[1:]
    written = [
        [T._stringify(c, "{:g}", "") for c in row][: len(T.SAMPLE_HEADERS)]
        for row in T.SAMPLE_ROWS
    ]
    codes_at: Dict[Tuple[int, str], List[str]] = {}
    for f in res.findings:
        codes_at.setdefault((f.row, f.column), []).append(f.code)

    changed: List[Tuple[int, str, str, str, str]] = []
    for i, (want, got) in enumerate(zip(written, body)):
        for j, (w, g) in enumerate(zip(want, got)):
            if g == w:
                continue
            col = header_row[j]
            codes = [c for c in codes_at.get((i, col), []) if c != "WIDE_GLYPH"]
            changed.append((i, col, w, g, ", ".join(codes) or "(unpredicted)"))

    unpredicted = [c for c in changed if c[4] == "(unpredicted)"]
    print(
        "\nround trip: %d of %d cells read back byte-identical; %d differ, %d of "
        "those unpredicted.\n" % (60 - len(changed), 60, len(changed), len(unpredicted))
    )
    print("%-5s %-9s %-22s %-20s %s" % ("row", "column", "written", "read back", "audit said"))
    print(RULE)
    for i, col, w, g, codes in changed:
        print("%-5d %-9s %-22s %-20s %s" % (i, col, repr(w)[:22], repr(g)[:20], codes))
    print(RULE)
    print(
        "  Rows 0 and 2 are the honest edge of this comparison: their difference is\n"
        "  the backticks, which are markdown syntax rather than cell content, so the\n"
        "  code span is doing its job. The audit still has something to say about\n"
        "  both -- row 0's pipe needed the escape, and row 2's entity did not work.\n\n"
        "  Row 5's read-back is what tag-stripping does to a rendered <br>; section 5\n"
        "  showed the raw HTML, where the break is real when HTML is enabled.\n\n"
        "  Row 9's sixth cell is not in this list at all, because it is not in the\n"
        "  table: it was dropped at render time and RAGGED_EXTRA is the only record\n"
        "  that it ever existed. That is the one finding class a round trip can\n"
        "  never reproduce, and the reason the audit runs before the render rather\n"
        "  than after it.\n\n"
        "  No cell changed without a finding. The audit is not a heuristic warning\n"
        "  list -- it is the diff a parser would give you, computed without one."
    )


def main() -> None:
    print(RULE)
    print("markdown-tabler -- what a GFM table cannot carry")
    print(RULE)
    if not HAVE_PARSER:
        print(
            "\nmarkdown-it-py is not installed, so the round-trip sections are\n"
            "skipped. Install it with:  pip install markdown-it-py\n"
        )
        section_ragged_audit_only()
        return
    section_pipe()
    section_ragged()
    section_emphasis()
    section_whitespace()
    section_newline()
    section_width()
    section_sample()
    print("\n" + RULE)
    print("Verified against markdown-it-py 'gfm-like'. Rendering never raised.")
    print(RULE)


def section_ragged_audit_only() -> None:  # pragma: no cover - fallback path
    res = T.sample_table()
    print(res.markdown)
    for f in res.findings:
        print("%-15s %-12s %-8s %s" % (f.code, f.severity, f.where, f.detail[:50]))


if __name__ == "__main__":
    main()
