"""Render tabular data as a GFM table, and report what the table cannot carry.

A markdown table is a lossy container. Seven kinds of cell content either change
meaning, lose characters, or stop being markdown when they are placed inside one,
and the GitHub Flavored Markdown table extension signals none of them -- it has no
error state. A row with too many cells is truncated. A cell with leading spaces is
trimmed. A cell reading ``_id_field_`` is italicised and loses both underscores.

This module renders the table and, separately, returns the list of cells whose
content did not survive intact. Rendering never raises; the audit is the channel.

Every claim in the FINDING table below was verified against markdown-it-py's
``gfm-like`` preset -- see ``evidence.py`` for the round-trip that checks them.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Finding",
    "TableResult",
    "display_width",
    "pad",
    "escape_cell",
    "audit_cell",
    "infer_alignment",
    "render",
]

# --------------------------------------------------------------------------
# Severities
# --------------------------------------------------------------------------
# LOSS        the rendered table does not contain what you put in
# PORTABILITY it renders here, but not under every conforming renderer
# COSMETIC    the rendered output is right; the markdown source is misaligned

LOSS = "LOSS"
PORTABILITY = "PORTABILITY"
COSMETIC = "COSMETIC"

_SEVERITY_ORDER = {LOSS: 0, PORTABILITY: 1, COSMETIC: 2}


@dataclass(frozen=True)
class Finding:
    """One cell, one thing a GFM table will not carry unchanged."""

    code: str
    severity: str
    row: int  # -1 is the header row
    column: str
    detail: str

    @property
    def where(self) -> str:
        return "header" if self.row < 0 else "row %d" % self.row


@dataclass
class TableResult:
    """A rendered table plus the audit of what it could not represent."""

    markdown: str
    findings: List[Finding] = field(default_factory=list)
    alignment: List[str] = field(default_factory=list)
    widths: List[int] = field(default_factory=list)

    def by_severity(self, severity: str) -> List[Finding]:
        return [f for f in self.findings if f.severity == severity]

    @property
    def loses_data(self) -> bool:
        return any(f.severity == LOSS for f in self.findings)

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for f in self.findings:
            out[f.code] = out.get(f.code, 0) + 1
        return out


# --------------------------------------------------------------------------
# 1. Display width
# --------------------------------------------------------------------------
# len() counts code points. A monospace terminal, and the person reading the
# markdown source in a diff, count columns. The two disagree for CJK (two
# columns), for combining marks (zero), and for emoji joined by ZWJ, where a
# single grapheme is several code points wide by len() and two columns wide on
# screen. Padding by len() misaligns the source for exactly those rows.

_ZERO_WIDTH = {"Mn", "Me", "Cf"}


def display_width(text: str, ambiguous_wide: bool = False) -> int:
    """Terminal column count for ``text``.

    ``ambiguous_wide`` covers the East Asian *Ambiguous* class (Greek, Cyrillic,
    box drawing, ``±``), which is one column in a Western font and two in a CJK
    font. There is no correct answer without knowing the reader's font, so it is
    a parameter rather than a default.
    """
    total = 0
    for ch in text:
        if ch == "\u200d":  # zero-width joiner: binds an emoji sequence
            continue
        if unicodedata.category(ch) in _ZERO_WIDTH:
            continue
        eaw = unicodedata.east_asian_width(ch)
        if eaw in ("W", "F"):
            total += 2
        elif eaw == "A":
            total += 2 if ambiguous_wide else 1
        else:
            total += 1
    return total


def pad(text: str, width: int, align: str = "left", ambiguous_wide: bool = False) -> str:
    """Pad ``text`` to ``width`` display columns (not code points)."""
    gap = max(0, width - display_width(text, ambiguous_wide))
    if align == "right":
        return " " * gap + text
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


# --------------------------------------------------------------------------
# 2. Escaping
# --------------------------------------------------------------------------
# The pipe is the only character GFM lets you escape inside a table cell, and
# the escape is processed before inline parsing -- so `a\|b` inside a code span
# renders as the code `a|b`. The HTML entity `&#124;` is the advice you will
# find in most snippets; it works in plain text and fails inside a code span,
# where entity references are not recognised and it renders as the literal
# characters `&#124;`. escape_cell() therefore always uses the backslash.

_PIPE = re.compile(r"(?<!\\)\|")
_ENTITY_PIPE = re.compile(r"&#(?:124|x7c|X7C);")
_CODE_SPAN = re.compile(r"`+[^`]*`+")

# Emphasis: GFM opens emphasis on `*` almost anywhere, and on `_` only at a word
# boundary. `_id_field_` italicises; `snake_case_name` does not. Both are common
# in a column of identifiers, and only one of them survives.
_STAR_EMPHASIS = re.compile(r"\*(?=\S)[^*]*\*")
_UNDERSCORE_EMPHASIS = re.compile(
    r"(?:^|(?<=[\s(\[]))_(?=\S).*_(?=$|[\s.,;:!?)\]])"
)


def escape_cell(text: str, newline: str = "br", escape_emphasis: bool = False) -> str:
    """Make ``text`` safe to sit between two pipes.

    ``newline`` is one of ``"br"`` (join with a ``<br>`` tag -- HTML, see
    ``NEWLINE`` below), ``"space"`` (join with a space, portable, loses the
    break) or ``"strip"`` (keep the first line only).

    ``escape_emphasis`` backslash-escapes the ``*`` and ``_`` runs that would
    otherwise italicise. It is off by default: escaping rewrites the source a
    human then has to read in a diff, and the caller may prefer to be told
    rather than have it done to them.
    """
    out = text
    if newline == "br":
        out = re.sub(r"\r\n|\r|\n", "<br>", out)
    elif newline == "space":
        out = re.sub(r"\s*(?:\r\n|\r|\n)\s*", " ", out)
    elif newline == "strip":
        out = re.split(r"\r\n|\r|\n", out)[0]
    else:  # pragma: no cover - guarded by the caller
        raise ValueError("newline must be br, space or strip")
    out = _PIPE.sub(r"\\|", out)
    if escape_emphasis and _emphasis_spans(out):
        out = _escape_emphasis_runs(out)
    return out


def _escape_emphasis_runs(text: str) -> str:
    """Escape only the delimiters that actually open emphasis, code spans intact."""
    spans: List[Tuple[int, int]] = []
    masked = _CODE_SPAN.sub(lambda m: "\x00" * len(m.group(0)), text)
    for rx in (_STAR_EMPHASIS, _UNDERSCORE_EMPHASIS):
        for m in rx.finditer(masked):
            spans.append((m.start(), m.end()))
    marks = set()
    for start, end in spans:
        # Escape the whole delimiter run at each edge, not one character of it:
        # `__dunder__` is strong emphasis, and escaping only the outer pair
        # leaves `_dunder_`, which is still emphasis.
        i = start
        while i < end and text[i] in "*_":
            marks.add(i)
            i += 1
        j = end - 1
        while j >= start and text[j] in "*_":
            marks.add(j)
            j -= 1
    return "".join(
        ("\\" + ch) if (i in marks and ch in "*_") else ch for i, ch in enumerate(text)
    )


def _emphasis_spans(text: str) -> List[str]:
    """Emphasis runs that will fire, ignoring anything inside a code span."""
    masked = _CODE_SPAN.sub(lambda m: "\x00" * len(m.group(0)), text)
    hits = [m.group(0) for m in _STAR_EMPHASIS.finditer(masked)]
    hits += [m.group(0).strip() for m in _UNDERSCORE_EMPHASIS.finditer(masked)]
    return hits


# --------------------------------------------------------------------------
# 3. The audit
# --------------------------------------------------------------------------


def audit_cell(
    text: str,
    row: int,
    column: str,
    newline: str = "br",
    ambiguous_wide: bool = False,
    escape_emphasis: bool = False,
) -> List[Finding]:
    """Everything about ``text`` that a GFM table will not carry unchanged."""
    found: List[Finding] = []

    def add(code: str, sev: str, detail: str) -> None:
        found.append(Finding(code, sev, row, column, detail))

    if "\n" in text or "\r" in text:
        n = len(re.split(r"\r\n|\r|\n", text))
        if newline == "br":
            add(
                "NEWLINE",
                PORTABILITY,
                "%d lines joined with <br>; that is an HTML tag, and renders as "
                "literal text wherever inline HTML is disabled" % n,
            )
        elif newline == "strip":
            add("NEWLINE", LOSS, "%d lines, %d discarded" % (n, n - 1))
        else:
            add("NEWLINE", LOSS, "%d lines flattened to one; the breaks are gone" % n)

    if _ENTITY_PIPE.search(text):
        inside = any(_ENTITY_PIPE.search(m.group(0)) for m in _CODE_SPAN.finditer(text))
        if inside:
            add(
                "ENTITY_IN_CODE",
                LOSS,
                "&#124; inside a code span is not an entity reference; it renders "
                "as those six characters. Use a backslash escape",
            )

    if text != text.strip():
        add(
            "EDGE_SPACE",
            LOSS,
            "leading/trailing whitespace is trimmed by the renderer and is not "
            "representable in a cell (%r)" % text[:24],
        )

    spans = _emphasis_spans(text)
    if spans:
        shown = ", ".join(sorted(set(spans))[:3])
        if escape_emphasis:
            add(
                "EMPHASIS",
                COSMETIC,
                "%s would italicise; the delimiters were backslash-escaped, so "
                "the rendered text is intact and the source carries a backslash" % shown,
            )
        else:
            add("EMPHASIS", LOSS, "%s becomes italic and the delimiters are eaten" % shown)

    if text.endswith("\\") and not text.endswith("\\\\"):
        add(
            "BACKSLASH_END",
            PORTABILITY,
            "a trailing backslash sits directly against the closing pipe; "
            "renderers differ on whether it escapes it",
        )

    if display_width(text, ambiguous_wide) != len(text):
        add(
            "WIDE_GLYPH",
            COSMETIC,
            "%d code points, %d display columns -- padding by len() misaligns "
            "the source" % (len(text), display_width(text, ambiguous_wide)),
        )

    if "|" in text:
        add("PIPE", COSMETIC, "%d pipe(s) escaped as \\|" % text.count("|"))

    return found


# --------------------------------------------------------------------------
# 4. Alignment
# --------------------------------------------------------------------------
# Alignment is a property of the column, declared once in the delimiter row.
# There is no per-cell alignment in GFM, so a column of mixed numbers and
# labels has to pick one.

_NUMERIC = re.compile(r"^[-+(]?\s*[$£€¥]?\s*\d[\d,_ ]*(?:\.\d+)?\s*%?\)?$")


def infer_alignment(values: Sequence[str]) -> str:
    """``"right"`` if every non-empty value reads as a number, else ``"left"``."""
    seen = [v.strip() for v in values if v.strip()]
    if not seen:
        return "left"
    return "right" if all(_NUMERIC.match(v) for v in seen) else "left"


_DELIMITER = {"left": ":---", "center": ":---:", "right": "---:", "none": "---"}


def _delimiter_cell(align: str, width: int) -> str:
    """Delimiter run padded to the column width, keeping the colons in place."""
    if align == "right":
        return "-" * max(3, width - 1) + ":"
    if align == "center":
        return ":" + "-" * max(3, width - 2) + ":"
    if align == "left":
        return ":" + "-" * max(3, width - 1)
    return "-" * max(3, width)


# --------------------------------------------------------------------------
# 5. Render
# --------------------------------------------------------------------------


def render(
    rows: Iterable[Sequence[object]],
    headers: Optional[Sequence[object]] = None,
    align: Optional[Sequence[str]] = None,
    newline: str = "br",
    pad_cells: bool = True,
    ambiguous_wide: bool = False,
    escape_emphasis: bool = False,
    float_fmt: str = "{:g}",
    none_as: str = "",
) -> TableResult:
    """Render ``rows`` as a GFM table and audit what did not survive.

    ``rows`` may be any iterable of sequences; a pandas DataFrame should be
    passed as ``render(df.values.tolist(), df.columns)``. Ragged rows are
    reported, never silently reshaped -- GFM truncates the excess and pads the
    shortfall, and both happen without a warning from the renderer.
    """
    body = [list(r) for r in rows]
    if headers is None:
        if not body:
            raise ValueError("cannot infer headers from zero rows")
        headers = ["col%d" % i for i in range(len(body[0]))]
    head = [_stringify(h, float_fmt, none_as) for h in headers]
    ncols = len(head)

    findings: List[Finding] = []

    for i, h in enumerate(head):
        if not h.strip():
            findings.append(
                Finding(
                    "EMPTY_HEADER",
                    PORTABILITY,
                    -1,
                    "col%d" % i,
                    "an empty header cell is legal but leaves the column "
                    "unnameable in any tool that reads the table back",
                )
            )
        findings.extend(
            audit_cell(h, -1, h or "col%d" % i, newline, ambiguous_wide, escape_emphasis)
        )

    # Ragged rows: report before reshaping, because GFM's own reshape is silent.
    grid: List[List[str]] = []
    for r_idx, raw in enumerate(body):
        cells = [_stringify(c, float_fmt, none_as) for c in raw]
        if len(cells) > ncols:
            dropped = cells[ncols:]
            findings.append(
                Finding(
                    "RAGGED_EXTRA",
                    LOSS,
                    r_idx,
                    head[-1] if head else "?",
                    "%d cell(s) past the header width are dropped by the "
                    "renderer with no warning: %s"
                    % (len(dropped), ", ".join(repr(d) for d in dropped[:3])),
                )
            )
            cells = cells[:ncols]
        elif len(cells) < ncols:
            missing = ncols - len(cells)
            findings.append(
                Finding(
                    "RAGGED_SHORT",
                    PORTABILITY,
                    r_idx,
                    head[len(cells)] if len(cells) < len(head) else "?",
                    "%d cell(s) short; the renderer inserts empties, which is "
                    "indistinguishable from a genuine blank" % missing,
                )
            )
            cells = cells + [""] * missing
        for c_idx, cell in enumerate(cells):
            findings.extend(
                audit_cell(cell, r_idx, head[c_idx], newline, ambiguous_wide, escape_emphasis)
            )
        grid.append(cells)

    if align is None:
        alignment = [
            infer_alignment([row[i] for row in grid]) if grid else "left"
            for i in range(ncols)
        ]
    else:
        alignment = list(align) + ["left"] * (ncols - len(align))
        alignment = alignment[:ncols]

    esc_head = [escape_cell(h, newline, escape_emphasis) for h in head]
    esc_grid = [[escape_cell(c, newline, escape_emphasis) for c in row] for row in grid]

    if pad_cells:
        widths = [
            max(
                [display_width(esc_head[i], ambiguous_wide)]
                + [display_width(row[i], ambiguous_wide) for row in esc_grid]
                + [3]
            )
            for i in range(ncols)
        ]
    else:
        widths = [3] * ncols

    def line(cells: Sequence[str], aligns: Sequence[str]) -> str:
        if not pad_cells:
            return "| " + " | ".join(cells) + " |"
        padded = [
            pad(c, widths[i], aligns[i], ambiguous_wide) for i, c in enumerate(cells)
        ]
        return "| " + " | ".join(padded) + " |"

    out = [
        line(esc_head, ["left"] * ncols),
        "| " + " | ".join(_delimiter_cell(alignment[i], widths[i]) for i in range(ncols)) + " |",
    ]
    out += [line(row, alignment) for row in esc_grid]

    findings.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], f.row, f.code))
    return TableResult("\n".join(out), findings, alignment, widths)


def _stringify(value: object, float_fmt: str, none_as: str) -> str:
    if value is None:
        return none_as
    if isinstance(value, float):
        # NaN is a float and formats as 'nan'; treat it as missing.
        if value != value:
            return none_as
        return float_fmt.format(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# --------------------------------------------------------------------------
# 6. Sample data
# --------------------------------------------------------------------------
# A lint export from a code-review bot: the kind of table that gets pasted into
# a pull request comment. Every hostile cell here is content a real linter
# emits -- a regex, a code span, an owner name, an indented snippet.

SAMPLE_HEADERS = ["file", "pattern", "note", "hits", "owner"]

SAMPLE_ROWS: List[List[object]] = [
    ["etl/load.py", "`a|b`", "alternation in a split", 12, "Chen Wei"],
    ["etl/load.py", "r'\\d+|\\w+'", "raw regex, two branches", 3, "Chen Wei"],
    ["api/auth.py", "`&#124;`", "entity used to escape a pipe", 1, "Ana Ruiz"],
    ["api/auth.py", "_id_field_", "leading+trailing underscore", 7, "Ana Ruiz"],
    ["jobs/nightly.py", "2*3*4", "star run in a literal", 2, "Ana Ruiz"],
    ["jobs/nightly.py", "retry\nthen alert", "two-line note", 4, "陈伟"],
    ["ui/table.tsx", "  indent", "leading spaces are the finding", 1, "Ana Ruiz"],
    ["ui/table.tsx", "C:\\path\\", "trailing backslash", 1, "Ana Ruiz"],
    ["ops/deploy.sh", "set -e", "no issue, control row", 0, "Ana Ruiz"],
    ["ops/deploy.sh", "grep -c", "ragged row follows", 5, "Ana Ruiz", "EXTRA-COLUMN"],
    ["docs/readme.md", "snake_case_ok", "underscores mid-word are safe", 9, "Ana Ruiz"],
    ["docs/readme.md", "emoji 🚦 label", "wide glyph in the cell", 6, "Ana Ruiz"],
]


def sample_table(**kwargs: object) -> TableResult:
    """The bundled sample, rendered with the default policy."""
    return render(SAMPLE_ROWS, SAMPLE_HEADERS, **kwargs)  # type: ignore[arg-type]
