"""Line endings: what counts as a line, and who decides.

A file is a byte string. "Lines" are not in the file - they are produced by a
splitter, and every runtime ships a different one. `wc -l` counts `\\n` bytes.
Python's text mode rewrites `\\r\\n` and `\\r` to `\\n` before you see them.
`str.splitlines()` also breaks on vertical tab, form feed, NEL and U+2028.
A CSV reader keeps a `\\r\\n` that sits inside a quoted field, and a naive
`split(b"\\n")` does not.

So three questions have three different answers, per runtime:

1. **how many lines is this file?**
2. **what is on each line?** (a trailing `\\r` is invisible and still there)
3. **is reading and writing it back the identity?** (it is not, and the change
   can land inside a data value rather than at the end of a line)

This module models ten real splitters over fifteen byte blobs and measures the
disagreement. Every claim is recomputed by `evidence.py` from this file.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

CR = b"\r"
LF = b"\n"
CRLF = b"\r\n"

# --------------------------------------------------------------------------
# Corpus: fifteen byte blobs, each one an export somebody actually receives.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Blob:
    id: int
    label: str
    data: bytes
    note: str

    @property
    def display(self) -> str:
        """The bytes with the invisible ones named."""
        out = []
        for b in self.data:
            ch = bytes([b])
            if ch == b"\r":
                out.append("<CR>")
            elif ch == b"\n":
                out.append("<LF>\n")
            elif b == 0x0B:
                out.append("<VT>")
            elif b == 0x0C:
                out.append("<FF>")
            elif b < 0x20:
                out.append(f"<{b:02X}>")
            elif b > 0x7E:
                out.append(f"<{b:02X}>")
            else:
                out.append(chr(b))
        return "".join(out)

    @property
    def one_line(self) -> str:
        return self.display.replace("\n", "")


CORPUS: Tuple[Blob, ...] = (
    Blob(1, "lf-only", b"id,name\n1,Alice\n2,Bob\n", "the file everyone assumes they have"),
    Blob(2, "crlf-only", b"id,name\r\n1,Alice\r\n2,Bob\r\n", "any export written on Windows"),
    Blob(3, "cr-only", b"id,name\r1,Alice\r2,Bob\r", "classic Mac, and some Excel-for-Mac exports"),
    Blob(4, "mixed-lf-crlf", b"id,name\n1,Alice\r\n2,Bob\n", "two editors, one file"),
    Blob(5, "no-trailing-newline", b"id,name\n1,Alice\n2,Bob", "the last line ends with nothing"),
    Blob(
        6,
        "crlf-inside-quotes",
        b'id,note\r\n1,"line one\r\nline two"\r\n2,plain\r\n',
        "a CRLF that is data, not structure",
    ),
    Blob(
        7,
        "lone-cr-in-value",
        b'id,name\n1,"Smith\rJones"\n2,Bob\n',
        "a stray CR pasted into a form field",
    ),
    Blob(8, "nel-u0085", b"id,name\n1,A\xc2\x85B\n", "U+0085 NEL, which splitlines() breaks on"),
    Blob(
        9,
        "ls-u2028",
        b'{"bio": "para one\xe2\x80\xa8para two"}\n',
        "U+2028 in a JSON string - legal JSON, and a line break to Python",
    ),
    Blob(10, "vertical-tab", b"id,addr\n1,Flat 3\x0bLondon\n", "VT in an address field"),
    Blob(11, "form-feed", b"page one\x0cpage two\n", "FF as a page separator, from a report"),
    Blob(12, "bom-crlf", b"\xef\xbb\xbfid,name\r\n1,Alice\r\n", "UTF-8 BOM in front of a CRLF file"),
    Blob(13, "lf-then-cr", b"a\n\rb\n\r", "LF CR, the wrong way round - a botched conversion"),
    Blob(14, "double-converted", b"a\r\r\nb\r\r\n", "CRLF converted to CRLF again, over FTP ASCII"),
    Blob(15, "blank-lines-mixed", b"a\r\n\r\nb\n\nc\r\n", "blank lines under two conventions"),
)

CORPUS_BY_ID: Dict[int, Blob] = {b.id: b for b in CORPUS}

# --------------------------------------------------------------------------
# Ten splitters, each one a real runtime's answer to "what is a line?"
# --------------------------------------------------------------------------


def _strip_bom(data: bytes) -> bytes:
    return data[3:] if data.startswith(b"\xef\xbb\xbf") else data


def split_lf(data: bytes) -> List[bytes]:
    """`data.split(b"\\n")` - the one most pipelines actually use.

    A CR before the LF stays on the end of the line, invisibly.
    """
    parts = data.split(LF)
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    return parts


def split_wc(data: bytes) -> List[bytes]:
    """`wc -l` - counts LF bytes, so a file with no trailing newline is one
    line short of what you see in an editor."""
    parts = data.split(LF)
    return parts[:-1]


def split_universal(data: bytes) -> List[bytes]:
    """Python text mode, `open(..., newline=None)`.

    CRLF and lone CR are rewritten to LF *before* your code sees the string,
    which is why the bug is invisible from inside Python.
    """
    text = _strip_bom(data).replace(CRLF, LF).replace(CR, LF)
    parts = text.split(LF)
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    return parts


def split_newline_empty(data: bytes) -> List[bytes]:
    """`open(..., newline="")` - splits on all three, translates none."""
    out: List[bytes] = []
    cur = bytearray()
    i = 0
    data = _strip_bom(data)
    while i < len(data):
        ch = data[i : i + 1]
        if ch == CR:
            if data[i + 1 : i + 2] == LF:
                i += 1
            out.append(bytes(cur))
            cur = bytearray()
        elif ch == LF:
            out.append(bytes(cur))
            cur = bytearray()
        else:
            cur += ch
        i += 1
    if cur:
        out.append(bytes(cur))
    return out


def split_splitlines(data: bytes) -> List[bytes]:
    """`str.splitlines()` - breaks on eight boundaries, not one.

    LF, CR, CRLF, VT, FF, FS, GS, RS, NEL, U+2028, U+2029. Any of those
    arriving inside a data value silently becomes a new row.
    """
    text = _strip_bom(data).decode("utf-8", "surrogateescape")
    return [ln.encode("utf-8", "surrogateescape") for ln in text.splitlines()]


def split_bytes_splitlines(data: bytes) -> List[bytes]:
    """`bytes.splitlines()` - the same call on bytes, and a *different*
    boundary set: no NEL, no U+2028, but VT and FF still split."""
    return _strip_bom(data).splitlines()


def split_java(data: bytes) -> List[bytes]:
    """`BufferedReader.readLine()` - LF, CR and CRLF, and it drops the final
    empty line."""
    return split_newline_empty(data)


def split_js(data: bytes) -> List[bytes]:
    """JavaScript's usual `text.split(/\\r?\\n/)` - a lone CR is left inside
    the line, where it will be printed and not seen."""
    text = _strip_bom(data)
    parts = text.replace(CRLF, LF).split(LF)
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    return parts


def split_csv(data: bytes) -> List[bytes]:
    """`csv.reader` over `newline=""` - the only splitter here that knows a
    line terminator inside quotes is data."""
    text = _strip_bom(data).decode("utf-8", "surrogateescape")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    return [",".join(r).encode("utf-8", "surrogateescape") for r in rows]


def split_git_text_auto(data: bytes) -> List[bytes]:
    """`* text=auto` - normalises CRLF to LF on commit, then LF-splits. The
    normalisation is what makes the diff say every line changed."""
    return split_lf(_strip_bom(data).replace(CRLF, LF))


def split_posix(data: bytes) -> List[bytes]:
    """POSIX: a line is a sequence ending in LF. Trailing bytes with no LF
    are an *incomplete line*, which is why `git` prints
    `\\ No newline at end of file`."""
    return split_wc(data)


@dataclass(frozen=True)
class Splitter:
    key: str
    models: str
    fn: Callable[[bytes], List[bytes]]
    note: str = ""


SPLITTERS: Tuple[Splitter, ...] = (
    Splitter("split_lf", 'data.split(b"\\n") in any pipeline', split_lf,
             "leaves a CR on the end of every line of a CRLF file"),
    Splitter("wc_l", "wc -l, and POSIX's definition of a line", split_wc,
             "counts LF bytes; an unterminated last line is not a line"),
    Splitter("py_universal", "Python open(newline=None), pandas, most readers", split_universal,
             "rewrites CRLF and CR to LF before your code sees the text"),
    Splitter("py_newline_empty", 'Python open(newline="")', split_newline_empty,
             "splits on all three, translates none"),
    Splitter("str_splitlines", "str.splitlines()", split_splitlines,
             "eleven boundaries including VT, FF, NEL and U+2028"),
    Splitter("bytes_splitlines", "bytes.splitlines()", split_bytes_splitlines,
             "a different boundary set from the str version"),
    Splitter("java_readline", "BufferedReader.readLine(), Go bufio.Scanner", split_java,
             "LF, CR and CRLF"),
    Splitter("js_split", "JS text.split(/\\r?\\n/)", split_js,
             "a lone CR stays inside the line"),
    Splitter("csv_reader", 'csv.reader over newline=""', split_csv,
             "the only one that treats a terminator inside quotes as data"),
    Splitter("git_text_auto", "git * text=auto (normalise on commit)", split_git_text_auto,
             "rewrites the file, then splits"),
)

SPLITTER_BY_KEY: Dict[str, Splitter] = {s.key: s for s in SPLITTERS}


def lines(blob: Blob, sp: Splitter) -> List[bytes]:
    return sp.fn(blob.data)


def line_count(blob: Blob, sp: Splitter) -> int:
    return len(lines(blob, sp))


def count_matrix(blobs: Sequence[Blob] = CORPUS) -> Dict[Tuple[int, str], int]:
    return {(b.id, s.key): line_count(b, s) for b in blobs for s in SPLITTERS}


# --------------------------------------------------------------------------
# 1. Verdicts
# --------------------------------------------------------------------------


class Verdict(str, Enum):
    """What every splitter agrees on, for one blob."""

    AGREED = "agreed"  # same count and same content everywhere
    CONTENT_DRIFT = "content-drift"  # same count, different bytes on the lines
    COUNT_DRIFT = "count-drift"  # they do not even agree how many lines
    DATA_SPLIT = "data-split"  # a terminator inside a value becomes a row


def content_sets(blob: Blob) -> Dict[str, Tuple[bytes, ...]]:
    return {s.key: tuple(lines(blob, s)) for s in SPLITTERS}


def verdict(blob: Blob) -> Verdict:
    counts = {line_count(blob, s) for s in SPLITTERS}
    csv_n = line_count(blob, SPLITTER_BY_KEY["csv_reader"])
    others = {line_count(blob, s) for s in SPLITTERS if s.key != "csv_reader"}
    if len(counts) > 1 and csv_n < min(others):
        return Verdict.DATA_SPLIT
    if len(counts) > 1:
        return Verdict.COUNT_DRIFT
    if len(set(content_sets(blob).values())) > 1:
        return Verdict.CONTENT_DRIFT
    return Verdict.AGREED


def verdict_counts(blobs: Sequence[Blob] = CORPUS) -> Dict[Verdict, int]:
    out = {v: 0 for v in Verdict}
    for b in blobs:
        out[verdict(b)] += 1
    return out


def agreeing_blobs(blobs: Sequence[Blob] = CORPUS) -> List[Blob]:
    return [b for b in blobs if verdict(b) is Verdict.AGREED]


def count_spread(blob: Blob) -> Tuple[int, int]:
    counts = [line_count(blob, s) for s in SPLITTERS]
    return min(counts), max(counts)


# --------------------------------------------------------------------------
# 2. The invisible carriage return
# --------------------------------------------------------------------------


def trailing_cr_lines(blob: Blob, sp: Splitter) -> List[bytes]:
    """Lines this splitter hands you with a CR still on the end."""
    return [ln for ln in lines(blob, sp) if ln.endswith(CR)]


def cr_contamination(blobs: Sequence[Blob] = CORPUS) -> Dict[str, int]:
    return {
        s.key: sum(len(trailing_cr_lines(b, s)) for b in blobs) for s in SPLITTERS
    }


def cr_typed_failures(blobs: Sequence[Blob] = CORPUS) -> List[Tuple[Blob, str, bytes]]:
    """Fields that stop being parseable because a CR rode along.

    The failure is not the CR; it is that `"1\\r"` looks exactly like `"1"`
    in every log line, error message and screenshot.
    """
    out = []
    for b in blobs:
        for s in SPLITTERS:
            for ln in trailing_cr_lines(b, s):
                last = ln.split(b",")[-1]
                out.append((b, s.key, last))
    return out


# --------------------------------------------------------------------------
# 3. Round-tripping: read it, write it back, compare bytes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundTrip:
    blob: int
    splitter: str
    changed: bool
    inside_value: bool
    before: int
    after: int


def roundtrip(blob: Blob, sp: Splitter, write_with: bytes = LF) -> RoundTrip:
    """Read with this splitter, write back with one terminator.

    `inside_value` is the part that matters: a change at the end of a line is
    a formatting change, and a change inside a quoted field is data loss.
    """
    out = write_with.join(lines(blob, sp))
    if blob.data.endswith((LF, CR)):
        out += write_with
    changed = out != _strip_bom(blob.data)
    csv_rows_before = line_count(blob, SPLITTER_BY_KEY["csv_reader"])
    csv_rows_after = len(split_csv(out))
    return RoundTrip(blob.id, sp.key, changed, csv_rows_before != csv_rows_after,
                     csv_rows_before, csv_rows_after)


def roundtrip_table(blobs: Sequence[Blob] = CORPUS) -> List[RoundTrip]:
    return [roundtrip(b, s) for b in blobs for s in SPLITTERS]


def roundtrip_totals(blobs: Sequence[Blob] = CORPUS) -> Dict[str, int]:
    rts = roundtrip_table(blobs)
    return {
        "runs": len(rts),
        "changed": sum(1 for r in rts if r.changed),
        "row_count_changed": sum(1 for r in rts if r.inside_value),
    }


# --------------------------------------------------------------------------
# 4. The diff that says every line changed
# --------------------------------------------------------------------------


def edit_one_field(data: bytes) -> bytes:
    return data.replace(b"Alice", b"Alicia").replace(b"line two", b"line 2")


def diff_lines(before: bytes, after: bytes, sp: Splitter) -> int:
    """How many lines a line-based diff calls changed."""
    a = split_lf(before) if sp is None else sp.fn(before)
    b = split_lf(after) if sp is None else sp.fn(after)
    changed = sum(1 for x, y in zip(a, b) if x != y)
    return changed + abs(len(a) - len(b))


def diff_blast(blobs: Sequence[Blob] = CORPUS) -> List[Tuple[Blob, int, int]]:
    """(blob, lines changed by the edit alone, lines changed when the same
    commit also converts the line endings)."""
    sp = SPLITTER_BY_KEY["split_lf"]
    out = []
    for b in blobs:
        edited = edit_one_field(b.data)
        content_only = diff_lines(b.data, edited, sp)
        converted = diff_lines(b.data, edited.replace(CRLF, LF).replace(CR, LF), sp)
        out.append((b, content_only, converted))
    return out


# --------------------------------------------------------------------------
# 5. Concatenation, and the line that eats the next one
# --------------------------------------------------------------------------


def unterminated(blobs: Sequence[Blob] = CORPUS) -> List[Blob]:
    return [b for b in blobs if b.data and not b.data.endswith((LF, CR))]


def concat_loss(blobs: Sequence[Blob] = CORPUS) -> Tuple[int, int, List[int]]:
    """`cat *.csv > all.csv` over the corpus.

    Returns (sum of the parts' line counts, the concatenation's line count,
    the ids whose last line was welded onto the next file's first line).
    """
    sp = SPLITTER_BY_KEY["py_universal"]
    parts = sum(line_count(b, sp) for b in blobs)
    joined = b"".join(b.data for b in blobs)
    welded = [b.id for b in unterminated(blobs)]
    return parts, len(sp.fn(joined)), welded


# --------------------------------------------------------------------------
# 6. Streaming: a CRLF split across a chunk boundary
# --------------------------------------------------------------------------


def naive_chunk_reader(data: bytes, chunk: int) -> List[bytes]:
    """A stream reader that splits each chunk on LF and strips a trailing CR
    per line - correct on whole chunks, wrong when CRLF straddles the seam."""
    out: List[bytes] = []
    carry = b""
    for i in range(0, len(data), chunk):
        block = carry + data[i : i + chunk]
        parts = block.split(LF)
        carry = parts.pop()
        out.extend(p[:-1] if p.endswith(CR) else p for p in parts)
    if carry:
        out.append(carry[:-1] if carry.endswith(CR) else carry)
    return out


def chunk_drift(
    blobs: Sequence[Blob] = CORPUS, sizes: Sequence[int] = (4, 8, 16)
) -> List[Tuple[Blob, int, int, int]]:
    """(blob, chunk size, correct line count, chunked line count) where they
    differ, or where the *content* differs."""
    sp = SPLITTER_BY_KEY["py_universal"]
    out = []
    for b in blobs:
        want = sp.fn(b.data)
        for n in sizes:
            got = naive_chunk_reader(b.data, n)
            if got != want:
                out.append((b, n, len(want), len(got)))
    return out


# --------------------------------------------------------------------------
# 7. Detecting "the" line ending, which may not exist
# --------------------------------------------------------------------------


def eol_histogram(data: bytes) -> Dict[str, int]:
    crlf = data.count(CRLF)
    return {
        "CRLF": crlf,
        "LF": data.count(LF) - crlf,
        "CR": data.count(CR) - crlf,
    }


def detect_first(data: bytes) -> Optional[str]:
    """What most detectors do: report the first terminator seen."""
    for i in range(len(data)):
        if data[i : i + 2] == CRLF:
            return "CRLF"
        if data[i : i + 1] == LF:
            return "LF"
        if data[i : i + 1] == CR:
            return "CR"
    return None


def detect_majority(data: bytes) -> Optional[str]:
    h = eol_histogram(data)
    if not any(h.values()):
        return None
    return max(h, key=lambda k: h[k])


def detect_strict(data: bytes) -> Optional[str]:
    """Only answer when the file uses exactly one terminator."""
    h = eol_histogram(data)
    present = [k for k, v in h.items() if v]
    return present[0] if len(present) == 1 else None


def detection_table(blobs: Sequence[Blob] = CORPUS) -> List[Tuple[Blob, Dict[str, int], str, str, str]]:
    out = []
    for b in blobs:
        out.append(
            (
                b,
                eol_histogram(b.data),
                detect_first(b.data) or "none",
                detect_majority(b.data) or "none",
                detect_strict(b.data) or "mixed - refuses to answer",
            )
        )
    return out


def detection_disagreements(blobs: Sequence[Blob] = CORPUS) -> List[Blob]:
    out = []
    for b in blobs:
        answers = {detect_first(b.data), detect_majority(b.data)}
        if len(answers) > 1 or detect_strict(b.data) is None:
            out.append(b)
    return out


# --------------------------------------------------------------------------
# 8. Findings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    severity: str  # "blocking" | "silent" | "advisory"
    title: str
    detail: str


def findings(blobs: Sequence[Blob] = CORPUS) -> List[Finding]:
    out: List[Finding] = []
    vc = verdict_counts(blobs)
    out.append(
        Finding(
            "silent",
            f"{vc[Verdict.COUNT_DRIFT] + vc[Verdict.DATA_SPLIT]} of {len(blobs)} files "
            "get a different line count depending on who reads them",
            "Not a different interpretation of the same lines - a different "
            "number of them. "
            + "; ".join(
                f"{b.label}: {count_spread(b)[0]}-{count_spread(b)[1]}"
                for b in blobs
                if count_spread(b)[0] != count_spread(b)[1]
            ),
        )
    )
    if vc[Verdict.CONTENT_DRIFT]:
        out.append(
            Finding(
                "silent",
                f"{vc[Verdict.CONTENT_DRIFT]} file(s) where every splitter agrees on the "
                "count and not on the contents",
                "Same number of lines, different bytes on them. This is the one "
                "that gets shipped, because every count-based check passes.",
            )
        )
    cr = cr_contamination(blobs)
    worst = max(cr, key=lambda k: cr[k])
    if cr[worst]:
        out.append(
            Finding(
                "blocking",
                f"{worst} hands back {cr[worst]} lines with a carriage return still "
                "on the end",
                "`int('1\\r')` raises; `'Alice\\r' == 'Alice'` is False; and both "
                "print identically in a log, an error message and a screenshot. "
                "Splitters affected: "
                + ", ".join(f"{k} ({v})" for k, v in cr.items() if v),
            )
        )
    ds = [b for b in blobs if verdict(b) is Verdict.DATA_SPLIT]
    if ds:
        out.append(
            Finding(
                "blocking",
                f"{len(ds)} file(s) where a line terminator is data, not structure",
                "; ".join(
                    f"{b.label}: csv.reader sees "
                    f"{line_count(b, SPLITTER_BY_KEY['csv_reader'])} rows, a line "
                    f"splitter sees {line_count(b, SPLITTER_BY_KEY['py_universal'])}"
                    for b in ds
                )
                + ". Splitting first and parsing second turns one row into two, "
                "with the second one short of columns.",
            )
        )
    exotic = []
    for b in blobs:
        a = line_count(b, SPLITTER_BY_KEY["str_splitlines"])
        c = line_count(b, SPLITTER_BY_KEY["py_universal"])
        if a != c:
            exotic.append((b, a, c))
    if exotic:
        out.append(
            Finding(
                "blocking",
                f"str.splitlines() finds more lines than a reader does in "
                f"{len(exotic)} file(s)",
                "It breaks on VT, FF, FS, GS, RS, NEL, U+2028 and U+2029 as well "
                "as the three real terminators. "
                + "; ".join(f"{b.label}: {a} vs {c}" for b, a, c in exotic)
                + ". Any of those inside a user-supplied value silently becomes a "
                "new row.",
            )
        )
    sl = line_count(CORPUS_BY_ID[10], SPLITTER_BY_KEY["str_splitlines"])
    bl = line_count(CORPUS_BY_ID[10], SPLITTER_BY_KEY["bytes_splitlines"])
    if sl or bl:
        out.append(
            Finding(
                "advisory",
                "str.splitlines() and bytes.splitlines() do not have the same "
                "boundary set",
                "The str version adds NEL, U+2028 and U+2029. Decoding before "
                "splitting therefore changes the row count, which makes "
                "`.decode()` a semantic operation and not a formatting one.",
            )
        )
    rt = roundtrip_totals(blobs)
    out.append(
        Finding(
            "silent",
            f"Read-then-write is not the identity in {rt['changed']} of {rt['runs']} runs",
            f"And in {rt['row_count_changed']} of them the CSV row count changes "
            "afterwards, which means the rewrite landed inside a value rather "
            "than at the end of a line. Text mode is a transformation, not a read.",
        )
    )
    blast = [(b, a, c) for b, a, c in diff_blast(blobs) if c > a]
    if blast:
        worst_b, worst_a, worst_c = max(blast, key=lambda t: t[2] - t[1])
        out.append(
            Finding(
                "silent",
                f"A one-field edit shows up as {worst_c} changed lines instead of "
                f"{worst_a} when the same commit normalises the endings",
                f"On {worst_b.label}. Review cost is the whole file, the real "
                "change is invisible inside it, and `git blame` now points at the "
                "conversion commit for every line. This is what `* text=auto` "
                "does the first time it is switched on.",
            )
        )
    parts, joined, welded = concat_loss(blobs)
    if welded:
        out.append(
            Finding(
                "blocking",
                f"cat over these files loses {parts - joined} line(s): "
                f"{len(welded)} file(s) do not end with a terminator",
                f"The last line of file {welded} is welded onto the first line of "
                "the next one, producing a row that parses cleanly and is wrong. "
                "POSIX says a text file's last line ends with a newline; git says "
                "`\\ No newline at end of file`; nothing enforces either.",
            )
        )
    cd = chunk_drift(blobs)
    if cd:
        out.append(
            Finding(
                "silent",
                f"A chunked reader disagrees with itself in {len(cd)} "
                "(file, chunk size) combinations",
                "A CRLF straddling a read boundary leaves the CR at the end of one "
                "chunk and the LF at the start of the next. The reader is correct "
                "at most buffer sizes, which is why it survives testing and fails "
                "on the one file that is a few bytes longer.",
            )
        )
    dd = detection_disagreements(blobs)
    out.append(
        Finding(
            "advisory",
            f"'Detect the line ending' has no answer for {len(dd)} of "
            f"{len(blobs)} files",
            "First-seen and majority-vote disagree, or the file simply uses more "
            "than one. A detector that always returns a single terminator is "
            "reporting a summary as if it were a fact; the honest return value is "
            "the histogram.",
        )
    )
    out.append(
        Finding(
            "advisory",
            "The fix is on the write path, not the read path",
            "Read with an explicit `newline=''` and a parser that knows about "
            "quoting; write with one chosen terminator; normalise once, in its own "
            "commit, with `.gitattributes` committed alongside. Detection is a "
            "diagnostic, not a strategy.",
        )
    )
    return out


def finding_counts(blobs: Sequence[Blob] = CORPUS) -> Dict[str, int]:
    out = {"blocking": 0, "silent": 0, "advisory": 0}
    for f in findings(blobs):
        out[f.severity] += 1
    return out


def splitter_disagreement(blobs: Sequence[Blob] = CORPUS) -> Dict[Tuple[str, str], int]:
    """How many blobs each pair of splitters reads differently."""
    m: Dict[Tuple[str, str], int] = {}
    for a, b in combinations(SPLITTERS, 2):
        n = sum(1 for x in blobs if tuple(lines(x, a)) != tuple(lines(x, b)))
        m[(a.key, b.key)] = n
        m[(b.key, a.key)] = n
    for s in SPLITTERS:
        m[(s.key, s.key)] = 0
    return m
