"""Every number quoted in the README, printed from the engine.

Run `python3 evidence.py`. Nothing in the README is typed by hand.
"""

from __future__ import annotations

from lineends import (
    CORPUS,
    CORPUS_BY_ID,
    SPLITTER_BY_KEY,
    SPLITTERS,
    Verdict,
    chunk_drift,
    concat_loss,
    cr_contamination,
    cr_typed_failures,
    detection_disagreements,
    detection_table,
    diff_blast,
    finding_counts,
    findings,
    line_count,
    lines,
    naive_chunk_reader,
    roundtrip_table,
    roundtrip_totals,
    splitter_disagreement,
    trailing_cr_lines,
    unterminated,
    verdict,
    verdict_counts,
)

RULE = "-" * 78


def head(n: int, title: str) -> None:
    print(f"\n{RULE}\n{n}. {title}\n{RULE}")


def wrap(text: str, width: int = 70):
    words, lines_, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines_.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines_.append(cur)
    return lines_


def s1_corpus() -> None:
    head(1, "The files")
    print(f"{len(CORPUS)} byte blobs, {len(SPLITTERS)} splitters.\n")
    for b in CORPUS:
        print(f"  {b.id:2d}  {b.label:22s} {len(b.data):3d} bytes  {b.note}")
        print(f"      {b.one_line}")


def s2_splitters() -> None:
    head(2, "The ten splitters")
    for s in SPLITTERS:
        print(f"  {s.key:18s} {s.models}")
        print(f"  {'':18s} {s.note}")


def s3_counts() -> None:
    head(3, "How many lines is this file?")
    keys = [s.key for s in SPLITTERS]
    print(f"  {'file':<22s}" + "".join(f"{k[:9]:>10s}" for k in keys))
    for b in CORPUS:
        row = "".join(f"{line_count(b, s):>10d}" for s in SPLITTERS)
        print(f"  {b.label:<22s}{row}")
    spreads = [
        (b, min(line_count(b, s) for s in SPLITTERS), max(line_count(b, s) for s in SPLITTERS))
        for b in CORPUS
    ]
    drifting = [t for t in spreads if t[1] != t[2]]
    print(
        f"\n  {len(drifting)} of {len(CORPUS)} files get a different line *count* "
        "depending on the reader:"
    )
    for b, lo, hi in drifting:
        print(f"    {b.label:<22s} {lo} to {hi} lines")


def s4_verdicts() -> None:
    head(4, "Four verdicts")
    print(
        "  agreed         same count and same bytes from all ten\n"
        "  content-drift  same count, different bytes - the one that ships\n"
        "  count-drift    they do not agree how many lines there are\n"
        "  data-split     a terminator inside a value becomes a new row\n"
    )
    for b in CORPUS:
        print(f"  {b.label:<22s} {verdict(b).value}")
    vc = verdict_counts()
    print()
    for v in Verdict:
        print(f"  {v.value:<14s} {vc[v]:2d} of {len(CORPUS)}")
    print(
        "\n  One file in fifteen is read identically by every runtime here, and it\n"
        "  is the one written with LF, terminated, with nothing exotic in a value."
    )


def s5_cr() -> None:
    head(5, "The carriage return that is still there")
    cr = cr_contamination()
    for k, v in cr.items():
        mark = "  <-- CR-blind" if v else ""
        print(f"  {k:18s} {v:3d} lines handed back with a trailing CR{mark}")
    b = CORPUS_BY_ID[2]
    raw = trailing_cr_lines(b, SPLITTER_BY_KEY["split_lf"])[1]
    clean = lines(b, SPLITTER_BY_KEY["py_universal"])[1]
    print(
        f"\n  {b.label}, line 2:\n"
        f"    split_lf      {raw!r}\n"
        f"    py_universal  {clean!r}\n"
        f"    printed       {raw.decode()!s}| vs {clean.decode()!s}|\n"
        f"    equal?        {raw == clean}\n"
        "\n  A field with a CR on the end prints identically in a log, an error\n"
        "  message and a screenshot. `int('2\\\\r')` raises; `'Bob\\\\r' == 'Bob'`\n"
        "  is False; a GROUP BY sees two customers."
    )
    print(f"\n  {len(cr_typed_failures())} contaminated (file, splitter, field) combinations")


def s6_data_split() -> None:
    head(6, "When the terminator is the data")
    csvs = SPLITTER_BY_KEY["csv_reader"]
    uni = SPLITTER_BY_KEY["py_universal"]
    for b in CORPUS:
        if verdict(b) is not Verdict.DATA_SPLIT:
            continue
        print(f"  {b.label}: {b.one_line}\n")
        print(f"    csv.reader   {line_count(b, csvs)} rows: "
              f"{[ln.decode() for ln in lines(b, csvs)]}")
        print(f"    py_universal {line_count(b, uni)} lines: "
              f"{[ln.decode() for ln in lines(b, uni)]}")
    print(
        "\n  Split first, parse second, and one row becomes two - the second one\n"
        "  short of columns, the first one truncated. Both are valid CSV rows on\n"
        "  their own, which is why nothing raises."
    )
    sl, uni_n = SPLITTER_BY_KEY["str_splitlines"], SPLITTER_BY_KEY["py_universal"]
    print("\n  And the other direction - splitlines() inventing rows:\n")
    for b in CORPUS:
        a, c = line_count(b, sl), line_count(b, uni_n)
        if a != c:
            print(f"    {b.label:<22s} splitlines {a}, reader {c}   ({b.note})")
    print(
        "\n  str.splitlines() breaks on LF, CR, CRLF, VT, FF, FS, GS, RS, NEL,\n"
        "  U+2028 and U+2029. bytes.splitlines() uses a different subset, so\n"
        "  `.decode()` before splitting changes the row count."
    )


def s7_roundtrip() -> None:
    head(7, "Read it, write it back, compare the bytes")
    tot = roundtrip_totals()
    print(
        f"  {tot['runs']} (file, splitter) runs\n"
        f"    bytes changed by the roundtrip:      {tot['changed']}\n"
        f"    CSV row count changed afterwards:    {tot['row_count_changed']}\n"
    )
    for r in roundtrip_table():
        if r.inside_value:
            b = CORPUS_BY_ID[r.blob]
            print(f"    {b.label:<22s} {r.splitter:<18s} "
                  f"{r.before} rows -> {r.after} rows")
    print(
        "\n  A rewrite that only touches line ends is a formatting change. A\n"
        "  rewrite that changes the row count landed inside a value. Text mode is\n"
        "  a transformation, not a read."
    )


def s8_diff() -> None:
    head(8, "The diff that says every line changed")
    print(f"  {'file':<22s}{'edit alone':>12s}{'edit + normalise':>20s}")
    for b, a, c in diff_blast():
        flag = "   <-- whole file" if c > a else ""
        print(f"  {b.label:<22s}{a:>12d}{c:>20d}{flag}")
    print(
        "\n  Same one-field edit in both columns. The second one also converts the\n"
        "  line endings, which is what `* text=auto` does the first time it is\n"
        "  switched on. Review cost is the whole file, the real change is hidden\n"
        "  inside it, and git blame now points at the conversion commit."
    )


def s9_concat() -> None:
    head(9, "cat, and the line that eats the next one")
    parts, joined, welded = concat_loss()
    print(
        f"  sum of the parts:  {parts} lines\n"
        f"  concatenated:      {joined} lines\n"
        f"  lost:              {parts - joined}\n"
    )
    for b in unterminated():
        print(f"    {b.label}: ends {b.data[-8:]!r} - no terminator")
    print(
        "\n  The last line of an unterminated file is welded to the first line of\n"
        "  the next one, and the result parses cleanly. POSIX says a text file's\n"
        "  last line ends with a newline; git prints `\\ No newline at end of\n"
        "  file`; nothing enforces either."
    )


def s10_chunks() -> None:
    head(10, "A CRLF split across a read boundary")
    cd = chunk_drift()
    print(f"  {len(cd)} (file, chunk size) combinations where a chunked reader is wrong\n")
    print(f"  {'file':<22s}{'chunk':>7s}{'correct':>9s}{'chunked':>9s}")
    for b, n, want, got in cd:
        print(f"  {b.label:<22s}{n:>7d}{want:>9d}{got:>9d}")
    b = CORPUS_BY_ID[2]
    print(
        f"\n  {b.label} at chunk 8: {naive_chunk_reader(b.data, 8)}\n"
        f"  {'':22s} correct: {lines(b, SPLITTER_BY_KEY['py_universal'])}\n"
        "\n  The reader is right at most buffer sizes. It fails on the file that is\n"
        "  a few bytes longer than the last one, which is why this ships."
    )


def s11_detection() -> None:
    head(11, "'Detect the line ending' - three answers, or none")
    print(f"  {'file':<22s}{'CRLF':>6s}{'LF':>5s}{'CR':>5s}   "
          f"{'first':<7s}{'majority':<10s}strict")
    for b, h, first, major, strict in detection_table():
        print(f"  {b.label:<22s}{h['CRLF']:>6d}{h['LF']:>5d}{h['CR']:>5d}   "
              f"{first:<7s}{major:<10s}{strict}")
    dd = detection_disagreements()
    print(
        f"\n  {len(dd)} of {len(CORPUS)} files have no single honest answer: first-seen\n"
        "  and majority disagree, or more than one terminator is present. A\n"
        "  detector that always returns one terminator is reporting a summary as\n"
        "  if it were a fact. The honest return value is the histogram."
    )


def s12_pairs() -> None:
    head(12, "Which splitters are interchangeable")
    m = splitter_disagreement()
    keys = [s.key for s in SPLITTERS]
    print(f"  {'':18s}" + "".join(f"{k[:8]:>9s}" for k in keys))
    for a in keys:
        print(f"  {a:<18s}" + "".join(f"{m[(a, b)]:>9d}" for b in keys))
    same = sorted((a, b) for (a, b), n in m.items() if a < b and n == 0)
    print(f"\n  pairs that read all {len(CORPUS)} files identically: {len(same)}")
    for a, b in same:
        print(f"    {a} == {b}")
    print(
        "\n  Note git_text_auto == js_split: both convert CRLF and both leave a\n"
        "  lone CR in place. Two tools nobody would call equivalent, agreeing\n"
        "  because they share the same blind spot."
    )


def s13_findings() -> None:
    head(13, "Findings")
    icon = {"blocking": "[blocking]", "silent": "[silent]  ", "advisory": "[advisory]"}
    rank = {"blocking": 0, "silent": 1, "advisory": 2}
    for f in sorted(findings(), key=lambda f: rank[f.severity]):
        print(f"\n  {icon[f.severity]} {f.title}")
        for line in wrap(f.detail):
            print(f"      {line}")
    c = finding_counts()
    print(
        f"\n  {c['blocking']} blocking, {c['silent']} silent, {c['advisory']} advisory.\n"
        "  The blocking ones raise something. The silent ones are a row count that\n"
        "  moved, a field with an invisible byte in it, and a diff nobody can read."
    )


def main() -> None:
    print("LINE-ENDING DETECTOR - what counts as a line, and who decides")
    print(f"{len(CORPUS)} byte blobs x {len(SPLITTERS)} splitters, every number computed")
    for fn in (
        s1_corpus, s2_splitters, s3_counts, s4_verdicts, s5_cr, s6_data_split,
        s7_roundtrip, s8_diff, s9_concat, s10_chunks, s11_detection, s12_pairs,
        s13_findings,
    ):
        fn()
    print()


if __name__ == "__main__":
    main()
