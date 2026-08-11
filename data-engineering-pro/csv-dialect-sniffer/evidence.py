"""Seven experiments, each isolating one way dialect detection returns a wrong
answer without raising. Every table in README.md is printed by this file.

    python3 evidence.py
"""

from __future__ import annotations

import csv
from typing import List, Optional, Sequence

import sniff

RULE = "-" * 78


def head(n: int, title: str) -> None:
    print("\n" + "=" * 78)
    print("{0}. {1}".format(n, title))
    print("=" * 78)


def table(headers: Sequence[str], rows: Sequence[Sequence[object]], widths: Sequence[int]) -> None:
    fmt = "  ".join("{{{0}:<{1}}}".format(i, w) for i, w in enumerate(widths))
    print(fmt.format(*headers))
    print(RULE)
    for r in rows:
        print(fmt.format(*[str(c) for c in r]))
    print(RULE)


# --------------------------------------------------------------------------- #


def exp1_contested() -> None:
    head(1, "Two parses, both clean, different widths - and Sniffer picks one")
    raw = sniff.sample_files()["sensor.csv"]
    text = raw.decode("utf-8")
    print("\nsensor.csv, an export with no header row:\n")
    for line in text.splitlines():
        print("    " + line)

    v = sniff.classify_delimiter(text)
    rows: List[List[object]] = []
    for s in sorted(v.all_shapes, key=lambda s: (s.delimiter, str(s.quotechar))):
        if s.quotechar != '"':
            continue
        rows.append([s.label.split(" /")[0], s.modal, "{0:.0%}".format(s.consistency),
                     "yes" if s.viable else "no", s.reason])
    print()
    table(["delimiter", "fields", "consistent", "viable", "why"],
          rows, [12, 6, 10, 6, 40])

    print("\nverdict: {0} - {1}".format(v.status.upper(), v.reason))
    print("csv.Sniffer().sniff() picks: {0!r}  ->  {1} columns".format(
        sniff.sniffer_says(text),
        sniff.shape_of(text, sniff.sniffer_says(text) or ",").modal))
    print("the file was written by a German ERP: the delimiter is ';' and '1,50' is one and a half.")
    print("\nRead the comma way, row 1 is:")
    print("   ", sniff.parse(text, ",")[0])
    print("Read the semicolon way, row 1 is:")
    print("   ", sniff.parse(text, ";")[0])
    print("\nBoth are 100% consistent across all 4 records. Nothing is ragged. Nothing raises.")


def exp2_header_breaks_tie() -> None:
    head(2, "A header row is 24 bytes and it settles the question")
    files = sniff.sample_files()
    for name in ("sensor.csv", "sales_eu.csv"):
        text = files[name].decode("utf-8")
        v = sniff.classify_delimiter(text)
        print("\n{0}  ({1} B)".format(name, len(files[name])))
        print("  first line : " + text.splitlines()[0])
        print("  status     : {0}".format(v.status))
        print("  viable     : " + ", ".join(
            "{0} -> {1} cols".format(s.label.split(" /")[0], s.modal) for s in v.viable))
        print("  sniffer    : {0!r}".format(sniff.sniffer_says(text)))
    print("\nThe header contains no commas, so the comma parse makes it a single field while the")
    print("body has three - ragged, and therefore ruled out. The same data without the header is")
    print("undecidable. Writing a header row is a data-quality control, not a formatting habit.")


def exp3_encoding() -> None:
    head(3, "A successful decode is not evidence; a failed one is")
    files = sniff.sample_files()
    for name in ("cp1252.csv", "utf8_umlaut.csv"):
        raw = files[name]
        r = sniff.probe_encoding(raw)
        print("\n{0}  ({1} B)   verdict: {2}".format(name, len(raw), r.verdict))
        print("  " + r.reason)
        rows: List[List[object]] = []
        for enc in sniff.ENCODINGS:
            ok = r.decodes[enc]
            sample = ""
            if ok:
                line = r.texts[enc].splitlines()[1] if len(r.texts[enc].splitlines()) > 1 else ""
                sample = line[:34]
            why = next((w for e, w in r.not_evidence if e == enc), "")
            rows.append([enc, "yes" if ok else "NO",
                         repr(sample)[:36] if ok else "-",
                         why[:30]])
        table(["encoding", "decodes", "row 1 as decoded", "success proves nothing because"],
              rows, [10, 7, 36, 30])

    print("\nThe C1 test is the discriminator that works: bytes 0x80-0x9f are printable characters")
    print("in cp1252 and undefined control codes in latin-1, and text does not contain control")
    print("codes. Its coverage in the other direction - utf-8 bytes read as latin-1 - is exactly")
    print("half, and the half it catches is not the half you would guess:")
    letters = [chr(c) for c in range(0xC0, 0x180)]
    caught = [c for c in letters if any(0x80 <= b <= 0x9F for b in c.encode("utf-8")[1:])]
    print("    {0} of {1} accented Latin letters encode with a continuation byte in 0x80-0x9f".format(
        len(caught), len(letters)))
    print("    caught : " + "".join(caught[:16]))
    print("    missed : " + "".join([c for c in letters if c not in caught][:16]))
    print("\nA code point's second utf-8 byte is 0x80 | (cp & 0x3f), so U+00C0-U+00DF land inside")
    print("the C1 range and U+00E0-U+00FF do not. Upper case is caught, lower case is not:")
    for a, b in sniff.mojibake_pairs(files["utf8_umlaut.csv"], "utf-8", "latin-1"):
        c1 = sum(1 for ch in b if "\x80" <= ch <= "\x9f")
        print("    utf-8 {0:<14} latin-1 {1:<20} C1 chars: {2}  {3}".format(
            repr(a), repr(b), c1, "DISQUALIFIED" if c1 else "passes, and is wrong"))
    print("\nOne caught character anywhere in the file disqualifies latin-1 for the whole file, so")
    print("on real prose the test usually fires. On a file of lower-case-only names it does not.")


def exp4_bom() -> None:
    head(4, "The BOM that becomes part of a column name")
    raw = sniff.sample_files()["bom.csv"]
    plain = raw.decode("utf-8")
    stripped = raw.decode("utf-8-sig")
    rows = sniff.parse(plain, ",")
    rows_s = sniff.parse(stripped, ",")
    print("\nfirst three bytes : {0!r}".format(raw[:3]))
    print("read as 'utf-8'      : header = {0!r}".format(rows[0]))
    print("read as 'utf-8-sig'  : header = {0!r}".format(rows_s[0]))
    print("\nprinted, they are identical:")
    print("    utf-8      -> " + "|".join(rows[0]))
    print("    utf-8-sig  -> " + "|".join(rows_s[0]))
    print("\nand compared, they are not:")
    print("    rows[0][0] == 'id'  ->  {0}".format(rows[0][0] == "id"))
    print("    len(rows[0][0])     ->  {0}   (for a two-character name)".format(len(rows[0][0])))
    print("\nSo df['id'] raises KeyError on a column that prints as id, in a file that opened")
    print("cleanly, on a dashboard that worked yesterday against an export from a different tool.")


def exp5_header_undecidable() -> None:
    head(5, "has_header() returns a bool for a question the file does not answer")
    files = sniff.sample_files()
    rows: List[List[object]] = []
    for name in ("sales_eu.csv", "years.csv", "alltext.csv", "sensor.csv"):
        text = files[name].decode("utf-8")
        v = sniff.classify_delimiter(text)
        d = v.preferred.delimiter if v.preferred else ","
        parsed = [r for r in sniff.parse(text, d) if r]
        h = sniff.classify_header(parsed, text)
        rows.append([name, text.splitlines()[0][:26], h.sniffer, h.status, h.basis])
    table(["file", "first line", "Sniffer", "this module", "basis"],
          rows, [14, 26, 8, 13, 18])
    print("\nyears.csv is the case that matters. Its first row is 2019,2020,2021 - column labels")
    print("that are integers, so every type test sees a numeric row over numeric rows and")
    print("concludes no header. alltext.csv is the mirror image: three text rows, and the first")
    print("one is data. Sniffer answers False for both, and it is wrong about one of them; which")
    print("one is not recoverable from the file.")
    print("\nWhat the two answers cost, on years.csv:")
    parsed = sniff.parse(files["years.csv"].decode("utf-8"), ",")
    print("    header=0    -> columns {0}, {1} data rows, sum of col 0 = {2}".format(
        parsed[0], len(parsed) - 1, sum(int(r[0]) for r in parsed[1:])))
    print("    header=None -> columns [0, 1, 2], {0} data rows, sum of col 0 = {1}".format(
        len(parsed), sum(int(r[0]) for r in parsed)))
    print("    a 2019 that is counted as a measurement inflates column 0 by {0}.".format(
        int(parsed[0][0])))


def exp6_sample_size() -> None:
    head(6, "Sniffing a prefix, and the letter i")
    text = sniff.sample_files()["late.csv"].decode("utf-8")
    rows: List[List[object]] = []
    for size in (64, 128, 256, 1024, len(text)):
        chunk = text[:size]
        label = "all ({0} B)".format(len(text)) if size >= len(text) else "{0} B".format(size)
        pick = sniff.sniffer_says(chunk)
        v = sniff.classify_delimiter(chunk)
        rows.append([label, repr(pick) if pick else "raises",
                     sniff.shape_of(chunk, pick).modal if pick else "-",
                     v.status, v.preferred.modal if v.preferred else "-"])
    table(["sample", "Sniffer picks", "its cols", "this module", "its cols"],
          rows, [14, 14, 9, 14, 9])
    print("\nAt 64 bytes Sniffer raises. At 128 bytes it returns {0!r} - not a delimiter it was".format(
        sniff.sniffer_says(text[:128])))
    print("offered, but a letter from the word 'Widget', because _guess_delimiter falls through to")
    print("any character with a consistent per-line frequency when no preferred candidate has one.")
    print("It does not raise: it returns a Dialect whose delimiter is 'i', and a 2-column frame.")
    print("\nThe file is 1,101 B. A 1 KB sniff sample - the common default - stops at record 60.")
    print("Record 61 is the only quoted field in the file:")
    print("    " + [ln for ln in text.splitlines() if '"' in ln][0])
    print("So the sample that chose the dialect never saw the row the dialect exists for.")


def exp7_line_counting() -> None:
    head(7, "Counting records needs the dialect; validating the dialect needs the count")
    files = sniff.sample_files()
    rows: List[List[object]] = []
    for name in ("quoted.csv", "mac.csv", "sales_eu.csv"):
        text = files[name].decode("utf-8")
        v = sniff.classify_delimiter(text)
        d = v.preferred.delimiter if v.preferred else ","
        t = sniff.probe_terminator(text, d)
        rows.append([name, t.verdict, t.naive_lines, t.records,
                     sum(t.inside.values()), t.records - t.naive_lines])
    table(["file", "terminator", "split('\\n')", "csv records", "nl in field", "delta"],
          rows, [14, 11, 12, 12, 12, 6])
    print("\nmac.csv uses a bare \\r, so str.split('\\n') returns one line for three records and")
    print("wc -l reports 0. quoted.csv has a newline inside a quoted address, so the naive count")
    print("is one too many. Both are off, in opposite directions, and neither raises.")
    print("\nThe circularity is the point: to know that a newline is inside a field you must know")
    print("the quotechar, which is part of the dialect you are trying to detect. scan_terminators()")
    print("takes the quotechar as an argument rather than pretending it can be inferred first.")

    text = files["quoted.csv"].decode("utf-8")
    print("\nAnd the setting this file does exercise, versus one it does not:")
    for name in ("quoted.csv", "sales_eu.csv"):
        v = sniff.classify_delimiter(files[name].decode("utf-8"))
        print("    {0:<14} untested: {1}".format(
            name, ", ".join(v.untested) if v.untested else "none - the file has quoted fields"))


def exp8_quotechar() -> None:
    head(8, "The wrong quotechar loses a row without changing the column count")
    files = sniff.sample_files()
    text = files["dutch.csv"].decode("utf-8")
    print("\ndutch.csv - Dutch surnames begin with an apostrophe:\n")
    for line in text.splitlines():
        print("    " + line)

    print('\nquotechar=\'"\' (4 records):')
    for r in sniff.parse(text, ",", '"'):
        print("    " + repr(r))
    print("\nquotechar=\"'\" (3 records):")
    for r in sniff.parse(text, ",", "'"):
        print("    " + repr(r))

    v = sniff.classify_delimiter(text)
    rows: List[List[object]] = []
    for s in v.viable:
        rows.append([s.label, s.records, s.modal, "{0:.0%}".format(s.consistency),
                     s.fields_with_newline])
    print()
    table(["dialect", "records", "fields", "consistent", "nl in field"], rows, [22, 9, 8, 11, 12])
    print("\nverdict: {0} - {1}".format(v.status.upper(), v.reason))
    print("\nBoth parses are 3 columns wide and 100% consistent. One of them has three records")
    print("where the file has four: the apostrophe on 't Hooft opened a quoted field that ran to")
    print("the apostrophe on 's Gravesande, absorbing the record terminator between them. Every")
    print("check that watches the column count passes. The row is simply gone, and the two names")
    print("it merged are now one field containing a newline - which is why fields_with_newline is")
    print("the signal, and why the tie-break prefers the parse that keeps more records.")


def ledger() -> None:
    head(9, "The ledger")
    files = sniff.sample_files()
    rows: List[List[object]] = [
        ["two clean parses, 3 vs 4 cols", "sensor.csv", "wrong column count", "silent"],
        ["utf-8 read as latin-1", "utf8_umlaut.csv", "AusfÃ¼hrung in every row", "silent"],
        ["latin-1 read for a cp1252 file", "cp1252.csv", "C1 controls in the text", "silent"],
        ["BOM kept as a name character", "bom.csv", "KeyError on a visible column", "raises late"],
        ["numeric header read as data", "years.csv", "col 0 inflated by 2019", "silent"],
        ["text data row read as header", "alltext.csv", "one row lost, names wrong", "silent"],
        ["apostrophe read as a quotechar", "dutch.csv", "1 of 4 records merged away", "silent"],
        ["prefix sniff picks 'i'", "late.csv", "1-column frame", "silent"],
        ["bare \\r line ending", "mac.csv", "3 records counted as 1", "silent"],
        ["newline inside a quoted field", "quoted.csv", "line count one too high", "silent"],
    ]
    table(["failure mode", "sample", "effect", "raises?"], rows, [32, 16, 26, 10])
    audits = {n: sniff.audit(r, n) for n, r in files.items()}
    undecided = [n for n, a in audits.items() if not a.decided]
    print("\n{0} of the {1} sample files are fully determined by their own bytes.".format(
        len(files) - len(undecided), len(files)))
    print("The other {0} are not: {1}".format(len(undecided), ", ".join(sorted(undecided))))
    print("Nine of the ten failure modes above produce a plausible answer with no exception, and")
    print("all ten are reproducible - re-running the load gives the same wrong number, so a")
    print("reconciliation against yesterday agrees.")


def main() -> None:
    exp1_contested()
    exp2_header_breaks_tie()
    exp3_encoding()
    exp4_bom()
    exp5_header_undecidable()
    exp6_sample_size()
    exp7_line_counting()
    exp8_quotechar()
    ledger()
    print()


if __name__ == "__main__":
    main()
