"""Every claim in the README, printed from the live truncators.

Run it:  python evidence.py
"""

from __future__ import annotations

import unicodedata

import uwidth as U


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show(text: str, width: int = 30) -> str:
    """Render a string so its invisible parts are visible on a terminal."""
    out = []
    for ch in text:
        cp = ord(ch)
        if ch == U.ZWJ:
            out.append("<ZWJ>")
        elif ch in (U.VS15, U.VS16):
            out.append("<VS>")
        elif 0xD800 <= cp <= 0xDFFF:
            out.append(f"<D{cp:04X}>")
        elif ch == "​":
            out.append("<ZWSP>")
        elif ch in U.BIDI_OPEN or ch in U.BIDI_CLOSE:
            out.append(f"<U+{cp:04X}>")
        elif cp < 0x20 or cp == 0x7F:
            out.append(f"<{cp:02X}>")
        else:
            out.append(ch)
    return "".join(out)


def section_1_roster() -> None:
    rule("1. Ten truncators. Same string, same N, ten answers.")
    print(f"{'truncator':<20} {'unit':<18} where you meet it")
    print("-" * 78)
    for t in U.TRUNCATORS:
        print(f"{t.name:<20} {t.unit:<18} {t.seen_in}")
    v = U.node_versions()
    print(f"\nUTF-16 and ICU truncators are a real node subprocess: "
          f"ICU {v['icu']}, Unicode {v['unicode']}.")
    print("Grapheme clusters on the Python side are regex's UAX #29 \\X.")
    print("\nNone of these is wrong. Each is correct for the limit it was written to protect.")


def section_2_one_string() -> None:
    rule("2. One bio, one N, six different strings")
    case = U.CASE_BY_NAME["emoji-family"]
    print(f"input : {show(case.text)}")
    print(f"        {case.source}, truncate to n={case.n}\n")
    cuts = U.cut_all(case)
    print(f"{'truncator':<20} {'output':<34} {'B':>3} {'cp':>3} {'gr':>3} {'col':>4}  note")
    print("-" * 100)
    for name, cut in cuts.items():
        note = []
        if not cut.well_formed:
            note.append("LONE SURROGATE" if cut.lone_surrogate else "U+FFFD")
        if cut.dangling:
            note.append(f"ends in {cut.dangling}")
        change = U.identity_change(case.text, cut.text)
        if change:
            note.append(change)
        print(f"{name:<20} {show(cut.text):<34} {cut.bytes_out:>3} {cut.code_points:>3} "
              f"{cut.grapheme_count:>3} {cut.columns_out:>4}  {'; '.join(note)}")
    print(f"\ndistinct strings out of one truncate(s, {case.n}): "
          f"{len({c.text for c in cuts.values()})}")


def section_3_unit_spread() -> None:
    rule("3. The limit is 20 what? One string, six lengths.")
    for name in ("emoji-family", "cjk-bio", "devanagari", "url"):
        case = U.CASE_BY_NAME[name]
        spread = U.unit_spread(case)
        print(f"\n{name}: {show(case.text)}")
        print("   " + "  ".join(f"{k}={v}" for k, v in spread.items()))
    print("\nSame value. A limit written as a bare integer is satisfied or violated")
    print("depending on which of these the layer enforcing it happens to count.")


def section_4_ddl() -> None:
    rule("4. The same DDL, six different capacities")
    print(f"{'sink':<26} {'unit':<15} note")
    print("-" * 78)
    for s in U.SINKS:
        print(f"{s.name:<26} {s.unit:<15} {s.note}")
    bio = U.CASE_BY_NAME["emoji-family"].text
    print("\nHow much of this bio fits in a limit of 20, per sink?")
    print(f"  {show(bio)}\n")
    for s in U.SINKS:
        kept = U.safe_truncate(bio, 20, s.name)
        print(f"  {s.name:<26} {s.measure(bio):>3} {s.unit:<14} -> keeps {show(kept)}")


def section_5_not_text() -> None:
    rule("5. Two ways a cut stops producing text")
    print("a) A byte cut inside a multi-byte sequence -> U+FFFD in the next consumer.")
    print("b) A UTF-16 cut inside a surrogate pair -> a lone surrogate, which has")
    print("   NO UTF-8 encoding at all. It is a valid JavaScript string and an")
    print("   impossible byte string; it raises at the first encode(), JSON round")
    print("   trip, or database write.\n")
    surr = [(c.name, n) for c in U.CORPUS for n, cut in U.cut_all(c).items() if cut.lone_surrogate]
    fffd = [(c.name, n) for c in U.CORPUS for n, cut in U.cut_all(c).items()
            if U.has_replacement(cut.text)]
    print(f"lone surrogate produced in {len(surr)} cuts, all by utf16_units:")
    for name, t in surr:
        print(f"    {name:<16} {t}")
    print(f"\nU+FFFD produced in {len(fffd)} cuts:")
    for name, t in fffd:
        print(f"    {name:<16} {t}")
    js = U._node_one("utf16_units", U.CASE_BY_NAME["emoji-family"].text, 9)
    print(f"\nlive: node .slice(0, 9) -> {show(js)}")
    try:
        js.encode("utf-8")
        print("    encodes to UTF-8: yes")
    except UnicodeEncodeError as exc:
        print(f"    encodes to UTF-8: NO -> {exc.reason}")


def section_6_overflow() -> None:
    rule("6. The byte truncator that exceeds its own byte limit")
    print("s.encode()[:n] cuts a 4-byte emoji after its first byte. That partial")
    print("byte is decoded to U+FFFD, which re-encodes to THREE bytes. The value")
    print("enforcing a limit of n comes back longer than n.\n")
    rows = [(c.name, name, cut.bytes_out, c.n)
            for c in U.CORPUS for name, cut in U.cut_all(c).items()
            if cut.overflows_own_limit]
    print(f"{'case':<18} {'truncator':<20} {'bytes out':>9} {'limit':>6}")
    print("-" * 60)
    for case, name, got, limit in rows:
        print(f"{case:<18} {name:<20} {got:>9} {limit:>6}   +{got - limit}")
    print(f"\n{len(rows)} of {len(U.CORPUS)} cases. Worst overrun: "
          f"+{max(got - lim for _, _, got, lim in rows)} bytes over a limit that was being enforced.")
    print("\nminimal:")
    s = "aa\U0001F600"
    cut = s.encode()[:3].decode("utf-8", "replace")
    print(f"    {s!r} -> encode()[:3] -> {cut!r} -> re-encode = {len(cut.encode())} bytes for n=3")


def section_7_identity() -> None:
    rule("7. Cuts that return a different valid thing")
    print("Nothing is malformed here. The output renders cleanly, passes every")
    print("encoding check, and means something else.\n")
    print(f"{'meant':<10} {'got':<10} what changed")
    print("-" * 60)
    for whole, part, desc in U.IDENTITY_PROBES:
        print(f"{whole:<10} {part:<10} {desc}")
    print("\nIn the corpus:")
    for case in U.CORPUS:
        for name, cut in U.cut_all(case).items():
            change = U.identity_change(case.text, cut.text)
            if change:
                print(f"    {case.name:<16} {name:<20} {change}")
    print("\nNo validator fires on any of these. There is nothing wrong with the result;")
    print("it is simply not the value that was stored.")


def section_8_dangling() -> None:
    rule("8. What the next concatenation does to a dangling joiner")
    rows = [(c.name, name, cut.dangling, cut.text)
            for c in U.CORPUS for name, cut in U.cut_all(c).items() if cut.dangling]
    print(f"{'case':<18} {'truncator':<20} trailing code point")
    print("-" * 70)
    for case, name, what, _ in rows:
        print(f"{case:<18} {name:<20} {what}")
    print(f"\n{len(rows)} cuts end in a code point that binds to whatever comes next.\n")
    fam = U.CASE_BY_NAME["emoji-family"].text
    stub = U._node_one("utf16_safe_cp", fam, 10)
    print(f"cut  : {show(stub)}")
    for suffix, label in (("…", "an ellipsis"), ("\U0001F467", "the next page of a paginated render")):
        joined = stub + suffix
        print(f"  + {label:<42} -> {show(joined)}  "
              f"({len(U.graphemes(stub))} + 1 clusters -> {len(U.graphemes(joined))})")
    print("\nThe ZWJ does not stay dangling. It joins, and two independent values")
    print("fuse into one glyph that was in neither of them.")


def section_9_normalisation() -> None:
    rule("9. The same visible name, two normal forms, two truncations")
    nfc = U.CASE_BY_NAME["accent-nfc"]
    nfd = U.CASE_BY_NAME["accent-nfd"]
    print(f"NFC input: {nfc.text!r}  ({len(nfc.text)} code points)")
    print(f"NFD input: {nfd.text!r}  ({len(nfd.text)} code points)")
    print(f"They render identically. Truncate both to {nfc.n} code points:\n")
    a = U.TRUNCATOR_BY_NAME["code_points"].cut(nfc.text, nfc.n)
    b = U.TRUNCATOR_BY_NAME["code_points"].cut(nfd.text, nfd.n)
    print(f"    NFC -> {a!r}")
    print(f"    NFD -> {b!r}")
    print(f"\n    equal? {a == b}")
    print(f"    visible characters kept: NFC {len(U.graphemes(a))}, NFD {len(U.graphemes(b))}")
    print("\nThe same limit of 12 stores 12 characters of one and 10 of the other,")
    print("because the marks are counted as characters in one form and not the other.\n")
    print("Move the cut to 9, where it lands between a letter and its mark:")
    print(f"    NFC -> {nfc.text[:9]!r}")
    print(f"    NFD -> {unicodedata.normalize('NFC', nfd.text[:9])!r}")
    print("\n'Muñ' became 'Mun'. The accent is gone, the name is a different name,")
    print("the output is perfectly well-formed and nothing raised.")
    print("Which normal form arrives is decided by the client OS, not by you:")
    print("macOS file APIs hand over NFD, most web forms hand over NFC.")


def section_10_segmenters() -> None:
    rule("10. Two UAX #29 implementations in one pipeline, two answers")
    v = U.node_versions()
    print(f"Python: regex \\X, UCD {unicodedata.unidata_version}")
    print(f"Node  : Intl.Segmenter, ICU {v['icu']} / Unicode {v['unicode']}\n")
    dis = U.segmenter_disagreements()
    print(f"{'case':<18} {'regex':>6} {'ICU':>6}  text")
    print("-" * 60)
    for name, py, js in dis:
        print(f"{name:<18} {py:>6} {js:>6}  {U.CASE_BY_NAME[name].text}")
    print(f"\n{len(dis)} of {len(U.CORPUS)} corpus strings are counted differently.")
    for probe in ("क्ष", "क्षि", "நி", "กำ"):
        py = len(U.graphemes(probe))
        js = int(U._node_one("count_graphemes", probe, 0))
        mark = "  <- disagree" if py != js else ""
        print(f"    {probe:<8} regex={py}  ICU={js}{mark}")
    print("\nBoth implement UAX #29 correctly for the UCD they ship. The Indic")
    print("conjunct rule (GB9c) changed the answer, so an API written in Node and")
    print("a worker written in Python disagree about how long a Hindi name is.")
    print("A limit of 6 'characters' is two different limits inside one service.")


def section_11_width() -> None:
    rule("11. Fitting N units is not fitting N columns")
    print(f"{'case':<16} {'n':>3} {'truncator':<20} {'unit len':>9} {'columns':>8}")
    print("-" * 64)
    for name in ("cjk-bio", "cjk-mixed", "emoji-run", "mixed-width"):
        case = U.CASE_BY_NAME[name]
        for t_name in ("code_points", "grapheme_columns"):
            cut = U.cut_all(case)[t_name]
            print(f"{name:<16} {case.n:>3} {t_name:<20} {cut.code_points:>9} {cut.columns_out:>8}")
    print("\nA CJK code point is two columns wide. Cutting a name to 12 code points")
    print("for a 12-wide report cell puts 24 columns in it and every following")
    print("column in the table shifts right.\n")
    fam = "\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466"
    print("And width is not always computable: wcwidth measures the family emoji")
    print(f"    {fam}  as {U.columns(fam)} columns.")
    print("    A terminal that renders the ZWJ sequence draws 2.")
    print(f"    A terminal that does not draws {U.columns(fam)}.")
    print("    Both exist. The string does not carry the answer; the renderer decides.")


def section_12_bidi() -> None:
    rule("12. A cut that leaves an unbalanced bidi override")
    case = U.CASE_BY_NAME["bidi-override"]
    print(f"input   : {show(case.text)}   (balance {U.bidi_balance(case.text)})")
    for t_name in ("code_points", "py_graphemes", "grapheme_columns"):
        cut = U.cut_all(case)[t_name]
        print(f"{t_name:<16}: {show(cut.text):<28} balance {cut.bidi_leak}")
    print("\nThe input opens an RTL override and closes it. A cut between the two")
    print("keeps the opener and drops the closer, so the override escapes the value")
    print("and reverses text that belongs to whatever renders it next - a table")
    print("header, a neighbouring cell, the rest of the line.")
    print("safe_truncate drops trailing clusters until the balance is zero again.")


def section_13_ellipsis() -> None:
    rule("13. The ellipsis has to come out of the budget")
    print("one ellipsis costs:", U.ellipsis_cost())
    over = U.naive_ellipsis_overflow()
    performed = sum(1 for c in U.CORPUS for t in U.TRUNCATORS if t.cut(c.text, c.n) != c.text)
    print(f"\ncut to exactly n, then append '...': over the limit in "
          f"{len(over)} of {performed} cuts that removed anything.")
    print("\nA '...' is 3 bytes, 1 code point, 1 UTF-16 unit and 1 column. Appending it")
    print("to a value that is exactly at the limit puts it over in every unit at once.")
    print("safe_truncate subtracts the marker from the budget before it starts.")


def section_14_sinks() -> None:
    rule("14. Truncating to 20 does not make it fit in a limit of 20")
    failed, total = U.sink_failure_rate()
    print(f"{len(U.TRUNCATORS)} truncators x {len(U.SINKS)} sinks x {len(U.CORPUS)} cases "
          f"= {total} runs")
    print(f"    still over the limit: {failed}  ({failed / total:.0%})\n")
    per_sink = {}
    for _, _, sink, _, _ in U.sink_failures():
        per_sink[sink] = per_sink.get(sink, 0) + 1
    print(f"{'sink':<26} {'over limit':>11} of {len(U.CORPUS) * len(U.TRUNCATORS)}")
    print("-" * 52)
    for s in U.SINKS:
        print(f"{s.name:<26} {per_sink.get(s.name, 0):>11}")
    print("\nA truncator is correct relative to one unit. Pointed at a sink that")
    print("counts a different unit it is not approximately right, it is unrelated.")


def section_15_answer() -> None:
    rule("15. The only question that decides a truncator")
    print("Not 'how do I cut a string safely'. There is no safe cut in the abstract.")
    print("The question is: WHAT DOES THE THING I AM PROTECTING COUNT?\n")
    for s in U.SINKS:
        print(f"    {s.name:<26} counts {s.unit:<15} -> use {U.choose_truncator(s.name)}")
    audit = U.safe_truncate_audit()
    ok = sum(1 for _, _, f, d, w in audit if f and d and w)
    print(f"\nsafe_truncate(text, n, sink) over {len(U.CORPUS)} cases x {len(U.SINKS)} sinks "
          f"= {len(audit)} runs")
    print(f"    fits the sink's own measure, no dangling joiner, no split boundary: "
          f"{ok} / {len(audit)}")
    print("\nIt does three things the roster truncators do not all do: measures in the")
    print("sink's unit, reserves the ellipsis inside the budget, and never splits a")
    print("cluster - then drops trailing clusters until nothing dangles and the bidi")
    print("balance is zero.\n")
    case = U.CASE_BY_NAME["emoji-family"]
    for s in U.SINKS:
        print(f"    {s.name:<26} -> {show(U.safe_truncate(case.text, case.n, s.name))}")


def section_16_census() -> None:
    rule("16. The corpus")
    distinct, total = U.distinct_output_count()
    print(f"{len(U.CORPUS)} strings x {len(U.TRUNCATORS)} truncators = {total} cuts, "
          f"{distinct} distinct outputs\n")
    print(f"{'case':<18} {'n':>3} {'out':>4} {'verdict':<17} detail")
    print("-" * 96)
    for case in U.CORPUS:
        v = U.verdict_for(case)
        print(f"{case.name:<18} {case.n:>3} {v.distinct_outputs:>4} {v.verdict:<17} {v.detail}")
    print("\nverdict census (the worst thing that happened to each case):")
    for k, n in U.verdict_census().items():
        print(f"    {k:<18} {n}")
    print("\nflag census (every thing that happened, cases may carry several):")
    for k, n in U.flag_census().items():
        print(f"    {k:<18} {n}")
    agreed = U.verdict_census()["agreed"]
    print(f"\n{agreed} of {len(U.CORPUS)} strings are cut identically by all ten truncators.")
    print("Both are pure ASCII. Every string in this corpus that is not ASCII has")
    print("more than one defensible truncation, and the integer n does not say which.")


def main() -> None:
    for fn in (
        section_1_roster, section_2_one_string, section_3_unit_spread, section_4_ddl,
        section_5_not_text, section_6_overflow, section_7_identity, section_8_dangling,
        section_9_normalisation, section_10_segmenters, section_11_width, section_12_bidi,
        section_13_ellipsis, section_14_sinks, section_15_answer, section_16_census,
    ):
        fn()
    print()


if __name__ == "__main__":
    main()
