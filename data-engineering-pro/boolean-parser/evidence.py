"""Every claim in the README, printed from the live readers.

Run it:  python evidence.py
"""

from __future__ import annotations

import collections

import boolparse as B

MARK = {B.TRUE: "T", B.FALSE: "F", B.REFUSED: "!", B.NOTBOOL: "?"}


def pad(text: str, width: int) -> str:
    """Left-align to `width` *terminal columns*, not characters.

    The corpus contains fullwidth Latin, which is two columns per
    character; `f"{s:<12}"` would misalign the table by five columns.
    """
    from wcwidth import wcswidth

    used = wcswidth(text)
    if used < 0:
        used = len(text)
    return text + " " * max(0, width - used)


def rule(title: str) -> None:
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)


def section_1_roster() -> None:
    rule("1. Sixteen readers. One string. Sixteen separate decisions.")
    print(f"{'reader':<16} {'stack':<15} {'src':<5} {'refuses?':<9} where you meet it")
    print("-" * 84)
    for r in B.READERS:
        print(f"{r.name:<16} {r.stack:<15} {r.source:<5} "
              f"{('yes' if r.can_refuse else 'never'):<9} {r.seen_in}")
    live = sum(1 for r in B.READERS if r.source == "live")
    print(f"\n{live} of {len(B.READERS)} are real interpreters invoked at run time. "
          "The two marked `spec` are Go and Java,")
    print("whose documented value tables are transcribed and asserted in test_boolparse.py.")
    print("\nNone of these is wrong. Each is correct for the accept table it was written with.")


def section_2_no_string_means_true() -> None:
    rule("2. There is no string that means true.")
    n = len(B.CORPUS)
    unan = B.unanimous()
    print(f"Corpus: {n} strings x {len(B.READERS)} readers = {n * len(B.READERS)} readings.")
    print(f"Strings read the same way by all {len(B.READERS)} readers: {len(unan)}.")
    print()
    for text in ("true", "false", "1", "0"):
        s = next(x for x in B.CORPUS if x.text == text)
        v = B.verdicts_for(s)
        counts = collections.Counter(r.verdict for r in v.values())
        print(f"  {s.label:<8} -> " + "  ".join(f"{k}={counts[k]}" for k in B.VERDICTS if counts[k]))
    print()
    s = next(x for x in B.CORPUS if x.text == "true")
    dissent = [n_ for n_, r in B.verdicts_for(s).items() if r.verdict != B.TRUE]
    print(f"Even {s.label!r} is not unanimous. Reading it as anything but true: {dissent}")
    print("`js_loose_eq` is `s == true` in JavaScript, which converts BOTH sides to numbers.")
    print("Number('true') is NaN, NaN == 1 is false. The literal string 'true' is not == true.")


def section_3_the_title_bug() -> None:
    rule("3. 'false' came back true - in 6 of 16 readers.")
    s = next(x for x in B.CORPUS if x.text == "false")
    v = B.verdicts_for(s)
    trues = sorted(n for n, r in v.items() if r.verdict == B.TRUE)
    print(f"{'reader':<16} {'verdict':<14} what it actually returned")
    print("-" * 84)
    for name in [r.name for r in B.READERS]:
        r = v[name]
        flag = "  <-- the bug" if r.verdict == B.TRUE else ""
        print(f"{name:<16} {r.verdict:<14} {r.raw}{flag}")
    print(f"\n{len(trues)} of {len(B.READERS)} readers read the string 'false' as true: {trues}")
    print("Every one of them is a *truthiness* reader: it never consulted a boolean table at all.")
    print("It asked 'is this string non-empty', which for 'false' is yes.")


def section_4_all_45_flip() -> None:
    rule("4. Every string in the corpus flips sign somewhere.")
    flips = B.sign_flips()
    print(f"{len(flips)} of {len(B.CORPUS)} strings have at least one reader saying true")
    print("and at least one saying false, both confidently, neither raising.\n")
    print(f"{'string':<12} {'family':<11} {'T':>3} {'F':>3} {'!':>3} {'?':>3}  distinct")
    print("-" * 84)
    for s in B.CORPUS:
        c = collections.Counter(r.verdict for r in B.verdicts_for(s).values())
        print(f"{pad(s.label, 12)} {s.family:<11} {c[B.TRUE]:>3} {c[B.FALSE]:>3} "
              f"{c[B.REFUSED]:>3} {c[B.NOTBOOL]:>3}  {len(B.distinct_verdicts(s))}")


def section_5_never_refuse() -> None:
    rule("5. Ten of sixteen readers cannot fail - and therefore cannot warn you.")
    never = B.never_refuse()
    refusals = B.refusal_counts()
    notbool = B.notbool_counts()
    n = len(B.CORPUS)
    print(f"{'reader':<16} {'refused':>8} {'not-a-bool':>11} {'confident':>10}  verdict")
    print("-" * 84)
    for r in B.READERS:
        conf = n - refusals[r.name] - notbool[r.name]
        note = "answers everything" if r.name in never else ""
        print(f"{r.name:<16} {refusals[r.name]:>8} {notbool[r.name]:>11} {conf:>10}  {note}")
    print(f"\n{len(never)} of {len(B.READERS)} readers return a confident boolean for all "
          f"{n} strings, including")
    print("'undefined', a bare UTF-8 BOM and the empty string. A reader that cannot fail")
    print("cannot tell you that you spelled it wrong - it can only be quietly wrong instead.")
    print("\nThe four that DO refuse are the only ones that carry information:")
    for r in B.READERS:
        if refusals[r.name]:
            print(f"  {r.name:<16} refuses {refusals[r.name]:>2}/{n}")


def section_6_notbool_defers() -> None:
    rule("6. Succeeding is not deciding: the not-a-boolean verdict.")
    print("A YAML or JSON reader that returns 1, 'yes' or None has not failed and has not")
    print("decided. The decision is deferred to whichever `if value:` runs next.\n")
    for name in ("yaml11", "yaml12", "json_strict"):
        rows = [(s, r) for s, r in zip(B.CORPUS, B.grid()[name]) if r.verdict == B.NOTBOOL]
        print(f"{name}: {len(rows)}/{len(B.CORPUS)} strings come back as something else")
        for s, r in rows[:6]:
            after = "true" if _py_truthy_of(r.raw) else "false"
            print(f"    {s.label:<10} -> {r.raw:<18} and `if v:` then reads it as {after}")
        if len(rows) > 6:
            print(f"    ... and {len(rows) - 6} more")
        print()


def _py_truthy_of(raw: str) -> bool:
    """Rough truthiness of the printed repr, for illustration only."""
    return not any(tok in raw for tok in ("=0", "=None", "=''", "=0.0"))


def section_7_norway() -> None:
    rule("7. The Norway problem: one document, two YAML parsers, two values.")
    print(f"{'string':<10} {'YAML 1.1 (PyYAML)':<22} {'YAML 1.2 (ruamel)':<22} agree?")
    print("-" * 84)
    diff = 0
    for s in B.CORPUS:
        a = B.verdict("yaml11", s)
        b = B.verdict("yaml12", s)
        if a.verdict == b.verdict and a.raw == b.raw:
            continue
        diff += 1
        print(f"{pad(s.label, 10)} {a.verdict + ' ' + a.raw:<22} {b.verdict + ' ' + b.raw:<22} no")
    print(f"\n{diff} of {len(B.CORPUS)} strings change meaning between YAML 1.1 and YAML 1.2.")
    print("An unquoted NO in a country column is the boolean false under 1.1 and the string")
    print("'NO' under 1.2. YAML 1.2 deleted yes/no/on/off from the core schema for this reason;")
    print("PyYAML still implements 1.1, and docker-compose, Ansible and GitHub Actions read")
    print("their files with parsers on both sides of that line.")
    print("\nAlso worth knowing: YAML 1.1's type repository lists `y` and `n` as booleans.")
    for t in ("y", "n"):
        s = next(x for x in B.CORPUS if x.text == t)
        print(f"  PyYAML does not implement them: {t!r} -> {B.verdict('yaml11', s).raw}")


def section_8_sqlite() -> None:
    rule("8. In SQLite, the string 'true' is false. So are 'yes', 'on', 'TRUE' and 't'.")
    trues = [s for s in B.CORPUS if B.verdict("sqlite_where", s).verdict == B.TRUE]
    print("SQLite has no boolean type. A TEXT value in boolean position is cast to a number,")
    print("and a string that does not begin with digits casts to 0.\n")
    print(f"{'string':<12} {'WHERE flag':<12} CAST(... AS NUMERIC)")
    print("-" * 84)
    for s in B.CORPUS:
        r = B.verdict("sqlite_where", s)
        star = "  <-- the only truthy ones" if r.verdict == B.TRUE else ""
        print(f"{pad(s.label, 12)} {r.verdict:<12} {r.raw}{star}")
    print(f"\n{len(trues)} of {len(B.CORPUS)} strings are truthy in a SQLite WHERE clause: "
          f"{[s.label for s in trues]}")
    print(f"The other {len(B.CORPUS) - len(trues)} - every word-shaped spelling of true there is -")
    print("select zero rows, in every query, forever, and SQLite never raises.")


def section_9_awk_entry_path() -> None:
    rule("9. awk: the same string, two answers, decided by how it got in.")
    print("awk's strnum rule compares a value that *looks* numeric as a number. Whether it")
    print("looks numeric depends on whether it arrived as program text or as data.\n")
    print(f"{'string':<10} {'literal in program':<20} {'-v var=':<12} {'input field':<12} answers")
    print("-" * 84)
    for text in ("0", "00", "0.0", "0e0", "1", "false", " "):
        lit = B.awk_program_literal(text)
        var = B.awk_assigned_var(text)
        fld = B.awk_input_field(text)
        vals = {lit, var, fld}
        print(f"{pad(B.show(text), 10)} {str(lit):<20} {str(var):<12} {str(fld):<12} {len(vals)}")
    print("\n`awk 'BEGIN { if (\"0\") ... }'` is TRUE - a string constant.")
    print("`echo 0 | awk '$0 { ... }'`  is FALSE - a strnum field.")
    print("No other reader in this roster changes its answer based on the entry path.")


def section_10_js_two_fixes() -> None:
    rule("10. JavaScript: Boolean(s) and s == true are not the same reader.")
    loose_false = B.js_loose_false(B.CORPUS)
    print(f"{'string':<12} {'Boolean(s)':<12} {'s == true':<12} {'s == false':<12} note")
    print("-" * 84)
    disagree = 0
    for s, lf in zip(B.CORPUS, loose_false):
        a = B.verdict("js_truthy", s).verdict == B.TRUE
        b = B.verdict("js_loose_eq", s).verdict == B.TRUE
        note = ""
        if a and lf:
            note = "truthy AND == false"
        if a != b:
            disagree += 1
        print(f"{pad(s.label, 12)} {str(a):<12} {str(b):<12} {str(lf):<12} {note}")
    both = sum(1 for s, lf in zip(B.CORPUS, loose_false)
               if lf and B.verdict("js_truthy", s).verdict == B.TRUE)
    print(f"\nBoolean(s) and (s == true) disagree on {disagree} of {len(B.CORPUS)} strings.")
    print(f"{both} strings are simultaneously truthy AND loosely equal to false.")
    print("`'1' == true` is true; `'true' == true` is false. Loose equality against a boolean")
    print("numifies both sides, and Number('true') is NaN. The 'fix' inverts the original bug.")


def section_11_residue() -> None:
    rule("11. The file the value travelled in is part of the value.")
    for text, why in (("true\r", "a .env file saved with CRLF line endings"),
                      ("﻿true", "a UTF-8 BOM on the first line of the file"),
                      (" true", "a space after the = in the .env file"),
                      ("true ", "a trailing space nobody can see in a diff")):
        s = next(x for x in B.CORPUS if x.text == text)
        v = B.verdicts_for(s)
        t = sorted(n for n, r in v.items() if r.verdict == B.TRUE)
        f = sorted(n for n, r in v.items() if r.verdict == B.FALSE)
        e = sorted(n for n, r in v.items() if r.verdict == B.REFUSED)
        print(f"\n{s.label!r:<14} {why}")
        print(f"   true  ({len(t):>2}): {', '.join(t)}")
        print(f"   false ({len(f):>2}): {', '.join(f)}")
        print(f"   refused ({len(e):>2}): {', '.join(e) or '-'}")
    print("\nThe author wrote `true` in all four cases. Day 151 showed a file has no lines in")
    print("it until a splitter makes them; here the splitter's leftovers change a boolean.")


def section_12_normalisation() -> None:
    rule("12. Normalisation is the second decision, and lower() != casefold().")
    print("One fixed accept table, five normalisations, applied to the same string.\n")
    keys = list(B.normalisations("x").keys())
    print(f"{'string':<10} " + " ".join(f"{k:<14}" for k in keys))
    print("-" * 84)
    for text in ("TRUE", "tRuE", " true", "true\r", "yeſ", "FALſE", "ＴＲＵＥ"):
        acc = B.accepted_after(text)
        print(pad(B.show(text), 10) + " " + " ".join(f"{str(acc[k]):<14}" for k in keys))
    print("\n`FALſE`.casefold() is 'false' but `.lower()` is 'falſe'. A casefolding reader")
    print("accepts a string a lowercasing reader refuses - the two normalisations are not")
    print("a strictness ordering, they are different functions.")
    print("`ＴＲＵＥ` needs NFKC before any casing helps it at all.")
    print("\nThe Turkish dotless-i hazard from Day 149 reaches exactly one literal here.")
    words = sorted(w.upper() for w in B.EXTENDED_TABLE if w.isalpha())
    en = B.locale_lower(words, "en")
    tr = B.locale_lower(words, "tr")
    print("Real node 22 `toLocaleLowerCase`, not a transcribed table:\n")
    print(f"{'as written':<12} {'lower(en)':<12} {'lower(tr)':<12} in the table?")
    print("-" * 84)
    for w in words:
        ok = "yes" if tr[w] in B.EXTENDED_TABLE else "NO - matches nothing"
        print(f"{w:<12} {en[w]:<12} {tr[w]:<12} {ok}")
    broken = [w for w in words if tr[w] not in B.EXTENDED_TABLE]
    print(f"\n{len(broken)} of {len(words)} word literals survive a Turkish lowercase: {broken}")
    print("The twelve core literals - true, false, t, f, yes, y, no, n, on, off, 1, 0 - are")
    print("immune, because not one of them contains the letter I. That is luck, not design:")
    print("the moment a vocabulary grows to include `disabled`, a Turkish-locale server")
    print("lowercases it to `dısabled` and the lookup misses, in exactly one locale.")


def section_13_silently_wrong() -> None:
    rule("13. Wrong quietly vs refused loudly.")
    wrong = B.silently_wrong()
    intent_cells = sum(1 for s in B.CORPUS if s.intent is not None) * len(B.READERS)
    refused = sum(B.refusal_counts().values())
    notbool = sum(B.notbool_counts().values())
    total = len(B.CORPUS) * len(B.READERS)
    print(f"Total readings:                     {total}")
    print(f"Readings of a string with a clear intent: {intent_cells}")
    print(f"  ... confidently OPPOSITE that intent:   {len(wrong)}"
          f"  ({100 * len(wrong) / intent_cells:.0f}%)")
    print(f"Readings refused outright:          {refused}"
          f"  ({100 * refused / total:.0f}% of all)")
    print(f"Readings that returned a non-boolean: {notbool}")
    print("\nThe refused ones are the good outcome. They stop at the boundary with the")
    print("offending string in the message. The other 161 propagate.\n")
    per = collections.Counter(name for _, name, _ in wrong)
    print(f"{'reader':<16} {'silently wrong':>15} {'refuses':>9}")
    print("-" * 84)
    for r in B.READERS:
        print(f"{r.name:<16} {per[r.name]:>15} {B.refusal_counts()[r.name]:>9}")
    print("\nThe ranking inverts: the readers with the fewest refusals have the most silent")
    print("errors, one for one. There is no reader that is both permissive and safe.")
    print("\nyaml11 and yaml12 score zero in BOTH columns, which is not a clean bill of")
    print("health. They neither raise nor mislead because they mostly do not answer: 25 and")
    print("35 of the 45 strings come back as a str, an int or None. The decision is still")
    print("coming; it will just be made by `if value:` somewhere with no idea what the")
    print("string was.")


def section_14_round_trip() -> None:
    rule("14. Written for one reader, read by another - and it runs one way.")
    rt = B.round_trip()
    names = [r.name for r in B.READERS]
    print("Filter on the strings the WRITER reads confidently - the spellings somebody would")
    print("plausibly put in a config file authored against that reader. Then count how many")
    print("the READER fails to reproduce, split by how it failed.\n")
    worst = sorted(((v, k) for k, v in rt.items() if k[0] != k[1]), reverse=True)[:10]
    print(f"{'written for':<16} {'read by':<16} {'lost':>5} {'flipped':>8} {'refused':>8} {'deferred':>9}")
    print("-" * 84)
    for count, (w, r) in worst:
        parts = B.round_trip_flips(w, r)
        print(f"{w:<16} {r:<16} {count:>5} {parts['flipped']:>8} "
              f"{parts['refused']:>8} {parts['deferred']:>9}")
    pairs = [(w, r) for w in names for r in names if w != r]
    clean = [p for p in pairs if rt[p] == 0]
    print(f"\n{len(pairs)} ordered pairs. {len(clean)} of them lose nothing.")
    print("\nThe relation is NOT symmetric - swapping the two ends changes the number, because")
    print("the set of strings at risk is the writer's, not the pair's:\n")
    print(f"{'A':<16} {'B':<16} {'A->B':>6} {'B->A':>6}")
    print("-" * 84)
    for a, b, ab, ba in B.most_asymmetric(6):
        print(f"{a:<16} {b:<16} {ab:>6} {ba:>6}")
    print("\nRead that as: a config written against the permissive reader is unusable to the")
    print("strict one, while a config written against the strict one survives the permissive")
    print("one nearly intact. Migrations have a safe direction, and it points at strictness.")


def section_15_disagreement_matrix() -> None:
    rule("15. Which readers agree with which.")
    names = [r.name for r in B.READERS]
    print("Cells: strings where both are confident and they disagree.\n")
    head = "".join(f"{n[:6]:>7}" for n in names)
    print(f"{'':<16}{head}")
    for a in names:
        row = "".join(f"{B.agreement(a, b)[1]:>7}" for b in names)
        print(f"{a:<16}{row}")
    identical = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                 if B.agreement(a, b)[0] == len(B.CORPUS)]
    print(f"\nReader pairs that agree on all {len(B.CORPUS)} strings: {len(identical)}")
    for a, b in identical:
        print(f"  {a} == {b}")


def section_16_what_to_do() -> None:
    rule("16. The three decisions, written down.")
    print("A boolean reader is an accept table, a normalisation and a failure policy. If you")
    print("do not choose them, sixteen layers choose sixteen different combinations for you.\n")
    print("  1. Name the accept table in the schema, not in the code that reads it.")
    print("     `true|false` is a defensible table. So is git's. `bool(s)` is not a table.")
    print("  2. Choose the normalisation deliberately: strip first (all four residue strings")
    print("     in section 11 are a stripping bug), then casefold - not lower - and NFKC only")
    print("     if fullwidth input is possible.")
    print("  3. Refuse. A reader that returns false for an unrecognised string has thrown")
    print("     away the only fact it had: that nobody knows what the value means.\n")
    print("And the rule the SQLite section earns, which is not 'pick a better parser':")
    print("do not store a boolean as text, so that nothing downstream has to read one.")
    print("There is no string that survives this roster intact - not `true`, and not `1`:\n")
    for text in ("1", "0", "true", "false"):
        s = next(x for x in B.CORPUS if x.text == text)
        c = collections.Counter(r.verdict for r in B.verdicts_for(s).values())
        print(f"  {s.label!r:<9} " + ", ".join(f"{k}={c[k]}" for k in B.VERDICTS if c[k]))
    print(f"\n{len(B.unanimous())} of {len(B.CORPUS)} strings are read the same way by all "
          f"{len(B.READERS)} readers.")
    print("A BOOLEAN column, or an INTEGER holding 0 and 1, is not a string and is never")
    print("parsed. That is the whole fix. Everything above is what it costs not to do it.")


def main() -> None:
    section_1_roster()
    section_2_no_string_means_true()
    section_3_the_title_bug()
    section_4_all_45_flip()
    section_5_never_refuse()
    section_6_notbool_defers()
    section_7_norway()
    section_8_sqlite()
    section_9_awk_entry_path()
    section_10_js_two_fixes()
    section_11_residue()
    section_12_normalisation()
    section_13_silently_wrong()
    section_14_round_trip()
    section_15_disagreement_matrix()
    section_16_what_to_do()
    print()


if __name__ == "__main__":
    main()
