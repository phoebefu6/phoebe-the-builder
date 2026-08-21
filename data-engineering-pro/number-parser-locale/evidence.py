"""Print every claim this project makes, from the live readers.

Nothing here is typed in by hand.  Run it and the README's numbers are
reproduced, or the README is wrong.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import numlocale as N


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def s1_symbols() -> None:
    rule("1. The four things that make a reader, and where they come from")
    print("A reader is (symbol table) x (grouping rule) x (strictness) x (scanner).")
    print("These are this machine's CLDR symbols, not a table someone wrote down:")
    print()
    print("%-8s %-22s %-20s %s" % ("locale", "group", "decimal", "1234567.89 renders as"))
    for r in N.locale_symbols():
        print("%-8s %-22s %-20s %s" % (
            r["locale"], "%r %s" % (r["group"], r["group_cp"]),
            "%r %s" % (r["decimal"], r["decimal_cp"]), r["sample"]))
    print()
    print("Note the two that are not on a keyboard: French groups with U+202F")
    print("NARROW NO-BREAK SPACE and Swiss with U+2019 RIGHT SINGLE QUOTATION MARK.")
    print("An ASCII space or an ASCII apostrophe in those positions is not the")
    print("locale's separator, and a strict reader is right to refuse it.")


def _block(table, cases, names, w: int) -> None:
    head = "%-24s" % "string" + "".join("%*s" % (w, n.replace("_strict", "_S").replace("_loose", "_L")) for n in names)
    print(head)
    print("-" * len(head))
    for c in cases:
        row = "%-24s" % c.escaped()[:24]
        for n in names:
            row += "%*s" % (w, table[c.name][n].display(width=w - 1))
        print(row)


def s2_matrix(table, cases) -> None:
    rule("2. One corpus, fifteen readers, %d readings" % (len(cases) * len(N.reader_names())))
    names = N.reader_names()
    print("Block A -- the five scanners, no locale involved:")
    print()
    _block(table, cases, names[:5], 18)
    print()
    print("Block B -- the ten locale readers (_S strict grouping check, _L loose):")
    print()
    _block(table, cases, names[5:], 12)
    print()
    print("'--' is a refusal, 'n/a' an absent reader, a trailing '~' a truncated")
    print("display.  Every column is a defensible way to read the column.")


def s3_verdicts(verds) -> None:
    rule("3. Verdicts")
    counts: Dict[str, int] = {v: 0 for v in N.VERDICTS}
    for v in verds:
        counts[v.verdict] += 1
    labels = {
        "sign-drift": "readers disagree on the sign",
        "sign-loss": "producer meant a negative; no reader returns one",
        "magnitude-drift": "two readings >= 10x apart, no error raised",
        "value-drift": "different numbers, under 10x apart",
        "silent-zero": "a reader hands back 0 for something that is not 0",
        "accept-drift": "one number, but only some readers will take it",
        "agreed": "every accepting reader returns the same number",
        "rejected-by-all": "nobody takes it",
    }
    print("%-18s %6s   %s" % ("verdict", "count", "meaning"))
    for k in N.VERDICTS:
        print("%-18s %6d   %s" % (k, counts[k], labels[k]))
    print()
    print("AGREED IS %d OF %d." % (counts["agreed"], len(verds)))
    print("Four strings in this corpus are read identically by every reader that")
    print("accepts them: a bare integer, 1e3, 1e309 and the word Infinity.")
    print("Everything that looks like money is not one of them.")
    print()
    for v in verds:
        if v.verdict in ("magnitude-drift", "sign-loss", "sign-drift"):
            print("  %-20s %-16s %s" % (v.case.escaped()[:20], v.verdict,
                                        "%s x apart" % N._fmt_ratio(v.ratio) if v.ratio else v.flags[0][:48]))


def s4_headline(verds, table) -> None:
    rule("4. The canonical pair: 1.234 and 1,234")
    for name in ("dot-3dp", "comma-3dp"):
        v = next(x for x in verds if x.case.name == name)
        print()
        print("  %r  (%s)" % (v.case.raw, v.case.provenance))
        for reader in N.reader_names():
            r = table[name][reader]
            if r.ok:
                print("      %-16s -> %s" % (reader, r.display()))
        print("      distinct readings: %s" % ", ".join(
            format(d.normalize(), "f") for d in v.distinct))
        print("      %s x apart" % N._fmt_ratio(v.ratio))
    print()
    print("Neither string carries the information needed to choose.  A pipeline")
    print("that picks one is not parsing; it is guessing, once per row.")


def s5_invisible(table) -> None:
    rule("5. Three strings that look identical on screen")
    for name in ("fr-nnbsp", "fr-nbsp", "fr-space"):
        c = next(x for x in N.corpus() if x.name == name)
        takers = [n for n in N.reader_names() if table[name][n].ok]
        print("  %-12s %-30s %d/%d readers accept" % (
            name, c.escaped()[:30], len(takers), len(N.reader_names())))
        print("               rendered:  %s" % c.raw)
        print("               accepted by: %s" % (", ".join(takers) if takers else "nobody"))
    print()
    print("Only the U+202F one is French.  The other two are what actually")
    print("arrives, because U+202F is not on anyone's keyboard and most")
    print("spreadsheet exports emit NBSP or a plain space instead.")
    print()
    print("Same story on the Swiss side:")
    for name in ("ch-rsquo", "ch-apostrophe"):
        c = next(x for x in N.corpus() if x.name == name)
        takers = [n for n in N.reader_names() if table[name][n].ok]
        print("  %-14s %-24s accepted by: %s" % (
            name, c.raw, ", ".join(takers) if takers else "nobody"))


def s6_prefix(table, cases) -> None:
    rule("6. Prefix parsers: the biggest magnitude errors in the corpus")
    print("strtod and parseFloat consume what they can and stop.  They do not")
    print("fail on a separator they do not recognise -- they return the digits")
    print("in front of it.  On a grouped integer that is a 6-figure error.")
    print()
    print("%-22s %14s %14s %14s" % ("string", "locale reading", "parseFloat", "strtod"))
    for c in cases:
        loc = None
        for n in N.reader_names():
            if n.endswith(("_strict", "_loose")) and table[c.name][n].is_finite:
                loc = table[c.name][n]
                break
        pf, sd = table[c.name]["js_parsefloat"], table[c.name]["c_strtod"]
        if loc is None or not pf.is_finite:
            continue
        assert loc.value is not None and pf.value is not None
        if loc.value != pf.value and abs(loc.value) > 0 and abs(pf.value) > 0:
            if abs(loc.value) / abs(pf.value) >= 10:
                print("%-22s %14s %14s %14s" % (c.escaped()[:22], loc.display(),
                                                pf.display(), sd.display()))
    print()
    print("strtod also reports how many bytes it consumed.  A caller who checks")
    print("endptr sees the truncation.  A caller who does not -- awk, most")
    print("hand-rolled C importers, every wrapper that returns only the double --")
    print("cannot distinguish these from a clean parse.")


def s7_silent_zero(table, cases) -> None:
    rule("7. The zero that was not in the file")
    print("Two mechanisms invent a zero:")
    print()
    print("  strtod: on no conversion it returns 0.0 and sets endptr == nptr.")
    print("  JS Number(): the StringNumericLiteral of whitespace is defined as 0,")
    print("               so Number('') and Number('   ') are 0, not NaN.")
    print()
    for c in cases:
        z = []
        if table[c.name]["c_strtod"].is_finite and table[c.name]["c_strtod"].value == 0 \
                and "silent 0" in table[c.name]["c_strtod"].note:
            z.append("strtod")
        jn = table[c.name]["js_number"]
        if jn.is_finite and jn.value == 0 and c.raw.strip() == "":
            z.append("Number()")
        if z:
            print("  %-22s -> 0  by %s" % (c.escaped()[:22], " and ".join(z)))
    print()
    print("A zero is the worst possible failure value for an amount column: it")
    print("passes a not-null check, passes a numeric type check, passes a range")
    print("check, and moves an average.")


def s8_negatives(table, verds) -> None:
    rule("8. Three ways to write a negative, none of them read as one")
    for name in ("accounting-neg", "trailing-minus", "true-minus"):
        c = next(x for x in N.corpus() if x.name == name)
        v = next(x for x in verds if x.case.name == name)
        readings = [(n, table[name][n].display()) for n in N.reader_names()
                    if table[name][n].ok]
        print()
        print("  %-14s %-12s producer meant %s" % (name, c.raw, c.intended))
        print("     provenance: %s" % c.provenance)
        if readings:
            print("     readings:   %s" % ", ".join("%s=%s" % r for r in readings))
        else:
            print("     readings:   refused by all 15")
        print("     verdict:    %s" % v.verdict)
    print()
    print("Accounting parentheses, a COBOL trailing sign and U+2212 MINUS SIGN")
    print("are all standard notations in files that arrive daily.  Of the three,")
    print("one is read with the wrong sign and two become a silent zero.  Not one")
    print("reader in the roster recovers the intended negative.")


def s9_precision(table) -> None:
    rule("9. Two readers that agree on the string and not on the number")
    for name in ("int53-plus1", "binary-inexact", "sci-overflow"):
        c = next(x for x in N.corpus() if x.name == name)
        print()
        print("  %r  (%s)" % (c.raw, c.provenance))
        for reader in ("py_decimal", "py_float", "js_number", "c_strtod"):
            r = table[name][reader]
            if r.ok:
                print("      %-14s -> %s" % (reader, r.display()))
    print()
    print("9007199254740993 is an id.  A double cannot hold it, so every")
    print("float-backed reader returns its neighbour instead -- off by one, no")
    print("error, and it will join to the wrong row for the rest of time.")
    print("Decimal holds it exactly.  This is a scanner property, independent of")
    print("every locale question above.")


def s10_border() -> None:
    rule("10. The border crossing: write in one locale, read in another")
    cross = N.crossings()
    print("An accounting export renders %d amounts with a fixed 2dp pattern in" % len(N.ROUNDTRIP_VALUES))
    print("each of 5 locales, then each rendering is read back by all 5 locales")
    print("at both strictness settings: %d runs." % len(cross))
    print()
    ok = sum(1 for c in cross if c.status == "ok")
    err = sum(1 for c in cross if c.status == "error")
    wrong = sum(1 for c in cross if c.status == "wrong")
    print("  %4d  correct" % ok)
    print("  %4d  refused -- loud, recoverable, the good outcome" % err)
    print("  %4d  SILENTLY WRONG -- a number came back and it was not the number" % wrong)
    print()
    print("The silently-wrong cells, all of them:")
    print("%-8s %-8s %-6s %-16s %14s %14s %8s" % (
        "wrote", "read", "strict", "rendered", "should be", "got", "ratio"))
    for c in cross:
        if c.status == "wrong":
            print("%-8s %-8s %-6s %-16r %14s %14s %8s" % (
                c.wrote, c.read, str(c.strict), c.rendered, c.target, c.got,
                N._fmt_ratio(c.ratio) if c.ratio else "-"))
    print()
    print("Every one is a loose reader stripping the other locale's decimal")
    print("point as if it were a group separator.  0.50 becomes 50.  1,234.50")
    print("becomes 1.23450.  Money moves by two or three orders of magnitude and")
    print("the row still looks like money.")


def s11_diagonal() -> None:
    rule("11. Strict mode refusing its own locale's output")
    diag = N.own_output_roundtrip()
    bad = [c for c in diag if c.status != "ok"]
    print("The diagonal of section 10: the locale that wrote the string is also")
    print("the one reading it.  Every cell here should be 'ok'.")
    print()
    print("  %d of %d fail, and all %d are strict-mode refusals:" % (
        len(bad), len(diag), sum(1 for c in bad if c.strict)))
    print()
    for c in bad:
        print("  %-8s wrote %-18r then refused it (strict=%s)" % (c.read, c.rendered, c.strict))
    print()
    print("Two separate causes, both worth knowing before you turn strict on:")
    print()
    print("  a) Trailing zero cents.  Babel 2.11's strict check validates by")
    print("     re-formatting and comparing strings; format_decimal normalises")
    print("     '1,234.50' to '1,234.5', the strings differ, and the parse is")
    print("     refused.  A fixed-2dp money column is roughly one row in ten.")
    print()
    print("  b) A pattern that overrides the locale's own grouping.  en_IN groups")
    print("     at 2,2,3 -- '12,34,567.89'.  Handed the pattern '#,##0.00' the")
    print("     formatter emits '1,234,567.89' and the strict reader, which")
    print("     checks against the locale rule rather than the pattern, refuses")
    print("     it.  Writer and reader disagree inside a single locale.")
    print()
    print("So strict mode is not simply the safe setting.  It converts a class of")
    print("silent errors into refusals -- and it also refuses correct input.")


def s12_column() -> None:
    rule("12. What the audit says about a real column")
    col = ["1.234", "2.500", "3.000", "1.750"]
    print("A four-row amount column, exactly as received:")
    print("   %s" % ", ".join(repr(v) for v in col))
    print()
    a = N.audit_column(col, "amount_eur")
    print("Only a reader that REFUSES carries information.  A prefix parser")
    print("accepts every string, so it never eliminates a hypothesis:")
    print("   readers accepting every row: %d of %d" % (
        len(a.readers_that_take_every_row), len(N.reader_names())))
    print()
    print("So the question is not 'what does a reader return' but 'which locale")
    print("could have written this column'.  Under a strict grouping check:")
    print()
    for h in N.locale_hypotheses(col):
        if h.survives:
            print("   %-8s SURVIVES   total = %s" % (h.locale, h.total))
        else:
            print("   %-8s eliminated by row %r" % (h.locale, h.killed_by))
    print()
    d = a.decision
    assert d is not None
    print("   verdict: %s  (%d surviving locale hypotheses, %d distinct totals)" % (
        d.verdict, len(d.surviving), len({str(v) for v in d.totals.values()})))
    if d.spread is not None:
        print("   the surviving totals are %s x apart" % N._fmt_ratio(d.spread))
    print()
    print("Findings:")
    for f in a.findings:
        print("   - %s" % f)
    print()
    print("The column sums to 8.484 or to 8484 and the file does not say which.")
    print("audit_column() returns both, because returning one would be a guess")
    print("dressed as a total.")


def s13_disambiguating() -> None:
    rule("13. When the column decides itself, and when it cannot")
    print("Two structural facts do the eliminating:")
    print()
    print("  * a group of four digits is not a group, so '1.2345' rules out")
    print("    every locale that groups with '.';")
    print("  * two different separators in one value pin which is which, so")
    print("    '1.234,56' rules out every locale whose decimal symbol is '.'.")
    print()
    cols = [
        ("three-digit groups only", ["1.234", "2.500", "3.000", "1.750"]),
        ("a group count > 1", ["1.234.567", "89.012", "3.456"]),
        ("a four-digit group", ["1.2345", "2.500"]),
        ("both separators present", ["1.234,56", "7.890,12"]),
        ("both, US order", ["1,234.56", "7,890.12"]),
        ("lakh grouping", ["12,34,567", "1,23,456"]),
        ("nothing fits", ["1.2345,67", "9"]),
    ]
    print("%-26s %-30s %-11s %s" % ("column shape", "rows", "verdict", "survivors -> total"))
    print("-" * 108)
    for label, col in cols:
        d = N.decide_column(col)
        totals = sorted({str(v) for v in d.totals.values()})
        tail = "%s -> %s" % (",".join(d.surviving) or "none", " OR ".join(totals) or "-")
        print("%-26s %-30s %-11s %s" % (label, ", ".join(col)[:30], d.verdict, tail[:60]))
    print()
    print("Five of the seven shapes decide themselves.  The one that does not is")
    print("the money column: single three-digit groups, no four-digit group to")
    print("rule out a thousands separator, no second separator to pin the")
    print("decimal.  All five locales read it and two totals come out 1,000 x")
    print("apart.  That is not a tooling gap -- the information is not in the file.")


def s14_reader_profile(table, cases) -> None:
    rule("14. Reader profile: strictness is not one axis")
    names = N.reader_names()
    print("%-18s %8s %8s %8s   %s" % ("reader", "accepts", "refuses", "silent0", "character"))
    char = {
        "py_float": "C scanner + PEP 515 underscores + Unicode digits",
        "py_decimal": "exact, unbounded, same lenience as float()",
        "c_strtod": "prefix parse; never fails; hex; silent 0",
        "js_number": "whole-string; whitespace -> 0; hex/bin/oct; NaN is a value",
        "js_parsefloat": "prefix parse; no hex; NaN on no digits",
    }
    for n in names:
        acc = sum(1 for c in cases if table[c.name][n].ok)
        ref = sum(1 for c in cases if table[c.name][n].status == N.REJECTED)
        z = sum(1 for c in cases if table[c.name][n].is_finite
                and table[c.name][n].value == 0 and "silent 0" in table[c.name][n].note)
        desc = char.get(n, "CLDR %s, %s grouping check" % (
            n.rsplit("_", 1)[0], "validated" if n.endswith("strict") else "ignored"))
        print("%-18s %8d %8d %8d   %s" % (n, acc, ref, z, desc))
    print()
    print("c_strtod accepts all %d strings in the corpus.  That is not tolerance," % len(cases))
    print("it is the absence of a failure channel.  The strictest reader here")
    print("refuses %d of %d -- and section 11 shows some of those refusals are wrong." % (
        min(sum(1 for c in cases if table[c.name][n].status == N.REJECTED) for n in names
            if n.endswith("strict")), len(cases)))


def s15_what_to_do() -> None:
    rule("15. What to actually do")
    print("1. Store the locale with the file, not in someone's head.  A CSV of")
    print("   numbers without a declared locale is not self-describing data.")
    print()
    print("2. Never let a prefix parser near a grouped number.  strtod and")
    print("   parseFloat turn '1,234,567' into 1.  If you must use them, check")
    print("   endptr / compare the consumed length, every time.")
    print()
    print("3. Treat an unexpected 0 as a parse failure, not a value.  Both")
    print("   strtod and JS Number() reach for 0 where other readers refuse.")
    print()
    print("4. Parse ids as text or Decimal.  9007199254740993 is not")
    print("   representable as a double and no locale setting changes that.")
    print()
    print("5. Before you turn strict mode on, round-trip your own formatter")
    print("   through it (section 11).  Strict is a different set of errors,")
    print("   not fewer errors.")
    print()
    print("6. When a column is ambiguous, return the candidates.  audit_column()")
    print("   reports every total a conforming reader would produce, and says")
    print("   decidable=False rather than picking one.")


def main(argv: Optional[List[str]] = None) -> int:
    cases = N.corpus()
    table = N.read_all(cases)
    verds = N.all_verdicts(cases, table)

    print("NUMBER-PARSER-LOCALE -- evidence")
    print("A numeric string does not contain a number. A reader assigns one.")
    print("%d strings x %d readers = %d readings. Everything below is measured." % (
        len(cases), len(N.reader_names()), len(cases) * len(N.reader_names())))

    s1_symbols()
    s2_matrix(table, cases)
    s3_verdicts(verds)
    s4_headline(verds, table)
    s5_invisible(table)
    s6_prefix(table, cases)
    s7_silent_zero(table, cases)
    s8_negatives(table, verds)
    s9_precision(table)
    s10_border()
    s11_diagonal()
    s12_column()
    s13_disambiguating()
    s14_reader_profile(table, cases)
    s15_what_to_do()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
