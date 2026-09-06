"""Every number quoted in the README, printed from the engine.

Run `python3 evidence.py` and the tables below are what appears. Nothing in the
README is typed by hand.
"""

from __future__ import annotations

from durations import (
    CORPUS,
    DAY24,
    DEFAULT_ANCHORS,
    GRAMMARS,
    PARSERS,
    REFERENCE_ANCHOR,
    Verdict,
    _fmt,
    audit,
    audit_corpus,
    best_single_grammar,
    parse_iso,
    safe_form,
)

RULE = "-" * 78


def head(n: int, title: str) -> None:
    print(f"\n{RULE}\n{n}. {title}\n{RULE}")


def show(text: str, width: int = 16) -> str:
    return repr(text).ljust(width)


# ---------------------------------------------------------------------------


def s1_one_string() -> None:
    head(1, "One string, eight parsers: '1h30'")
    for g in GRAMMARS:
        r = PARSERS[g.name]("1h30")
        if r.ok:
            print(f"  {g.name:<11} {_fmt(r.exact_s):>10}  = {r.exact_s:>9,.0f}s   {g.kind}")
        else:
            print(f"  {g.name:<11} {'refused':>10}    {r.error}")
    a = audit("1h30")
    print(f"\n  verdict: {a.verdict.value}   values: {sorted(a.distinct_values())}")
    print("  Nobody is wrong. systemd gives the trailing 30 the default unit (seconds);")
    print("  Jira reads a bare number as minutes; a human meant minutes too, by a")
    print("  different rule. Five of the eight refuse the string outright.")


def s2_bare_number() -> None:
    head(2, "The same digits, five orders of magnitude apart: '90'")
    a = audit("90")
    for r in a.readings:
        if r.ok:
            print(f"  {r.grammar:<11} {_fmt(r.exact_s):>10}  = {r.exact_s:>12,.0f}s")
    print(f"\n  ratio max/min: {a.spread_ratio:,.0f}")
    print("  A JSON field called `duration: 90` with no unit and no schema is any of these.")


def s3_shift_key() -> None:
    head(3, "One shift key: '1m' against '1M', and 'P1M' against 'PT1M'")
    for pair in (("1m", "1M"), ("PT1M", "P1M")):
        vals = []
        for t in pair:
            a = audit(t)
            g = a.accepted[0]
            vals.append((t, g.grammar, g.resolve(REFERENCE_ANCHOR)))
        (t1, g1, v1), (t2, g2, v2) = vals
        print(f"  {t1:>5} ({g1}) = {_fmt(v1):>8}   {t2:>5} ({g2}) = {_fmt(v2):>8}   "
              f"factor {max(v1, v2) / min(v1, v2):,.0f}")
    print("\n  In systemd the difference is case. In ISO 8601 it is position relative")
    print("  to the T. Neither is a typo anybody notices in review.")


def s4_colons() -> None:
    head(4, "Colon fields fill from opposite ends: '1:30'")
    for name in ("ffmpeg", "excel"):
        r = PARSERS[name]("1:30")
        print(f"  {name:<11} {_fmt(r.exact_s):>8}  {r.notes[0] if r.notes else ''}")
    print("\n  Same two characters of separator, 60x apart. A timesheet exported as")
    print("  1:30 and read by a media tool becomes ninety seconds of work.")


def s5_headline() -> None:
    head(5, "The headline run: 28 duration strings from one repository")
    rep = audit_corpus()
    print(f"  {'string':<16} {'verdict':<10} {'readers':>7}  {'low':>10} {'high':>12}  x-grammar")
    print("  (low/high span every reading over every anchor; x-grammar is max/min at one anchor)")
    for a, (_, origin) in zip(rep.audits, CORPUS):
        if a.verdict is Verdict.REJECTED:
            print(f"  {show(a.text)} {'rejected':<10} {0:>7}  {'-':>10} {'-':>12}  -")
            continue
        ratio = a.spread_ratio or 1.0
        rat = f"{ratio:,.0f}" if ratio >= 10 else f"{ratio:.4g}"
        print(f"  {show(a.text)} {a.verdict.value:<10} {len(a.accepted):>7}  "
              f"{_fmt(a.anchor_min_s):>10} {_fmt(a.anchor_max_s):>12}  {rat}")
    print(f"\n  verdicts: {rep.verdicts}")
    lonely = rep.lonely()
    print(f"  exact with two or more readers agreeing: {len(rep.unanimous()) - len(lonely)}")
    print(f"  exact only because one parser could read it at all: {len(lonely)}"
          f"  {[a.text for a in lonely]}")
    print(f"  ambiguous (two parsers, two numbers, both succeed): {len(rep.contested())}"
          f"  {[a.text for a in rep.contested()]}")


def s6_single_library() -> None:
    head(6, "Pick one library, as every codebase does")
    rep = audit_corpus()
    print(f"  {'grammar':<11} {'kind':<15} {'accepts':>8} {'of':>4}  {'silently differs':>17}")
    for g in GRAMMARS:
        acc = rep.accepted_by[g.name]
        wrong = 0
        for a in rep.audits:
            mine = next((r for r in a.accepted if r.grammar == g.name), None)
            if mine is None:
                continue
            others = [r for r in a.accepted if r.grammar != g.name]
            if any(round(o.resolve(REFERENCE_ANCHOR), 6) != round(mine.resolve(REFERENCE_ANCHOR), 6)
                   for o in others):
                wrong += 1
        print(f"  {g.name:<11} {g.kind:<15} {acc:>8} {rep.total:>4}  {wrong:>17}")
    name, acc, wrong = best_single_grammar(rep)
    print(f"\n  best single parser: {name}, {acc} of {rep.total} accepted, {wrong} of those read")
    print(f"  differently by another parser. No grammar reads all {rep.total}.")


def s7_pairs() -> None:
    head(7, "Which pairs of parsers disagree, and how often")
    rep = audit_corpus()
    pairs = sorted(rep.disagreements.items(), key=lambda kv: (-kv[1], kv[0]))
    print(f"  {'pair':<26} {'strings both accept and read differently':>42}")
    for (a, b), n in pairs:
        print(f"  {a + ' vs ' + b:<26} {n:>42}")
    agree_only = [
        (g1.name, g2.name)
        for i, g1 in enumerate(GRAMMARS)
        for g2 in GRAMMARS[i + 1:]
        if tuple(sorted((g1.name, g2.name))) not in rep.disagreements
    ]
    print(f"\n  pairs that never disagreed on this corpus: {len(agree_only)}")
    print("  (mostly because they never both accepted the same string)")


def s8_anchor() -> None:
    head(8, "The anchored ones: length depends on when you start")
    for text in ("P1D", "P1M", "P1Y", "P1W"):
        r = parse_iso(text)
        vals = [(a, r.resolve(a)) for a in DEFAULT_ANCHORS]
        lo = min(vals, key=lambda kv: kv[1])
        hi = max(vals, key=lambda kv: kv[1])
        print(f"  {text:<5} {_fmt(lo[1]):>10} from {lo[0]:%Y-%m-%d %H:%M}   "
              f"{_fmt(hi[1]):>10} from {hi[0]:%Y-%m-%d %H:%M}   "
              f"spread {_fmt(hi[1] - lo[1])}")
    print("\n  Compare the fixed substitutions other grammars make for the same words:")
    for g, text in (("prometheus", "1d"), ("prometheus", "1y"), ("systemd", "1M"), ("systemd", "1y"), ("jira", "1d")):
        r = PARSERS[g](text)
        print(f"  {g:<11} {text:<4} = {_fmt(r.exact_s):>10}  (exact by definition, so never flagged)")
    iso_day = parse_iso("P1D")
    spring = [a for a in DEFAULT_ANCHORS if (a.month, a.day) == (3, 9)][0]
    print(f"\n  A prometheus `1d` is {_fmt(DAY24)} always. The calendar day starting "
          f"{spring:%Y-%m-%d %H:%M}\n  is {_fmt(iso_day.resolve(spring))} - so a retention window set in `1d` units "
          "keeps an hour\n  more or less than a day, twice a year, silently.")


def s9_findings() -> None:
    head(9, "Eighteen findings, and how often each fires on the corpus")
    rep = audit_corpus()
    from durations import DESCRIPTION_OF, SEVERITY_OF

    print(f"  {'code':<28} {'severity':<9} {'fires':>5}  description")
    for code in rep.finding_counts:
        print(f"  {code:<28} {SEVERITY_OF[code]:<9} {rep.finding_counts[code]:>5}  {DESCRIPTION_OF[code]}")
    print("\n  `silent` is the severity that matters: every parser involved returned")
    print("  successfully and they returned different numbers.")


def s10_fix() -> None:
    head(10, "The only unambiguous way to write it down")
    for seconds in (30, 5400, 86400, 2592000):
        text = safe_form(seconds)
        readings = {g.name: PARSERS[g.name](text) for g in GRAMMARS}
        agree = {n for n, r in readings.items() if r.ok and round(r.resolve(REFERENCE_ANCHOR), 6) == seconds}
        print(f"  {text:<10} agreed by {len(agree)} grammars: {', '.join(sorted(agree))}")
    print("\n  Integer seconds, no calendar unit, no colon, no bare number. It is also")
    print("  unreadable, which is exactly why configuration is not written that way -")
    print("  and why a parser should return a verdict rather than a number.")


def main() -> None:
    print("Duration Parser - evidence for every number in the README")
    print(f"reference anchor: {REFERENCE_ANCHOR:%Y-%m-%d %H:%M %Z}, "
          f"{len(DEFAULT_ANCHORS)} anchors in the sweep")
    for fn in (s1_one_string, s2_bare_number, s3_shift_key, s4_colons, s5_headline,
               s6_single_library, s7_pairs, s8_anchor, s9_findings, s10_fix):
        fn()
    print()


if __name__ == "__main__":
    main()
