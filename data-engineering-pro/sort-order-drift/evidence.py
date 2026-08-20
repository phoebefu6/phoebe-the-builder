"""Every number quoted in the README, printed from the engine.

Run `python3 evidence.py`. Nothing in the README is typed by hand.
"""

from __future__ import annotations

from itertools import combinations

from collate import (
    CORPUS,
    PAGE_SIZES,
    COLLATION_BY_NAME,
    COLLATIONS,
    Verdict,
    bmp_flip,
    distinct_count,
    dirty_offset_runs,
    drift_matrix,
    finding_counts,
    findings,
    first_row,
    flips,
    identical_pairs,
    keyset_pagination,
    libc_agreement,
    libc_distinct_orders,
    libc_probe,
    libc_refused_names,
    max_displacement,
    normalization_gap,
    offset_pagination,
    order,
    pagination_totals,
    positions,
    range_counts,
    range_drift,
    sort_key,
    tie_groups,
    tied_rows,
    top1_answers,
    turkish_case_breakage,
    unique_violations,
    verdict,
    verdict_counts,
)

RULE = "-" * 78
PAIRS = len(list(combinations(CORPUS, 2)))


def head(n: int, title: str) -> None:
    print(f"\n{RULE}\n{n}. {title}\n{RULE}")


def s1_corpus() -> None:
    head(1, "The column")
    print(f"{len(CORPUS)} rows, {PAIRS} row pairs, {len(COLLATIONS)} collations.\n")
    for r in CORPUS:
        cps = " ".join(f"U+{ord(c):04X}" for c in r.name)
        print(f"  {r.id:2d}  {r.display:<16s} {r.note}")
        if any(ord(c) > 0x7F for c in r.name):
            print(f"      {cps}")


def s2_ten_orders() -> None:
    head(2, "Ten collations, ten answers to `ORDER BY name`")
    for c in COLLATIONS:
        seq = ", ".join(r.display for r in order(CORPUS, c)[:8])
        print(f"\n  {c.key_name:13s} {c.models}")
        print(f"  {'':13s} {seq}, ...")
    print()
    ident = identical_pairs()
    print(
        f"  Collation pairs returning the identical row sequence: {len(ident)} of "
        f"{len(list(combinations(COLLATIONS, 2)))}"
    )
    for a, b in ident:
        print(f"    {a} == {b}")


def s3_top1() -> None:
    head(3, "`ORDER BY name LIMIT 1`, which is also `MIN(name)`")
    ans = top1_answers()
    for c in COLLATIONS:
        print(f"  {c.key_name:13s} -> {first_row(c).display!r}")
    print(f"\n  distinct answers: {len(set(ans.values()))} ({', '.join(sorted(set(ans.values())))})")


def s4_verdicts() -> None:
    head(4, "Four verdicts")
    print(
        "  stable-total  deterministic forever, but the order is not linguistic\n"
        "  total         linguistic and injective: safe to paginate\n"
        "  tied          ties exist, so row order inside a tie is the plan's choice\n"
        "  merging       ties AND nondeterministic equality: row counts change too\n"
    )
    for c in COLLATIONS:
        v = verdict(c)
        print(
            f"  {c.key_name:13s} {v.value:13s} tie groups {len(tie_groups(c)):2d}  "
            f"tied rows {tied_rows(c):2d}  COUNT(DISTINCT name) {distinct_count(c):2d}"
        )
    counts = verdict_counts()
    print()
    for v in Verdict:
        print(f"  {v.value:13s} {counts[v]:2d} of {len(COLLATIONS)}")
    print(
        "\n  No linguistic collation is `total` here, and the reason is two rows:\n"
        "  rows 12 and 13 are the same string in different normal forms, so every\n"
        "  Unicode-aware collation must call them equal. Only the byte orders,\n"
        "  which do not know what a string is, avoid the tie."
    )


def s5_ties() -> None:
    head(5, "The ties themselves")
    for c in COLLATIONS:
        groups = tie_groups(c)
        if not groups:
            continue
        print(f"\n  {c.key_name} ({'nondeterministic' if not c.deterministic else 'deterministic'})")
        for g in groups:
            print(f"    {' = '.join(r.display for r in g)}")
    ai = COLLATION_BY_NAME["ai_ci"]
    v = unique_violations(ai)
    print(
        f"\n  Under {ai.key_name} the ties are equality, so a UNIQUE(name) index "
        f"rejects {len(v)} row pairs"
    )
    for a, b in v:
        print(f"    {a.display!r} = {b.display!r}")
    print(
        f"\n  COUNT(DISTINCT name): {distinct_count(ai)} under {ai.key_name}, "
        f"{len({r.name for r in CORPUS})} under every deterministic collation.\n"
        "  A deterministic collation's ties are ordering only: PostgreSQL still\n"
        "  compares bytes for `=`, so the rows stay distinct and only their order\n"
        "  is undefined. That is why this class of bug shows up as a report whose\n"
        "  rows moved, not as an error."
    )


def s6_drift() -> None:
    head(6, "Pairwise drift: how many row pairs come back in the opposite order")
    m = drift_matrix()
    names = [c.key_name for c in COLLATIONS]
    w = max(len(n) for n in names)
    print(" " * (w + 2) + " ".join(f"{n[:6]:>6s}" for n in names))
    for a in names:
        row = " ".join(f"{m[(a, b)]:6d}" for b in names)
        print(f"  {a:<{w}s}{row}")
    ranked = sorted(((v, k) for k, v in m.items() if k[0] < k[1]), reverse=True)
    print(f"\n  worst pairs (out of {PAIRS} row pairs):")
    for v, k in ranked[:6]:
        print(f"    {k[0]:13s} vs {k[1]:13s} {v:4d}  ({100 * v / PAIRS:.1f}%)")
    print("\n  closest non-identical pairs:")
    for v, k in [x for x in ranked if x[0] > 0][-5:]:
        print(f"    {k[0]:13s} vs {k[1]:13s} {v:4d}")
    de = flips(COLLATION_BY_NAME["de_DIN"], COLLATION_BY_NAME["de_phonebook"])
    print(
        f"\n  Same language, two standards: de_DIN vs de_phonebook disagree on "
        f"{len(de)} pairs"
    )
    for a, b in de:
        print(f"    {a.display!r} vs {b.display!r}")


def s7_displacement() -> None:
    head(7, "How far one row can move")
    for r, gap, a, b in max_displacement()[:8]:
        print(f"  {r.display:<16s} {gap:2d} positions   {a} vs {b}")
    pos = {c.key_name: positions(c) for c in COLLATIONS}
    print("\n  positions of four rows, per collation (0 = first):")
    watch = [2, 5, 21, 27]
    print(f"    {'collation':<14s}" + "".join(f"{CORPUS[i - 1].name[:11]:>13s}" for i in watch))
    for c in COLLATIONS:
        cells = "".join(f"{pos[c.key_name][i]:>13d}" for i in watch)
        print(f"    {c.key_name:<14s}{cells}")


def s8_pagination() -> None:
    head(8, "Paginating a tied sort")
    print(
        "  Each page is a separate query execution, so each may be handed a\n"
        "  different physical row order (insertion order, a backward index scan,\n"
        "  a table rewritten by VACUUM FULL). A stable sort preserves whatever it\n"
        "  is given, so inside a tie group the physical order IS the result order.\n"
    )
    print(f"  page sizes swept: {', '.join(str(n) for n in PAGE_SIZES)}\n")
    print("  Every (collation, page size) run where OFFSET paging is wrong:\n")
    print(f"  {'collation':<14s}{'page':>5s}  {'never returned':<24s} returned twice")
    for name, n, lost, dup in dirty_offset_runs():
        lo = ", ".join(CORPUS[i - 1].name for i in lost)
        du = ", ".join(CORPUS[i - 1].name for i in dup)
        print(f"  {name:<14s}{n:>5d}  {lo:<24s} {du}")
    print(
        "\n  Whether a tie group straddles a page boundary depends on the page\n"
        "  size, so the same query is exact at one page size and lossy at\n"
        "  another. That is why this reaches production: the test suite picked\n"
        "  a page size, and it was one of the clean ones.\n"
    )
    print(f"  {'collation':<14s}{'page':>5s} | {'OFFSET':>22s} | {'+ , id':>8s} | "
          f"{'keyset >':>10s} | {'keyset >=':>12s}")
    for c in COLLATIONS:
        for n in (4, 6, 8):
            off = offset_pagination(c, n)
            tb = offset_pagination(c, n, tiebreak=True)
            ks = keyset_pagination(c, n, strict=True)
            kl = keyset_pagination(c, n, strict=False)
            print(
                f"  {c.key_name:<14s}{n:>5d} | "
                f"lost {len(off.lost)}, repeated {len(off.duplicated):<7d} | "
                f"{'clean' if tb.clean else 'DIRTY':>8s} | "
                f"lost {len(ks.lost):<5d} | "
                f"repeated {len(kl.duplicated)}{' + stall' if kl.stalled else ''}"
            )
    tot = pagination_totals()
    print(
        f"\n  Summed over {tot['runs']} (collation, page size) runs:\n"
        f"    OFFSET               {tot['offset_lost']} rows never returned, "
        f"{tot['offset_dup']} returned twice; {tot['clean_offset']} of {tot['runs']} runs clean\n"
        f"    OFFSET + `, id`      {tot['tiebreak_lost']} lost, {tot['tiebreak_dup']} repeated\n"
        f"    keyset with `>`      {tot['keyset_strict_lost']} rows never returned\n"
        f"    keyset with `>=`     {tot['keyset_loose_dup']} rows repeated, "
        f"{tot['keyset_loose_stalls']} of {tot['runs']} runs stall and never terminate"
    )
    ai = COLLATION_BY_NAME["ai_ci"]
    a = offset_pagination(ai, 6)
    print(
        f"\n  Worked example - {ai.key_name}, page size 6:\n"
        f"    rows never returned: {[CORPUS[i - 1].name for i in a.lost]}\n"
        f"    rows returned twice: {[CORPUS[i - 1].name for i in a.duplicated]}\n"
        "    Both pages were individually correct. Nothing raised an error."
    )
    print(
        "\n  The `>=` stall is not about ties at all: the last row of a page\n"
        "  always satisfies `name >= $last`, so it opens the next page forever.\n"
        f"  It stalls under C too, which has {len(tie_groups(COLLATION_BY_NAME['C']))} ties."
    )


def s9_range() -> None:
    head(9, "A range predicate is collation-dependent too")
    counts = range_counts()
    print("  WHERE name >= 'A' AND name < 'N'\n")
    for c in COLLATIONS:
        print(f"    {c.key_name:13s} {counts[c.key_name]:2d} rows")
    print(
        f"\n  {min(counts.values())} to {max(counts.values())} rows from the same "
        "table and the same predicate.\n"
    )
    for r, yes, no in range_drift():
        print(f"    {r.display:<16s} in: {', '.join(yes)}")
        print(f"    {'':<16s} out: {', '.join(no)}")
    print(
        "\n  Every A-M / N-Z split inherits this: shard keys, archive sweeps,\n"
        "  alphabetical index tabs, partition bounds."
    )


def s10_normalisation() -> None:
    head(10, "Two rows that are the same string")
    for a, b, gaps in normalization_gap():
        print(f"  {a.display!r} (row {a.id}) and {b.display!r} (row {b.id})")
        print(f"    code points: {' '.join(f'U+{ord(c):04X}' for c in a.name)}")
        print(f"                 {' '.join(f'U+{ord(c):04X}' for c in b.name)}")
        print(f"    byte lengths: {len(a.name.encode())} vs {len(b.name.encode())}")
        print("    positions apart, per collation:")
        for k, v in gaps.items():
            print(f"      {k:13s} {v}")
    print(
        "\n  Normalisation is a write-path decision. A collation cannot undo it:\n"
        "  under C the two spellings are two rows a UNIQUE(name) index accepts,\n"
        "  and every later lookup finds one of them."
    )


def s11_binary_orders() -> None:
    head(11, "Two binary orders that are not the same binary order")
    fl = bmp_flip()
    c8, c16 = COLLATION_BY_NAME["C"], COLLATION_BY_NAME["UTF16_BIN"]
    print(f"  pairs ordered differently by UTF-8 bytes and UTF-16 code units: {len(fl)}")
    for a, b in fl:
        print(f"\n    {a.display!r}  max code point U+{max(ord(c) for c in a.name):04X}")
        print(f"    {b.display!r}  max code point U+{max(ord(c) for c in b.name):04X}")
        first = "a" if sort_key(a.name, c8) < sort_key(b.name, c8) else "b"
        print(f"      C          puts {'the first' if first == 'a' else 'the second'} first")
        first16 = "a" if sort_key(a.name, c16) < sort_key(b.name, c16) else "b"
        print(f"      UTF16_BIN  puts {'the first' if first16 == 'a' else 'the second'} first")
    print(
        "\n  UTF-16 leads a supplementary character with a surrogate at U+D800, so\n"
        "  everything above the BMP sorts below U+E000..U+FFFF. A Java service, a\n"
        "  JavaScript sort and a SQL Server *_BIN2 index therefore disagree with a\n"
        "  PostgreSQL C index - and both are 'just binary'."
    )


def s12_case() -> None:
    head(12, "LOWER() is locale-dependent, and so is the index built on it")
    for r, root, tr in turkish_case_breakage():
        print(f"  {r.display:<16s} root: {root!r:<16s} tr_TR: {tr!r}")
    print(
        "\n  A functional index on LOWER(name) is only valid for the LC_CTYPE it\n"
        "  was built under. Same for a case-insensitive comparison done in the\n"
        "  application: two services in different locales disagree on whether\n"
        "  'Istanbul' and 'ISTANBUL' are the same name."
    )


def s13_host() -> None:
    head(13, "The model against this host's own libc")
    for key_name, loc, agree, total, status in libc_agreement():
        if total:
            print(f"  {key_name:13s} vs {loc:13s} {agree}/{total} pairs agree  ({status})")
        else:
            print(f"  {key_name:13s} vs {loc:13s} {status}")
    print()
    for loc, status, refused, distinct in libc_probe():
        print(f"  {loc:13s} {status:22s} names refused: {refused}   distinct orders so far: {distinct}")
    print(
        f"\n  This host's installed locales produce {libc_distinct_orders()} different "
        "orders over the same rows,\n  which is the point: the tailoring is real, not a "
        "property of this model."
    )
    refused = libc_refused_names()
    if refused:
        print(
            f"\n  And {len({n for _l, n in refused})} name(s) this host's libc will not "
            "transform at all:"
        )
        for loc, n in refused:
            print(f"    {loc:13s} strxfrm({n!r}) -> OSError")
        print(
            "    It succeeds under C. Anything built on strcoll cannot place that\n"
            "    row, so its position is whatever the error path leaves behind."
        )
    print(
        "\n  Where libc could compare, the model agrees on the great majority of\n"
        "  pairs. The pairs it cannot compare are the interesting ones, which is\n"
        "  why the model exists: it is a readable stand-in for ICU, and the claims\n"
        "  above are about the structure of the disagreement, not about matching\n"
        "  ICU weight for weight."
    )


def s14_findings() -> None:
    head(14, "Findings")
    counts = finding_counts()
    order_ = {"blocking": 0, "silent": 1, "advisory": 2}
    icon = {"blocking": "[blocking]", "silent": "[silent]  ", "advisory": "[advisory]"}
    for f in sorted(findings(), key=lambda f: order_[f.severity]):
        print(f"\n  {icon[f.severity]} {f.title}")
        for line in _wrap(f.detail, 70):
            print(f"      {line}")
    print(
        f"\n  {counts['blocking']} blocking, {counts['silent']} silent, "
        f"{counts['advisory']} advisory.\n"
        "  The blocking ones you would find from a stack trace or a constraint\n"
        "  violation. The silent ones ship: a report whose rows moved, a page of\n"
        "  results that never arrived, a range that lost a customer."
    )


def _wrap(text: str, width: int) -> list:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main() -> None:
    print("SORT-ORDER DRIFT - what `ORDER BY name` actually returns")
    print(f"{len(CORPUS)} rows x {len(COLLATIONS)} collations, every number computed")
    for fn in (
        s1_corpus,
        s2_ten_orders,
        s3_top1,
        s4_verdicts,
        s5_ties,
        s6_drift,
        s7_displacement,
        s8_pagination,
        s9_range,
        s10_normalisation,
        s11_binary_orders,
        s12_case,
        s13_host,
        s14_findings,
    ):
        fn()
    print()


if __name__ == "__main__":
    main()
