"""Every number in the README, computed. Each experiment isolates one mechanism.

Run: python3 evidence.py
"""

from __future__ import annotations

import unicodedata
from typing import List

import slug as S

RULE = "-" * 78


def head(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


# ---------------------------------------------------------------------------


def exp1_the_fold_deletes_what_it_cannot_decompose() -> None:
    head(1, "It is not accent-stripping. It is 'keep whatever happens to decompose'.")
    print("Django slugify() normalises NFKD, then drops every non-ASCII byte.\n")
    print(f"{'title':26} {'NFKD':30} {'slug':22} {'letters lost'}")
    print(RULE)
    for t in ("café", "Ångström", "Straße", "Łódź", "Søren", "Encyclopædia", "Ærø"):
        nfkd = unicodedata.normalize("NFKD", t)
        out = S.django_ascii(t)
        lost = "".join(
            ch for ch in nfkd if ord(ch) > 127 and not unicodedata.combining(ch)
        )
        print(f"{t:26} {nfkd!r:30} {out:22} {lost or '-'}")
    print(RULE)
    print("a composed letter decomposes to base + mark, so the base survives.")
    print("an atomic letter has no decomposition, so nothing survives. it is deleted.\n")
    print("consequence, on a corpus that contains both:")
    a, b = "Łódź: our new datacentre", "Odz - a naming retrospective"
    print(f"  {a!r:34} -> /{S.django_ascii(a)}")
    print(f"  {b!r:34} -> /{S.django_ascii(b)}")
    print(f"  collide: {S.django_ascii(a).split('-')[0] == S.django_ascii(b).split('-')[0]}"
          f"  (both begin /odz)")


def exp2_lower_versus_casefold() -> None:
    head(2, "Moving one line decides whether two titles share a URL.")
    print("`str.lower()` is a 1:1 mapping. `str.casefold()` applies the full")
    print("Unicode case-folding table, which expands ß to ss and ﬁ to fi.\n")
    print(f"{'title':22} {'lower() then NFKD':22} {'casefold() then NFKD':22}")
    print(RULE)
    for t in ("Straße", "STRASSE", "ﬁle handles", "Weiß"):
        print(f"{t:22} {S.django_ascii(t):22} {S.casefold_ascii(t):22}")
    print(RULE)
    pair = ["Straße oder Strasse", "STRASSE ODER STRASSE"]
    for name in ("django_ascii", "casefold_ascii"):
        r = S.audit(pair, name)
        got = sorted(set(r.slugs.values()))
        print(f"{name:18} {got}  -> {'collide' if len(got) == 1 else 'two URLs'}")
    print("\nsame characters, same steps, different order. one of these is a")
    print("uniqueness-constraint violation and the other is two live pages.")


def exp3_nfc_versus_nfd() -> None:
    head(3, "Two byte strings, one text, two URLs - if you skip the normalise step.")
    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", "café")
    print(f"NFC  {nfc!r:16} {len(nfc)} chars  {nfc.encode('utf-8').hex()}")
    print(f"NFD  {nfd!r:16} {len(nfd)} chars  {nfd.encode('utf-8').hex()}")
    print(f"equal as text (NFC-folded): {unicodedata.normalize('NFC', nfd) == nfc}")
    print(f"equal as Python strings:    {nfd == nfc}\n")
    print(f"{'profile':22} {'normalises?':12} {'NFC':12} {'NFD':12} {'split?'}")
    print(RULE)
    for name, p in S.PROFILES.items():
        a, b = p(nfc), p(nfd)
        print(f"{name:22} {str(p.normalises):12} {a:12} {b:12} {'YES' if a != b else '-'}")
    print(RULE)
    print("macOS filesystems hand back NFD. Most Linux tooling and most web forms")
    print("hand back NFC. A title pasted from Finder and the same title typed into")
    print("the CMS are different bytes, and the hand-rolled slugifier is the only")
    print("one that notices - by giving them different URLs.")


def exp4_punctuation_annihilation() -> None:
    head(4, "The punctuation is the whole title.")
    for group in (
        ["C++ for data engineers", "C# for data engineers", "C for data engineers"],
        ["Node.js at scale", "NodeJS at scale", "Node JS at scale"],
        ["What's next for our platform?", "Whats next for our platform"],
        ["Hello --- World", "Hello, World!", "Hello World"],
    ):
        print()
        for t in group:
            print(f"  {t:36} -> /{S.django_ascii(t)}")
        r = S.audit(group)
        n = sum(len(f.titles) for f in r.of_kind(S.Kind.COLLISION))
        print(f"  {n} of {len(group)} collide")


def exp5_one_title_seven_profiles() -> None:
    head(5, "A Japanese title, put through seven published algorithms.")
    t = "データ契約の基礎"
    print(f"title: {t}\n")
    print(f"{'profile':22} {'slug':46} {'len'}")
    print(RULE)
    for name, p in S.PROFILES.items():
        out = p(t)
        shown = out if out else "(empty string)"
        print(f"{name:22} {shown[:46]:46} {len(out)}")
    print(RULE)
    print("three different failure modes, none of them an error:")
    print("  deleted       - django_ascii, casefold_ascii: the row needs a fallback")
    print("  '?'-collapsed - rails_parameterize: hyphens, then stripped to nothing")
    print("  encoded       - wordpress: unique and permanent, and 72 characters of hex")
    print("  preserved     - django_unicode, github_anchor: readable, and non-ASCII in a URL")


def exp6_corpus_audit() -> None:
    head(6, f"The whole corpus: {len(S.CORPUS)} ordinary titles, seven algorithms.")
    print("Nothing in this corpus is adversarial. Every line is a headline")
    print("somebody would ship.\n")
    print(f"{'profile':22} {'verdict':11} {'coll':>5} {'empty':>6} {'route':>6} "
          f"{'split':>6} {'distinct URLs':>14}")
    print(RULE)
    for name in S.PROFILES:
        r = S.audit(S.CORPUS, name)
        c = r.counts()
        distinct = len({s for s in r.slugs.values() if s})
        print(
            f"{name:22} {r.verdict.value:11} "
            f"{c.get('COLLISION', 0):>5} {c.get('EMPTY_SLUG', 0):>6} "
            f"{c.get('ROUTE_SHADOW', 0):>6} {c.get('CONFUSABLE_SPLIT', 0):>6} "
            f"{distinct:>14}"
        )
    print(RULE)
    r = S.audit(S.CORPUS, "django_ascii")
    lost = len(S.CORPUS) - len({s for s in r.slugs.values() if s})
    print(f"django_ascii: {len(S.CORPUS)} titles -> "
          f"{len({s for s in r.slugs.values() if s})} distinct URLs. "
          f"{lost} titles have nowhere of their own to live.")
    print(f"verdict: {r.verdict.value} - {r.reason}")


def exp7_findings_in_full() -> None:
    head(7, "Every finding on the corpus, django_ascii.")
    r = S.audit(S.CORPUS, "django_ascii")
    for sev in (S.Severity.CRITICAL, S.Severity.HIGH, S.Severity.MEDIUM):
        block = [f for f in r.findings if f.severity is sev]
        if not block:
            continue
        print(f"\n{sev.value.upper()}  ({len(block)})")
        for f in block:
            print(f"  {f.kind.value:20} {f.detail}")
            if len(f.titles) > 1:
                for t in f.titles:
                    print(f"    - {t}")


def exp8_the_inverse_problem() -> None:
    head(8, "The failure a uniqueness constraint cannot catch.")
    pair = ["Аpple silicon benchmarks", "Apple silicon benchmarks"]
    print("Two titles. The first character of the first one is U+0410,")
    print("CYRILLIC CAPITAL LETTER A. On screen they are the same headline.\n")
    for t in pair:
        print(f"  {t!r}")
        print(f"    first char  U+{ord(t[0]):04X} {unicodedata.name(t[0])}")
        print(f"    django_ascii   -> /{S.django_ascii(t)}")
        print(f"    django_unicode -> /{S.django_unicode(t)}")
    r = S.audit(pair)
    print(f"\n  verdict: {r.verdict.value} - every slug is distinct, so no")
    print("  uniqueness constraint fires, no import warns, and nothing in a list")
    print("  view distinguishes the two rows.")
    for f in r.of_kind(S.Kind.CONFUSABLE_SPLIT):
        print(f"\n  {f.kind.value}: {f.detail}")


def exp9_truncation() -> None:
    head(9, "The schema change that adds collisions.")
    a = "The complete guide to building resilient data pipelines in production"
    b = "The complete guide to building resilient data pipelines on Kubernetes"
    print(f"  {a}\n  {b}\n")
    for cap in (None, 60, 50, 40):
        r = S.audit([a, b], cap=cap)
        got = sorted(set(r.slugs.values()))
        label = "no cap" if cap is None else f"cap {cap}"
        print(f"  {label:8} {got[0]}")
        if len(got) > 1:
            print(f"           {got[1]}")
        else:
            print(f"           ^ both titles, one URL "
                  f"({r.of_kind(S.Kind.TRUNCATION_COLLISION)[0].kind.value})")
    print("\nthe whole corpus, as the column shrinks:\n")
    caps = [255, 200, 120, 80, 60, 50, 40, 30, 25, 20, 15, 10]
    curve = S.truncation_curve(S.CORPUS, "django_ascii", caps)
    print("  cap        " + " ".join(f"{c:>5}" for c, _, _, _ in curve))
    print("  groups     " + " ".join(f"{g:>5}" for _, g, _, _ in curve))
    print("  titles hit " + " ".join(f"{n:>5}" for _, _, n, _ in curve))
    print("  distinct   " + " ".join(f"{d:>5}" for _, _, _, d in curve))
    print("\ngroup count wobbles - shrinking merges two groups as often as it")
    print("splits a new one. Titles-hit only rises. VARCHAR(255) -> VARCHAR(50)")
    print("is a migration nobody reviews for slugs.")


def exp10_order_dependence() -> None:
    head(10, "The URL is not a property of the post.")
    titles = ["Hello, World!", "Hello --- World", "Hello World"]
    print("three titles, one slug between them. import order decides who gets what:\n")
    for label, order in (
        ("as listed", titles),
        ("reversed", list(reversed(titles))),
        ("sorted by title", sorted(titles)),
    ):
        got = S.assign(order)
        print(f"  {label:16} " + "  ".join(f"{t!r}=/{got[t]}" for t in titles))

    print("\nwhole corpus, four plausible import orders (as listed, reversed,")
    print("alphabetical, reverse-alphabetical):\n")
    c = list(S.CORPUS)
    orders = [c, list(reversed(c)), sorted(c), sorted(c, reverse=True)]
    n, unstable = S.order_sensitivity(c, orders)
    print(f"  {n} of {len(c)} titles received more than one URL across the four runs")
    for t, urls in unstable[:6]:
        print(f"    {t[:40]:42} {urls}")
    print("\nre-importing from a backup that iterates in a different order does not")
    print("preserve URLs. Nothing errors; the old ones 404.")

    print("\ndeletion, and who inherits the links:\n")
    out = S.deletion_promotes_nobody(
        ["Hello, World!", "Hello --- World"], "Hello, World!", "Hello World"
    )
    for k, v in out.items():
        print(f"  {k:20} {v}")
    print("\nthe runner-up is not promoted - stored slugs persist. the bare slug is")
    print("free, and the next post to claim it inherits every inbound link,")
    print("bookmark and cached search result that pointed at the deleted one.")


def exp11_migration_diff() -> None:
    head(11, "Changing slugifier is a URL migration. Here is its size.")
    pairs = [
        ("django_ascii", "django_unicode"),
        ("django_ascii", "rails_parameterize"),
        ("django_ascii", "casefold_ascii"),
        ("django_ascii", "wordpress"),
        ("naive_regex", "django_ascii"),
    ]
    print(f"{'from -> to':44} {'URLs that change':>18}")
    print(RULE)
    for a, b in pairs:
        d = S.disagreements(S.CORPUS, a, b)
        print(f"{a + ' -> ' + b:44} {len(d):>10} of {len(S.CORPUS)}")
    print(RULE)
    d = S.disagreements(S.CORPUS, "django_ascii", "casefold_ascii")
    print("\ndjango_ascii -> casefold_ascii, the one-line change:")
    for t, x, y in d:
        print(f"  {t[:34]:36} /{x:28} -> /{y}")


def exp12_properties_checked() -> None:
    head(12, "Two properties everyone assumes. Checked, not assumed.")
    print("idempotence - slugify(slugify(x)) == slugify(x). Every CMS re-runs the")
    print("slugifier on edit; if this fails, a no-op edit moves a live URL.\n")
    print(f"{'profile':22} {'idempotent on all ' + str(len(S.CORPUS)) + ' titles'}")
    print(RULE)
    for name, p in S.PROFILES.items():
        bad = [t for t in S.CORPUS if p(p(t)) != p(t)]
        print(f"{name:22} {'yes' if not bad else f'NO - {len(bad)} failures'}")
    print(RULE)
    print("\nholds everywhere here, and it is not free: it holds because every")
    print("profile's output is already a fixed point of its own character filter.")
    print("The shape that breaks it is a slugifier that percent-encodes without")
    print("checking for existing escapes - it double-encodes on the second pass:\n")
    t = "北京 office"
    once = S.wordpress(t)
    naive_reencode = "".join(
        ("%%%02x" % ord(ch)) if ch == "%" else ch for ch in once
    )
    print(f"  wordpress({t!r}) = {once}")
    print(f"  a re-encode that does not skip '%'  = {naive_reencode}")
    print(f"  stable under this implementation: {S.wordpress(once) == once}")

    print("\nsecond property - canonical equivalence. Two strings that are the same")
    print("text must get the same slug:\n")
    nfc, nfd = unicodedata.normalize("NFC", "café"), unicodedata.normalize("NFD", "café")
    for name, p in S.PROFILES.items():
        ok = p(nfc) == p(nfd)
        print(f"  {name:22} {'holds' if ok else 'FAILS'}")


def exp13_cross_check() -> None:
    head(13, "The collision detector, checked against a second one.")
    import itertools

    total_groups = 0
    total_cases = 0
    for name in S.PROFILES:
        for cap in (None, 50, 30, 12):
            r = S.audit(S.CORPUS, name, cap=cap)
            titles = list(S.CORPUS)
            fast = sorted(
                tuple(sorted(titles.index(t) for t in f.titles))
                for f in r.findings
                if f.kind in (S.Kind.COLLISION, S.Kind.TRUNCATION_COLLISION)
            )
            slugs = [S._truncate(S.profile(name)(t), cap) for t in titles]
            groups: List[set] = []
            for i, j in itertools.combinations(range(len(titles)), 2):
                if slugs[i] == "" or slugs[i] != slugs[j]:
                    continue
                for g in groups:
                    if i in g or j in g:
                        g.update({i, j})
                        break
                else:
                    groups.append({i, j})
            slow = sorted(tuple(sorted(g)) for g in groups)
            assert fast == slow, (name, cap)
            total_groups += len(fast)
            total_cases += 1
    print(f"  {total_cases} profile x cap combinations")
    print(f"  {total_groups} collision groups found by the hash-map grouping in audit()")
    print(f"  {total_groups} found by an independent O(n^2) pairwise scan")
    print("  identical in every case")


def main() -> None:
    print("SLUG COLLIDER - evidence")
    print(f"corpus: {len(S.CORPUS)} titles, {len(S.PROFILES)} profiles")
    exp1_the_fold_deletes_what_it_cannot_decompose()
    exp2_lower_versus_casefold()
    exp3_nfc_versus_nfd()
    exp4_punctuation_annihilation()
    exp5_one_title_seven_profiles()
    exp6_corpus_audit()
    exp7_findings_in_full()
    exp8_the_inverse_problem()
    exp9_truncation()
    exp10_order_dependence()
    exp11_migration_diff()
    exp12_properties_checked()
    exp13_cross_check()
    print()


if __name__ == "__main__":
    main()
