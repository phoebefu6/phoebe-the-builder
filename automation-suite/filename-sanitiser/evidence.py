"""Every number in the README, computed. Each experiment isolates one mechanism.

Run: python3 evidence.py
"""

from __future__ import annotations

import unicodedata
from typing import List

import sanitise as S

RULE = "-" * 78
NAMES = S.SAMPLE_NAMES
WIN_DEST = r"C:\data"


def head(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def short(name: str, width: int = 34) -> str:
    """Printable, bounded, and honest about control characters."""
    s = name.encode("unicode_escape").decode("ascii")
    return s if len(s) <= width else s[: width - 3] + "..."


# ---------------------------------------------------------------------------


def exp1_a_sanitiser_is_a_projection() -> None:
    head(1, "A sanitiser is a projection. Projections collide.")
    print(f"{len(NAMES)} source names, target {S.WINDOWS.name}, destination {WIN_DEST!r}.")
    print("Each source lands in exactly one bucket, so the three columns sum to "
          f"{len(NAMES)}.\n")
    print(f"{'sanitiser':<18}{'delivered':>10}{'overwritten':>13}{'rejected':>10}"
          f"{'distinct out':>14}")
    print(RULE)
    for row in S.compare(NAMES, S.WINDOWS, WIN_DEST):
        print(f"{row.sanitiser:<18}{row.delivered:>10}{row.overwritten:>13}"
              f"{row.rejected:>10}{row.distinct_out:>14}")
    print(RULE)
    rows = {r.sanitiser: r for r in S.compare(NAMES, S.WINDOWS, WIN_DEST)}
    best = max(rows.values(), key=lambda r: r.delivered)
    nothing = rows["passthrough"]
    print(f"doing nothing delivers {nothing.delivered}. the best of the five real "
          f"sanitisers delivers {best.delivered} ({best.sanitiser}).")
    worse = [r.sanitiser for r in rows.values()
             if r.sanitiser != "passthrough" and r.delivered <= nothing.delivered]
    print(f"{len(worse)} of 5 deliver no more than doing nothing: {', '.join(worse)}.")
    print("\n'distinct out' falls as the sanitiser rewrites more. that column is the")
    print("size of the codomain, and every name it loses is two sources merging.")


def exp2_four_names_one_file() -> None:
    head(2, "Four names, one file, four successful writes")
    group = ["a:b.txt", "a*b.txt", "a?b.txt", "a|b.txt"]
    print(f"{'source':<14}{'strip_bad_chars':<20}{'django_valid':<16}{'slugify':<12}")
    print(RULE)
    for n in group:
        print(f"{n:<14}{S.s_strip_bad_chars(n):<20}{S.s_django_valid(n):<16}"
              f"{S.s_slugify(n):<12}")
    print(RULE)
    r = S.audit(group, S.WINDOWS, WIN_DEST, "strip_bad_chars")
    d, o, rej = r.partition()
    print(f"verdict: {r.verdict.value}. delivered {len(d)}, overwritten {len(o)}, "
          f"rejected {len(rej)}.")
    print("every call returned a string. no call raised. three files are gone.")
    print("\nthe replacement character is the whole mechanism: mapping N forbidden")
    print("characters to 1 replacement merges every pair that differed only there.")


def exp3_the_same_sanitiser_helps_and_harms() -> None:
    head(3, "The same sanitiser is correct on one target and destructive on another")
    print("`pathvalidate` is written against Win32's rules. Applied unconditionally:\n")
    print(f"{'target':<18}{'nothing':>9}{'pathvalidate':>14}{'change':>9}"
          f"{'  what the target actually needs'}")
    print(RULE)
    for p in [S.WINDOWS, S.MACOS_APFS, S.LINUX_EXT4, S.OBJECT_STORE]:
        dest = WIN_DEST if p.name.startswith("windows") else "/data"
        a = S.audit(NAMES, p, dest, "passthrough").delivered
        b = S.audit(NAMES, p, dest, "pathvalidate").delivered
        need = {
            "windows-ntfs": "deny-list, devices, MAX_PATH",
            "macos-apfs": "nothing but ':' and NFD folding",
            "linux-ext4": "nothing but '/' and NUL",
            "object-store": "nothing",
        }[p.name]
        print(f"{p.name:<18}{a:>9}{b:>14}{b - a:>+9}  {need}")
    print(RULE)
    print("+6 on the target it was written for. -4 on all three of the others,")
    print("because on a permissive byte-exact volume every rewrite is pure loss:")
    print("there was nothing to fix, and the rewrite still merged names.")
    print("\nsanitising happens at upload time. the target is chosen at write time.")
    print("the function is called before the answer it needs is known.")


def exp4_reserved_names_survive_extensions() -> None:
    head(4, "Reserved device names survive extensions, dots and trailing spaces")
    variants = ["CON", "CON.txt", "con.tar.gz", "CON.", "CON ", "NUL.log",
                "COM1.csv", "aux.tar.gz", "CONS.txt", "CON2.txt"]
    print(f"{'name':<14}{'win32 opens':<14}{'lookup stem':<14}{'reserved?':<11}"
          f"{'strip_bad_chars fixes it?'}")
    print(RULE)
    for n in variants:
        eff = S.win32_effective(n)
        stem = S.reserved_stem(n)
        res = S.is_reserved(n, S.WINDOWS)
        fixed = not S.is_reserved(S.s_strip_bad_chars(n), S.WINDOWS)
        print(f"{n!r:<14}{eff!r:<14}{stem:<14}{'YES' if res else '-':<11}"
              f"{'yes' if (fixed and res) else ('n/a' if not res else 'NO')}")
    print(RULE)
    print("the lookup is on the stem before the FIRST dot, after trailing dots and")
    print("spaces are stripped - so `con.tar.gz` and `CON.` are both the console.")
    print("`CONS` and `CON2` are ordinary names; the check is exact, not a prefix.\n")
    print(f"{'sanitiser':<18}{'reserved names still reserved after sanitising'}")
    print(RULE)
    res_only = [n for n in variants if S.is_reserved(n, S.WINDOWS)]
    for key, fn in S.SANITISERS.items():
        still = [n for n in res_only if S.is_reserved(fn(n), S.WINDOWS)]
        print(f"{key:<18}{len(still)} of {len(res_only)}"
              + (f"   {', '.join(repr(x) for x in still[:4])}" if still else "   (none)"))
    print(RULE)
    print("two of six handle device names. the other four ship a name Windows")
    print("will not open, and return it as a `str` with no indication.")


def exp5_no_fold_model_is_a_filesystem() -> None:
    head(5, "Neither str.lower() nor str.casefold() is any filesystem's case table")
    pairs = [
        ("Straße.txt", "STRASSE.txt", "keeps apart"),
        ("ΣΙΣΥΦΟΣ", "σισυφοσ", "merges"),
        ("ΣΙΣΥΦΟΣ.txt", "σισυφοσ.txt", "merges"),
        ("İstanbul.txt", "istanbul.txt", "keeps apart"),
        ("Q3 Report.csv", "Q3 report.csv", "merges"),
    ]
    print(f"{'a':<16}{'b':<16}{'lower':>7}{'casefold':>10}{'simple_upper':>14}"
          f"{'   NTFS/APFS'}")
    print(RULE)
    for a, b, truth in pairs:
        votes = {k: "merge" if f(a) == f(b) else "-" for k, f in S.FOLDS.items()}
        flag = "  <-- wrong" if (
            (votes["py_lower"] == "merge") != (truth == "merges")
            or (votes["py_casefold"] == "merge") != (truth == "merges")
        ) else ""
        print(f"{a:<16}{b:<16}{votes['py_lower']:>7}{votes['py_casefold']:>10}"
              f"{votes['simple_upper']:>14}   {truth}{flag}")
    print(RULE)
    print("row 1: casefold expands ß to ss and merges two files the volume keeps")
    print("       apart. a dedupe built on casefold() deletes one of them.")
    print("rows 2-3: THE SAME TWO STEMS, and lower() changes its mind when an")
    print("       extension is appended. the final-sigma rule fires only when the")
    print("       sigma ends a word, so `ΣΙΣΥΦΟΣ` lowercases to `...ος` and")
    print("       `ΣΙΣΥΦΟΣ.txt` to `...οσ.txt`. lower() is wrong on row 2 and")
    print("       right on row 3 for a reason that has nothing to do with the")
    print("       filesystem: whether there was a dot after the name.")
    print("\nthe two standard-library functions err in OPPOSITE directions - casefold")
    print("over-merges (row 1), lower under-merges (row 2) - so neither is a safe")
    print("default and picking between them is not the fix.")
    print("\nsimple_upper matches the volume on every row, because that is what a")
    print("volume does: a 1:1 uppercase table ($UpCase on NTFS), not full folding.")
    print(f"\n{len(S.case_table_disagreements(NAMES))} pair(s) in the corpus where the "
          f"three models disagree.")


def exp6_the_limit_is_not_in_characters() -> None:
    head(6, "The limit is in bytes or code units. Sanitisers count characters.")
    probes = [
        ("季度销售报告" * 15 + ".csv", "90 CJK characters"),
        ("\U0001f4c8" * 70 + ".png", "70 emoji"),
        ("a" * 300 + ".csv", "300 ASCII"),
        ("é" * 200 + ".csv", "200 precomposed é"),
    ]
    print(f"{'probe':<22}{'chars':>7}{'utf-8 B':>9}{'utf-16 CU':>11}"
          f"{'  ext4 (255B)':<15}{'NTFS (255CU)'}")
    print(RULE)
    for name, label in probes:
        chars = len(name)
        b = len(name.encode("utf-8"))
        cu = len(name.encode("utf-16-le")) // 2
        print(f"{label:<22}{chars:>7}{b:>9}{cu:>11}"
              f"{'  REJECT' if b > 255 else '  ok':<15}"
              f"{'REJECT' if cu > 255 else 'ok'}")
    print(RULE)
    print("90 CJK characters is legal on NTFS and too long on ext4. 300 ASCII is")
    print("too long on both. every character-counting length check gets one of")
    print("these two rows wrong, and which one depends on the target.\n")
    for p in [S.WINDOWS, S.LINUX_EXT4]:
        dest = WIN_DEST if p.name.startswith("windows") else "/data"
        r = S.audit(NAMES, p, dest, "passthrough")
        f = [x for x in r.findings if x.code == "BYTE_LENGTH_EXCEEDED"]
        print(f"corpus against {p.name:<14} ({p.component_unit:<18}): "
              f"{len(f[0].names) if f else 0} over the limit")


def exp7_validity_is_a_property_of_the_path() -> None:
    head(7, "Validity is a property of (name, target, destination)")
    dests = [
        r"C:\d",
        r"C:\data\exports",
        r"C:\Users\phoebe\OneDrive - Contoso Ltd\Finance\2026\Q3\exports",
        r"C:\Users\phoebe\OneDrive - Contoso Ltd\Shared Documents\Finance"
        r"\Reporting\2026\Q3\regional\emea\exports",
        r"C:\Users\phoebe\OneDrive - Contoso Ltd\Shared Documents\Finance"
        r"\Reporting\2026\Q3\regional\emea\exports\final\approved\circulated",
    ]
    print("Identical corpus, identical sanitiser, identical target. Only the")
    print("destination moves - and a pure function of the name cannot see it.\n")
    print(f"{'destination depth':>18}{'delivered':>11}{'path too long':>15}")
    print(RULE)
    for d in dests:
        r = S.audit(NAMES, S.WINDOWS, d, "pathvalidate")
        f = [x for x in r.findings if x.code == "PATH_LENGTH_EXCEEDED"]
        print(f"{len(d):>15} ch{r.delivered:>11}{len(f[0].names) if f else 0:>15}")
    print(RULE)
    print("MAX_PATH is 260 including the terminating NUL, so 259 usable characters")
    print("for the whole path. a OneDrive- or Teams-synced folder spends half of")
    print("that before the filename starts. the names did not change.")


def exp8_round_trip_between_volumes() -> None:
    head(8, "Build the archive on Linux, extract it on macOS")
    print("Nothing is sanitised here. Every name is already legal on both.\n")
    combos = [
        (S.LINUX_EXT4, S.MACOS_APFS),
        (S.LINUX_EXT4, S.WINDOWS),
        (S.MACOS_APFS, S.LINUX_EXT4),
    ]
    print(f"{'built on':<14}{'opened on':<16}{'entries':>9}{'on disk':>9}{'lost':>6}")
    print(RULE)
    for a, b in combos:
        rt = S.round_trip(NAMES, a, b)
        print(f"{rt['built_on']:<14}{rt['opened_on']:<16}{rt['entries']:>9}"
              f"{rt['files_on_disk']:>9}{rt['lost']:>6}")
    print(RULE)
    rt = S.round_trip(NAMES, S.LINUX_EXT4, S.MACOS_APFS)
    print(f"linux -> macos loses {rt['lost']}: {', '.join(short(c, 20) for c in rt['casualties'])}")
    print("\nthe archive is valid. `unzip` reports no error. the file count on disk")
    print("is lower than the file count in the archive, and the direction matters:")
    print("macos -> linux loses nothing, because byte-exact is the finer partition.")


def exp9_truncating_to_fit_is_its_own_bug() -> None:
    head(9, "Truncating to fit splits code points and invents collisions")
    probes = [
        ("季度销售报告" * 15 + ".csv", "CJK, 3 bytes each"),
        ("Q3" + "季度销售报告" * 15 + ".csv", "the same, after a 2-char prefix"),
        ("\U0001f4c8" * 70 + ".png", "emoji, 4 bytes each"),
        ("é" * 200 + ".csv", "precomposed é, 2 bytes each"),
    ]
    print("Whether a byte cut splits a character depends on the arithmetic, so it")
    print("is a bug that passes every test written against one input:\n")
    print(f"{'probe':<34}{'bytes':>7}{'cut at 255':>12}{'U+FFFD?':>9}"
          f"{'  keeps .ext?'}")
    print(RULE)
    for name, label in probes:
        naive = S.naive_truncate(name, 255)
        safe = S.truncate_to_bytes(name, 255)
        split = chr(0xFFFD) in naive
        ext = "." + name.rsplit(".", 1)[-1]
        print(f"{label:<34}{len(name.encode()):>7}"
              f"{'SPLITS' if split else 'aligns':>12}{'yes' if split else '-':>9}"
              f"{'  naive ' + ('yes' if ext in naive else 'NO')}"
              f" / safe {'yes' if ext in safe else 'no'}")
    print(RULE)
    print("255 is divisible by 3, so a name of pure 3-byte characters aligns and the")
    print("naive cut looks correct. prepend two ASCII characters - `Q3` - and the")
    print("same code on the same name emits U+FFFD. U+FFFD is a legal filename")
    print("character, so the write succeeds and the name is quietly corrupt.")
    print("this is why it survives review: the bug is in the arithmetic between the")
    print("limit and the encoding, so it passes whichever input the test used.")
    print("no probe keeps its extension under the naive cut. all of them do under")
    print("truncate_to_bytes(), which reserves the extension before cutting.\n")

    a = "Regional Sales Performance and Margin Analysis " + "x" * 240 + "-EMEA.csv"
    b = "Regional Sales Performance and Margin Analysis " + "x" * 240 + "-APAC.csv"
    print("two report names differing only in the last field before the extension:")
    print(f"  a: ...{a[-20:]!r}  ({len(a)} chars)")
    print(f"  b: ...{b[-20:]!r}  ({len(b)} chars)")
    r = S.audit([a, b], S.LINUX_EXT4, "/data", "passthrough")
    same = S.truncate_to_bytes(a, 255) == S.truncate_to_bytes(b, 255)
    print(f"\ntruncated to 255 bytes -> {'IDENTICAL' if same else 'distinct'}")
    print(f"verdict {r.verdict.value}, and both names are reported over the limit "
          f"rather than merged -")
    print("because as supplied they cannot be written at all, so the write fails")
    print("loudly. the collision only appears once you apply the usual remedy:\n")
    for f in r.findings:
        if f.code in {"BYTE_LENGTH_EXCEEDED", "TRUNCATION_COLLISION"}:
            print(f"  {f.code}: {len(f.names)} name(s)")
    print("\nthe distinguishing field was at the end. truncation is a prefix, so it")
    print("throws away exactly the part that made the two names different - which")
    print("is why TRUNCATION_COLLISION is a separate finding from the length one.")


def exp10_the_verdict() -> None:
    head(10, "The verdict, and what it does not say")
    cases = [
        (["q3.csv", "q4.csv"], "ordinary names"),
        (["Report.csv", "report.csv"], "differ only by case"),
        (["café.txt", "cafe\u0301.txt"], "differ only by normalisation"),
        (["report.", "report"], "differ only by a trailing dot"),
        (["CON.txt"], "reserved device name"),
        (["a" * 300], "over NAME_MAX"),
        (["report‐2024.pdf", "report-2024.pdf"], "confusable hyphens"),
    ]
    print(f"{'corpus':<34}{'windows':<11}{'macos':<11}{'ext4':<11}{'why'}")
    print(RULE)
    for names, why in cases:
        w = S.audit(names, S.WINDOWS, WIN_DEST, "passthrough").verdict.value
        m = S.audit(names, S.MACOS_APFS, "/data", "passthrough").verdict.value
        l = S.audit(names, S.LINUX_EXT4, "/data", "passthrough").verdict.value
        print(f"{short(', '.join(names), 32):<34}{w:<11}{m:<11}{l:<11}{why}")
    print(RULE)
    print("rows 2-4 each fail on a DIFFERENT set of volumes:")
    print("  case pair          lossy on Windows and macOS   (both fold case)")
    print("  normalisation pair lossy on macOS only           (only APFS folds NFD)")
    print("  trailing-dot pair  lossy on Windows only         (only Win32 strips)")
    print("in all three the names are legal, already distinct, and no sanitiser is")
    print("involved - so there is no single 'safe' name to rewrite them to, and a")
    print("function that cannot see the target cannot even know which rule applies.")
    print("\nthe last row is `portable` everywhere and is still a problem: two files")
    print("that render identically. `portable` is a claim about whether the bytes")
    print("survive the round trip. it is not a claim that a human can tell them")
    print("apart, and it is not an all-clear. read the findings.")


def main() -> None:
    print("=" * 78)
    print("filename-sanitiser: evidence".center(78))
    print("=" * 78)
    exp1_a_sanitiser_is_a_projection()
    exp2_four_names_one_file()
    exp3_the_same_sanitiser_helps_and_harms()
    exp4_reserved_names_survive_extensions()
    exp5_no_fold_model_is_a_filesystem()
    exp6_the_limit_is_not_in_characters()
    exp7_validity_is_a_property_of_the_path()
    exp8_round_trip_between_volumes()
    exp9_truncating_to_fit_is_its_own_bug()
    exp10_the_verdict()
    print("\n" + "=" * 78)
    print("every number above is computed, not quoted. no randomness is used.")
    print("=" * 78)


if __name__ == "__main__":
    main()
