"""Tests for sanitise.py.

Two groups earn their place beyond the obvious unit checks:

`test_partition_*` - the three buckets must partition the corpus exactly, for
every sanitiser against every profile. This is the invariant that caught the
first version reporting 6 and 10 for the same quantity, because `compare()`
derived `overwritten` as a residual while `Report.lost` counted merge groups.

`test_simple_upper_*` - `fold_simple_upper` claims to model a 1:1 case table, so
it must preserve length for every code point in the BMP. A table that expands is
not a 1:1 table, and the whole §5 argument rests on that distinction.
"""

from __future__ import annotations

import unicodedata

import pytest

import sanitise as S

ALL_PROFILES = [S.WINDOWS, S.WINDOWS_LONG, S.MACOS_APFS, S.LINUX_EXT4, S.OBJECT_STORE]
ALL_SANITISERS = list(S.SANITISERS)


def dest_for(p: S.Profile) -> str:
    return r"C:\data" if p.name.startswith("windows") else "/data"


# ---------------------------------------------------------------------------
# The partition invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=lambda p: p.name)
@pytest.mark.parametrize("san", ALL_SANITISERS)
def test_partition_is_exhaustive(profile: S.Profile, san: str) -> None:
    r = S.audit(S.SAMPLE_NAMES, profile, dest_for(profile), san)
    d, o, rej = r.partition()
    assert len(d) + len(o) + len(rej) == len(S.SAMPLE_NAMES)


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=lambda p: p.name)
@pytest.mark.parametrize("san", ALL_SANITISERS)
def test_partition_is_disjoint(profile: S.Profile, san: str) -> None:
    r = S.audit(S.SAMPLE_NAMES, profile, dest_for(profile), san)
    d, o, rej = (set(x) for x in r.partition())
    assert d & o == set()
    assert d & rej == set()
    assert o & rej == set()


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=lambda p: p.name)
def test_partition_covers_every_source_name(profile: S.Profile) -> None:
    r = S.audit(S.SAMPLE_NAMES, profile, dest_for(profile), "pathvalidate")
    d, o, rej = r.partition()
    assert set(d) | set(o) | set(rej) == set(S.SAMPLE_NAMES)


def test_compare_rows_agree_with_report_partition() -> None:
    """`compare()` and `Report` must not compute the same quantity two ways."""
    for row in S.compare(S.SAMPLE_NAMES, S.WINDOWS, r"C:\data"):
        r = S.audit(S.SAMPLE_NAMES, S.WINDOWS, r"C:\data", row.sanitiser)
        d, o, rej = r.partition()
        assert (row.delivered, row.overwritten, row.rejected) == (
            len(d), len(o), len(rej)
        )


def test_overwritten_counts_all_names_in_a_merge_group() -> None:
    """Not just the losers: which name survives depends on write order, so every
    name in the group is at risk."""
    r = S.audit(["a:b.txt", "a*b.txt", "a?b.txt"], S.WINDOWS, r"C:\d", "strip_bad_chars")
    d, o, rej = r.partition()
    assert len(o) == 3
    assert d == []


# ---------------------------------------------------------------------------
# fold_simple_upper models a 1:1 table, so it must preserve length
# ---------------------------------------------------------------------------


def test_simple_upper_preserves_length_across_the_bmp() -> None:
    for cp in range(0x0000, 0x10000):
        if 0xD800 <= cp <= 0xDFFF:  # surrogates are not characters
            continue
        ch = chr(cp)
        assert len(S.fold_simple_upper(ch)) == 1, f"U+{cp:04X} expanded"


def test_simple_upper_preserves_length_above_the_bmp() -> None:
    for cp in list(range(0x10000, 0x10100)) + list(range(0x1F300, 0x1F400)):
        assert len(S.fold_simple_upper(chr(cp))) == 1


def test_full_folding_does_expand_which_is_the_point() -> None:
    """The contrast the §5 argument depends on."""
    assert len("ß".casefold()) == 2
    assert len(S.fold_simple_upper("ß")) == 1
    assert len("İ".lower()) == 2
    assert len(S.fold_simple_upper("İ")) == 1


def test_casefold_over_merges_relative_to_a_volume() -> None:
    a, b = "Straße.txt", "STRASSE.txt"
    assert a.casefold() == b.casefold()                    # casefold merges
    assert S.fold_simple_upper(a) != S.fold_simple_upper(b)  # NTFS does not


def test_lower_under_merges_relative_to_a_volume() -> None:
    a, b = "ΣΙΣΥΦΟΣ", "σισυφοσ"
    assert a.lower() != b.lower()                           # lower keeps apart
    assert S.fold_simple_upper(a) == S.fold_simple_upper(b)  # NTFS merges


def test_final_sigma_rule_is_context_dependent() -> None:
    """The same two stems, and lower() changes its answer when a dot follows."""
    assert "ΣΙΣΥΦΟΣ".lower() != "σισυφοσ".lower()
    assert "ΣΙΣΥΦΟΣ.txt".lower() == "σισυφοσ.txt".lower()


# ---------------------------------------------------------------------------
# Reserved device names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["CON", "con", "CON.txt", "con.tar.gz", "CON.", "CON ", "CON. ",
     "NUL.log", "COM1.csv", "LPT9", "aux.tar.gz", "PRN.pdf"],
)
def test_reserved_names_are_reserved(name: str) -> None:
    assert S.is_reserved(name, S.WINDOWS)


@pytest.mark.parametrize(
    "name", ["CONS.txt", "CON2.txt", "MYCON", "COM.txt", "COM10.csv", "CONSOLE",
             "NULL.log", "report.CON"],
)
def test_lookalikes_are_not_reserved(name: str) -> None:
    """The check is exact on the stem, not a prefix or substring match.

    `COM.txt` is not reserved (the device names are COM0-COM9), `COM10` is not
    reserved, and a reserved word in the *extension* position is irrelevant.
    """
    assert not S.is_reserved(name, S.WINDOWS)


def test_reserved_names_are_windows_only() -> None:
    for p in [S.MACOS_APFS, S.LINUX_EXT4, S.OBJECT_STORE]:
        assert not S.is_reserved("CON.txt", p)


def test_reserved_stem_takes_everything_before_the_first_dot() -> None:
    assert S.reserved_stem("con.tar.gz") == "CON"
    assert S.reserved_stem("CON.") == "CON"
    assert S.reserved_stem("a.CON") == "A"


# ---------------------------------------------------------------------------
# win32 trailing-strip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "given,opens",
    [("report.", "report"), ("report ", "report"), ("report. ", "report"),
     ("report..", "report"), ("report", "report"), (".hidden", ".hidden"),
     (" leading", " leading"), ("a.b.", "a.b")],
)
def test_win32_effective(given: str, opens: str) -> None:
    assert S.win32_effective(given) == opens


def test_trailing_strip_creates_a_collision_no_sanitiser_caused() -> None:
    r = S.audit(["report.", "report"], S.WINDOWS, r"C:\d", "passthrough")
    assert r.verdict is S.Verdict.LOSSY
    assert any(f.code == "TRAILING_STRIP_COLLISION" for f in r.findings)


def test_trailing_strip_is_not_applied_on_posix() -> None:
    r = S.audit(["report.", "report"], S.LINUX_EXT4, "/d", "passthrough")
    assert r.verdict is S.Verdict.PORTABLE


# ---------------------------------------------------------------------------
# Length: the unit is never characters
# ---------------------------------------------------------------------------


def test_component_length_counts_utf8_bytes_on_posix() -> None:
    assert S.component_length("季", S.LINUX_EXT4) == 3
    assert S.component_length("\U0001f4c8", S.LINUX_EXT4) == 4
    assert S.component_length("é", S.LINUX_EXT4) == 2


def test_component_length_counts_utf16_code_units_on_ntfs() -> None:
    assert S.component_length("季", S.WINDOWS) == 1
    assert S.component_length("\U0001f4c8", S.WINDOWS) == 2  # surrogate pair
    assert S.component_length("é", S.WINDOWS) == 1


def test_same_name_opposite_verdicts_by_unit() -> None:
    """90 CJK characters: legal on NTFS, over NAME_MAX on ext4."""
    name = "季度销售报告" * 15 + ".csv"
    assert S.component_length(name, S.WINDOWS) <= 255
    assert S.component_length(name, S.LINUX_EXT4) > 255


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 7, 16, 31, 64, 100, 255, 254, 253])
@pytest.mark.parametrize(
    "name",
    ["季度销售报告" * 40, "Q3" + "季度销售报告" * 40, "\U0001f4c8" * 80,
     "é" * 300, "a" * 400, "aé季\U0001f4c8" * 80],
)
def test_truncate_to_bytes_never_exceeds_and_always_decodes(name: str, limit: int) -> None:
    out = S.truncate_to_bytes(name, limit)
    assert len(out.encode("utf-8")) <= limit
    assert out.encode("utf-8").decode("utf-8") == out
    assert chr(0xFFFD) not in out


def test_truncate_to_bytes_is_a_prefix_or_prefix_plus_extension() -> None:
    name = "report-" + "x" * 400 + ".csv"
    out = S.truncate_to_bytes(name, 255)
    assert out.endswith(".csv")
    assert name.startswith(out[: -len(".csv")])


def test_truncate_to_bytes_is_identity_under_the_limit() -> None:
    assert S.truncate_to_bytes("short.csv", 255) == "short.csv"


def test_naive_truncate_splits_where_the_arithmetic_says_so() -> None:
    """255 % 3 == 0, so pure CJK aligns and the bug hides."""
    aligned = S.naive_truncate("季" * 100, 255)
    assert chr(0xFFFD) not in aligned
    shifted = S.naive_truncate("Q3" + "季" * 100, 255)
    assert chr(0xFFFD) in shifted


def test_truncation_drops_the_distinguishing_suffix() -> None:
    a = "R" + "x" * 300 + "-EMEA.csv"
    b = "R" + "x" * 300 + "-APAC.csv"
    assert a != b
    assert S.truncate_to_bytes(a, 255) == S.truncate_to_bytes(b, 255)


# ---------------------------------------------------------------------------
# Validity depends on the destination
# ---------------------------------------------------------------------------


def test_path_length_depends_on_the_destination() -> None:
    name = "R" * 200 + ".csv"
    shallow = S.audit([name], S.WINDOWS, r"C:\d", "passthrough")
    deep = S.audit([name], S.WINDOWS, "C:\\" + "d" * 100, "passthrough")
    assert not any(f.code == "PATH_LENGTH_EXCEEDED" for f in shallow.findings)
    assert any(f.code == "PATH_LENGTH_EXCEEDED" for f in deep.findings)


def test_max_path_does_not_apply_when_long_paths_are_enabled() -> None:
    name = "R" * 200 + ".csv"
    deep = "C:\\" + "d" * 100
    assert any(
        f.code == "PATH_LENGTH_EXCEEDED"
        for f in S.audit([name], S.WINDOWS, deep, "passthrough").findings
    )
    assert not any(
        f.code == "PATH_LENGTH_EXCEEDED"
        for f in S.audit([name], S.WINDOWS_LONG, deep, "passthrough").findings
    )


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def test_clean_corpus_is_portable() -> None:
    assert S.audit(["q3.csv", "q4.csv"]).verdict is S.Verdict.PORTABLE


def test_case_pair_is_lossy_on_windows_and_portable_on_ext4() -> None:
    names = ["Report.csv", "report.csv"]
    assert S.audit(names, S.WINDOWS, r"C:\d").verdict is S.Verdict.LOSSY
    assert S.audit(names, S.LINUX_EXT4, "/d").verdict is S.Verdict.PORTABLE


def test_normalisation_pair_is_lossy_on_apfs_only() -> None:
    names = ["café.txt", "cafe\u0301.txt"]
    assert S.audit(names, S.MACOS_APFS, "/d").verdict is S.Verdict.LOSSY
    assert S.audit(names, S.LINUX_EXT4, "/d").verdict is S.Verdict.PORTABLE
    assert S.audit(names, S.WINDOWS, r"C:\d").verdict is S.Verdict.PORTABLE


def test_reserved_name_is_rejected() -> None:
    assert S.audit(["CON.txt"], S.WINDOWS, r"C:\d").verdict is S.Verdict.REJECTED


def test_rejected_outranks_lossy() -> None:
    """A fatal finding pins the verdict regardless of how many names merge."""
    names = ["Report.csv", "report.csv", "CON.txt"]
    assert S.audit(names, S.WINDOWS, r"C:\d").verdict is S.Verdict.REJECTED


def test_confusables_are_portable_and_still_reported() -> None:
    names = ["report\u20102024.pdf", "report-2024.pdf"]
    r = S.audit(names, S.LINUX_EXT4, "/d")
    assert r.verdict is S.Verdict.PORTABLE
    assert any(f.code == "CONFUSABLE_PAIR" for f in r.findings)


def test_empty_corpus_is_portable_and_finds_nothing() -> None:
    r = S.audit([])
    assert r.verdict is S.Verdict.PORTABLE
    assert r.findings == []


# ---------------------------------------------------------------------------
# Collision reasons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,reason",
    [
        ("report.", "report", "trailing"),
        ("report ", "report", "trailing"),
        ("café.txt", "cafe\u0301.txt", "nfc"),
        ("Report.csv", "report.csv", "case"),
        ("a_b.txt", "a_b.txt", "identical"),
        ("a:b.txt", "a|b.txt", "sanitiser"),
    ],
)
def test_collision_reason(a: str, b: str, reason: str) -> None:
    assert S.collision_reason(a, b) == reason


def test_collision_reason_applies_the_rules_cumulatively() -> None:
    """A pair needing two rules is classified by the one that finally merges it,
    not dropped to `sanitiser` because no single rule matched on its own."""
    assert S.collision_reason("CAF\u00c9.txt.", "CAF\u00c9.txt") == "trailing"
    # needs the trailing strip, then NFD: decomposed E-acute vs precomposed
    assert S.collision_reason("CAFE\u0301.txt.", "CAF\u00c9.txt") == "nfc"
    # needs the trailing strip, then the case fold
    assert S.collision_reason("CAF\u00c9.txt.", "caf\u00e9.txt") == "case"


# ---------------------------------------------------------------------------
# Sanitisers: documented behaviour, including where they fail
# ---------------------------------------------------------------------------


def test_replacement_character_merges_every_forbidden_character() -> None:
    outs = {S.s_strip_bad_chars(f"a{c}b.txt") for c in '<>:"|?*'}
    assert outs == {"a_b.txt"}


def test_only_two_sanitisers_handle_reserved_names() -> None:
    handles = {
        k for k, fn in S.SANITISERS.items() if not S.is_reserved(fn("CON.txt"), S.WINDOWS)
    }
    assert handles == {"werkzeug_secure", "pathvalidate"}


def test_ascii_folding_sanitisers_destroy_cjk_entirely() -> None:
    assert S.s_werkzeug_secure("季度销售报告.csv") == "csv"
    assert S.s_django_valid("季度销售报告.csv") == "季度销售报告.csv"
    # slugify empties the stem and leaves a bare extension - a dot-file, which
    # is invisible in `ls` and in every file picker.
    assert S.s_slugify("季度销售报告.csv") == ".csv"
    assert any(
        f.code == "LEADING_DASH_OR_DOT"
        for f in S.audit(["季度销售报告.csv"], S.LINUX_EXT4, "/d", "slugify").findings
    )


def test_ascii_folding_collapses_a_cjk_corpus_to_one_target() -> None:
    names = ["销售.csv", "报告.csv", "季度.csv"]
    r = S.audit(names, S.LINUX_EXT4, "/d", "werkzeug_secure")
    assert r.verdict is S.Verdict.LOSSY
    assert r.delivered == 0


@pytest.mark.parametrize("san", ALL_SANITISERS)
def test_sanitisers_are_idempotent(san: str) -> None:
    """Two layers both sanitising is the normal case, not an exotic one: a web
    framework on upload and a storage client on write. A non-idempotent
    sanitiser gives a different name depending on how many layers ran."""
    fn = S.SANITISERS[san]
    for name in S.SAMPLE_NAMES:
        once = fn(name)
        assert fn(once) == once, f"{san} is not idempotent on {name!r}"


@pytest.mark.parametrize("san", ALL_SANITISERS)
def test_sanitiser_output_has_no_separators_or_is_reported(san: str) -> None:
    """A sanitiser may leave a separator in - but then the audit must say so."""
    fn = S.SANITISERS[san]
    r = S.audit(S.SAMPLE_NAMES, S.WINDOWS, r"C:\d", san)
    leaky = [n for n in S.SAMPLE_NAMES if len(S.__dict__["re"].split(r"[/\\]", fn(n))) > 1]
    reported = {n for f in r.findings if f.code == "PATH_TRAVERSAL" for n in f.names}
    assert set(leaky) <= reported


# ---------------------------------------------------------------------------
# Round trip between volumes
# ---------------------------------------------------------------------------


def test_byte_exact_to_insensitive_can_lose_files() -> None:
    rt = S.round_trip(S.SAMPLE_NAMES, S.LINUX_EXT4, S.MACOS_APFS)
    assert rt["lost"] > 0
    assert rt["files_on_disk"] < rt["entries"]


def test_insensitive_to_byte_exact_loses_nothing() -> None:
    """Byte-exact is the finer partition, so the direction is not symmetric."""
    rt = S.round_trip(S.SAMPLE_NAMES, S.MACOS_APFS, S.LINUX_EXT4)
    assert rt["lost"] == 0


def test_round_trip_to_the_same_profile_is_lossless() -> None:
    for p in ALL_PROFILES:
        assert S.round_trip(S.SAMPLE_NAMES, p, p)["lost"] == 0


# ---------------------------------------------------------------------------
# The claim that sanitising can make things worse
# ---------------------------------------------------------------------------


def test_sanitising_helps_on_windows_and_harms_on_ext4() -> None:
    """The same function, the same corpus, opposite sign."""
    win_gain = (
        S.audit(S.SAMPLE_NAMES, S.WINDOWS, r"C:\data", "pathvalidate").delivered
        - S.audit(S.SAMPLE_NAMES, S.WINDOWS, r"C:\data", "passthrough").delivered
    )
    ext4_gain = (
        S.audit(S.SAMPLE_NAMES, S.LINUX_EXT4, "/data", "pathvalidate").delivered
        - S.audit(S.SAMPLE_NAMES, S.LINUX_EXT4, "/data", "passthrough").delivered
    )
    assert win_gain > 0
    assert ext4_gain < 0


def test_rewriting_more_shrinks_the_codomain() -> None:
    rows = {r.sanitiser: r for r in S.compare(S.SAMPLE_NAMES, S.WINDOWS, r"C:\data")}
    assert rows["slugify"].distinct_out < rows["strip_bad_chars"].distinct_out
    assert rows["strip_bad_chars"].distinct_out < rows["passthrough"].distinct_out


def test_passthrough_is_the_identity_on_a_byte_exact_volume() -> None:
    """`distinct_out` counts *writable* outputs, so it equals the corpus size only
    where every name has a writable form. On Windows two of these names are only
    dots and spaces, which Win32 strips to nothing - so 40, not 42, and that is
    the profile talking rather than the sanitiser."""
    on_ext4 = {r.sanitiser: r for r in S.compare(S.SAMPLE_NAMES, S.LINUX_EXT4, "/data")}
    assert on_ext4["passthrough"].distinct_out == len(set(S.SAMPLE_NAMES))

    on_win = {r.sanitiser: r for r in S.compare(S.SAMPLE_NAMES, S.WINDOWS, r"C:\data")}
    assert on_win["passthrough"].distinct_out == len(set(S.SAMPLE_NAMES)) - 2
    assert any(
        f.code == "SANITISER_EMPTIED_NAME"
        for f in S.audit(["..."], S.WINDOWS, r"C:\d", "passthrough").findings
    )


def test_collision_key_is_an_equivalence_relation() -> None:
    """Grouping by key assumes transitivity. Key equality gives it for free -
    this test exists so that stays true if the key ever grows a step."""
    names = S.SAMPLE_NAMES
    for p in ALL_PROFILES:
        keys = {n: S.collision_key(n, p) for n in names}
        for a in names[:12]:
            for b in names[:12]:
                for c in names[:12]:
                    if keys[a] == keys[b] and keys[b] == keys[c]:
                        assert keys[a] == keys[c]


def test_findings_are_ordered_by_severity() -> None:
    r = S.audit(S.SAMPLE_NAMES, S.WINDOWS, r"C:\data", "strip_bad_chars")
    rank = {S.Severity.CRITICAL: 0, S.Severity.WARNING: 1, S.Severity.INFO: 2}
    seen = [rank[f.severity] for f in r.findings]
    assert seen == sorted(seen)


def test_every_finding_names_at_least_one_source() -> None:
    for p in ALL_PROFILES:
        r = S.audit(S.SAMPLE_NAMES, p, dest_for(p), "strip_bad_chars")
        for f in r.findings:
            assert f.names
            assert set(f.names) <= set(S.SAMPLE_NAMES)


# ---------------------------------------------------------------------------
# Drive-relative paths, which is why ':' is forbidden
# ---------------------------------------------------------------------------


def test_a_colon_name_is_a_drive_relative_path_not_a_filename() -> None:
    """`a:b.txt` on Win32 means `b.txt` in drive A:'s current directory.

    Passing it through does not create an oddly-named file; it writes somewhere
    else entirely. The notebook's first simplified audit missed this and
    delivered one more file than the engine, which is how it was found.
    """
    r = S.audit(["a:b.txt"], S.WINDOWS, r"C:\data", "passthrough")
    assert r.verdict is S.Verdict.REJECTED
    assert any(f.code == "PATH_TRAVERSAL" for f in r.findings)


def test_only_a_single_letter_before_the_colon_is_a_drive() -> None:
    """`ab:c.txt` is not drive-relative - drive letters are one character."""
    traversal = {
        f.code for f in S.audit(["ab:c.txt"], S.WINDOWS, r"C:\d", "passthrough").findings
    }
    assert "PATH_TRAVERSAL" not in traversal
    assert "RESERVED_CHARACTER" in traversal


def test_replacing_the_colon_removes_the_drive_relative_reading() -> None:
    for san in ["strip_bad_chars", "pathvalidate", "django_valid"]:
        r = S.audit(["a:b.txt"], S.WINDOWS, r"C:\data", san)
        assert not any(f.code == "PATH_TRAVERSAL" for f in r.findings), san
