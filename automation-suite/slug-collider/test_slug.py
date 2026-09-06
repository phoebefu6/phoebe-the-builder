"""Tests for slug.py.

The ones that matter are at the bottom: a second, obviously-correct collision
detector cross-checked against the fast one, and the two properties every
slugifier is assumed to have and only some of them do.

Run: python3 -m pytest test_slug.py -q
"""

from __future__ import annotations

import itertools
import unicodedata

import pytest
import slug as S

# ---------------------------------------------------------------------------
# Profile behaviour - the documented algorithms, pinned
# ---------------------------------------------------------------------------


def test_django_deletes_letters_it_cannot_decompose():
    # The headline fact. NFKD has no decomposition for these, so the ASCII
    # filter removes them entirely rather than folding them.
    assert S.django_ascii("Straße") == "strae"
    assert S.django_ascii("Łódź") == "odz"
    assert S.django_ascii("Søren") == "sren"
    assert S.django_ascii("Encyclopædia") == "encyclopdia"


def test_django_keeps_letters_it_can_decompose():
    assert S.django_ascii("café") == "cafe"
    assert S.django_ascii("Ångström") == "angstrom"


def test_nfkd_is_not_only_accent_stripping():
    # Compatibility decomposition rewrites semantics, not just diacritics.
    assert S.django_ascii("Ⅻ lessons") == "xii-lessons"
    assert S.django_ascii("①②③ steps") == "123-steps"
    assert S.django_ascii("ﬁle handles") == "file-handles"


def test_casefold_before_normalise_changes_the_slug():
    assert S.django_ascii("Straße") == "strae"
    assert S.casefold_ascii("Straße") == "strasse"
    # ...which decides whether these two titles collide.
    assert S.django_ascii("Straße") != S.django_ascii("STRASSE")
    assert S.casefold_ascii("Straße") == S.casefold_ascii("STRASSE")


def test_turkish_capital_i_grows_a_combining_mark():
    lowered = "İstanbul".lower()
    assert len(lowered) == 9  # 8 letters plus U+0307
    assert "̇" in lowered
    # Django's NFKD pass drops the mark; the naive regex turns it into a hyphen.
    assert S.django_ascii("İstanbul") == "istanbul"
    assert S.naive_regex("İstanbul") == "i-stanbul"


def test_punctuation_annihilation():
    assert S.django_ascii("C++") == S.django_ascii("C#") == "c"
    assert S.django_ascii("Node.js") == S.django_ascii("NodeJS") == "nodejs"
    assert S.django_ascii("Node JS") == "node-js"


def test_scripts_are_erased_or_encoded_depending_on_profile():
    assert S.django_ascii("データ契約の基礎") == ""
    assert S.django_unicode("データ契約の基礎") == "データ契約の基礎"
    assert S.rails_parameterize("データ契約の基礎") == ""
    wp = S.wordpress("データ契約の基礎")
    assert wp.startswith("%") and len(wp) == 8 * 9  # 8 chars, 3 bytes each, %xx


def test_emoji_slugify_to_nothing_in_every_ascii_profile():
    for name in ("django_ascii", "naive_regex", "casefold_ascii", "rails_parameterize"):
        assert S.PROFILES[name]("🎉🎉🎉") == ""


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_only_the_unnormalised_profiles_split_on_nfc_vs_nfd():
    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", "café")
    assert nfc != nfd  # different bytes
    for name, p in S.PROFILES.items():
        if p.normalises:
            assert p(nfc) == p(nfd), f"{name} split on a canonical equivalence"
        else:
            assert p(nfc) != p(nfd), f"{name} unexpectedly survived NFD"
    # And the concrete pair: three characters versus four.
    assert S.naive_regex(nfc) == "caf"
    assert S.naive_regex(nfd) == "cafe"


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


def test_verdict_is_three_valued():
    assert S.audit(["Alpha", "Beta"]).verdict is S.Verdict.INJECTIVE
    assert S.audit(["Hello, World!", "Hello --- World"]).verdict is S.Verdict.DEDUPED
    assert S.audit(["New", "Beta"]).verdict is S.Verdict.LOSSY
    assert S.audit(["🎉", "Beta"]).verdict is S.Verdict.LOSSY


def test_empty_slugs_are_not_reported_as_a_collision():
    # Three titles land on "". That is three separate failures with three
    # separate fallbacks, not one collision group.
    r = S.audit(["🎉", "🚀", "データ"])
    assert len(r.of_kind(S.Kind.EMPTY_SLUG)) == 3
    assert r.of_kind(S.Kind.COLLISION) == []


def test_truncation_collision_is_distinguished_from_a_plain_one():
    a = "The complete guide to building resilient data pipelines in production"
    b = "The complete guide to building resilient data pipelines on Kubernetes"
    assert S.audit([a, b]).of_kind(S.Kind.TRUNCATION_COLLISION) == []
    capped = S.audit([a, b], cap=50)
    assert len(capped.of_kind(S.Kind.TRUNCATION_COLLISION)) == 1
    assert capped.of_kind(S.Kind.COLLISION) == []


def test_confusable_split_finds_the_inverse_of_a_collision():
    r = S.audit(["Аpple silicon", "Apple silicon"])  # first A is U+0410
    f = r.of_kind(S.Kind.CONFUSABLE_SPLIT)
    assert len(f) == 1
    assert r.verdict is S.Verdict.INJECTIVE  # no uniqueness constraint would fire
    assert set(r.slugs.values()) == {"apple-silicon", "pple-silicon"}


def test_route_shadow():
    r = S.audit(["New", "Search", "API"])
    assert len(r.of_kind(S.Kind.ROUTE_SHADOW)) == 3


# ---------------------------------------------------------------------------
# Assignment and order
# ---------------------------------------------------------------------------


def test_assign_suffixes_collisions():
    got = S.assign(["Hello, World!", "Hello --- World", "Hello World"])
    assert sorted(got.values()) == ["hello-world", "hello-world-2", "hello-world-3"]


def test_assign_uses_the_fallback_for_empty_slugs():
    got = S.assign(["🎉", "🚀"])
    assert sorted(got.values()) == ["untitled", "untitled-2"]


def test_the_url_is_a_function_of_insertion_order():
    titles = ["Hello, World!", "Hello --- World"]
    forward = S.assign(titles)
    backward = S.assign(list(reversed(titles)))
    assert forward != backward
    assert forward[titles[0]] == "hello-world"
    assert backward[titles[0]] == "hello-world-2"


def test_order_sensitivity_counts_the_unstable_titles():
    titles = list(S.CORPUS)
    n, unstable = S.order_sensitivity(titles, [titles, list(reversed(titles))])
    assert n > 0
    assert all(len(v) > 1 for _, v in unstable)


def test_deleting_the_holder_frees_the_slug_for_a_stranger():
    out = S.deletion_promotes_nobody(
        ["Hello, World!", "Hello --- World"], "Hello, World!", "Hello World"
    )
    assert out["reused"] == "yes"
    # The runner-up keeps its suffix; it is not promoted.
    assert out["runner_up_before"] == out["runner_up_after"] == "hello-world-2"


# ---------------------------------------------------------------------------
# Cross-check: a second collision detector that is obviously correct
# ---------------------------------------------------------------------------


def _pairwise_groups(titles, prof, cap=None):
    """O(n^2) grouping. Shares nothing with `audit` except the profile."""
    slugs = [S._truncate(prof(t), cap) for t in titles]
    groups = []
    used = set()
    for i, j in itertools.combinations(range(len(titles)), 2):
        if slugs[i] == "" or slugs[i] != slugs[j]:
            continue
        for g in groups:
            if i in g or j in g:
                g.update({i, j})
                break
        else:
            groups.append({i, j})
        used |= {i, j}
    return sorted(tuple(sorted(g)) for g in groups)


@pytest.mark.parametrize("name", sorted(S.PROFILES))
@pytest.mark.parametrize("cap", [None, 50, 30, 12])
def test_fast_and_slow_collision_detectors_agree(name, cap):
    titles = list(S.CORPUS)
    r = S.audit(titles, name, cap=cap)
    fast = sorted(
        tuple(sorted(titles.index(t) for t in f.titles))
        for f in r.findings
        if f.kind in (S.Kind.COLLISION, S.Kind.TRUNCATION_COLLISION)
    )
    slow = _pairwise_groups(titles, S.profile(name), cap)
    assert fast == slow, f"{name} @ cap={cap}"


# ---------------------------------------------------------------------------
# Properties every slugifier is assumed to have
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(S.PROFILES))
def test_idempotence(name):
    """slugify(slugify(x)) == slugify(x).

    Every CMS re-runs the slugifier when a post is edited. If this fails, a
    no-op edit changes a live URL.
    """
    p = S.profile(name)
    for t in S.CORPUS:
        once = p(t)
        assert p(once) == once, f"{name}: {t!r} -> {once!r} -> {p(once)!r}"


@pytest.mark.parametrize("name", sorted(S.PROFILES))
def test_output_is_url_safe(name):
    """No character in the output needs percent-encoding when placed in a path.

    `wordpress` is the exception by design: it percent-encodes, so `%` is legal
    in its output and only there.
    """
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
    if name == "wordpress":
        allowed |= {"%"}
    for t in S.CORPUS:
        out = S.profile(name)(t)
        if name in ("django_unicode", "github_anchor"):
            continue  # these emit non-ASCII on purpose
        assert set(out) <= allowed, f"{name}: {t!r} -> {out!r}"


def test_truncation_curve_is_monotone_in_the_wrong_direction():
    caps = [200, 100, 60, 40, 25, 15]
    curve = S.truncation_curve(S.CORPUS, "django_ascii", caps)
    colliding = [c for _, _, c, _ in curve]
    distinct = [d for _, _, _, d in curve]
    # Titles that lose their own URL only ever increase as the cap shrinks.
    assert colliding == sorted(colliding)
    assert distinct == sorted(distinct, reverse=True)
    assert colliding[-1] > colliding[0]
    # Group *count* is not monotone - shrinking merges groups as well as
    # creating them, which is why it is the wrong thing to watch.
    groups = [g for _, g, _, _ in curve]
    assert max(groups) >= groups[-1]


def test_byte_truncation_does_not_emit_a_broken_character():
    s = S.django_unicode("データ契約の基礎")
    for n in range(1, 30):
        S.byte_truncate(s, n).encode("utf-8")  # must not raise
