"""Slug generation that reports the collisions it creates.

A slugifier returns a string. It cannot return the fact that two different
titles just landed on the same string, that a third one landed on the empty
string, that two titles which render identically on screen landed on *different*
strings, or that the URL a post ends up with depends on the order the posts were
imported rather than on the post.

Core ideas
----------
1. Slugifying one title is easy and almost never the bug.
2. Slugifying a *corpus* is the bug. The function is not injective, and nothing
   in its signature says so.
3. The verdict is three-valued:
   `injective`   - every title maps to a distinct, non-empty, non-reserved slug.
   `deduped`     - collisions exist and are resolvable by suffixing, at the cost
                   of making the URL depend on insertion order.
   `lossy`       - some title maps to nothing a suffix can rescue: an empty
                   slug, or a slug that shadows an application route.

Standard library only: `unicodedata`, `re`, `dataclasses`, `enum`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
#
# Each profile is a reimplementation of a slugifier that exists in the wild,
# written from its documented algorithm. They are *models*, not vendored code:
# the point is not byte-fidelity with a particular release, it is that the
# published algorithms disagree with each other on ordinary input, and the
# disagreements are structural rather than incidental.


def django_ascii(title: str) -> str:
    """Django's `slugify(value)` - the default, and the most deployed.

    NFKD-normalise, drop everything that is not ASCII, lowercase, strip
    punctuation, collapse whitespace and hyphens.

    The second step is the interesting one. NFKD decomposes a *composed* letter
    into a base plus a combining mark, so `é` survives as `e`. A letter with no
    decomposition - `ß`, `ø`, `æ`, `Ł`, `þ`, `đ` - has nothing to fall back to
    and is deleted outright. The fold is not "strip accents"; it is "keep what
    happens to decompose".
    """
    v = unicodedata.normalize("NFKD", title)
    v = v.encode("ascii", "ignore").decode("ascii")
    v = re.sub(r"[^\w\s-]", "", v.lower())
    return re.sub(r"[-\s]+", "-", v).strip("-_")


def django_unicode(title: str) -> str:
    """Django's `slugify(value, allow_unicode=True)`.

    NFKC-normalise, strip non-word characters with the Unicode-aware `\\w`,
    lowercase, collapse. Nothing is deleted for being foreign, so no title is
    erased - at the cost of URLs that are only distinguishable to a reader who
    can see the script.
    """
    v = unicodedata.normalize("NFKC", title)
    v = re.sub(r"[^\w\s-]", "", v.lower())
    return re.sub(r"[-\s]+", "-", v).strip("-_")


def naive_regex(title: str) -> str:
    """The hand-rolled one. No normalisation step at all.

    `re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')`

    Written in five seconds in a hundred thousand codebases, and correct for
    every title anyone tests it on. It is the only profile here whose output
    depends on the *normalisation form of the input bytes* rather than on the
    text, which makes it the only one that can give two different URLs to two
    strings that are canonically equivalent.
    """
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def casefold_ascii(title: str) -> str:
    """`django_ascii` with one line moved: casefold *before* normalising.

    `str.casefold()` applies the Unicode full case folding table, which expands
    `ß` to `ss` and `ﬁ` to `fi`. `str.lower()` does not - it is a 1:1 mapping
    that leaves `ß` alone for the ASCII filter to delete. Same characters, same
    steps, different order, different slug.
    """
    v = unicodedata.normalize("NFKD", title.casefold())
    v = v.encode("ascii", "ignore").decode("ascii")
    v = re.sub(r"[^\w\s-]", "", v)
    return re.sub(r"[-\s]+", "-", v).strip("-_")


_RAILS_MAP = {
    "ß": "ss", "æ": "ae", "Æ": "AE", "ø": "o", "Ø": "O", "å": "a", "Å": "A",
    "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "þ": "th", "Þ": "TH",
    "ð": "d", "Ð": "D", "œ": "oe", "Œ": "OE",
}


def rails_parameterize(title: str) -> str:
    """Rails' `String#parameterize`, which routes through `I18n.transliterate`.

    Transliteration is table-driven, so the atomic letters `django_ascii`
    deletes are *spelled out* here: `ß` becomes `ss`, `ø` becomes `o`. Anything
    with no table entry becomes `?` - Rails' documented default replacement -
    which the separator pass then turns into a hyphen. A Japanese title does not
    vanish; it becomes a row of hyphens, which strips to the empty string by a
    different route.
    """
    v = "".join(_RAILS_MAP.get(ch, ch) for ch in title)
    v = unicodedata.normalize("NFKD", v)
    v = "".join(ch if ord(ch) < 128 else ("" if unicodedata.combining(ch) else "?") for ch in v)
    v = re.sub(r"[^a-zA-Z0-9\-_]+", "-", v)
    return re.sub(r"-{2,}", "-", v).strip("-").lower()


def wordpress(title: str) -> str:
    """WordPress `sanitize_title_with_dashes`.

    `remove_accents()` folds Latin-1 and Latin Extended-A through an explicit
    table, then anything still non-ASCII is *percent-encoded* rather than
    dropped. A CJK title therefore never collapses to empty - it becomes a run
    of lowercase hex that is unique, permanent, and unreadable.
    """
    v = "".join(_RAILS_MAP.get(ch, ch) for ch in title)
    out: List[str] = []
    for ch in unicodedata.normalize("NFKD", v):
        if unicodedata.combining(ch):
            continue
        if ord(ch) < 128:
            out.append(ch)
        else:
            out.append("".join("%%%02x" % b for b in ch.encode("utf-8")))
    v = "".join(out).lower()
    v = re.sub(r"[^a-z0-9%\s\-_]", "", v)
    return re.sub(r"[\s\-]+", "-", v).strip("-")


def github_anchor(title: str) -> str:
    """GitHub's heading-anchor algorithm.

    Lowercase, delete anything that is not a word character, space or hyphen -
    Unicode-aware, so scripts survive - then spaces to hyphens. Notably it does
    *not* strip leading or trailing hyphens, so a heading that ends in a colon
    produces an anchor that ends in a hyphen.
    """
    v = re.sub(r"[^\w\- ]", "", title.lower(), flags=re.UNICODE)
    return v.replace(" ", "-")


@dataclass(frozen=True)
class Profile:
    name: str
    fn: Callable[[str], str]
    origin: str
    normalises: bool  # does it fold input to a canonical form before slugifying?

    def __call__(self, title: str) -> str:
        return self.fn(title)


PROFILES: Dict[str, Profile] = {
    p.name: p
    for p in (
        Profile("django_ascii", django_ascii, "Django slugify() default", True),
        Profile("django_unicode", django_unicode, "Django slugify(allow_unicode=True)", True),
        Profile("naive_regex", naive_regex, "hand-rolled [^a-z0-9]+", False),
        Profile("casefold_ascii", casefold_ascii, "django_ascii, casefold first", True),
        Profile("rails_parameterize", rails_parameterize, "Rails String#parameterize", True),
        Profile("wordpress", wordpress, "WordPress sanitize_title_with_dashes", True),
        Profile("github_anchor", github_anchor, "GitHub heading anchors", False),
    )
}


def profile(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; have {sorted(PROFILES)}")
    return PROFILES[name]


# ---------------------------------------------------------------------------
# Reserved route segments
# ---------------------------------------------------------------------------
#
# A slug does not live in its own namespace. It lives under a route prefix that
# already has verbs in it, and a post titled "New" claims one of them.

RESERVED: Set[str] = {
    "new", "edit", "delete", "create", "update", "index", "search", "admin",
    "api", "login", "logout", "signup", "settings", "static", "assets",
    "feed", "rss", "sitemap", "robots", "favicon", "health", "status",
}


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"


class Kind(Enum):
    COLLISION = "COLLISION"
    EMPTY_SLUG = "EMPTY_SLUG"
    ROUTE_SHADOW = "ROUTE_SHADOW"
    CONFUSABLE_SPLIT = "CONFUSABLE_SPLIT"
    TRUNCATION_COLLISION = "TRUNCATION_COLLISION"
    NORMALISATION_SPLIT = "NORMALISATION_SPLIT"
    NOT_IDEMPOTENT = "NOT_IDEMPOTENT"
    ORDER_DEPENDENT = "ORDER_DEPENDENT"


SEVERITY_OF: Dict[Kind, Severity] = {
    Kind.COLLISION: Severity.CRITICAL,
    Kind.EMPTY_SLUG: Severity.CRITICAL,
    Kind.ROUTE_SHADOW: Severity.CRITICAL,
    Kind.CONFUSABLE_SPLIT: Severity.HIGH,
    Kind.TRUNCATION_COLLISION: Severity.HIGH,
    Kind.NORMALISATION_SPLIT: Severity.HIGH,
    Kind.NOT_IDEMPOTENT: Severity.MEDIUM,
    Kind.ORDER_DEPENDENT: Severity.HIGH,
}


@dataclass(frozen=True)
class Finding:
    kind: Kind
    titles: Tuple[str, ...]
    slug: str
    detail: str

    @property
    def severity(self) -> Severity:
        return SEVERITY_OF[self.kind]

    def __str__(self) -> str:
        return f"[{self.severity.value:8}] {self.kind.value:22} {self.detail}"


class Verdict(Enum):
    INJECTIVE = "injective"
    DEDUPED = "deduped"
    LOSSY = "lossy"


@dataclass
class Report:
    profile: str
    titles: Tuple[str, ...]
    slugs: Dict[str, str]
    findings: List[Finding]
    verdict: Verdict
    reason: str
    cap: Optional[int] = None

    def of_kind(self, kind: Kind) -> List[Finding]:
        return [f for f in self.findings if f.kind == kind]

    @property
    def collision_groups(self) -> List[List[str]]:
        return [list(f.titles) for f in self.of_kind(Kind.COLLISION)]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for f in self.findings:
            out[f.kind.value] = out.get(f.kind.value, 0) + 1
        return out


# ---------------------------------------------------------------------------
# Confusable folding - for detecting the *inverse* of a collision
# ---------------------------------------------------------------------------
#
# A collision is two titles sharing a URL. The inverse is two titles that render
# identically sharing nothing, which is worse: it is not caught by a uniqueness
# constraint, produces two live pages, and is invisible in every list view.
#
# A subset of the Unicode confusables table, restricted to the substitutions
# that actually occur in copy-pasted titles: Cyrillic and Greek letters that
# share a glyph with Latin ones.

_CONFUSABLES = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "а": "a", "е": "e",
    "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ѕ": "s",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "ο": "o", " ": " ", "‐": "-", "‑": "-", "–": "-",
    "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
}


def skeleton(title: str) -> str:
    """The confusable skeleton: what the title *looks like*, not what it is."""
    v = unicodedata.normalize("NFKC", title)
    return "".join(_CONFUSABLES.get(ch, ch) for ch in v).casefold()


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


def _truncate(s: str, cap: Optional[int]) -> str:
    """Cut to `cap` characters and tidy the hyphen the cut may have exposed.

    Note what this cannot do: a cap expressed in *bytes* rather than characters
    (VARCHAR(50) in a `latin1` column, an S3 key limit) can cut a multi-byte
    character in half. `byte_truncate` below models that separately.
    """
    if cap is None or len(s) <= cap:
        return s
    return s[:cap].rstrip("-")


def byte_truncate(s: str, cap_bytes: int) -> str:
    """Truncate at a byte limit, dropping the partial character at the edge."""
    raw = s.encode("utf-8")[:cap_bytes]
    return raw.decode("utf-8", "ignore")


def audit(
    titles: Sequence[str],
    profile_name: str = "django_ascii",
    cap: Optional[int] = None,
    reserved: Optional[Set[str]] = None,
) -> Report:
    """Slugify a corpus and report everything the return value cannot carry."""
    prof = profile(profile_name)
    reserved = RESERVED if reserved is None else reserved
    titles = tuple(titles)

    full = {t: prof(t) for t in titles}
    slugs = {t: _truncate(s, cap) for t, s in full.items()}

    findings: List[Finding] = []

    # 1. Collisions on the final slug.
    by_slug: Dict[str, List[str]] = {}
    for t in titles:
        by_slug.setdefault(slugs[t], []).append(t)

    for s, group in sorted(by_slug.items()):
        if s == "" or len(group) < 2:
            continue
        # Distinguish a collision that the cap created from one that was
        # already there. They have different fixes: one is a schema change.
        pre = {full[t] for t in group}
        if cap is not None and len(pre) > 1:
            findings.append(
                Finding(
                    Kind.TRUNCATION_COLLISION,
                    tuple(group),
                    s,
                    f"{len(group)} titles have distinct slugs that become "
                    f"identical at {cap} characters -> /{s}",
                )
            )
        else:
            findings.append(
                Finding(
                    Kind.COLLISION,
                    tuple(group),
                    s,
                    f"{len(group)} titles produce the same slug -> /{s}",
                )
            )

    # 2. Empty slugs. Every one of these lands in the same bucket, and the
    #    fallback (post id, "untitled") is outside the slugifier's contract.
    empty = by_slug.get("", [])
    for t in empty:
        findings.append(
            Finding(
                Kind.EMPTY_SLUG,
                (t,),
                "",
                f"{t!r} slugifies to the empty string; the URL comes from a "
                f"fallback this function does not define",
            )
        )

    # 3. Route shadowing.
    for t in titles:
        if slugs[t] in reserved:
            findings.append(
                Finding(
                    Kind.ROUTE_SHADOW,
                    (t,),
                    slugs[t],
                    f"{t!r} -> /{slugs[t]}, which is a reserved route segment",
                )
            )

    # 4. Homoglyph split: identical skeletons, different slugs.
    by_skel: Dict[str, List[str]] = {}
    for t in titles:
        by_skel.setdefault(skeleton(t), []).append(t)
    for skel, group in sorted(by_skel.items()):
        if len(group) < 2:
            continue
        if len({slugs[t] for t in group}) > 1:
            findings.append(
                Finding(
                    Kind.CONFUSABLE_SPLIT,
                    tuple(group),
                    "",
                    f"{len(group)} titles fold to the same skeleton (case + "
                    f"confusables) but produce "
                    f"{sorted({slugs[t] for t in group})}",
                )
            )

    # 5. Normalisation split: the same text in NFC and NFD.
    for t in titles:
        nfc, nfd = unicodedata.normalize("NFC", t), unicodedata.normalize("NFD", t)
        if nfc == nfd:
            continue
        a, b = _truncate(prof(nfc), cap), _truncate(prof(nfd), cap)
        if a != b:
            findings.append(
                Finding(
                    Kind.NORMALISATION_SPLIT,
                    (t,),
                    a,
                    f"{t!r} gives /{a} in NFC and /{b} in NFD - same text, "
                    f"different bytes, different URL",
                )
            )

    # 6. Idempotence: re-slugifying an existing slug must be a no-op, because
    #    every CMS re-runs the slugifier when the post is edited.
    for t in titles:
        s = slugs[t]
        if s and _truncate(prof(s), cap) != s:
            findings.append(
                Finding(
                    Kind.NOT_IDEMPOTENT,
                    (t,),
                    s,
                    f"/{s} re-slugifies to /{_truncate(prof(s), cap)}; a no-op "
                    f"edit changes the URL",
                )
            )

    verdict, reason = _verdict(findings, titles)
    return Report(profile_name, titles, slugs, findings, verdict, reason, cap)


def _verdict(findings: Sequence[Finding], titles: Sequence[str]) -> Tuple[Verdict, str]:
    lossy = [f for f in findings if f.kind in (Kind.EMPTY_SLUG, Kind.ROUTE_SHADOW)]
    if lossy:
        n_empty = sum(1 for f in lossy if f.kind is Kind.EMPTY_SLUG)
        n_route = len(lossy) - n_empty
        bits = []
        if n_empty:
            bits.append(f"{n_empty} title(s) slugify to nothing")
        if n_route:
            bits.append(f"{n_route} title(s) shadow a route")
        return Verdict.LOSSY, "; ".join(bits) + " - no suffix rescues these"
    dup = [f for f in findings if f.kind in (Kind.COLLISION, Kind.TRUNCATION_COLLISION)]
    if dup:
        n = sum(len(f.titles) for f in dup)
        return (
            Verdict.DEDUPED,
            f"{n} of {len(titles)} titles fall into {len(dup)} collision group(s); "
            f"resolvable by suffixing, which makes the URL a function of import order",
        )
    return Verdict.INJECTIVE, f"all {len(titles)} titles map to distinct, usable slugs"


# ---------------------------------------------------------------------------
# Assignment - what actually ends up in the database
# ---------------------------------------------------------------------------


def assign(
    titles: Sequence[str],
    profile_name: str = "django_ascii",
    cap: Optional[int] = None,
    fallback: str = "untitled",
) -> Dict[str, str]:
    """Resolve collisions the way every CMS does: append -2, -3, ...

    This is where the URL stops being a property of the post. The suffix is
    decided by whoever got inserted first, so the same corpus loaded in a
    different order produces a different set of URLs.
    """
    prof = profile(profile_name)
    taken: Set[str] = set()
    out: Dict[str, str] = {}
    for t in titles:
        base = _truncate(prof(t), cap) or fallback
        cand, n = base, 1
        while cand in taken:
            n += 1
            cand = f"{base}-{n}"
        taken.add(cand)
        out[t] = cand
    return out


def order_sensitivity(
    titles: Sequence[str],
    orders: Sequence[Sequence[str]],
    profile_name: str = "django_ascii",
    cap: Optional[int] = None,
) -> Tuple[int, List[Tuple[str, List[str]]]]:
    """How many titles get a different URL depending on the insertion order.

    Returns the count and the per-title set of URLs it was seen under.
    """
    seen: Dict[str, Set[str]] = {t: set() for t in titles}
    for order in orders:
        got = assign(order, profile_name, cap)
        for t in titles:
            seen[t].add(got[t])
    unstable = [(t, sorted(v)) for t, v in seen.items() if len(v) > 1]
    return len(unstable), unstable


def deletion_promotes_nobody(
    titles: Sequence[str], deleted: str, newcomer: str, profile_name: str = "django_ascii"
) -> Dict[str, str]:
    """The link-rot mechanism nobody models.

    Delete the post holding the bare slug and the post holding `-2` is *not*
    promoted - so the bare slug is free, and the next post to claim it inherits
    every inbound link, bookmark and search result pointing at the deleted one.
    """
    before = assign(titles, profile_name)

    # Stored slugs persist. Deleting a row does not re-slugify anything else,
    # so the survivors keep exactly the URLs they were given.
    after = {t: before[t] for t in titles if t != deleted}

    prof = profile(profile_name)
    taken = set(after.values())
    base = prof(newcomer) or "untitled"
    cand, n = base, 1
    while cand in taken:
        n += 1
        cand = f"{base}-{n}"
    after[newcomer] = cand

    runner_up = next((t for t in titles if t != deleted), "")
    return {
        "deleted_had": before[deleted],
        "newcomer_gets": after[newcomer],
        "reused": "yes" if after[newcomer] == before[deleted] else "no",
        "runner_up_before": before.get(runner_up, ""),
        "runner_up_after": after.get(runner_up, ""),
    }


# ---------------------------------------------------------------------------
# Cross-profile comparison
# ---------------------------------------------------------------------------


def compare(titles: Sequence[str], names: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, str]]:
    """title -> {profile: slug}. The migration diff, before you migrate."""
    names = list(PROFILES) if names is None else list(names)
    return {t: {n: PROFILES[n](t) for n in names} for t in titles}


def disagreements(titles: Sequence[str], a: str, b: str) -> List[Tuple[str, str, str]]:
    """Titles whose URL changes when you move from profile `a` to profile `b`."""
    pa, pb = profile(a), profile(b)
    return [(t, pa(t), pb(t)) for t in titles if pa(t) != pb(t)]


def truncation_curve(
    titles: Sequence[str], profile_name: str = "django_ascii", caps: Sequence[int] = ()
) -> List[Tuple[int, int, int, int]]:
    """`(cap, groups, titles_in_a_collision, distinct_urls)` as the cap shrinks.

    Group *count* is the wrong thing to watch: shortening the cap merges two
    groups into one as often as it creates a new one, so the count wobbles
    while the damage grows. The number of titles that no longer have a URL of
    their own is monotone, and it is what the reader loses.
    """
    prof = profile(profile_name)
    full = [prof(t) for t in titles]
    out = []
    for cap in caps:
        by: Dict[str, int] = {}
        for s in full:
            k = _truncate(s, cap)
            if k:
                by[k] = by.get(k, 0) + 1
        groups = sum(1 for v in by.values() if v > 1)
        colliding = sum(v for v in by.values() if v > 1)
        out.append((cap, groups, colliding, len(by)))
    return out


# ---------------------------------------------------------------------------
# Sample corpus
# ---------------------------------------------------------------------------
#
# Ordinary titles from an ordinary engineering blog. Nothing here is adversarial
# and nothing is invalid; every one of them is a headline somebody would ship.

CORPUS: Tuple[str, ...] = (
    "Why we moved to Postgres",
    "Why we moved to Postgres!",
    "C++ for data engineers",
    "C# for data engineers",
    "Node.js at scale",
    "NodeJS at scale",
    "Node JS at scale",
    "Straße oder Strasse",
    "STRASSE ODER STRASSE",
    "Łódź: our new datacentre",
    "Odz - a naming retrospective",
    "Søren on schema design",
    "Encyclopædia of failure modes",
    "Ångström-scale profiling",
    "café culture and code review",       # NFC
    "café culture and code review",  # NFD - same text, different bytes
    "北京 office opening",
    "東京 office opening",
    "Привет from the Moscow team",
    "データ契約の基礎",
    "デ\u30fc\u30bf品質の測り方",
    "🎉🎉🎉",
    "🎉 We raised a Series B",
    "🚀 We raised a Series B",
    "New",
    "Search",
    "API",
    "The complete guide to building resilient data pipelines in production",
    "The complete guide to building resilient data pipelines on Kubernetes",
    "How we cut our warehouse bill by 40 percent in six weeks",
    "How we cut our warehouse bill by 40 percent without rewriting a query",
    "Building a data platform: part one, ingestion",
    "Building a data platform: part two, transformation",
    "Building a data platform: part three, serving",
    "Аpple silicon benchmarks",   # Cyrillic А
    "Apple silicon benchmarks",   # Latin A
    "İstanbul engineering offsite",
    "Istanbul engineering offsite",
    "Ⅻ lessons from a decade of on-call",
    "①②③ steps to a data contract",
    "What's next for our platform?",
    "Whats next for our platform",
    "Hello --- World",
    "Hello, World!",
)
