"""Sort-order drift: what `ORDER BY name` actually returns.

`ORDER BY name` reads like a total order on a column. It is not. It is a
*collation* applied to bytes, and a collation is three separate decisions the
SQL never states:

1. which sequence the characters are in (locale tailoring),
2. how many levels of difference count as a difference (strength), and
3. whether two strings that compare equal are the *same value* for `=`,
   `DISTINCT`, `GROUP BY` and `UNIQUE` (determinism).

Decision 1 makes the same rows come back in different orders on two servers.
Decision 2 creates ties, and a tie means the row order inside it is whatever the
plan happened to produce - so paginating a tied sort drops rows and repeats
others with no error anywhere. Decision 3 changes the *number* of rows a
report returns.

This module models ten real collations as sort-key functions and measures the
disagreement. The tailorings here are deliberately small models, not ICU: the
claim is about the *structure* of the disagreement (which pairs flip, where the
ties are, what pagination does to them), and every claim is recomputed by
`evidence.py` from this file. Where the host has the matching libc locale,
`libc_agreement()` checks the model against the real thing.
"""

from __future__ import annotations

import locale
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Corpus: one text column out of a customer table, chosen so that every row
# is ordinary and at least one pair of rows disagrees under some collation.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    id: int
    name: str
    note: str

    @property
    def display(self) -> str:
        """The name with invisible structure made visible."""
        out = []
        for ch in self.name:
            if unicodedata.combining(ch):
                out.append(f"<U+{ord(ch):04X}>")
            else:
                out.append(ch)
        return "".join(out)


CORPUS: Tuple[Row, ...] = (
    Row(1, "Aaberg", "plain ASCII, sorts first almost everywhere"),
    Row(2, "Åberg", "A-ring: with A in en/de, a letter after Z in sv"),
    Row(3, "aaberg", "same letters as row 1, different case"),
    Row(4, "Ahtari", "unaccented twin of row 5"),
    Row(5, "Ähtäri", "A-diaeresis: with A in de, after Z in sv"),
    Row(6, "Muench", "sits between Mueller and Muller under the phonebook rule"),
    Row(7, "Mueller", "the ae-spelling"),
    Row(8, "Müller", "u-diaeresis: = Muller in DIN 5007-1, = Mueller in the phonebook order"),
    Row(9, "Muller", "the unaccented spelling"),
    Row(10, "Straße", "sharp s: expands to ss in the phonebook order"),
    Row(11, "Strasse", "the ss-spelling of row 10"),
    Row(12, "José", "NFC: e-acute is one code point (U+00E9)"),
    Row(13, "José", "NFD: the same string as row 12, as e + U+0301"),
    Row(14, "Jose", "no accent at all"),
    Row(15, "Istanbul", "capital I - lowercases to i everywhere but Turkish"),
    Row(16, "Işık", "dotless i and s-cedilla: separate letters in Turkish"),
    Row(17, "Isik", "the ASCII spelling of row 16"),
    Row(18, "van der Berg", "spaces: variable-weighted in ICU, ignored by glibc"),
    Row(19, "vanderBerg", "the same letters as row 18 with the spaces removed"),
    Row(20, "Van Der Berg", "the same letters again, different case"),
    Row(21, "Item 9", "digit run: after Item 10 lexicographically, before it numerically"),
    Row(22, "Item 10", "two digits"),
    Row(23, "Item 100", "three digits"),
    Row(24, "Čapek", "C-caron: with C in en, its own letter in cs"),
    Row(25, "Capek", "the unaccented twin of row 24"),
    Row(26, "Ａ Corp", "fullwidth A (U+FF21) - inside the BMP"),
    Row(27, "\U0001f40d Python Co", "U+1F40D - above the BMP, so a surrogate pair in UTF-16"),
    Row(28, "Zoe", "last under most tailorings, not under all"),
)

CORPUS_BY_ID: Dict[int, Row] = {r.id: r for r in CORPUS}
NAMES: Tuple[str, ...] = tuple(r.name for r in CORPUS)

# --------------------------------------------------------------------------
# Collation model
# --------------------------------------------------------------------------

BASE_ALPHABET = "abcdefghijklmnopqrstuvwxyz"

# Combining marks -> secondary weight. 1 means "no accent".
ACCENT: Dict[str, int] = {
    "\u0301": 2,  # acute
    "\u0300": 3,  # grave
    "\u0302": 4,  # circumflex
    "\u0303": 5,  # tilde
    "\u0308": 6,  # diaeresis
    "\u030a": 7,  # ring above
    "\u0327": 8,  # cedilla
    "\u030c": 9,  # caron
    "\u0306": 10,  # breve
    "\u0307": 11,  # dot above
}

# Primary weight classes, so every weight is a comparable (class, value) pair.
# The order of the classes is DUCET's: punctuation, then symbols, then digits,
# then the tailored Latin alphabet, then every other script by code point.
CLS_VARIABLE = 0  # space and punctuation, when not ignored
CLS_SYMBOL = 1
CLS_DIGIT = 2
CLS_LETTER = 3
CLS_OTHER = 4  # other scripts and anything with no fold

# Letters with no canonical decomposition, folded to a base plus a secondary
# difference - which is what ICU's root collation does with them.
FOLD_SECONDARY = 12
BASE_FOLD = {
    "\u0131": "i",   # dotless i
    "\u00f8": "o",   # o with stroke
    "\u00e6": "ae",
    "\u0153": "oe",
    "\u00df": "ss",  # sharp s
    "\u0111": "d",
    "\u0142": "l",
    "\u00f0": "d",
    "\u00fe": "th",
}

# Sentinel code points used to lift a tailored letter out of its base letter.
SENT = {
    "aring": "\ue001",
    "adia": "\ue002",
    "odia": "\ue003",
    "udia": "\ue004",
    "ccedil": "\ue005",
    "gbreve": "\ue006",
    "dotless": "\ue007",
    "scedil": "\ue008",
    "ccaron": "\ue009",
}


class Verdict(str, Enum):
    """What `ORDER BY name` is, under one collation, over one corpus."""

    STABLE_TOTAL = "stable-total"  # deterministic forever, but not linguistic
    TOTAL = "total"  # linguistic and injective here: safe to paginate
    TIED = "tied"  # ties: row order inside a tie is the plan's choice
    MERGING = "merging"  # ties AND nondeterministic: row counts change too


@dataclass(frozen=True)
class Collation:
    key_name: str
    models: str
    kind: str = "uca"  # "bytes" | "utf16" | "uca"
    alphabet: str = BASE_ALPHABET
    pre_map: Tuple[Tuple[str, str], ...] = ()
    variable: str = "shifted"  # "shifted" | "non-ignorable" | "ignored"
    numeric: bool = False
    strength: int = 3  # 1 primary, 2 +accents, 3 +case
    deterministic: bool = True
    lower: str = "root"  # "root" | "tr"
    libc: Optional[str] = None
    note: str = ""
    _pre: Dict[str, str] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_pre", dict(self.pre_map))


DE_PHONEBOOK = (
    ("ä", "ae"),
    ("ö", "oe"),
    ("ü", "ue"),
    ("ß", "ss"),
)
SV_TAILOR = (
    ("å", SENT["aring"]),
    ("ä", SENT["adia"]),
    ("ö", SENT["odia"]),
)
TR_TAILOR = (
    ("ç", SENT["ccedil"]),
    ("ğ", SENT["gbreve"]),
    ("ı", SENT["dotless"]),
    ("ö", SENT["odia"]),
    ("ş", SENT["scedil"]),
    ("ü", SENT["udia"]),
)

COLLATIONS: Tuple[Collation, ...] = (
    Collation(
        key_name="C",
        models='PostgreSQL COLLATE "C", SQLite BINARY, LC_ALL=C sort',
        kind="bytes",
        note="UTF-8 byte order. All uppercase before all lowercase; no locale, "
        "no version, no ties.",
    ),
    Collation(
        key_name="UTF16_BIN",
        models="Java String.compareTo, JS Array#sort, SQL Server *_BIN2 on nvarchar",
        kind="utf16",
        note="UTF-16 code-unit order. Agrees with C on the BMP and disagrees "
        "above it, because a surrogate pair starts at U+D800.",
    ),
    Collation(
        key_name="en_US_icu",
        models="ICU root/en collation, PostgreSQL ICU en-US-x-icu, MySQL utf8mb4_0900_as_cs",
        libc="en_US.UTF-8",
        note="Three levels: base letter, then accent, then case.",
    ),
    Collation(
        key_name="de_DIN",
        models="DIN 5007-1, ICU de",
        libc="de_DE.UTF-8",
        note="Umlauts are the base vowel with a secondary difference.",
    ),
    Collation(
        key_name="de_phonebook",
        models="DIN 5007-2, ICU de-u-co-phonebk",
        pre_map=DE_PHONEBOOK,
        note="Umlauts expand: ae, oe, ue, ss. Same language as de_DIN, "
        "different answer.",
    ),
    Collation(
        key_name="sv_SE",
        models="ICU sv, PostgreSQL sv-SE-x-icu",
        alphabet=BASE_ALPHABET + SENT["aring"] + SENT["adia"] + SENT["odia"],
        pre_map=SV_TAILOR,
        libc="sv_SE.UTF-8",
        note="Three letters after Z, not three accented vowels.",
    ),
    Collation(
        key_name="tr_TR",
        models="ICU tr, PostgreSQL tr-TR-x-icu",
        alphabet="abc"
        + SENT["ccedil"]
        + "defg"
        + SENT["gbreve"]
        + "h"
        + SENT["dotless"]
        + "ijklmno"
        + SENT["odia"]
        + "pqrs"
        + SENT["scedil"]
        + "tu"
        + SENT["udia"]
        + "vwxyz",
        pre_map=TR_TAILOR,
        lower="tr",
        libc="tr_TR.UTF-8",
        note="Dotless i is a letter before i, and I lowercases to it.",
    ),
    Collation(
        key_name="ai_ci",
        models="MySQL 8 default utf8mb4_0900_ai_ci, PostgreSQL nondeterministic ICU colStrength=primary",
        strength=1,
        deterministic=False,
        note="Primary level only. Accent- and case-insensitive, and equality "
        "follows the collation, so DISTINCT and UNIQUE change meaning.",
    ),
    Collation(
        key_name="glibc_en_US",
        models="glibc en_US.UTF-8 (PostgreSQL default on most Linux hosts)",
        variable="ignored",
        note="Punctuation and space are dropped before comparing, so two "
        "different strings can be exactly equal at every level.",
    ),
    Collation(
        key_name="icu_numeric",
        models="ICU kn-true (en-u-kn-true), natural sort",
        numeric=True,
        note="A digit run is one number, not a sequence of characters.",
    ),
)

COLLATION_BY_NAME: Dict[str, Collation] = {c.key_name: c for c in COLLATIONS}


def locale_lower(ch: str, mode: str) -> str:
    """Lowercase one character the way `mode` does it."""
    if mode == "tr":
        if ch == "I":
            return "ı"
        if ch == "İ":  # I with dot above
            return "i"
    return ch.lower()


def _is_variable(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("Z") or cat.startswith("P")


def sort_key(s: str, coll: Collation) -> Tuple:
    """The comparison key `ORDER BY s COLLATE coll` actually compares."""
    if coll.kind == "bytes":
        return tuple(s.encode("utf-8"))
    if coll.kind == "utf16":
        raw = s.encode("utf-16-be")
        return tuple((raw[i] << 8) | raw[i + 1] for i in range(0, len(raw), 2))

    text = unicodedata.normalize("NFC", s)
    # Tailoring runs on the composed form, before decomposition, or the
    # decomposition would destroy the very letter being tailored.
    buf: List[Tuple[str, int]] = []  # (char, case weight)
    for ch in text:
        low = locale_lower(ch, coll.lower)
        case_w = 1 if low == ch else 2
        mapped = coll._pre.get(low, low)
        for m in mapped:
            buf.append((m, case_w))

    primary: List[Tuple[int, int]] = []
    secondary: List[int] = []
    tertiary: List[int] = []
    quaternary: List[int] = []

    i = 0
    while i < len(buf):
        ch, case_w = buf[i]
        # A combining mark attaches to the base letter that came before it.
        if unicodedata.combining(ch):
            if secondary:
                secondary[-1] = ACCENT.get(ch, 12)
            else:  # a mark with no base: treat as its own element
                primary.append((CLS_OTHER, ord(ch)))
                secondary.append(1)
                tertiary.append(1)
            i += 1
            continue

        decomposed = unicodedata.normalize("NFD", ch)
        base = decomposed[0]
        marks = [m for m in decomposed[1:] if unicodedata.combining(m)]

        if _is_variable(base):
            if coll.variable == "ignored":
                pass  # never compared at any level
            elif coll.variable == "shifted":
                quaternary.append(ord(base))
            else:  # non-ignorable: punctuation sorts before every letter
                primary.append((CLS_VARIABLE, ord(base)))
                secondary.append(1)
                tertiary.append(1)
            i += 1
            continue

        if base.isdigit():
            if coll.numeric:
                run = ""
                while i < len(buf) and buf[i][0].isdigit():
                    run += buf[i][0]
                    i += 1
                primary.append((CLS_DIGIT, int(run)))
                secondary.append(1)
                tertiary.append(1)
                continue
            primary.append((CLS_DIGIT, ord(base)))
            secondary.append(1)
            tertiary.append(1)
            i += 1
            continue

        acc = max([ACCENT.get(m, 12) for m in marks], default=1)
        idx = coll.alphabet.find(base)
        if idx >= 0:
            primary.append((CLS_LETTER, idx))
            secondary.append(acc)
            tertiary.append(case_w)
            i += 1
            continue

        # No tailoring for this character. Two fallbacks, in ICU's spirit:
        # a compatibility decomposition (fullwidth A is an A), then an
        # explicit fold for the letters Unicode never decomposes (dotless i
        # is an i, sharp s is ss) - each with a secondary difference so the
        # fold is not silent.
        fold = BASE_FOLD.get(base)
        if fold is None:
            kd = "".join(
                c for c in unicodedata.normalize("NFKD", base) if not unicodedata.combining(c)
            )
            if kd != base and all(c.lower() in coll.alphabet for c in kd if c.isalpha()):
                fold = "".join(c.lower() for c in kd)
        if fold is not None:
            for j, fch in enumerate(fold):
                primary.append((CLS_LETTER, coll.alphabet.find(fch)))
                secondary.append(FOLD_SECONDARY if j == 0 else 1)
                tertiary.append(case_w)
            i += 1
            continue

        cat = unicodedata.category(base)
        cls = CLS_SYMBOL if cat.startswith("S") else CLS_OTHER
        primary.append((cls, ord(base)))
        secondary.append(acc)
        tertiary.append(case_w)
        i += 1

    levels: List[Tuple] = [tuple(primary)]
    if coll.strength >= 2:
        levels.append(tuple(secondary))
    if coll.strength >= 3:
        levels.append(tuple(tertiary))
    if coll.variable == "shifted":
        levels.append(tuple(quaternary))
    return tuple(levels)


def order(rows: Sequence[Row], coll: Collation) -> List[Row]:
    """`SELECT ... ORDER BY name COLLATE coll` - a stable sort, as every
    engine's is once it has committed to a plan."""
    return sorted(rows, key=lambda r: sort_key(r.name, coll))


def positions(coll: Collation, rows: Sequence[Row] = CORPUS) -> Dict[int, int]:
    return {r.id: i for i, r in enumerate(order(rows, coll))}


# --------------------------------------------------------------------------
# 1. Ties: the rows a stable sort was never given an order for
# --------------------------------------------------------------------------


def tie_groups(coll: Collation, rows: Sequence[Row] = CORPUS) -> List[List[Row]]:
    """Sets of rows whose keys are equal, so their relative order is the
    plan's choice and nothing else."""
    groups: Dict[Tuple, List[Row]] = {}
    for r in rows:
        groups.setdefault(sort_key(r.name, coll), []).append(r)
    return [g for g in groups.values() if len(g) > 1]


def tied_rows(coll: Collation, rows: Sequence[Row] = CORPUS) -> int:
    return sum(len(g) for g in tie_groups(coll, rows))


def distinct_count(coll: Collation, rows: Sequence[Row] = CORPUS) -> int:
    """`SELECT COUNT(DISTINCT name)` - which is collation-dependent only when
    the collation is nondeterministic."""
    if coll.deterministic:
        return len({r.name for r in rows})
    return len({sort_key(r.name, coll) for r in rows})


def unique_violations(coll: Collation, rows: Sequence[Row] = CORPUS) -> List[Tuple[Row, Row]]:
    """Row pairs a `UNIQUE(name)` index would reject under this collation.

    A deterministic collation compares equal but *is* not equal: PostgreSQL
    still uses byte equality for `=`, so a tie is not a uniqueness violation.
    A nondeterministic collation makes the tie the truth.
    """
    if coll.deterministic:
        return []
    out = []
    for a, b in combinations(rows, 2):
        if a.name != b.name and sort_key(a.name, coll) == sort_key(b.name, coll):
            out.append((a, b))
    return out


def verdict(coll: Collation, rows: Sequence[Row] = CORPUS) -> Verdict:
    ties = tie_groups(coll, rows)
    if not ties:
        return Verdict.STABLE_TOTAL if coll.kind in ("bytes", "utf16") else Verdict.TOTAL
    return Verdict.TIED if coll.deterministic else Verdict.MERGING


def verdict_counts(rows: Sequence[Row] = CORPUS) -> Dict[Verdict, int]:
    out = {v: 0 for v in Verdict}
    for c in COLLATIONS:
        out[verdict(c, rows)] += 1
    return out


# --------------------------------------------------------------------------
# 2. Drift between two collations
# --------------------------------------------------------------------------


def flips(a: Collation, b: Collation, rows: Sequence[Row] = CORPUS) -> List[Tuple[Row, Row]]:
    """Row pairs that a and b order in opposite directions.

    A pair tied under either collation is not counted as a flip: it has no
    direction to disagree with. That is measured separately, by `tie_groups`.
    """
    out = []
    for x, y in combinations(rows, 2):
        ax, ay = sort_key(x.name, a), sort_key(y.name, a)
        bx, by = sort_key(x.name, b), sort_key(y.name, b)
        if ax == ay or bx == by:
            continue
        if (ax < ay) != (bx < by):
            out.append((x, y))
    return out


def drift_matrix(rows: Sequence[Row] = CORPUS) -> Dict[Tuple[str, str], int]:
    m: Dict[Tuple[str, str], int] = {}
    for a, b in combinations(COLLATIONS, 2):
        n = len(flips(a, b, rows))
        m[(a.key_name, b.key_name)] = n
        m[(b.key_name, a.key_name)] = n
    for c in COLLATIONS:
        m[(c.key_name, c.key_name)] = 0
    return m


def identical_pairs(rows: Sequence[Row] = CORPUS) -> List[Tuple[str, str]]:
    """Collation pairs that return the exact same row sequence here."""
    out = []
    for a, b in combinations(COLLATIONS, 2):
        if [r.id for r in order(rows, a)] == [r.id for r in order(rows, b)]:
            out.append((a.key_name, b.key_name))
    return out


def max_displacement(rows: Sequence[Row] = CORPUS) -> List[Tuple[Row, int, str, str]]:
    """For each row, the largest position gap between any two collations."""
    pos = {c.key_name: positions(c, rows) for c in COLLATIONS}
    out = []
    for r in rows:
        best = (0, "", "")
        for a, b in combinations(COLLATIONS, 2):
            gap = abs(pos[a.key_name][r.id] - pos[b.key_name][r.id])
            if gap > best[0]:
                best = (gap, a.key_name, b.key_name)
        out.append((r, best[0], best[1], best[2]))
    return sorted(out, key=lambda t: -t[1])


def first_row(coll: Collation, rows: Sequence[Row] = CORPUS) -> Row:
    """`ORDER BY name LIMIT 1` - which is also what MIN(name) returns."""
    return order(rows, coll)[0]


def top1_answers(rows: Sequence[Row] = CORPUS) -> Dict[str, str]:
    return {c.key_name: first_row(c, rows).name for c in COLLATIONS}


# --------------------------------------------------------------------------
# 3. Pagination over a tied sort
# --------------------------------------------------------------------------


def plan_orders(rows: Sequence[Row] = CORPUS) -> List[List[Row]]:
    """Three physical row orders one table can hand the sort.

    Nothing exotic: insertion order, the reverse (a backward index scan or a
    table rewritten by VACUUM FULL), and clustered by name length (a different
    index picked by the planner). A stable sort preserves whichever it is
    given, so inside a tie group the physical order *is* the result order.
    """
    base = list(rows)
    return [base, list(reversed(base)), sorted(base, key=lambda r: (len(r.name), r.id))]


@dataclass(frozen=True)
class PageAudit:
    collation: str
    page_size: int
    lost: Tuple[int, ...]
    duplicated: Tuple[int, ...]
    stalled: bool = False

    @property
    def clean(self) -> bool:
        return not self.lost and not self.duplicated and not self.stalled


def offset_pagination(
    coll: Collation, page_size: int, rows: Sequence[Row] = CORPUS, tiebreak: bool = False
) -> PageAudit:
    """`ORDER BY name LIMIT n OFFSET k`, one query per page.

    Each page is a separate execution, so each may get a different physical
    order. Only rows inside a tie group can move - which is exactly why this
    is invisible in testing on a table with no duplicate names.
    """
    plans = plan_orders(rows)
    seen: List[int] = []
    for page_no in range(0, (len(rows) + page_size - 1) // page_size):
        physical = plans[page_no % len(plans)]
        if tiebreak:
            ordered = sorted(physical, key=lambda r: (sort_key(r.name, coll), r.id))
        else:
            ordered = sorted(physical, key=lambda r: sort_key(r.name, coll))
        seen.extend(r.id for r in ordered[page_no * page_size : (page_no + 1) * page_size])
    lost = tuple(sorted({r.id for r in rows} - set(seen)))
    dup = tuple(sorted(i for i in set(seen) if seen.count(i) > 1))
    return PageAudit(coll.key_name, page_size, lost, dup)


def keyset_pagination(
    coll: Collation, page_size: int, rows: Sequence[Row] = CORPUS, strict: bool = True
) -> PageAudit:
    """`WHERE name > $last ORDER BY name LIMIT n` - the recommended cure for
    OFFSET, which has its own failure when the key is not unique.

    `>` skips the rest of the tie group the page ended inside. `>=` re-reads
    all of it. With a non-unique sort key there is no third option.
    """
    plans = plan_orders(rows)
    seen: List[int] = []
    last_key: Optional[Tuple] = None
    prev_page: Optional[Tuple[int, ...]] = None
    stalled = False
    for page_no in range(len(rows) + 1):
        physical = plans[page_no % len(plans)]
        ordered = sorted(physical, key=lambda r: sort_key(r.name, coll))
        if last_key is not None:
            if strict:
                ordered = [r for r in ordered if sort_key(r.name, coll) > last_key]
            else:
                ordered = [r for r in ordered if sort_key(r.name, coll) >= last_key]
        page = ordered[:page_size]
        if not page:
            break
        ids = tuple(r.id for r in page)
        if prev_page is not None and set(ids) == set(prev_page):
            # The cursor cannot advance: `>=` keeps re-reading the same tie
            # group. In production this is a loop, not a wrong answer.
            stalled = True
            break
        seen.extend(ids)
        prev_page = ids
        last_key = sort_key(page[-1].name, coll)
    lost = tuple(sorted({r.id for r in rows} - set(seen)))
    dup = tuple(sorted(i for i in set(seen) if seen.count(i) > 1))
    return PageAudit(coll.key_name, page_size, lost, dup, stalled)


PAGE_SIZES: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 10)


def pagination_table(
    page_sizes: Sequence[int] = PAGE_SIZES, rows: Sequence[Row] = CORPUS
) -> List[Tuple[str, int, PageAudit, PageAudit, PageAudit, PageAudit]]:
    """Per collation and page size: OFFSET, OFFSET with a unique tiebreak,
    keyset with `>`, keyset with `>=`."""
    out = []
    for c in COLLATIONS:
        for n in page_sizes:
            out.append(
                (
                    c.key_name,
                    n,
                    offset_pagination(c, n, rows),
                    offset_pagination(c, n, rows, tiebreak=True),
                    keyset_pagination(c, n, rows, strict=True),
                    keyset_pagination(c, n, rows, strict=False),
                )
            )
    return out


def dirty_offset_runs(
    rows: Sequence[Row] = CORPUS,
) -> List[Tuple[str, int, Tuple[int, ...], Tuple[int, ...]]]:
    """The (collation, page size) combinations where OFFSET paging is wrong.

    Whether a tie group straddles a page boundary depends on the page size, so
    the same query is exact at one page size and lossy at another. This is the
    reason the bug reaches production: the test suite picked a page size.
    """
    out = []
    for name, n, off, _tb, _ks, _kl in pagination_table(rows=rows):
        if not off.clean:
            out.append((name, n, off.lost, off.duplicated))
    return out


def pagination_totals(rows: Sequence[Row] = CORPUS) -> Dict[str, int]:
    """Row-level damage summed over every collation and page size."""
    tot = {"offset_lost": 0, "offset_dup": 0, "tiebreak_lost": 0, "tiebreak_dup": 0,
           "keyset_strict_lost": 0, "keyset_loose_dup": 0, "keyset_loose_stalls": 0,
           "runs": 0, "clean_offset": 0}
    for _, _, off, tb, ks, kl in pagination_table(rows=rows):
        tot["runs"] += 1
        tot["clean_offset"] += 1 if off.clean else 0
        tot["offset_lost"] += len(off.lost)
        tot["offset_dup"] += len(off.duplicated)
        tot["tiebreak_lost"] += len(tb.lost)
        tot["tiebreak_dup"] += len(tb.duplicated)
        tot["keyset_strict_lost"] += len(ks.lost)
        tot["keyset_loose_dup"] += len(kl.duplicated)
        tot["keyset_loose_stalls"] += 1 if kl.stalled else 0
    return tot


# --------------------------------------------------------------------------
# 4. Predicates: a range is collation-dependent too
# --------------------------------------------------------------------------


def in_range(name: str, coll: Collation, lo: str = "A", hi: str = "N") -> bool:
    """`WHERE name >= lo AND name < hi` under this collation - the shape of
    every shard boundary, archive sweep and alphabetical tab."""
    k = sort_key(name, coll)
    return sort_key(lo, coll) <= k < sort_key(hi, coll)


def range_drift(
    rows: Sequence[Row] = CORPUS, lo: str = "A", hi: str = "N"
) -> List[Tuple[Row, Tuple[str, ...], Tuple[str, ...]]]:
    """Rows whose membership in [lo, hi) is not agreed on by all ten."""
    out = []
    for r in rows:
        yes = tuple(c.key_name for c in COLLATIONS if in_range(r.name, c, lo, hi))
        no = tuple(c.key_name for c in COLLATIONS if not in_range(r.name, c, lo, hi))
        if yes and no:
            out.append((r, yes, no))
    return out


def range_counts(
    rows: Sequence[Row] = CORPUS, lo: str = "A", hi: str = "N"
) -> Dict[str, int]:
    return {
        c.key_name: sum(1 for r in rows if in_range(r.name, c, lo, hi)) for c in COLLATIONS
    }


# --------------------------------------------------------------------------
# 5. Two rows that are the same string, and one index that disagrees
# --------------------------------------------------------------------------


def normalization_pairs(rows: Sequence[Row] = CORPUS) -> List[Tuple[Row, Row]]:
    """Row pairs that are canonically equivalent under Unicode - the same
    string - but not equal byte for byte."""
    out = []
    for a, b in combinations(rows, 2):
        if a.name == b.name:
            continue
        if unicodedata.normalize("NFC", a.name) == unicodedata.normalize("NFC", b.name):
            out.append((a, b))
    return out


def normalization_gap(rows: Sequence[Row] = CORPUS) -> List[Tuple[Row, Row, Dict[str, int]]]:
    """How far apart each canonically equivalent pair lands, per collation."""
    pos = {c.key_name: positions(c, rows) for c in COLLATIONS}
    out = []
    for a, b in normalization_pairs(rows):
        gaps = {
            c.key_name: abs(pos[c.key_name][a.id] - pos[c.key_name][b.id])
            for c in COLLATIONS
        }
        out.append((a, b, gaps))
    return out


def bmp_flip(rows: Sequence[Row] = CORPUS) -> List[Tuple[Row, Row]]:
    """Pairs ordered one way by UTF-8 bytes and the other by UTF-16 code
    units - the disagreement between PostgreSQL C and Java/JS/SQL Server."""
    c8 = COLLATION_BY_NAME["C"]
    c16 = COLLATION_BY_NAME["UTF16_BIN"]
    return flips(c8, c16, rows)


def turkish_case_breakage(rows: Sequence[Row] = CORPUS) -> List[Tuple[Row, str, str]]:
    """Rows where `LOWER(name)` is locale-dependent.

    This matters beyond ORDER BY: a functional index on `LOWER(name)` built
    under one LC_CTYPE and queried under another silently misses rows.
    """
    out = []
    for r in rows:
        root = "".join(locale_lower(ch, "root") for ch in r.name)
        tr = "".join(locale_lower(ch, "tr") for ch in r.name)
        if root != tr:
            out.append((r, root, tr))
    return out


# --------------------------------------------------------------------------
# 6. Findings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    severity: str  # "blocking" | "silent" | "advisory"
    title: str
    detail: str


def findings(rows: Sequence[Row] = CORPUS) -> List[Finding]:
    out: List[Finding] = []
    tot = pagination_totals(rows)
    tied = [c for c in COLLATIONS if verdict(c, rows) in (Verdict.TIED, Verdict.MERGING)]
    merging = [c for c in COLLATIONS if verdict(c, rows) is Verdict.MERGING]

    if tot["offset_lost"]:
        out.append(
            Finding(
                "silent",
                f"OFFSET pagination never returns {tot['offset_lost']} rows",
                "Summed over every collation and page size, OFFSET paging drops "
                f"{tot['offset_lost']} rows and repeats {tot['offset_dup']}. No "
                "error is raised: each page is individually correct.",
            )
        )
    if tot["keyset_strict_lost"]:
        out.append(
            Finding(
                "silent",
                f"Keyset paging with `>` drops {tot['keyset_strict_lost']} rows",
                "`WHERE name > $last` skips the remainder of the tie group the "
                "previous page ended inside. Switching to `>=` trades that for "
                f"{tot['keyset_loose_dup']} repeated rows.",
            )
        )
    if tot["tiebreak_lost"] == 0 and tot["tiebreak_dup"] == 0:
        out.append(
            Finding(
                "advisory",
                "A unique tiebreak fixes all of it",
                "`ORDER BY name, id` makes every one of the ten collations a "
                "total order over this corpus: 0 lost, 0 repeated, at every "
                "page size tested.",
            )
        )
    for c in merging:
        v = unique_violations(c, rows)
        out.append(
            Finding(
                "blocking",
                f"{c.key_name}: {len(v)} row pairs collide as one value",
                f"{c.key_name} is nondeterministic, so its ties are equality. "
                f"COUNT(DISTINCT name) reads {distinct_count(c, rows)} instead of "
                f"{len({r.name for r in rows})}, and a UNIQUE(name) index rejects "
                f"{len(v)} pairs that are different strings.",
            )
        )
    for c in tied:
        if c in merging:
            continue
        out.append(
            Finding(
                "silent",
                f"{c.key_name}: {len(tie_groups(c, rows))} tie groups, {tied_rows(c, rows)} rows",
                "Deterministic, so `=` and UNIQUE are unaffected - the rows stay "
                "distinct. Only their order is undefined, which is why this "
                "surfaces as a moving report and not as an error.",
            )
        )
    drift = drift_matrix(rows)
    worst = max(((k, v) for k, v in drift.items() if k[0] < k[1]), key=lambda kv: kv[1])
    out.append(
        Finding(
            "blocking",
            f"{worst[0][0]} vs {worst[0][1]}: {worst[1]} of "
            f"{len(list(combinations(rows, 2)))} row pairs come back in the "
            "opposite order",
            "Both are collations a production server can be configured with. "
            "The rows are identical; only the ORDER BY resolution differs.",
        )
    )
    nf = normalization_pairs(rows)
    if nf:
        a, b = nf[0]
        c_coll = COLLATION_BY_NAME["C"]
        pos_c = positions(c_coll, rows)
        lo, hi = sorted((pos_c[a.id], pos_c[b.id]))
        between = [r for r in rows if lo < pos_c[r.id] < hi]
        adjacent = [
            r
            for r in rows
            if r.id not in (a.id, b.id) and min(abs(pos_c[r.id] - lo), abs(pos_c[r.id] - hi)) == 1
        ]
        where = (
            f"and byte order seats {len(between)} other row(s) between them: "
            + ", ".join(repr(r.name) for r in between)
            if between
            else "and byte order seats them next to "
            + ", ".join(repr(r.name) for r in adjacent)
        )
        out.append(
            Finding(
                "silent",
                f"Rows {a.id} and {b.id} are the same string, {where}",
                f"{a.display} and {b.display} are canonically equivalent - any "
                "Unicode-aware comparison calls them one value, and every UCA "
                "collation here does. Byte order calls them two different strings "
                f"{abs(hi - lo)} position(s) apart, so a UNIQUE(name) index under "
                "C accepts both spellings and every later lookup finds one of "
                "them. Normalisation is a write-path decision; no collation can "
                "undo skipping it.",
            )
        )
    bf = bmp_flip(rows)
    if bf:
        out.append(
            Finding(
                "advisory",
                f"C and UTF16_BIN disagree on {len(bf)} pair(s)",
                "Both are 'binary' orders. They agree across the whole BMP and "
                "part company above U+FFFF, where UTF-16 leads with a surrogate "
                "at U+D800. A Java service and a PostgreSQL C index do not agree "
                "on which of these two rows is first.",
            )
        )
    tb = turkish_case_breakage(rows)
    if tb:
        out.append(
            Finding(
                "blocking",
                f"LOWER() changes {len(tb)} row(s) when LC_CTYPE is tr_TR",
                "; ".join(f"{r.name!r} -> {root!r} or {tr!r}" for r, root, tr in tb)
                + ". A functional index on LOWER(name) is only valid for the "
                "locale it was built under.",
            )
        )
    rd = range_drift(rows)
    if rd:
        counts = range_counts(rows)
        out.append(
            Finding(
                "silent",
                f"WHERE name >= 'A' AND name < 'N' returns "
                f"{min(counts.values())} to {max(counts.values())} rows depending "
                "on the collation",
                f"{len(rd)} rows are inside the range under some collations and "
                "outside under others. Any A-M / N-Z split - a shard key, an "
                "archive sweep, an alphabetical tab - moves with the collation.",
            )
        )
    tot_pag = pagination_totals(rows)
    if tot_pag["keyset_loose_stalls"]:
        out.append(
            Finding(
                "blocking",
                f"Keyset paging with `>=` never terminates: "
                f"{tot_pag['keyset_loose_stalls']} of {tot_pag['runs']} runs stall",
                "The last row of every page satisfies `name >= $last`, so it comes "
                "back as the first row of the next page forever. This one is not "
                "about ties at all - the two byte collations, which have none, "
                "stall too. A cursor needs a comparison it can strictly advance.",
            )
        )
    refused = libc_refused_names(rows)
    if refused:
        names = sorted({n for _loc, n in refused})
        out.append(
            Finding(
                "blocking",
                f"This host's libc has no sort key at all for {len(names)} "
                f"name(s): {', '.join(repr(n) for n in names)}",
                f"`strxfrm` fails on it in {len({loc for loc, _ in refused})} of "
                "the installed UTF-8 locales, and succeeds under C. Anything "
                "built on strcoll - GNU sort, a C extension, a locale-aware "
                "comparator - cannot place that row, so its position is whatever "
                "the error path leaves behind.",
            )
        )
    out.append(
        Finding(
            "advisory",
            "A collation is versioned data, not a setting",
            "glibc 2.28 (RHEL 8, Ubuntu 18.10) changed en_US.UTF-8 ordering, "
            "which silently invalidated existing PostgreSQL text indexes and "
            "required a REINDEX; PostgreSQL 13+ records collation versions and "
            "warns. The same class of change lands with every ICU upgrade.",
        )
    )
    return out


def finding_counts(rows: Sequence[Row] = CORPUS) -> Dict[str, int]:
    out = {"blocking": 0, "silent": 0, "advisory": 0}
    for f in findings(rows):
        out[f.severity] += 1
    return out


# --------------------------------------------------------------------------
# 7. Checking the model against the host's own libc
# --------------------------------------------------------------------------


def libc_probe(rows: Sequence[Row] = CORPUS) -> List[Tuple[str, str, int, int]]:
    """What this host's own libc does with the corpus, per locale.

    Returns (locale, status, names refused, distinct orders seen so far). A
    refused name is not a curiosity: `strxfrm` returning an error means libc
    has no sort key for that string, so nothing built on `strcoll` can place
    it. This is the check that keeps the model honest about the machine it is
    running on.
    """
    saved = locale.setlocale(locale.LC_COLLATE)
    out: List[Tuple[str, str, int, int]] = []
    seqs: List[Tuple[str, ...]] = []
    try:
        for c in COLLATIONS:
            if not c.libc:
                continue
            try:
                locale.setlocale(locale.LC_COLLATE, c.libc)
            except locale.Error:
                out.append((c.libc, "locale not installed", 0, len(set(seqs))))
                continue
            usable, refused = [], 0
            for r in rows:
                try:
                    locale.strxfrm(r.name)
                    usable.append(r.name)
                except (OSError, ValueError):
                    refused += 1
            seqs.append(tuple(sorted(usable, key=locale.strxfrm)))
            out.append((c.libc, "compared", refused, len(set(seqs))))
    finally:
        locale.setlocale(locale.LC_COLLATE, saved)
    return out


def libc_refused_names(rows: Sequence[Row] = CORPUS) -> List[Tuple[str, str]]:
    """(locale, name) pairs this host's libc has no sort key for at all."""
    saved = locale.setlocale(locale.LC_COLLATE)
    out = []
    try:
        for c in COLLATIONS:
            if not c.libc:
                continue
            try:
                locale.setlocale(locale.LC_COLLATE, c.libc)
            except locale.Error:
                continue
            for r in rows:
                try:
                    locale.strxfrm(r.name)
                except (OSError, ValueError):
                    out.append((c.libc, r.name))
    finally:
        locale.setlocale(locale.LC_COLLATE, saved)
    return out


def libc_distinct_orders(rows: Sequence[Row] = CORPUS) -> int:
    """How many different orders this host's installed locales produce."""
    probe = libc_probe(rows)
    return max((p[3] for p in probe), default=0)


def libc_agreement(rows: Sequence[Row] = CORPUS) -> List[Tuple[str, str, int, int, str]]:
    """Compare each modelled collation that names a libc locale against the
    real `strcoll` order on this host.

    Returns (collation, locale, agreeing pairs, total pairs, status). Absent
    locales are reported, not skipped - a missing locale is itself the reason
    two hosts sort differently.
    """
    out = []
    saved = locale.setlocale(locale.LC_COLLATE)
    try:
        for c in COLLATIONS:
            if not c.libc:
                continue
            try:
                locale.setlocale(locale.LC_COLLATE, c.libc)
            except locale.Error:
                out.append((c.key_name, c.libc, 0, 0, "locale not installed"))
                continue
            agree = 0
            total = 0
            skipped = 0
            for x, y in combinations(rows, 2):
                kx, ky = sort_key(x.name, c), sort_key(y.name, c)
                if kx == ky:
                    continue
                try:
                    lx, ly = locale.strxfrm(x.name), locale.strxfrm(y.name)
                except (OSError, ValueError):
                    # strxfrm refuses some strings on some platforms - macOS
                    # rejects anything outside the BMP. A collation that
                    # cannot transform a string cannot order it either.
                    skipped += 1
                    continue
                if lx == ly:
                    continue
                total += 1
                if (kx < ky) == (lx < ly):
                    agree += 1
            status = "compared" if not skipped else f"compared, {skipped} pairs strxfrm refused"
            out.append((c.key_name, c.libc, agree, total, status))
    finally:
        locale.setlocale(locale.LC_COLLATE, saved)
    return out
