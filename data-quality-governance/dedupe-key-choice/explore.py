"""Six match keys for "is this the same customer?", against one hand-labelled customer table.

NOT a deduper (see `dedup-pipeline`, Day 67, which merges clusters into golden records). This compares the KEY
CHOICES a deduper is built on - exact, normalised, phonetic (Soundex, Metaphone), fuzzy (Jaro-Winkler, token-set
ratio) - over one corpus, and reports which pairs each key merges that the others keep apart.

Each row is one CRM record: the customer name as typed and the true customer it belongs to (hand-labelled from who
the person is - the same person typed three ways, or two different people who share a name). A key is judged on
the same two errors as Days 202-204:

- false split: same person, two different keys  -> one customer becomes two
- false merge: different people, one key        -> two customers become one

Exact and normalised keys are functions of one row. Phonetic keys are too. The fuzzy keys are PAIRWISE: a pair
matches when its similarity clears a threshold, and the table is then clustered by connected components, which is
what every "fuzzy dedupe" script does whether it says so or not. Every name is fictional. There is no randomness.
"""

from __future__ import annotations

import re
import unicodedata
from itertools import combinations
from typing import Callable, Dict, List, Set, Tuple

import jellyfish as jf
from rapidfuzz import fuzz

# (name as typed, true customer id, what the variation is)
CORPUS: List[Tuple[str, str, str]] = [
    # one person, case / whitespace / punctuation
    ("Mary O'Brien", "P01", "as filed"),
    ("mary o'brien", "P01", "lower case"),
    ("MARY  O'BRIEN", "P01", "caps, double space"),
    ("Mary OBrien", "P01", "apostrophe dropped"),
    # one person, accents folded by a web form
    ("José Núñez", "P02", "as filed"),
    ("Jose Nunez", "P02", "accents stripped"),
    ("JOSE NUNEZ", "P02", "accents stripped, caps"),
    # one person, order swapped by an import
    ("Jonathan Smith", "P03", "as filed"),
    ("Smith, Jonathan", "P03", "family, given"),
    ("Jonathon Smith", "P03", "one-letter typo"),
    ("Jon Smith", "P03", "short form"),
    # a DIFFERENT person who shares the surname and a similar given name
    ("Johnathan Smith", "P04", "as filed"),
    ("John Smith", "P05", "as filed"),
    ("John Smith", "P06", "as filed (a second John Smith)"),
    # one person, nickname: no string key gets this
    ("William Jones", "P07", "as filed"),
    ("Bill Jones", "P07", "nickname"),
    ("W. Jones", "P07", "initial"),
    # phonetic twins who are the same person (a form that heard the name)
    ("Catherine Lee", "P08", "as filed"),
    ("Kathryn Lee", "P08", "spelled as heard"),
    # phonetic twins who are different people
    ("Catherine Leigh", "P09", "as filed"),
    ("Stephen Baker", "P10", "as filed"),
    ("Steven Baker", "P10", "spelled as heard"),
    ("Stefan Becker", "P11", "as filed (sounds like P10 to Soundex)"),
    # hyphen vs space vs one surname kept
    ("Anna Müller-Schmidt", "P12", "as filed"),
    ("Anna Muller Schmidt", "P12", "hyphen to space, umlaut folded"),
    ("Anna Mueller-Schmidt", "P12", "umlaut transliterated"),
    ("Anna Schmidt", "P12", "one surname kept"),
    ("Anna Müller", "P13", "as filed (a different Anna)"),
    # same person, a middle name appears and disappears
    ("Priya Raman", "P14", "as filed"),
    ("Priya S. Raman", "P14", "middle initial"),
    ("Priya Subramanian Raman", "P14", "middle name"),
    ("Priya Ramen", "P14", "one-letter typo"),
    # one surname, two people with short given names
    ("Li Wei", "P15", "as filed"),
    ("Wei Li", "P15", "order swapped by an import"),
    ("Li Wen", "P16", "as filed (a different person)"),
    # transliteration variants of one person
    ("Mohammed Al-Sayed", "P17", "as filed"),
    ("Mohamed Alsayed", "P17", "transliteration"),
    ("Muhammad Al Sayed", "P17", "transliteration"),
    ("Mohammed Al-Said", "P18", "as filed (a different person)"),
    # trailing whitespace and a title
    ("Dr. Elena Petrova", "P19", "title"),
    ("Elena Petrova ", "P19", "trailing space"),
    ("Elena Petrov", "P20", "as filed (a different person)"),
    # a short name that fuzzy keys over-reach on
    ("Sam Ng", "P21", "as filed"),
    ("Sam Ong", "P22", "as filed (a different person)"),
    ("Sam Ngo", "P23", "as filed (a different person)"),
]

TITLES = {"dr", "mr", "mrs", "ms", "prof", "sir"}


def _fold(s: str) -> str:
    """Case-fold, strip accents, drop punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s.casefold())
    return " ".join(s.split())


def _tokens(s: str) -> List[str]:
    return [t for t in _fold(s).split() if t not in TITLES]


def _exact(s: str) -> str:
    """The string as typed - what a SQL GROUP BY name does."""
    return s


def _normalised(s: str) -> str:
    """Folded, titles removed, tokens SORTED so 'Smith, Jonathan' and 'Jonathan Smith' agree."""
    return " ".join(sorted(_tokens(s)))


def _soundex(s: str) -> str:
    """Soundex of every token, sorted."""
    return " ".join(sorted(jf.soundex(t) for t in _tokens(s)))


def _metaphone(s: str) -> str:
    """Metaphone of every token, sorted."""
    return " ".join(sorted(jf.metaphone(t) for t in _tokens(s)))


ROW_KEYS: Dict[str, Callable[[str], str]] = {
    "exact": _exact,
    "normalised": _normalised,
    "soundex": _soundex,
    "metaphone": _metaphone,
}

JW_T = 0.90
TSR_T = 85


def _jw(a: str, b: str) -> float:
    return jf.jaro_winkler_similarity(_normalised(a), _normalised(b))


def _tsr(a: str, b: str) -> float:
    return fuzz.token_set_ratio(_fold(a), _fold(b))


PAIR_KEYS: Dict[str, Tuple[Callable[[str, str], float], float]] = {
    f"jaro-winkler >= {JW_T}": (_jw, JW_T),
    f"token-set ratio >= {TSR_T}": (_tsr, TSR_T),
}
NAMES: List[str] = list(ROW_KEYS) + list(PAIR_KEYS)


def _components(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    """Connected components by union-find: the cluster id of every row."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        parent[find(a)] = find(b)
    return [find(i) for i in range(n)]


def clusters(name: str) -> List[str]:
    """One cluster label per row under this key."""
    if name in ROW_KEYS:
        return [ROW_KEYS[name](s) for s, _, _ in CORPUS]
    sim, t = PAIR_KEYS[name]
    edges = [(i, j) for i, j in combinations(range(len(CORPUS)), 2) if sim(CORPUS[i][0], CORPUS[j][0]) >= t]
    return [str(c) for c in _components(len(CORPUS), edges)]


def score(name: str) -> Dict:
    lab = clusters(name)
    splits, merges = [], []
    for i, j in combinations(range(len(CORPUS)), 2):
        same_truth = CORPUS[i][1] == CORPUS[j][1]
        same_key = lab[i] == lab[j]
        if same_truth and not same_key:
            splits.append((CORPUS[i][0], CORPUS[j][0]))
        if same_key and not same_truth:
            merges.append((CORPUS[i][0], CORPUS[j][0]))
    chained = 0
    if name in PAIR_KEYS:
        sim, t = PAIR_KEYS[name]
        chained = sum(
            1
            for i, j in combinations(range(len(CORPUS)), 2)
            if lab[i] == lab[j] and sim(CORPUS[i][0], CORPUS[j][0]) < t
        )
    return {
        "false_splits": splits,
        "false_merges": merges,
        "customers": len(set(lab)),
        "chained_pairs": chained,  # in one cluster although the pair itself is below the threshold
        "labels": lab,
    }


def merged_pairs(name: str) -> Set[Tuple[int, int]]:
    lab = clusters(name)
    return {(i, j) for i, j in combinations(range(len(CORPUS)), 2) if lab[i] == lab[j]}


def disagreement() -> Dict[str, Dict[str, int]]:
    """[A][B] = pairs A merges that B keeps apart."""
    m = {n: merged_pairs(n) for n in NAMES}
    return {a: {b: len(m[a] - m[b]) for b in NAMES} for a in NAMES}


def frontier(name: str) -> List[Dict]:
    """For a pairwise key: at every threshold its similarities take, the splits and merges of a direct
    (non-transitive) match. Shows whether any threshold reaches zero of both."""
    sim, _ = PAIR_KEYS[name]
    pairs = [(sim(CORPUS[i][0], CORPUS[j][0]), CORPUS[i][1] == CORPUS[j][1]) for i, j in combinations(range(len(CORPUS)), 2)]
    pos = sum(1 for _, same in pairs if same)
    out = []
    for t in sorted({round(s, 3) for s, _ in pairs}, reverse=True):
        matched = [same for s, same in pairs if s >= t]
        out.append(
            {
                "threshold": t,
                "false_splits": pos - sum(matched),
                "false_merges": sum(1 for same in matched if not same),
            }
        )
    return out


def ceiling() -> Dict:
    """What no function of the name string can settle.

    two_truths: identical strings that belong to different people (any key merges them).
    best_frontier: for each pairwise key, the smallest splits + merges any single threshold reaches, and whether
    a zero-error threshold exists at all."""
    by_s: Dict[str, Set[str]] = {}
    for s, t, _ in CORPUS:
        by_s.setdefault(s, set()).add(t)
    two = sorted(s for s, ts in by_s.items() if len(ts) > 1)
    best = {}
    for n in PAIR_KEYS:
        fr = frontier(n)
        b = min(fr, key=lambda d: (d["false_splits"] + d["false_merges"], -d["threshold"]))
        best[n] = {
            "threshold": b["threshold"],
            "false_splits": b["false_splits"],
            "false_merges": b["false_merges"],
            "zero_exists": any(d["false_splits"] == 0 and d["false_merges"] == 0 for d in fr),
        }
    true_pairs = sum(1 for i, j in combinations(range(len(CORPUS)), 2) if CORPUS[i][1] == CORPUS[j][1])
    return {"two_truths": two, "true_pairs": true_pairs, "best_frontier": best}


def direct(name: str) -> Dict[str, int]:
    """A pairwise key's errors if each pair were judged on its own (no transitive closure), at its threshold."""
    sim, t = PAIR_KEYS[name]
    pos = sum(1 for i, j in combinations(range(len(CORPUS)), 2) if CORPUS[i][1] == CORPUS[j][1])
    hit = [
        CORPUS[i][1] == CORPUS[j][1]
        for i, j in combinations(range(len(CORPUS)), 2)
        if sim(CORPUS[i][0], CORPUS[j][0]) >= t
    ]
    return {"false_splits": pos - sum(hit), "false_merges": sum(1 for h in hit if not h)}


def never_merged() -> List[Tuple[str, str]]:
    """True pairs that no key in this study joins: the floor every key here shares."""
    m = set.intersection(*[set(combinations(range(len(CORPUS)), 2)) - merged_pairs(n) for n in NAMES])
    return [(CORPUS[i][0], CORPUS[j][0]) for i, j in sorted(m) if CORPUS[i][1] == CORPUS[j][1]]


def study() -> Dict:
    per = {n: score(n) for n in NAMES}
    return {
        "n": len(CORPUS),
        "true_customers": len({t for _, t, _ in CORPUS}),
        "per": per,
        "disagreement": disagreement(),
        "ceiling": ceiling(),
        "frontier": {n: frontier(n) for n in PAIR_KEYS},
        "direct": {n: direct(n) for n in PAIR_KEYS},
        "never_merged": never_merged(),
    }
