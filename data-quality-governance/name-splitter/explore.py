"""Seven ways to split a full name into given + family, against a hand-labelled corpus of naming conventions.

Truth is two fields per row, labelled from the person's own convention (W3C, "Personal names around the world"):

- given  : the name you greet them by ("Dear ___")
- family : the inherited surname as it is filed, particles included; empty when the convention has none
           (mononyms, Malay bin/binti patronymics)

Every row is fictional. The corpus deliberately contains the SAME string twice with DIFFERENT truths ("Chen Lin"
written family-first for one person and given-first for another; "Thomas Edward Barlow" with a middle name vs
"Rupert Ashby Lowe" with an unhyphenated double-barrelled surname), because those rows set the ceiling: no
function of the string alone can be right on both. No randomness anywhere: every number is exact.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Tuple

from nameparser import HumanName

# (full name as typed into one field, group, true given, true family)
CORPUS: List[Tuple[str, str, str, str]] = [
    # plain given + family
    ("Grace Holloway", "plain", "Grace", "Holloway"),
    ("Daniel Okafor", "plain", "Daniel", "Okafor"),
    ("Priya Raman", "plain", "Priya", "Raman"),
    ("Lucas Moreau", "plain", "Lucas", "Moreau"),
    ("Hannah Becker", "plain", "Hannah", "Becker"),
    ("Samuel Adeyemi", "plain", "Samuel", "Adeyemi"),
    # middle names
    ("Thomas Edward Barlow", "middle name", "Thomas", "Barlow"),
    ("Claire Louise Fenwick", "middle name", "Claire", "Fenwick"),
    ("Michael James Ortega", "middle name", "Michael", "Ortega"),
    ("Aisha Noor Rahman", "middle name", "Aisha", "Rahman"),
    # surname particles (Dutch, German, French, Italian, Arabic)
    ("Pieter van den Broek", "particle", "Pieter", "van den Broek"),
    ("Anna von Halden", "particle", "Anna", "von Halden"),
    ("Luc de la Fontaine", "particle", "Luc", "de la Fontaine"),
    ("Marco di Stefano", "particle", "Marco", "di Stefano"),
    ("Sophie De Smet", "particle", "Sophie", "De Smet"),
    ("Omar al Rashid", "particle", "Omar", "al Rashid"),
    ("Vincent van der Berg", "particle", "Vincent", "van der Berg"),
    # Iberian: two surnames (paternal + maternal), sometimes two given names
    ("Ana María Pérez López", "two surnames", "Ana María", "Pérez López"),
    ("Carlos Ruiz Delgado", "two surnames", "Carlos", "Ruiz Delgado"),
    ("Lucía Fernández Ortiz", "two surnames", "Lucía", "Fernández Ortiz"),
    ("João Silva Santos", "two surnames", "João", "Silva Santos"),
    ("José Luis Navarro Gil", "two surnames", "José Luis", "Navarro Gil"),
    # family name first, as written natively (Chinese, Korean, Japanese, Hungarian, Vietnamese)
    ("Wang Fang", "family first", "Fang", "Wang"),
    ("Kim Min-jun", "family first", "Min-jun", "Kim"),
    ("Tanaka Yuki", "family first", "Yuki", "Tanaka"),
    ("Nagy Eszter", "family first", "Eszter", "Nagy"),
    ("Nguyen Van An", "family first", "An", "Nguyen"),
    ("Chen Lin", "family first", "Lin", "Chen"),
    # East Asian names already turned round into Western order
    ("Lin Chen", "Western-ordered East Asian", "Lin", "Chen"),
    ("Chen Lin", "Western-ordered East Asian", "Chen", "Lin"),
    ("Min-jun Park", "Western-ordered East Asian", "Min-jun", "Park"),
    # titles and suffixes
    ("Dr. Robert Hale Jr.", "title or suffix", "Robert", "Hale"),
    ("Margaret Osei PhD", "title or suffix", "Margaret", "Osei"),
    ("James Whitmore III", "title or suffix", "James", "Whitmore"),
    ("Mrs. Ellen Park", "title or suffix", "Ellen", "Park"),
    ("Prof. Henrik Dahl", "title or suffix", "Henrik", "Dahl"),
    # one name only
    ("Wiratmo", "mononym", "Wiratmo", ""),
    ("Suwarni", "mononym", "Suwarni", ""),
    ("Teodora", "mononym", "Teodora", ""),
    # Malay patronymics: "bin/binti <father>" is not a family name
    ("Abdullah bin Ahmad", "patronymic", "Abdullah", ""),
    ("Siti binti Hassan", "patronymic", "Siti", ""),
    ("Farid bin Yusof", "patronymic", "Farid", ""),
    # hyphenated surnames
    ("Emily Clarke-Hughes", "hyphenated", "Emily", "Clarke-Hughes"),
    ("Hamid Al-Farsi", "hyphenated", "Hamid", "Al-Farsi"),
    ("Jean-Luc Marchand", "hyphenated", "Jean-Luc", "Marchand"),
    # English double-barrelled surnames WITHOUT a hyphen (same shape as a middle name)
    ("Rupert Ashby Lowe", "double-barrelled", "Rupert", "Ashby Lowe"),
    ("Harriet Vane Tempest", "double-barrelled", "Harriet", "Vane Tempest"),
    # exports that write "Family, Given"
    ("van den Broek, Pieter", "comma inverted", "Pieter", "van den Broek"),
    ("Hale, Robert", "comma inverted", "Robert", "Hale"),
    ("Pérez López, Ana María", "comma inverted", "Ana María", "Pérez López"),
    ("Okafor, Daniel", "comma inverted", "Daniel", "Okafor"),
]

# Declared share of a customer table in each group. An ASSUMPTION for an international B2C list, not data:
# change it to your own mix (the app lets you).
SHARE: Dict[str, float] = {
    "plain": 0.51,
    "middle name": 0.12,
    "particle": 0.04,
    "two surnames": 0.06,
    "family first": 0.05,
    "Western-ordered East Asian": 0.06,
    "title or suffix": 0.03,
    "mononym": 0.005,
    "patronymic": 0.01,
    "hyphenated": 0.04,
    "double-barrelled": 0.005,
    "comma inverted": 0.07,
}

PARTICLES = {"van", "von", "de", "der", "den", "la", "le", "du", "di", "da", "del", "al", "el", "ter", "ten"}
TITLES = {"dr.", "dr", "mr.", "mr", "mrs.", "mrs", "ms.", "ms", "prof.", "prof"}
SUFFIXES = {"jr.", "jr", "sr.", "sr", "ii", "iii", "iv", "phd", "md"}
PATRONYM = {"bin", "binti", "ibn"}
CJK_SURNAMES = {"wang", "li", "zhang", "liu", "chen", "yang", "kim", "lee", "park", "nguyen", "tanaka", "lin"}

Split = Tuple[str, str]


def _first_space(s: str) -> Split:
    """given, family = s.split(" ", 1) - the one-liner in most import scripts."""
    p = s.split(" ", 1)
    return p[0], p[1] if len(p) > 1 else ""


def _last_space(s: str) -> Split:
    """given, family = s.rsplit(" ", 1) - 'the surname is the last word'."""
    p = s.rsplit(" ", 1)
    return (p[0], p[1]) if len(p) > 1 else (p[0], "")


def _first_last(s: str) -> Split:
    """Spreadsheet formula: LEFT to the first space, RIGHT from the last space; middle words dropped."""
    t = s.split()
    return t[0], t[-1] if len(t) > 1 else ""


def _strip_affixes(s: str) -> List[str]:
    return [w for w in s.replace(",", " ").split() if w.lower() not in TITLES | SUFFIXES]


def _particle_aware(s: str) -> Split:
    """Comma-aware, strips titles and suffixes, and pulls lowercase particles into the surname."""
    if "," in s:
        fam, giv = (x.strip() for x in s.split(",", 1))
        return giv, fam
    t = _strip_affixes(s)
    if len(t) == 1:
        return t[0], ""
    i = len(t) - 1
    while i > 1 and t[i - 1].lower() in PARTICLES:
        i -= 1
    return " ".join(t[:i]), " ".join(t[i:])


def _cjk_order(s: str) -> Split:
    """Particle-aware, plus: if the first word is a common East Asian surname, treat the name as family-first."""
    t = s.split()
    if "," not in s and len(t) in (2, 3) and t[0].lower() in CJK_SURNAMES:
        return t[-1] if len(t) == 3 else t[1], t[0]
    return _particle_aware(s)


def _nameparser(s: str) -> Split:
    h = HumanName(s)
    return h.first, h.last


SPLITTERS: Dict[str, Callable[[str], Split]] = {
    'split(" ", 1)': _first_space,
    'rsplit(" ", 1)': _last_space,
    "first + last word": _first_last,
    "particle-aware rules": _particle_aware,
    "+ East Asian surname list": _cjk_order,
    "nameparser 2.5": _nameparser,
}


def shape(s: str) -> str:
    """What a rule over word POSITIONS can see: comma, word count, and each word's class (not the word itself)."""
    cls = []
    for w in s.replace(",", " ").split():
        lw = w.lower()
        cls.append(
            "t"
            if lw in TITLES
            else "s"
            if lw in SUFFIXES
            else "p"
            if lw in PARTICLES
            else "b"
            if lw in PATRONYM
            else "W"
        )
    return ("," if "," in s else "") + "".join(cls)


def roles(s: str, given: str, family: str) -> str:
    """Which word of `s` is given (G), family (F) or neither (-): the split expressed as positions."""
    g, f = given.split(), family.split()
    out = []
    for w in s.replace(",", " ").split():
        if w in g:
            out.append("G")
            g.remove(w)
        elif w in f:
            out.append("F")
            f.remove(w)
        else:
            out.append("-")
    return "".join(out)


def ceiling() -> Dict:
    """Two upper bounds on any splitter that sees only the string.

    by string: the same string with two truths ("Chen Lin") - nothing can get both.
    by shape : rows a position rule cannot tell apart ("Thomas Edward Barlow" vs "Rupert Ashby Lowe" vs
               "Carlos Ruiz Delgado" are all W W W) - the best any rule over word classes can do, without a
               dictionary of which words are surnames.
    """
    by_s: Dict[str, Counter] = {}
    by_shape: Dict[str, Counter] = {}
    for s, _, g, f in CORPUS:
        by_s.setdefault(s, Counter())[(g, f)] += 1
        by_shape.setdefault(shape(s), Counter())[roles(s, g, f)] += 1
    return {
        "by_string": sum(c.most_common(1)[0][1] for c in by_s.values()),
        "ambiguous_strings": sorted(s for s, c in by_s.items() if len(c) > 1),
        "by_shape": sum(c.most_common(1)[0][1] for c in by_shape.values()),
        "WWW": dict(by_shape.get("WWW", Counter())),
    }


def score(name: str, share: Dict[str, float] = SHARE) -> Dict:
    """Per splitter: rows with given wrong, family wrong, either wrong; and the share of customers greeted wrong."""
    f = SPLITTERS[name]
    rows = [(s, grp, g, fam, *f(s)) for s, grp, g, fam in CORPUS]
    n_grp = Counter(grp for _, grp, _, _ in CORPUS)
    given_wrong = [(s, grp, pg) for s, grp, g, _, pg, _ in rows if pg != g]
    family_wrong = [(s, grp, pf) for s, grp, _, fam, _, pf in rows if pf != fam]
    both_right = sum(1 for s, _, g, fam, pg, pf in rows if pg == g and pf == fam)
    greet_lost = sum(share[grp] / n_grp[grp] for _, grp, _ in given_wrong)
    return {
        "given_wrong": given_wrong,
        "family_wrong": family_wrong,
        "both_right": both_right,
        "greeted_wrong": greet_lost / sum(share.values()),
        "groups_failed": sorted({grp for _, grp, _ in given_wrong} | {grp for _, grp, _ in family_wrong}),
        "greeted_as_affix": [
            (s, pg) for s, _, pg in given_wrong if pg.split()[-1].lower() in PARTICLES | TITLES if pg
        ],
    }


def study(share: Dict[str, float] = SHARE) -> Dict:
    per = {n: score(n, share) for n in SPLITTERS}
    groups = list(dict.fromkeys(grp for _, grp, _, _ in CORPUS))
    by_group = {
        n: {
            grp: sum(1 for s, gg, g, fam in CORPUS if gg == grp and SPLITTERS[n](s) == (g, fam))
            / sum(1 for _, gg, _, _ in CORPUS if gg == grp)
            for grp in groups
        }
        for n in SPLITTERS
    }
    wrong_by_all = [s for s, _, g, fam in CORPUS if all(SPLITTERS[n](s) != (g, fam) for n in SPLITTERS)]
    return {
        "n": len(CORPUS),
        "groups": groups,
        "per": per,
        "by_group": by_group,
        "ceiling": ceiling(),
        "wrong_by_all": wrong_by_all,
        "best": max(per, key=lambda n: (per[n]["both_right"], -per[n]["greeted_wrong"])),
    }
