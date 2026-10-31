"""How few quasi-identifiers make a row unique, counted on one fictional population rather than assumed.

NOT a redactor (see `pii-redactor`, Day 124, and `pii-detector`, Day 5, which find and mask direct identifiers).
This is the step those tools leave behind: the export has no names, and the columns that are left - postcode,
date of birth, sex - still point at one person. The study builds a fictional town of 60,000 residents with a
skewed postcode geography and a realistic age pyramid, takes a 5% clinic export of it, and COUNTS:

- how many export rows are unique on each subset of the three quasi-identifiers, in the export and in the town
- what each step of generalisation (date -> month -> year -> decade; postcode -> 4 digits -> sector) buys, as the
  smallest class size k, the rows below k = 5, and the rows that would have to be suppressed to reach it
- how far "unique in the export" overstates "unique in the town" at every sampling fraction
- how many k-anonymous classes still disclose the diagnosis because everyone in them shares it (homogeneity)
- what a linkage attack with a public register gets right, gets WRONG, and cannot settle, when the register is
  complete and when it is missing 20% of the town

Everything is seeded (Random(206)), so every number is reproducible to the digit. Every person is fictional.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from datetime import date, timedelta
from itertools import combinations, product
from typing import Callable, Dict, List, Optional, Tuple

SEED = 206
POP_N = 60_000
EXPORT_FRAC = 0.05
K = 5
TODAY = date(2026, 10, 31)

Row = Tuple[int, str, date, str, str]  # person id, postcode, date of birth, sex, diagnosis

AGE_BANDS = [((0, 19), 20), ((20, 39), 27), ((40, 59), 29), ((60, 79), 19), ((80, 95), 5)]
DIAGNOSES = ["routine check", "hypertension", "diabetes", "asthma", "depression", "fracture", "pregnancy", "HIV"]
DIAG_W = [40, 18, 12, 8, 9, 6, 5, 2]
SENSITIVE = {"depression", "HIV"}

# generalisation hierarchies: level name -> function of the raw value
POSTCODE_LEVELS: Dict[str, Callable[[str], str]] = {
    "6 digits": lambda p: p,
    "4 digits": lambda p: p[:4],
    "sector (2 digits)": lambda p: p[:2],
    "dropped": lambda p: "*",
}
DOB_LEVELS: Dict[str, Callable[[date], str]] = {
    "full date": lambda d: d.isoformat(),
    "month": lambda d: d.strftime("%Y-%m"),
    "year": lambda d: str(d.year),
    "decade": lambda d: f"{d.year // 10 * 10}s",
    "dropped": lambda d: "*",
}
SEX_LEVELS: Dict[str, Callable[[str], str]] = {"kept": lambda s: s, "dropped": lambda s: "*"}
QI = ("postcode", "date of birth", "sex")
FULL = ("6 digits", "full date", "kept")


def population(n: int = POP_N, seed: int = SEED) -> List[Row]:
    """A fictional town: 20 sectors x 10 sub-areas x 3 buildings = 600 postcodes (SSAABB, so the first four digits
    are the sub-area and the first two the sector) with Zipf-skewed sizes (a tower block holds hundreds, a terrace a
    handful), an age pyramid by band, and a skewed diagnosis mix."""
    rng = random.Random(seed)
    codes = [f"{s:02d}{a:02d}{b:02d}" for s in range(20) for a in range(10) for b in range(3)]
    weights = [1 / (r + 1) ** 0.8 for r in range(len(codes))]
    rng.shuffle(weights)
    bands = [b for b, _ in AGE_BANDS]
    bw = [w for _, w in AGE_BANDS]
    rows: List[Row] = []
    for pid in range(n):
        pc = rng.choices(codes, weights)[0]
        lo, hi = rng.choices(bands, bw)[0]
        age = rng.randint(lo, hi)
        dob = date(TODAY.year - age, 1, 1) + timedelta(days=rng.randrange(365))
        if dob > TODAY:
            dob = TODAY
        sex = rng.choice("MF")
        diag = rng.choices(DIAGNOSES, DIAG_W)[0]
        if diag == "pregnancy" and (sex == "M" or not 18 <= age <= 45):
            diag = "routine check"
        rows.append((pid, pc, dob, sex, diag))
    return rows


def sample(pop: List[Row], frac: float, seed: int = SEED + 1) -> List[Row]:
    rng = random.Random(seed)
    return rng.sample(pop, round(len(pop) * frac))


def key(row: Row, levels: Tuple[str, str, str], use: Tuple[bool, bool, bool] = (True, True, True)) -> Tuple[str, ...]:
    pl, dl, sl = levels
    return (
        POSTCODE_LEVELS[pl](row[1]) if use[0] else "*",
        DOB_LEVELS[dl](row[2]) if use[1] else "*",
        SEX_LEVELS[sl](row[3]) if use[2] else "*",
    )


def profile(export: List[Row], pop: List[Row], levels: Tuple[str, str, str], use=(True, True, True)) -> Dict:
    """One generalisation, measured on the export and projected onto the town it was drawn from."""
    ek = [key(r, levels, use) for r in export]
    ec = Counter(ek)
    pc = Counter(key(r, levels, use) for r in pop)
    n = len(export)
    small = [k for k, c in ec.items() if c < K]
    diag_by_class: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    for r, k in zip(export, ek):
        diag_by_class[k][r[4]] += 1
    kept = {k for k, c in ec.items() if c >= K}  # the release after suppressing the classes below K
    homog = {k for k in kept if len(diag_by_class[k]) == 1}
    # attribute disclosure in the release: the share of the class's commonest diagnosis is what an attacker who has
    # placed a person in the class learns about them, k or no k
    near = sum(ec[k] for k in kept if diag_by_class[k].most_common(1)[0][1] / ec[k] >= 0.8)
    sens_confident = sum(
        c for k in kept for d, c in diag_by_class[k].items() if d in SENSITIVE and c / ec[k] >= 0.5
    )
    sens_rows = sum(c for k in kept for d, c in diag_by_class[k].items() if d in SENSITIVE)
    return {
        "levels": levels,
        "classes": len(ec),
        "k_min": min(ec.values()),
        "unique_export": sum(1 for k in ek if ec[k] == 1),
        "unique_pop": sum(1 for k in ek if pc[k] == 1),
        "below_k": sum(ec[k] for k in small),
        "below_k_pop": sum(1 for k in ek if pc[k] < K),
        "released": sum(ec[k] for k in kept),
        "homog_classes": len(homog),
        "homog_rows": sum(ec[k] for k in homog),
        "near_homog_rows": near,
        "sensitive_rows_released": sens_rows,
        "sensitive_confident_rows": sens_confident,
        "n": n,
    }


def qi_subsets(export: List[Row], pop: List[Row]) -> List[Dict]:
    """Every non-empty subset of the three columns at full granularity: how few columns make a row unique."""
    out = []
    for r in (1, 2, 3):
        for idx in combinations(range(3), r):
            use = tuple(i in idx for i in range(3))
            p = profile(export, pop, FULL, use)
            p["columns"] = [QI[i] for i in idx]
            out.append(p)
    return out


def ladder(export: List[Row], pop: List[Row]) -> List[Dict]:
    """All 40 generalisations, most detailed first."""
    out = []
    for pl, dl, sl in product(POSTCODE_LEVELS, DOB_LEVELS, SEX_LEVELS):
        out.append(profile(export, pop, (pl, dl, sl)))
    return out


def first_k_anonymous(lad: List[Dict]) -> List[Dict]:
    """The generalisations that reach k >= K in the export without suppressing anything, most detailed first
    (most classes = least information lost)."""
    ok = [p for p in lad if p["below_k"] == 0]
    return sorted(ok, key=lambda p: -p["classes"])


def k_anonymous_with_suppression(lad: List[Dict], budget: float = 0.05) -> List[Dict]:
    """The generalisations that reach k >= K in the export once at most `budget` of the rows are suppressed,
    most detailed first."""
    ok = [p for p in lad if p["below_k"] <= budget * p["n"]]
    return sorted(ok, key=lambda p: (-p["classes"], p["below_k"]))


def sampling_sweep(pop: List[Row], fracs=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)) -> List[Dict]:
    """Unique in the export vs unique in the town, at full granularity and at sector + year, as the export grows."""
    out = []
    for f in fracs:
        ex = sample(pop, f)
        for lv in (FULL, ("sector (2 digits)", "year", "kept")):
            p = profile(ex, pop, lv)
            out.append(
                {
                    "frac": f,
                    "levels": lv,
                    "n": p["n"],
                    "unique_export_pct": 100 * p["unique_export"] / p["n"],
                    "unique_pop_pct": 100 * p["unique_pop"] / p["n"],
                }
            )
    return out


def linkage(export: List[Row], pop: List[Row], levels: Tuple[str, str, str], coverage: float, seed: int = SEED + 2) -> Dict:
    """An attacker holds a public register (name + the three columns) covering `coverage` of the town and looks
    every export row up by its generalised key. right = exactly one match and it is the person; wrong = exactly one
    match and it is someone else; ambiguous = several matches; none = nobody in the register has that key."""
    rng = random.Random(seed)
    reg = pop if coverage >= 1 else rng.sample(pop, round(len(pop) * coverage))
    by_key: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for r in reg:
        by_key[key(r, levels)].append(r[0])
    out = Counter()
    for r in export:
        m = by_key.get(key(r, levels), [])
        if len(m) == 0:
            out["none"] += 1
        elif len(m) > 1:
            out["ambiguous"] += 1
        elif m[0] == r[0]:
            out["right"] += 1
        else:
            out["wrong"] += 1
    return {"levels": levels, "coverage": coverage, "register_n": len(reg), **{k: out[k] for k in ("right", "wrong", "ambiguous", "none")}}


def smallest_postcodes(pop: List[Row], top: int = 3) -> List[Tuple[str, int]]:
    c = Counter(r[1] for r in pop)
    return sorted(c.items(), key=lambda kv: (kv[1], kv[0]))[:top]


def postcode_sizes(pop: List[Row]) -> Dict:
    c = Counter(r[1] for r in pop)
    sizes = sorted(c.values())
    return {"postcodes": len(c), "min": sizes[0], "median": sizes[len(sizes) // 2], "max": sizes[-1]}


def study(pop: Optional[List[Row]] = None) -> Dict:
    pop = pop or population()
    export = sample(pop, EXPORT_FRAC)
    lad = ladder(export, pop)
    ok = first_k_anonymous(lad)
    sup = k_anonymous_with_suppression(lad)
    best = ok[0]
    release = sup[0]
    link_levels = [FULL, ("6 digits", "year", "kept"), ("sector (2 digits)", "year", "kept"), release["levels"], best["levels"]]
    return {
        "pop_n": len(pop),
        "export_n": len(export),
        "sensitive_base": sum(1 for r in export if r[4] in SENSITIVE),
        "postcodes": postcode_sizes(pop),
        "smallest_postcodes": smallest_postcodes(pop),
        "subsets": qi_subsets(export, pop),
        "ladder": lad,
        "k_anonymous": ok,
        "best": best,
        "with_suppression": sup,
        "release": release,
        "sweep": sampling_sweep(pop),
        "linkage": [linkage(export, pop, lv, cov) for lv in link_levels for cov in (1.0, 0.8)],
    }
