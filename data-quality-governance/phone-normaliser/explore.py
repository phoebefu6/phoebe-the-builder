"""Six ways to turn a phone number as typed into a match key, against a hand-labelled customer table.

Each row is one CRM record: the phone string as the customer typed it, the record's country field (the SHIPPING
address, which is what most CRMs have; blank on some rows), and the true E.164 number, labelled from who the
customer is. Two rows with the same true number are the same customer typed twice; two rows with different true
numbers are different customers. A normaliser is judged on three things:

- e164 right : did it produce the true number (or, for the strict one, refuse rather than guess)
- false split: same true number, two different keys -> one customer becomes two
- false merge: different true numbers, one key      -> two customers become one

Every number is a fictional or documentation range (Ofcom 020 7946 0xxx, NANP 555-01xx, ACMA 0491 570 1xx, and
libphonenumber's own example numbers). The ambiguity sweep uses a seeded generator, so every number is exact.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

import phonenumbers as pn

# (phone as typed, country field on the record, true E.164, format)
CORPUS: List[Tuple[str, str, str, str]] = [
    # London landline, one customer, six ways of writing it
    ("+44 20 7946 0958", "GB", "+442079460958", "international +"),
    ("020 7946 0958", "GB", "+442079460958", "national, trunk 0"),
    ("0044 20 7946 0958", "GB", "+442079460958", "00 prefix"),
    ("+44 (0)20 7946 0958", "GB", "+442079460958", "international + (0)"),
    ("44 20 7946 0958", "SG", "+442079460958", "country code, no +"),
    ("020 7946 0958", "SG", "+442079460958", "national, trunk 0"),
    ("020 7946 0321", "GB", "+442079460321", "national, trunk 0"),
    # Washington DC, one customer, five ways
    ("202-555-0123", "US", "+12025550123", "NANP 10 digits"),
    ("(202) 555-0123", "US", "+12025550123", "NANP 10 digits"),
    ("1-202-555-0123", "US", "+12025550123", "NANP with 1"),
    ("+1 202 555 0123", "", "+12025550123", "international +"),
    ("202.555.0123 x204", "US", "+12025550123", "with extension"),
    ("415-555-0132", "US", "+14155550132", "NANP 10 digits"),
    # Toronto: same +1 country code, different country
    ("416-555-0199", "CA", "+14165550199", "NANP 10 digits"),
    ("416 555 0199", "", "+14165550199", "NANP 10 digits"),
    # Australian mobiles
    ("0491 570 156", "AU", "+61491570156", "national, trunk 0"),
    ("+61 491 570 156", "AU", "+61491570156", "international +"),
    ("61491570156", "AU", "+61491570156", "country code, no +"),
    ("0491 570 159", "AU", "+61491570159", "national, trunk 0"),
    ("0491570156", "", "+61491570156", "national, trunk 0"),
    # France vs Spain: the same nine digits after the French trunk 0
    ("06 12 34 56 78", "FR", "+33612345678", "national, trunk 0"),
    ("+33 6 12 34 56 78", "FR", "+33612345678", "international +"),
    ("0033612345678", "FR", "+33612345678", "00 prefix"),
    ("612 345 678", "ES", "+34612345678", "national, no trunk"),
    ("+34 612 34 56 78", "ES", "+34612345678", "international +"),
    # Singapore vs Hong Kong: the same eight digits, both valid mobiles
    ("9123 4567", "SG", "+6591234567", "national, no trunk"),
    ("+65 9123 4567", "SG", "+6591234567", "international +"),
    ("9123 4567", "HK", "+85291234567", "national, no trunk"),
    ("+852 9123 4567", "HK", "+85291234567", "international +"),
    # India, Germany, Malaysia
    ("98765 43210", "IN", "+919876543210", "national, no trunk"),
    ("+91 98765 43210", "IN", "+919876543210", "international +"),
    ("09876543210", "IN", "+919876543210", "national, trunk 0"),
    ("030 123456", "DE", "+4930123456", "national, trunk 0"),
    ("+49 30 123456", "DE", "+4930123456", "international +"),
    ("012-345 6789", "MY", "+60123456789", "national, trunk 0"),
    ("+60 12-345 6789", "MY", "+60123456789", "international +"),
    # a UK expat whose record carries the Singapore shipping address and no +
    ("07400 123456", "SG", "+447400123456", "national, trunk 0"),
    ("+44 7400 123456", "GB", "+447400123456", "international +"),
    # the country field is the SHIPPING address, not the phone's country
    ("612 345 678", "FR", "+34612345678", "national, no trunk"),
    ("9123 4567", "SG", "+85291234567", "national, no trunk"),
]

HQ = "SG"  # the default region a single-market import script assumes
REGIONS = sorted(pn.SUPPORTED_REGIONS)
E164 = pn.PhoneNumberFormat.E164


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s.split("x")[0])


def _digits_only(s: str, country: str, hq: str) -> Optional[str]:
    """re.sub(r'\\D', '', phone): the key most dedupe scripts start with."""
    return _digits(s)


def _last_nine(s: str, country: str, hq: str) -> Optional[str]:
    """Match on the last 9 digits - the 'ignore the prefix, compare the subscriber part' hack."""
    return _digits(s)[-9:]


def _plus_one(s: str, country: str, hq: str) -> Optional[str]:
    """A US-built script: 10 digits -> +1, 11 digits starting 1 -> +, anything else -> + digits."""
    d = _digits(s)
    if len(d) == 10:
        return "+1" + d
    return "+" + d.lstrip("0")


def _lib(s: str, region: Optional[str]) -> Optional[str]:
    try:
        n = pn.parse(s, region)
    except pn.NumberParseException:
        return None
    return pn.format_number(n, E164) if pn.is_valid_number(n) else None


def _lib_hq(s: str, country: str, hq: str) -> Optional[str]:
    """phonenumbers.parse(phone, HQ): one default region for every row."""
    return _lib(s, hq)


def _lib_country(s: str, country: str, hq: str) -> Optional[str]:
    """phonenumbers.parse(phone, record's country field), falling back to HQ when the field is blank."""
    return _lib(s, country or hq)


def _lib_none(s: str, country: str, hq: str) -> Optional[str]:
    """phonenumbers.parse(phone, None): accept only strings that carry their own country code, refuse the rest."""
    return _lib(s, None)


NORMALISERS: Dict[str, Callable[[str, str, str], Optional[str]]] = {
    "digits only": _digits_only,
    "last 9 digits": _last_nine,
    "US script (+1 if 10 digits)": _plus_one,
    f"phonenumbers, region={HQ}": _lib_hq,
    "phonenumbers, region=country field": _lib_country,
    "phonenumbers, no region (+ only)": _lib_none,
}
KEY_ONLY = {"digits only", "last 9 digits"}  # produce a match key, never an E.164 number


def readings(s: str) -> Dict[str, List[str]]:
    """Every distinct valid E.164 number the string can be, across all regions -> the regions that read it so."""
    out: Dict[str, List[str]] = {}
    for r in REGIONS:
        e = _lib(s, r)
        if e:
            out.setdefault(e, []).append(r)
    return out


def score(name: str, hq: str = HQ) -> Dict:
    f = NORMALISERS[name]
    keys = [f(s, c, hq) for s, c, _, _ in CORPUS]
    right = sum(1 for k, (_, _, t, _) in zip(keys, CORPUS) if k == t)
    refused = sum(1 for k in keys if k is None)
    if name in KEY_ONLY:
        right = None
    splits, merges = [], []
    for i, j in combinations(range(len(CORPUS)), 2):
        ki, kj, ti, tj = keys[i], keys[j], CORPUS[i][2], CORPUS[j][2]
        if ki is None or kj is None:
            continue
        if ti == tj and ki != kj:
            splits.append((CORPUS[i][0], CORPUS[j][0]))
        if ti != tj and ki == kj:
            merges.append((CORPUS[i][0], CORPUS[j][0]))
    true_pairs = sum(1 for i, j in combinations(range(len(CORPUS)), 2) if CORPUS[i][2] == CORPUS[j][2])
    # customers the table ends up with: distinct keys, refused rows each count as their own record
    customers = len({k for k in keys if k is not None}) + refused
    return {
        "right": right,
        "wrong": None if right is None else len(CORPUS) - right - refused,
        "refused": refused,
        "false_splits": splits,
        "false_merges": merges,
        "true_pairs": true_pairs,
        "customers": customers,
        "keys": keys,
    }


def sweep(n: int = 600, seed: int = 204) -> Dict:
    """How many countries a typed national number is valid in, for seeded random strings of two common shapes."""
    rng = random.Random(seed)
    out = {}
    for label, make in [
        (
            "0 + 9 digits (trunk-0 national)",
            lambda: (
                "0" + "".join(rng.choice("123456789") + "".join(rng.choice("0123456789") for _ in range(8)))
            ),
        ),
        (
            "8 digits, no trunk",
            lambda: rng.choice("2345689") + "".join(rng.choice("0123456789") for _ in range(7)),
        ),
    ]:
        counts = [len(readings(make())) for _ in range(n)]
        c = Counter(counts)
        out[label] = {
            "n": n,
            "valid_somewhere": sum(1 for x in counts if x >= 1) / n,
            "valid_3plus": sum(1 for x in counts if x >= 3) / n,
            "median_numbers": sorted(counts)[n // 2],
            "max_numbers": max(counts),
            "hist": dict(sorted(c.items())),
        }
    return out


def ceiling() -> Dict:
    """Upper bounds for anything that reads only the string, or the string plus the country field.

    A row counts as recoverable when its input has a single truth in this table and, for the string-only bound, a
    single valid reading across all 245 regions."""
    by_in: Dict[Tuple[str, str], Counter] = {}
    for s, c, t, _ in CORPUS:
        by_in.setdefault((s, c), Counter())[t] += 1
    return {
        "by_string": sum(1 for s, _, t, _ in CORPUS if list(readings(s)) == [t]),
        "by_string_and_field": sum(c.most_common(1)[0][1] for c in by_in.values()),
        "two_truths": sorted({f"{s} [{c}]" for (s, c), k in by_in.items() if len(k) > 1}),
    }


def study(hq: str = HQ, sweep_n: int = 600) -> Dict:
    per = {n: score(n, hq) for n in NORMALISERS}
    amb = {s: len(readings(s)) for s, _, _, _ in CORPUS}
    true_customers = len({t for _, _, t, _ in CORPUS})
    return {
        "n": len(CORPUS),
        "hq": hq,
        "true_customers": true_customers,
        "per": per,
        "ambiguity": amb,
        "ceiling": ceiling(),
        "sweep": sweep(sweep_n) if sweep_n else {},
    }
