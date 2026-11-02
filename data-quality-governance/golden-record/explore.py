"""Which customer record is real? Six survivorship rule-sets scored field by field against a known truth.

NOT a merger: `dedup-pipeline` (Day 67) implements survivorship rules and merges clusters. Nothing in the catalog
asks whether the value a rule keeps is the TRUE one. This builds fictional customers whose true details change over
time (moves, new phones, new emails), lets three systems capture them the way real systems do (CRM: entered by
sales, then touched for opt-ins; web: self-typed, fresh, nicknames and swapped dates; billing: verified address,
switchboard phones, invoice mailboxes, and sometimes an address synced from CRM), and scores what each rule-set
keeps against the customer's details today. Seeded Random(208); every person fictional.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

SEED = 208
N_CUSTOMERS = 400
TODAY = 1000  # day number; every capture happens on days 0..1000
FIELDS = ["name", "dob", "email", "phone", "address"]
SOURCES = ["crm", "billing", "web"]
VOLATILE = ["email", "phone", "address"]

GIVEN = [("Elizabeth", "Liz"), ("William", "Will"), ("Katherine", "Kate"), ("Robert", "Rob"), ("Margaret", "Maggie"),
         ("Jonathan", "Jon"), ("Alexandra", "Alex"), ("Nicholas", "Nick"), ("Rebecca", "Becky"), ("Thomas", "Tom"),
         ("Victoria", "Vicky"), ("Benjamin", "Ben"), ("Jennifer", "Jen"), ("Samuel", "Sam"), ("Patricia", "Pat")]
FAMILY = ["Okafor", "Lindqvist", "Tan", "Moreau", "Haddad", "Kowalski", "Nakamura", "Osei", "Brennan", "Castillo",
          "Petrov", "Achebe", "Lim", "Varga", "Duarte", "Nguyen", "Fischer", "Rahman", "Silva", "Ward"]
STREETS = ["Elm Row", "Quarry Lane", "Harbour Walk", "Mill Street", "Orchard Close", "Station Road", "Kiln Yard",
           "Fern Avenue", "Lantern Way", "Copper Hill", "Weaver Court", "Signal Terrace"]

Record = Dict[str, object]  # source, updated_at, field values, and per-field capture day under "_ts"
Rule = Callable[[List[Record], str], Optional[str]]


# --- the world ---------------------------------------------------------------------------------------------------

def _address(rng: random.Random) -> str:
    return f"{rng.randint(1, 180)} {rng.choice(STREETS)}, {rng.choice('ABCDEFGH')}{rng.randint(1, 19)} {rng.randint(1, 9)}XY"


def _history(rng: random.Random, p_change: float, make: Callable[[], str]) -> List[Tuple[int, str]]:
    """(from_day, value) steps. A value changes at most twice in the window."""
    steps = [(0, make())]
    for _ in range(2):
        if rng.random() < p_change:
            day = rng.randint(steps[-1][0] + 60, TODAY) if steps[-1][0] + 60 < TODAY else TODAY
            steps.append((day, make()))
    return steps


def value_at(steps: List[Tuple[int, str]], day: int) -> str:
    return [v for d, v in steps if d <= day][-1]


def make_world(seed: int = SEED) -> List[dict]:
    """Customers with true histories, each captured by 2-3 systems. Returns one dict per customer (= one cluster)."""
    rng = random.Random(seed)
    out: List[dict] = []
    used_emails = set()
    for cid in range(N_CUSTOMERS):
        given, nick = rng.choice(GIVEN)
        family = rng.choice(FAMILY)
        name = f"{given} {family}"
        y, m, d = rng.randint(1950, 2002), rng.randint(1, 12), rng.randint(1, 28)
        dob = f"{y}-{m:02d}-{d:02d}"

        def email() -> str:
            while True:
                e = f"{given[0].lower()}{family.lower()}{rng.randint(1, 999)}@{rng.choice(['mailbox.test', 'post.test', 'inbox.test'])}"
                if e not in used_emails:
                    used_emails.add(e)
                    return e

        def phone() -> str:
            return f"+44 7700 9{rng.randint(0, 99999):05d}"

        hist = {"email": _history(rng, 0.15, email), "phone": _history(rng, 0.25, phone),
                "address": _history(rng, 0.35, lambda: _address(rng))}
        truth = {"name": name, "dob": dob, **{f: value_at(h, TODAY) for f, h in hist.items()}}

        def at(day: int) -> Dict[str, str]:
            return {"name": name, "dob": dob, **{f: value_at(h, day) for f, h in hist.items()}}

        recs: List[Record] = []
        crm_cap = None
        # CRM: keyed in by sales early; later "touched" for an opt-in change that refreshes no customer field.
        if rng.random() < 0.9:
            cap = rng.randint(0, 600)
            crm_cap = cap
            v = at(cap)
            if rng.random() < 0.03:
                v["dob"] = f"{y}-{m:02d}-{min(28, d % 28 + 1):02d}"  # keyed one day off
            touched = rng.randint(cap, TODAY) if rng.random() < 0.5 else cap
            recs.append({"source": "crm", "updated_at": touched, **v, "_ts": {f: cap for f in FIELDS}})
        # Web: typed by the customer, recent. Nicknames, day/month swapped, dob often skipped.
        if rng.random() < 0.8:
            cap = rng.randint(300, TODAY)
            v = at(cap)
            if rng.random() < 0.2:
                v["name"] = f"{nick} {family}"
            if rng.random() < 0.5:
                v["dob"] = None
            elif d <= 12 and m != d and rng.random() < 0.25:
                v["dob"] = f"{y}-{d:02d}-{m:02d}"
            recs.append({"source": "web", "updated_at": cap, **v, "_ts": {f: cap for f in FIELDS}})
        # Billing: address verified by a payment, but sometimes overwritten by a CRM sync; company phones and
        # invoice mailboxes; almost never a dob.
        if rng.random() < 0.7:
            cap = rng.randint(100, TODAY)
            v = at(cap)
            ts = {f: cap for f in FIELDS}
            lineage = cap
            if crm_cap is not None and rng.random() < 0.3:
                v["address"] = at(crm_cap)["address"]  # the nightly CRM -> billing sync wrote CRM's copy
                ts["address"] = cap  # ... and stamped it with the sync day, so it LOOKS fresh
                lineage = crm_cap  # the day the value was really captured, if lineage had been kept
            if rng.random() < 0.2:
                v["phone"] = "+44 20 7946 0" + f"{rng.randint(0, 999):03d}"
            if rng.random() < 0.25:
                v["email"] = f"accounts@{family.lower()}-ltd.test"
            if rng.random() < 0.85:
                v["dob"] = None
            v["name"] = name.upper() if rng.random() < 0.3 else v["name"]  # some ledgers store upper case
            recs.append({"source": "billing", "updated_at": cap, **v, "_ts": ts, "_lineage": lineage})
        if len(recs) >= 2:
            out.append({"cid": cid, "truth": truth, "records": recs})
    return out


# --- survivorship rules: given a cluster and a field, keep one value ---------------------------------------------

def _cands(recs: List[Record], f: str) -> List[Record]:
    return [r for r in recs if r.get(f) not in (None, "")]


def most_recent(recs: List[Record], f: str) -> Optional[str]:
    c = _cands(recs, f)
    return max(c, key=lambda r: (r["updated_at"], r["source"]))[f] if c else None


PRIORITY = {"crm": 0, "billing": 1, "web": 2}


def source_priority(recs: List[Record], f: str) -> Optional[str]:
    c = _cands(recs, f)
    return min(c, key=lambda r: PRIORITY[r["source"]])[f] if c else None


def most_complete(recs: List[Record], f: str) -> Optional[str]:
    """The record with the most filled fields wins every field it has (ties -> most recent)."""
    c = _cands(recs, f)
    return max(c, key=lambda r: (sum(r.get(g) not in (None, "") for g in FIELDS), r["updated_at"]))[f] if c else None


def longest(recs: List[Record], f: str) -> Optional[str]:
    c = _cands(recs, f)
    return max(c, key=lambda r: (len(str(r[f])), r["updated_at"]))[f] if c else None


def majority(recs: List[Record], f: str) -> Optional[str]:
    c = _cands(recs, f)
    if not c:
        return None
    n = Counter(r[f] for r in c)
    return max(c, key=lambda r: (n[r[f]], r["updated_at"]))[f]


TRUST = {"name": ["crm", "billing", "web"], "dob": ["crm", "web", "billing"], "email": ["crm", "web", "billing"]}


def per_field(recs: List[Record], f: str) -> Optional[str]:
    """A declared rule per field: stable fields by source trust, volatile fields by when THAT FIELD was captured."""
    c = _cands(recs, f)
    if not c:
        return None
    if f in ("phone", "address"):
        return max(c, key=lambda r: (r["_ts"][f], r["source"]))[f]
    return min(c, key=lambda r: TRUST[f].index(r["source"]))[f]


def most_recent_field_ts(recs: List[Record], f: str, lineage: bool = False) -> Optional[str]:
    """most_recent, ordered by the field's own capture day instead of the record's updated_at. With lineage=True a
    synced address carries the day it was first captured, not the day the sync wrote it."""
    c = _cands(recs, f)
    day = (lambda r: r.get("_lineage", r["_ts"][f])) if lineage and f == "address" else (lambda r: r["_ts"][f])
    return max(c, key=lambda r: (day(r), r["source"]))[f] if c else None


RULES: Dict[str, Rule] = {"most_recent": most_recent, "source_priority": source_priority,
                          "most_complete": most_complete, "longest": longest, "majority": majority,
                          "per_field": per_field}
GENERIC = ["most_recent", "source_priority", "most_complete", "longest", "majority"]


def golden(recs: List[Record], rule: Rule) -> Dict[str, Optional[str]]:
    return {f: rule(recs, f) for f in FIELDS}


# --- the study ---------------------------------------------------------------------------------------------------

def study(seed: int = SEED) -> dict:
    world = make_world(seed)
    n = len(world)
    acc = {k: {f: 0 for f in FIELDS} for k in RULES}
    whole = {k: 0 for k in RULES}
    ceiling = {f: 0 for f in FIELDS}
    distinct = Counter()
    disagree_field = {f: 0 for f in FIELDS}
    ts_fix = {"record_ts": 0, "field_ts": 0, "lineage_ts": 0}
    stale_sync = {"clusters": 0, "majority_stale": 0}
    for c in world:
        recs, truth = c["records"], c["truth"]
        gs = {k: golden(recs, r) for k, r in RULES.items()}
        for k, g in gs.items():
            ok = [g[f] == truth[f] for f in FIELDS]
            for f, o in zip(FIELDS, ok):
                acc[k][f] += o
            whole[k] += all(ok)
        for f in FIELDS:
            ceiling[f] += any(r.get(f) == truth[f] for r in recs)
            disagree_field[f] += len({gs[k][f] for k in GENERIC}) > 1
        distinct[len({tuple(gs[k].values()) for k in GENERIC})] += 1
        ts_fix["record_ts"] += most_recent(recs, "address") == truth["address"]
        ts_fix["field_ts"] += most_recent_field_ts(recs, "address") == truth["address"]
        ts_fix["lineage_ts"] += most_recent_field_ts(recs, "address", lineage=True) == truth["address"]
        by = {r["source"]: r for r in recs}
        if {"crm", "billing", "web"} <= set(by) and by["crm"]["address"] == by["billing"]["address"] != truth["address"] \
                and by["web"]["address"] == truth["address"]:
            stale_sync["clusters"] += 1
            stale_sync["majority_stale"] += majority(recs, "address") != truth["address"]
    missing_any = sum(not all(any(r.get(f) == c["truth"][f] for r in c["records"]) for f in FIELDS) for c in world)
    oracle_whole = sum(all(any(r.get(f) == c["truth"][f] for r in c["records"]) for f in FIELDS) for c in world)
    return {
        "customers": N_CUSTOMERS, "clusters": n, "records": sum(len(c["records"]) for c in world),
        "acc": {k: {f: v / n for f, v in d.items()} for k, d in acc.items()},
        "whole": {k: v / n for k, v in whole.items()},
        "ceiling": {f: v / n for f, v in ceiling.items()}, "oracle_whole": oracle_whole / n,
        "missing_any": missing_any,
        "distinct": {int(k): v for k, v in sorted(distinct.items())},
        "disagree_field": {f: v / n for f, v in disagree_field.items()},
        "all_agree": distinct[1] / n,
        "ts_fix": {k: v / n for k, v in ts_fix.items()},
        "stale_sync": stale_sync,
        "example": example(world),
    }


def example(world: List[dict]) -> dict:
    """The first cluster where the five generic rule-sets produce the most distinct golden records."""
    best = max(world, key=lambda c: (len({tuple(golden(c["records"], RULES[k]).values()) for k in GENERIC}), -c["cid"]))
    return {"cid": best["cid"], "truth": best["truth"],
            "records": [{k: v for k, v in r.items() if not k.startswith("_")} for r in best["records"]],
            "golden": {k: golden(best["records"], r) for k, r in RULES.items()}}
