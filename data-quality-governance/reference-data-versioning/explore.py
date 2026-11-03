"""The country list changed: can last year's regional report be reproduced from today's reference table?

A fictional territory list (40 territories, 5 regions) changes over 36 months: territories move region (some moves
are recorded months after they took effect), codes are renamed, territories split and merge, and one code is retired
and later reused for a different territory. Sales facts carry the code that was valid on the day of the sale.

Each month a regional revenue report was published from the reference table as it was known that day. Today we try
to rebuild those reports four ways, and to restate them to today's regions one way, and score both against a truth
we can only know because the world is simulated. Seeded Random(209); every territory is fictional.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

SEED = 209
MONTHS = 36
REGIONS = ["North", "South", "East", "West", "Central"]
OPEN = 10**6
STRATEGIES = ["overwrite", "code_only", "scd2_asof", "bitemporal"]
Row = Dict[str, object]
Report = Dict[int, Dict[str, float]]


def make_codes(rng: random.Random, n: int) -> List[str]:
    out: List[str] = []
    while len(out) < n:
        c = rng.choice("BDFGKLMNPRSTVZ") + rng.choice("AEIOU") + rng.choice("BDFGKLMNPRSTVZ")
        if c not in out:
            out.append(c)
    return out


def make_world(seed: int = SEED) -> dict:
    """Reference-data events (effective month, recorded month) and the sales facts they govern."""
    rng = random.Random(seed)
    codes = make_codes(rng, 47)
    base, spare = codes[:40], codes[40:]
    init = {c: {"entity": f"T{i:02d}", "region": REGIONS[i % 5]} for i, c in enumerate(base)}
    pool = base[:]
    rng.shuffle(pool)
    ev: List[dict] = []
    other = lambda r: rng.choice([x for x in REGIONS if x != r])  # noqa: E731
    for i in range(6):  # region moves; the first three are recorded 2-4 months late (backdated)
        c, eff = pool.pop(), rng.randint(4, 28)
        ev.append({"kind": "move", "eff": eff, "rec": eff + (rng.randint(2, 4) if i < 3 else 0), "code": c,
                   "region": other(init[c]["region"])})
    for _ in range(2):  # renames: same territory, new code
        c, eff = pool.pop(), rng.randint(6, 30)
        ev.append({"kind": "rename", "eff": eff, "rec": eff, "code": c, "new": spare.pop()})
    for k in range(2):  # splits: one territory becomes two, the second in another region
        c, eff = pool.pop(), rng.randint(14, 24)
        r0 = init[c]["region"]
        ev.append({"kind": "split", "eff": eff, "rec": eff, "code": c,
                   "into": [(spare.pop(), f"S{k}a", r0), (spare.pop(), f"S{k}b", other(r0))]})
    a, b, eff = pool.pop(), pool.pop(), rng.randint(10, 26)  # merge into a new territory
    ev.append({"kind": "merge", "eff": eff, "rec": eff, "codes": [a, b], "new": spare.pop(), "entity": "M0",
               "region": init[a]["region"]})
    q, n, eff = pool.pop(), pool.pop(), rng.randint(6, 12)  # q absorbed into n, then the code q is reused
    ev.append({"kind": "absorb", "eff": eff, "rec": eff, "code": q, "into": n})
    ev.append({"kind": "create", "eff": eff + rng.randint(8, 14), "rec": None, "code": q, "entity": "R0",
               "region": other(init[n]["region"])})
    ev[-1]["rec"] = ev[-1]["eff"]
    ev.sort(key=lambda e: e["eff"])

    # sales facts, generated from the TRUE world (effective dates, not recorded ones)
    base_rev = {e: rng.uniform(50_000, 150_000) for e in [v["entity"] for v in init.values()]}
    state = {c: dict(v) for c, v in init.items()}  # code -> entity, region (true, today's knowledge irrelevant)
    facts: List[dict] = []
    for m in range(MONTHS):
        for e in [e for e in ev if e["eff"] == m]:
            if e["kind"] == "rename":
                state[e["new"]] = state.pop(e["code"])
            elif e["kind"] == "split":
                old = state.pop(e["code"])
                sh = 0.75 - 0.4 * m / e["eff"]
                for (code, ent, reg), part in zip(e["into"], (sh, 1 - sh)):
                    state[code] = {"entity": ent, "region": reg}
                    base_rev[ent] = base_rev[old["entity"]] * part
            elif e["kind"] == "merge":
                base_rev[e["entity"]] = sum(base_rev[state.pop(c)["entity"]] for c in e["codes"])
                state[e["new"]] = {"entity": e["entity"], "region": e["region"]}
            elif e["kind"] == "absorb":
                base_rev[state[e["into"]]["entity"]] += base_rev[state.pop(e["code"])["entity"]]
            elif e["kind"] == "create":
                state[e["code"]] = {"entity": e["entity"], "region": e["region"]}
                base_rev[e["entity"]] = rng.uniform(50_000, 150_000)
        split_pending = {e["code"]: e for e in ev if e["kind"] == "split" and e["eff"] > m}
        for code, s in state.items():
            amt = round(base_rev[s["entity"]] * (1 + 0.01 * m) * rng.gauss(1, 0.08))
            if code in split_pending:  # the sale happened in one of the two future halves; nobody records which
                e = split_pending[code]
                sh = 0.75 - 0.4 * m / e["eff"]
                a_amt = round(amt * sh)
                facts.append({"month": m, "code": code, "amount": a_amt, "area": e["into"][0][1]})
                facts.append({"month": m, "code": code, "amount": amt - a_amt, "area": e["into"][1][1]})
            else:
                facts.append({"month": m, "code": code, "amount": amt, "area": s["entity"]})
    return {"init": init, "events": ev, "facts": facts}


def rows_known(world: dict, t: Optional[int] = None) -> List[Row]:
    """The SCD2 reference table as it stood after month t was recorded (None = today)."""
    t = OPEN if t is None else t
    rows: List[Row] = []
    live: Dict[str, Row] = {}

    def open_(code: str, ent: str, reg: str, vf: int) -> None:
        live[code] = {"code": code, "entity": ent, "region": reg, "vf": vf, "vt": OPEN}
        rows.append(live[code])

    def close(code: str, at: int) -> Row:
        r = live.pop(code)
        r["vt"] = at
        return r

    for c, v in world["init"].items():
        open_(c, v["entity"], v["region"], 0)
    for e in world["events"]:
        if e["rec"] > t:
            continue
        m = e["eff"]
        if e["kind"] == "move":
            r = close(e["code"], m)
            open_(e["code"], r["entity"], e["region"], m)
        elif e["kind"] == "rename":
            r = close(e["code"], m)
            open_(e["new"], r["entity"], r["region"], m)
        elif e["kind"] == "split":
            close(e["code"], m)
            for code, ent, reg in e["into"]:
                open_(code, ent, reg, m)
        elif e["kind"] == "merge":
            for c in e["codes"]:
                close(c, m)
            open_(e["new"], e["entity"], e["region"], m)
        elif e["kind"] == "absorb":
            close(e["code"], m)
        elif e["kind"] == "create":
            open_(e["code"], e["entity"], e["region"], m)
    return [r for r in rows if r["vf"] < r["vt"]]


def lookup(rows: List[Row], code: str, month: Optional[int]) -> List[Row]:
    """Rows a join would match. month=None: join on code alone (every version)."""
    return [r for r in rows if r["code"] == code and (month is None or r["vf"] <= month < r["vt"])]


def report(world: dict, strategy: str) -> Tuple[Report, int]:
    """Revenue by region and month under one rebuild strategy, plus revenue the join dropped."""
    today = rows_known(world)
    current = [r for r in today if r["vt"] == OPEN]
    known = {m: rows_known(world, m) for m in range(MONTHS)}
    out: Report = defaultdict(lambda: defaultdict(float))
    lost = 0
    for f in world["facts"]:
        m = f["month"]
        hits = {"overwrite": lambda: lookup(current, f["code"], None),
                "code_only": lambda: lookup(today, f["code"], None),
                "scd2_asof": lambda: lookup(today, f["code"], m),
                "bitemporal": lambda: lookup(known[m], f["code"], m)}[strategy]()
        if not hits:
            lost += f["amount"]
        for r in hits:
            out[m][r["region"]] += f["amount"]
    return out, lost


def published(world: dict) -> Report:
    """What was actually sent each month: the table as known on the run day, as-of the sale month."""
    return report(world, "bitemporal")[0]


def successors(world: dict) -> Dict[str, List[str]]:
    """Entity lineage the reference data can carry: retired entity -> today's entities."""
    nxt: Dict[str, List[str]] = {}
    ent = {c: v["entity"] for c, v in world["init"].items()}
    for e in world["events"]:
        if e["kind"] == "rename":
            ent[e["new"]] = ent[e["code"]]
        elif e["kind"] == "split":
            nxt[ent[e["code"]]] = [x[1] for x in e["into"]]
        elif e["kind"] == "merge":
            for c in e["codes"]:
                nxt[ent[c]] = [e["entity"]]
        elif e["kind"] == "absorb":
            nxt[ent[e["code"]]] = [ent[e["into"]]]
    return nxt


def restate(world: dict, split_rule: str = "post_share") -> Report:
    """As-is: every past sale moved to the region its territory sits in TODAY, following lineage.

    A split has no unique successor; split_rule allocates it: 'post_share' by each half's revenue in the six months
    after the split, 'even' 50/50, 'truth' by the sub-area the sale really came from (unknowable in practice).
    """
    today = rows_known(world)
    region_now = {r["entity"]: r["region"] for r in today if r["vt"] == OPEN}
    nxt = successors(world)
    first6: Dict[str, float] = defaultdict(float)
    for f in world["facts"]:
        for e in world["events"]:
            if e["kind"] == "split" and e["eff"] <= f["month"] < e["eff"] + 6 and f["code"] in [x[0] for x in e["into"]]:
                first6[f["area"]] += f["amount"]
    out: Report = defaultdict(lambda: defaultdict(float))
    for f in world["facts"]:
        m = f["month"]
        if split_rule == "truth":
            ents = {f["area"]: 1.0}
        else:
            r = lookup(today, f["code"], m)[0]
            kids = nxt.get(r["entity"], [r["entity"]])
            if len(kids) == 1:
                ents = {kids[0]: 1.0}
            elif split_rule == "even":
                ents = {k: 1 / len(kids) for k in kids}
            else:
                tot = sum(first6[k] for k in kids)
                ents = {k: first6[k] / tot for k in kids}
        for e, w in ents.items():
            e = nxt.get(e, [e])[0] if e not in region_now else e
            out[m][region_now[e]] += f["amount"] * w
    return out


def cells_equal(a: Report, b: Report, tol: float) -> Tuple[int, int]:
    """(region-month cells within tol relative, months where all five are)."""
    cells = months = 0
    for m in range(MONTHS):
        ok = [abs(a[m][r] - b[m][r]) <= tol * max(1.0, abs(b[m][r])) for r in REGIONS]
        cells += sum(ok)
        months += all(ok)
    return cells, months


def study(seed: int = SEED) -> dict:
    w = make_world(seed)
    ledger = sum(f["amount"] for f in w["facts"])
    pub = published(w)
    res: dict = {"facts": len(w["facts"]), "ledger": ledger, "events": {}, "asw": {}, "asis": {}}
    for e in w["events"]:
        res["events"][e["kind"]] = res["events"].get(e["kind"], 0) + 1
    res["backdated"] = sum(1 for e in w["events"] if e["rec"] > e["eff"])
    for s in STRATEGIES:
        rep, lost = report(w, s)
        total = sum(sum(v.values()) for v in rep.values())
        c, mo = cells_equal(rep, pub, 1e-9)
        res["asw"][s] = {"cells": c, "months": mo, "reported": total / ledger, "lost": lost / ledger}
    truth = restate(w, "truth")
    for s, rep in (("overwrite", report(w, "overwrite")[0]), ("scd2_asof", pub), ("even", restate(w, "even")),
                   ("post_share", restate(w, "post_share"))):
        c, mo = cells_equal(rep, truth, 0.01)
        res["asis"][s] = {"cells": c, "months": mo}
    split_codes = {e["code"] for e in w["events"] if e["kind"] == "split"}
    pre = [f for f in w["facts"] if f["code"] in split_codes]
    pre_rev = sum(f["amount"] for f in pre)
    res["split_pre_rev"] = pre_rev / ledger
    for rule in ("even", "post_share"):
        rep = restate(w, rule)
        moved = sum(abs(rep[m][r] - truth[m][r]) for m in range(MONTHS) for r in REGIONS) / 2
        res[f"misalloc_{rule}"] = moved / pre_rev
    ab = next(e for e in w["events"] if e["kind"] == "absorb")
    cr = next(e for e in w["events"] if e["kind"] == "create")
    old_q = sum(f["amount"] for f in w["facts"] if f["code"] == ab["code"] and f["month"] < ab["eff"])
    res["reuse"] = {"code": ab["code"], "retired": ab["eff"], "reused": cr["eff"], "rev": old_q,
                    "old_region": w["init"][ab["code"]]["region"], "new_region": cr["region"]}
    res["rows_bitemporal"] = len(rows_known(w)) + sum(1 for e in w["events"] if e["rec"] > e["eff"])
    res["rows_snapshots"] = sum(sum(1 for r in rows_known(w, m) if r["vf"] <= m < r["vt"]) for m in range(MONTHS))
    return res
