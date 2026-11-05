"""The regions do not sum to the total: rolling postings up a cost-centre hierarchy extracted with defects.

A fictional organisation (5 regions, ~25 areas, ~150 cost centres) posts a year of costs. The hierarchy arrives as a
parent-child extract with the defects these extracts usually carry: cost centres that moved and kept their old row
too (two parents), cost centres whose parent was retired (orphans), one area whose parent field was keyed with one of
its own children (a cycle), postings made at area level, and postings to new cost centres not yet in the extract.

Five roll-up methods are scored against a truth we only know because the world is simulated: the region totals, the
grand total, and whether the check everyone runs ("do the regions sum to the total?") passes. Seeded Random(210).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

SEED = 210
ROOT = "TOTAL"
REGIONS = ["North", "South", "East", "West", "Central"]
UNASSIGNED = "Unassigned"
OTHER = "Other (plug)"
METHODS = ["naive_walk", "plug", "first_parent", "latest_parent", "validated"]
Row = Dict[str, object]


def make_world(seed: int = SEED) -> dict:
    """True tree, the defective extract of it, and the postings."""
    rng = random.Random(seed)
    parent: Dict[str, str] = {r: ROOT for r in REGIONS}
    areas: List[str] = []
    leaves: List[str] = []
    n = 100
    for ri, r in enumerate(REGIONS):
        for ai in range(rng.randint(4, 6)):
            a = f"A{ri + 1}{ai + 1}"
            parent[a] = r
            areas.append(a)
            for _ in range(rng.randint(3, 8)):
                leaves.append(f"CC{n}")
                parent[f"CC{n}"] = a
                n += 1
    extract: List[Row] = [{"child": c, "parent": p, "load": rng.randint(0, 40)} for c, p in parent.items()]
    by_child = {row["child"]: row for row in extract}
    pool = leaves[:]
    rng.shuffle(pool)
    moved, orphans = pool[:12], pool[12:18]
    for c in moved:  # moved to another area; the old row was never end-dated
        old = by_child[c]
        new = rng.choice([a for a in areas if a != old["parent"]])
        parent[c] = new
        extract.append({"child": c, "parent": new, "load": int(old["load"]) + rng.randint(30, 200)})
    for i, c in enumerate(orphans):  # parent area retired and removed from the extract; the child row was not
        by_child[c]["parent"] = f"Z{i + 1:02d}"
    cyc = rng.choice(areas)
    kid = next(c for c in leaves if parent[c] == cyc and c not in moved and c not in orphans)
    by_child[cyc]["parent"] = kid  # the area's parent field was keyed with one of its own children
    rng.shuffle(extract)
    new_ccs = []
    for _ in range(4):  # new cost centres that post before they reach the extract
        c = f"CC{n}"
        n += 1
        parent[c] = rng.choice(areas)
        new_ccs.append(c)
    posts: List[Tuple[str, int]] = []
    for c in leaves + new_ccs:
        size = rng.choice([1, 1, 2, 3])
        posts += [(c, size * rng.randint(400, 9000)) for _ in range(rng.randint(8, 16))]
    for a in areas:  # area-level adjustments (accruals, shared costs) posted above the leaves
        if rng.random() < 0.4:
            posts += [(a, rng.randint(2000, 25000)) for _ in range(rng.randint(1, 3))]
    return {"parent": parent, "extract": extract, "posts": posts, "moved": moved, "orphans": orphans,
            "cycle": (cyc, kid), "new_ccs": new_ccs, "areas": areas, "leaves": leaves}


def region_of(parent: Dict[str, str], node: str) -> Optional[str]:
    """Walk up a single-parent map; None if it never reaches a region (orphan, cycle, unknown)."""
    seen: Set[str] = set()
    while node not in REGIONS:
        if node in seen or node not in parent:
            return None
        seen.add(node)
        node = parent[node]
    return node


def amounts(posts: List[Tuple[str, int]]) -> Dict[str, float]:
    out: Dict[str, float] = defaultdict(float)
    for c, a in posts:
        out[c] += a
    return out


def truth(w: dict) -> Dict[str, float]:
    out: Dict[str, float] = defaultdict(float)
    for c, a in amounts(w["posts"]).items():
        out[region_of(w["parent"], c)] += a
    return out


def walk(kids: Dict[str, List[str]], amt: Dict[str, float], node: str, path: Set[str], leaves_only: bool) -> float:
    """Recursive descent along every edge, as a recursive CTE does. The path guard stops at a cycle; nothing stops a
    child listed under two parents from being summed twice."""
    if node in path:
        return 0.0
    ch = kids.get(node, [])
    own = amt.get(node, 0.0) if (not ch or not leaves_only) else 0.0
    return own + sum(walk(kids, amt, c, path | {node}, leaves_only) for c in ch)


def one_parent(extract: List[Row], how: str) -> Dict[str, str]:
    """Keep one row per child: the first in file order, or the latest load."""
    out: Dict[str, str] = {}
    best: Dict[str, int] = {}
    for row in extract:
        c, p, ld = str(row["child"]), str(row["parent"]), int(row["load"])
        if how == "first" and c in out:
            continue
        if how == "latest" and c in best and best[c] >= ld:
            continue
        out[c], best[c] = p, ld
    return out


def rollup(w: dict, method: str, amt: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Region totals plus whatever bucket the method uses, and the method's own grand total under ROOT."""
    amt = amounts(w["posts"]) if amt is None else amt
    ledger = sum(amt.values())
    out: Dict[str, float] = {}
    if method in ("naive_walk", "plug"):
        kids: Dict[str, List[str]] = defaultdict(list)
        for row in w["extract"]:
            kids[str(row["parent"])].append(str(row["child"]))
        for r in REGIONS:
            out[r] = walk(kids, amt, r, set(), leaves_only=True)
        out[ROOT] = sum(out[r] for r in REGIONS)  # the report total, built by the same walk
        if method == "plug":
            out[OTHER] = ledger - out[ROOT]
            out[ROOT] = ledger
        return out
    par = one_parent(w["extract"], "first" if method == "first_parent" else "latest")
    for r in REGIONS:
        out[r] = 0.0
    lost = 0.0
    for c, a in amt.items():
        r = region_of(par, c)
        if r is None:
            lost += a
        else:
            out[r] += a
    if method == "validated":
        out[UNASSIGNED] = lost
    out[ROOT] = sum(v for k, v in out.items() if k != ROOT)
    return out


def defects(w: dict) -> Dict[str, List[str]]:
    """What a validator finds in the extract before any roll-up: the checks `validated` quarantines on."""
    counts: Dict[str, int] = defaultdict(int)
    nodes = {str(r["child"]) for r in w["extract"]} | {ROOT}
    for row in w["extract"]:
        counts[str(row["child"])] += 1
    par = one_parent(w["extract"], "latest")
    orphan = sorted(c for c, p in par.items() if p not in nodes)
    cyc = sorted(c for c in par if region_of(par, c) is None and c not in orphan and par[c] in nodes
                 and _on_cycle(par, c))
    return {"multi_parent": sorted(c for c, k in counts.items() if k > 1), "orphan": orphan, "cycle": cyc,
            "unknown_code": sorted({c for c, _ in w["posts"]} - nodes)}


def _on_cycle(par: Dict[str, str], c: str) -> bool:
    seen: Set[str] = set()
    node = c
    while node in par and node not in seen:
        seen.add(node)
        node = par[node]
    return node in seen


def breakdown(w: dict, method: str) -> Dict[str, float]:
    """Where each posted dollar went: the true region, a wrong region, nowhere, or a second region on top.

    Net region error hides this: a dollar filed under the wrong region and a dollar dropped look the same once every
    region is short. So each node is rolled up on its own and its dollars are split four ways."""
    amt = amounts(w["posts"])
    out = {"right": 0.0, "misfiled": 0.0, "dropped": 0.0, "double": 0.0}
    for c, a in amt.items():
        rep = rollup(w, method, {c: 1.0})
        counted = sum(rep.get(r, 0.0) for r in REGIONS)
        right = min(1.0, rep.get(region_of(w["parent"], c), 0.0))
        out["right"] += a * right
        out["misfiled"] += a * (min(counted, 1.0) - right)
        out["dropped"] += a * (1.0 - min(counted, 1.0))
        out["double"] += a * max(0.0, counted - 1.0)
    return out


def score(rep: Dict[str, float], tru: Dict[str, float], ledger: float, bd: Dict[str, float]) -> dict:
    regions_sum = sum(v for k, v in rep.items() if k != ROOT)
    return {"sums_to_total": abs(regions_sum - rep[ROOT]) < 0.5, "total_is_ledger": abs(rep[ROOT] - ledger) < 0.5,
            "regions_right": sum(abs(rep.get(r, 0) - tru[r]) <= 0.005 * tru[r] for r in REGIONS),
            "total": rep[ROOT] / ledger, **{k: v / ledger for k, v in bd.items()},
            "wrong": (bd["misfiled"] + bd["dropped"] + bd["double"]) / ledger}


def study(seed: int = SEED) -> dict:
    w = make_world(seed)
    amt = amounts(w["posts"])
    ledger = sum(amt.values())
    tru = truth(w)
    res: dict = {"nodes": len(w["parent"]), "leaves": len(w["leaves"]), "areas": len(w["areas"]),
                 "rows": len(w["extract"]), "posts": len(w["posts"]), "ledger": ledger, "methods": {}}
    for m in METHODS:
        rep = rollup(w, m)
        res["methods"][m] = {**score(rep, tru, ledger, breakdown(w, m)), "buckets": {k: round(v) for k, v in rep.items()}}
    d = defects(w)
    res["defects"] = {k: len(v) for k, v in d.items()}
    cyc, _ = w["cycle"]
    share = lambda cs: sum(amt.get(c, 0.0) for c in cs) / ledger  # noqa: E731
    cyc_nodes = [c for c in w["parent"] if c == cyc or w["parent"].get(c) == cyc]
    res["share"] = {"moved": share(w["moved"]), "orphan": share(w["orphans"]), "cycle": share(cyc_nodes),
                    "unknown": share(w["new_ccs"]), "internal": share(w["areas"])}
    res["plug_net"] = abs(res["methods"]["plug"]["buckets"][OTHER]) / ledger
    first, latest = one_parent(w["extract"], "first"), one_parent(w["extract"], "latest")
    res["first_wrong_moves"] = sum(1 for c in w["moved"] if first[c] != w["parent"][c])
    res["latest_wrong_moves"] = sum(1 for c in w["moved"] if latest[c] != w["parent"][c])
    res["quarantined"] = res["methods"]["validated"]["buckets"][UNASSIGNED] / ledger
    return res


def seeds(n: int = 30) -> dict:
    """Across seeds: how often the plug looks small while the regions are badly wrong."""
    ratios, valid_err = [], []
    for s in range(1, n + 1):
        r = study(s)
        ratios.append(r["methods"]["plug"]["wrong"] / max(r["plug_net"], 1e-9))
        valid_err.append(r["methods"]["validated"]["wrong"])
    ratios.sort()
    return {"n": n, "gross_over_net_median": ratios[n // 2], "gross_over_net_min": ratios[0],
            "validated_err_min": min(valid_err), "validated_err_max": max(valid_err)}
