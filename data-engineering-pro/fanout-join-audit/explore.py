"""Revenue tripled after the join: which checks catch join fan-out BEFORE the aggregate.

A fictional shop (2,000 customers, 200 products, 20,000 orders; seeded) is reported as "revenue by customer tier
and product category". The report joins the orders to a customer dimension, a product dimension and a promo-code
bridge. Seven versions of that join are built - one correct, five with a common defect, two where two defects
cancel - and six pre-aggregate checks are run against every version.

Every number in evidence.txt comes from this file. Every customer, product and order is fictional.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

SEED = 212
N_CUST, N_PROD, N_ORD = 2000, 200, 20000
TIERS = ["standard", "gold", "platinum"]
CATS = ["home", "garden", "kitchen", "tools", "outdoor"]
SCENARIOS = ["clean", "scd2_no_filter", "dup_load", "promo_bridge", "late_inner", "rows_offset", "revenue_offset",
             "orphan_dups"]
CHECKS = ["row_count", "revenue_total", "declared_unique", "join_key_unique", "referenced_key_unique", "pk_preserved"]
TOL = 0.001  # revenue_total passes within 0.1%


def make_world(seed: int = SEED) -> Dict[str, pd.DataFrame]:
    """Orders, a type-2 customer history, a product dimension and a promo bridge."""
    rng = np.random.default_rng(seed)
    tier = rng.choice(3, N_CUST, p=[0.75, 0.2, 0.05])
    cust = pd.DataFrame({"customer_id": np.arange(N_CUST), "tier": [TIERS[t] for t in tier]})
    # Bigger tiers order more often and spend more per order.
    w = np.array([1.0, 2.0, 4.0])[tier]
    oc = rng.choice(N_CUST, N_ORD, p=w / w.sum())
    amount = np.round(rng.lognormal(3.6, 0.6, N_ORD) * np.array([1.0, 2.5, 6.0])[tier[oc]], 2)
    orders = pd.DataFrame({"order_id": np.arange(N_ORD), "customer_id": oc,
                           "product_id": rng.integers(0, N_PROD, N_ORD), "amount": amount})
    # Type-2 history: customers who were upgraded keep their old row, flagged not current.
    # Upgrades happen to big spenders, so the duplicated customers are the valuable ones.
    up = np.flatnonzero((tier > 0) & (rng.random(N_CUST) < 0.18))
    old = cust.loc[up].assign(tier=[TIERS[t - 1] for t in tier[up]], valid_from="2024-01-01", is_current=False)
    cur = cust.assign(valid_from=np.where(np.isin(cust.customer_id, up), "2025-06-01", "2024-01-01"), is_current=True)
    scd2 = pd.concat([old, cur], ignore_index=True).sort_values(["customer_id", "valid_from"], ignore_index=True)
    prod = pd.DataFrame({"product_id": np.arange(N_PROD), "category": [CATS[i % 5] for i in range(N_PROD)]})
    # Promo bridge: most orders have no code, a few have one, fewer have two.
    r = rng.random(N_ORD)
    one = np.flatnonzero(r < 0.075)
    two = np.flatnonzero(r < 0.015)
    promo = pd.DataFrame({"order_id": np.concatenate([one, two]),
                          "promo_code": ["SPRING"] * len(one) + ["LOYAL"] * len(two)})
    return {"orders": orders, "scd2": scd2, "prod": prod, "promo": promo, "upgraded": pd.Series(up)}


def build(world: Dict[str, pd.DataFrame], scenario: str) -> Dict[str, object]:
    """One version of the report query. Returns the joined rows and the dimensions it actually used."""
    o, scd2, prod = world["orders"], world["scd2"], world["prod"]
    cust = scd2[scd2.is_current]  # the correct customer dimension: current rows only
    how = "left"
    if scenario in ("scd2_no_filter", "rows_offset", "revenue_offset"):
        cust = scd2  # forgot `WHERE is_current`
    if scenario == "dup_load":
        prod = pd.concat([prod, prod.iloc[[3, 41, 77, 102, 150, 188]]], ignore_index=True)  # six rows loaded twice
    if scenario == "orphan_dups":
        used = set(o.product_id)
        prod = pd.concat([prod, pd.DataFrame({"product_id": [900, 900, 901, 901], "category": "home"})],
                         ignore_index=True)  # duplicated discontinued products that no order references
        assert not used & {900, 901}
    if scenario in ("late_inner", "rows_offset", "revenue_offset"):
        how = "inner"
        o = late_arrivals(world, scenario)
    j = o.merge(cust[["customer_id", "tier"]], on="customer_id", how=how)
    j = j.merge(prod, on="product_id", how=how)
    dims = {"customer": (cust, ["customer_id"]), "product": (prod, ["product_id"])}
    if scenario == "promo_bridge":
        j = j.merge(world["promo"], on="order_id", how="left")  # "show the promo code on the report"
        dims["promo"] = (world["promo"], ["order_id"])
    return {"joined": j, "orders": o, "dims": dims}


def late_arrivals(world: Dict[str, pd.DataFrame], scenario: str) -> pd.DataFrame:
    """Orders whose customer has not landed in the dimension yet get an unknown id; an inner join drops them.

    rows_offset and revenue_offset are tuned on purpose: late orders are drawn from customers the missing filter
    does NOT duplicate, until the dropped rows (or the dropped revenue) equal what the duplication adds. How often
    real defects cancel is not measured here; the point is that the check CAN be fooled."""
    o = world["orders"].copy()
    rng = np.random.default_rng(SEED + 1)
    dup = o.customer_id.isin(world["upgraded"]).to_numpy()
    if scenario == "late_inner":
        late = rng.choice(len(o), int(0.03 * len(o)), replace=False)
    else:
        pool = rng.permutation(np.flatnonzero(~dup))
        if scenario == "rows_offset":
            late = pool[: int(dup.sum())]
        else:
            target = o.amount.to_numpy()[dup].sum()
            k = int(np.searchsorted(np.cumsum(o.amount.to_numpy()[pool]), target))
            late = pool[:k]
    n = len(late)
    o.loc[late, "customer_id"] = -1 - np.arange(n)
    return o


def checks(world: Dict[str, pd.DataFrame], b: Dict[str, object]) -> Dict[str, bool]:
    """True = the check FIRES (raises an alarm). Every check runs before anything is aggregated."""
    j, src = b["joined"], world["orders"]
    o_used = b["orders"]
    out: Dict[str, bool] = {}
    out["row_count"] = bool(len(j) != len(src))
    out["revenue_total"] = bool(abs(j.amount.sum() / src.amount.sum() - 1) > TOL)
    # The test a team actually declares: SCD2 tables are unique on (key, valid_from), bridges on (order, code).
    declared = {"customer": ["customer_id", "valid_from"], "product": ["product_id"], "promo": ["order_id", "promo_code"]}
    fire = False
    for name, (d, _) in b["dims"].items():
        cols = [c for c in declared[name] if c in d.columns]
        fire |= bool(d.duplicated(cols).any())
    out["declared_unique"] = fire
    # pandas merge(validate="many_to_one"): the join key itself must be unique in the dimension.
    out["join_key_unique"] = any(bool(d.duplicated(k).any()) for d, k in b["dims"].values())
    # Same test restricted to keys the fact table actually references.
    out["referenced_key_unique"] = any(
        bool(d[d[k[0]].isin(o_used[k[0]])].duplicated(k).any()) for d, k in b["dims"].values())
    # The grain check: every source order appears exactly once in the joined rows.
    out["pk_preserved"] = bool(not (j.order_id.is_unique and len(j) == len(src)))
    return out


def damage(world: Dict[str, pd.DataFrame], b: Dict[str, object], truth: Dict[str, pd.Series]) -> Dict[str, float]:
    j, src = b["joined"], world["orders"]
    by_tier = j.groupby("tier").amount.sum().reindex(TIERS, fill_value=0)
    by_cat = j.groupby("category").amount.sum().reindex(CATS, fill_value=0)
    cells = pd.concat([by_tier / truth["tier"] - 1, by_cat / truth["cat"] - 1])
    worst = cells.abs().idxmax()
    return {"rows": len(j) / len(src) - 1, "revenue": j.amount.sum() / src.amount.sum() - 1,
            "worst_cell": str(worst), "worst": float(cells[worst])}


def run(world: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, object]]:
    clean = build(world, "clean")["joined"]
    truth = {"tier": clean.groupby("tier").amount.sum().reindex(TIERS),
             "cat": clean.groupby("category").amount.sum().reindex(CATS)}
    res: Dict[str, Dict[str, object]] = {}
    for s in SCENARIOS:
        b = build(world, s)
        res[s] = {**damage(world, b, truth), "fires": checks(world, b)}
    return res


def study() -> Dict[str, object]:
    w = make_world()
    r = run(w)
    up = w["orders"].customer_id.isin(w["upgraded"])
    return {"scenarios": r, "n_upgraded": int(len(w["upgraded"])), "scd2_rows": int(len(w["scd2"])),
            "upgraded_order_share": float(up.mean()),
            "upgraded_revenue_share": float(w["orders"].amount[up].sum() / w["orders"].amount.sum())}


def seeds(n: int = 30) -> Dict[str, object]:
    """Across worlds: how much does the row count understate the revenue error, per defect?"""
    ratio: Dict[str, List[float]] = {s: [] for s in ("scd2_no_filter", "dup_load", "promo_bridge")}
    offset_rev: List[float] = []
    fires = {s: {c: 0 for c in CHECKS} for s in SCENARIOS}
    for k in range(n):
        r = run(make_world(1000 + k))
        for s in SCENARIOS:
            for c in CHECKS:
                fires[s][c] += int(r[s]["fires"][c])
        for s in ratio:
            ratio[s].append(r[s]["revenue"] / r[s]["rows"])
        offset_rev.append(r["rows_offset"]["revenue"])
        assert not r["rows_offset"]["fires"]["row_count"] and not r["revenue_offset"]["fires"]["revenue_total"]
    return {"n": n, "ratio_median": {s: float(np.median(v)) for s, v in ratio.items()},
            "ratio_min": {s: float(np.min(v)) for s, v in ratio.items()},
            "offset_revenue_median": float(np.median(offset_rev)), "offset_revenue_min": float(np.min(offset_rev)),
            "fires": fires}
