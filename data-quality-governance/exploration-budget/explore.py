"""What does a random-approval slice cost, and what does it buy? Pricing the fix that selective-labels found.

Day 200 (`selective-labels`) showed that when past approvals used an unrecorded signal, the labeled rows rank a
worse credit model first, IPW cannot fix it, and a 5% random-approval slice can. Risk then asks the next question:
what does that slice cost in bad loans, and what is the smallest one that still ranks the models right?

The declared world is copied from `selective-labels` (each build must run standalone): bureau score `x` (recorded),
interviewer read `u` (never recorded), group B 30%; officers ("production") approved 40% on x + u with a 0.6 sd
higher bar for B. Two challengers are fitted on a training period: the oracle (all outcomes, truly better) and the
label-trained model (what a team can fit). On a fresh evaluation period, two ways to buy unbiased labels:

  uniform  - approve a random share `e` of ALL applicants regardless of score (Day 200's design)
  targeted - production's own approvals are already labeled for free; approve a random share `q` of only the rows
             a challenger would approve and production refused, and weight those by 1/q (stratified sampling)

Each design point is replayed REPS times (fresh random slices, same applicants). Reported per point: extra loans
made, their net dollars (declared unit economics), the chance the slice ranks the oracle above the label-trained
model, and the bias of each challenger's repay estimate. Then the one thing targeted cannot do: a fairness audit,
whose denominator includes good payers that neither production nor any challenger would ever approve.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

SEEDS = [200, 201, 202]
N = 40_000  # applicants per period
APPROVE = 0.40
REPS = 400  # replayed random slices per design point
UNIFORM = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]  # share of all applicants approved at random
TARGETED = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]  # share of challenger-only rows approved at random
GAIN, LOSS = 300.0, 1_500.0  # declared: margin on a repaid loan, loss on a default (break-even repay 83.3%)
PER = 10_000  # costs reported per 10,000 applicants
Z = 1.96

WORLD: Dict[str, float] = {"share_b": 0.30, "sel_u": 1.0, "bar_b": 0.6, "officer_noise": 0.5, "noise": 0.8}


def draw(n: int, rng: np.random.Generator, w: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    """One period: applicants, true outcome, and production's decision (label observed iff approved)."""
    w = w or WORLD
    g = (rng.random(n) < w["share_b"]).astype(int)
    x = rng.normal(0, 1, n)
    u = rng.normal(0, 1, n)
    y = (0.8 * x + 0.8 * u + rng.normal(0, w["noise"], n) > -0.4).astype(int)  # 1 = repays
    s = x + w["sel_u"] * u - w["bar_b"] * g + rng.normal(0, w["officer_noise"], n)
    return {"x": x, "u": u, "g": g, "y": y, "prod": s >= np.quantile(s, 1 - APPROVE)}


def fit(d: Dict[str, np.ndarray], mask: Optional[np.ndarray] = None) -> HistGradientBoostingClassifier:
    m = np.ones(len(d["y"]), bool) if mask is None else mask
    return HistGradientBoostingClassifier(random_state=0).fit(np.column_stack([d["x"], d["g"]])[m], d["y"][m])


def decide(model: HistGradientBoostingClassifier, d: Dict[str, np.ndarray]) -> np.ndarray:
    p = model.predict_proba(np.column_stack([d["x"], d["g"]]))[:, 1]
    return p >= np.quantile(p, 1 - APPROVE)


def net(y: np.ndarray) -> np.ndarray:
    return np.where(y == 1, GAIN, -LOSS)


def uniform_est(y: np.ndarray, a: np.ndarray, sl: np.ndarray) -> np.ndarray:
    """Repay among the challenger's approvals, read on a random slice. sl: (reps, n) bool."""
    k = (sl & a).sum(1)
    return np.where(k > 0, (sl & a & (y == 1)).sum(1) / np.maximum(k, 1), np.nan)


def targeted_est(y: np.ndarray, a: np.ndarray, prod: np.ndarray, sl: np.ndarray, q: float) -> np.ndarray:
    """Overlap with production is fully labeled; challenger-only rows are sampled at q and weighted 1/q."""
    free = float((a & prod & (y == 1)).sum())
    sampled = (sl & a & ~prod & (y == 1)).sum(1) / q
    return (free + sampled) / a.sum()


def tpr_b(y: np.ndarray, a: np.ndarray, g: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """Share of group B's good payers the challenger approves, from weighted labeled rows. wt: (reps, n)."""
    good = (y == 1) & (g == 1)
    return (wt * (good & a)).sum(1) / np.maximum((wt * good).sum(1), 1e-9)


def one_seed(seed: int, n: int = N, reps: int = REPS) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    train, ev = draw(n, rng), draw(n, rng)
    a_o, a_l = decide(fit(train), ev), decide(fit(train, train["prod"]), ev)
    y, g, prod = ev["y"], ev["g"], ev["prod"]
    truth = {"oracle": float(y[a_o].mean()), "label": float(y[a_l].mean())}
    good_b = (y == 1) & (g == 1)
    out: Dict[str, object] = {
        "truth": truth,
        "true_tpr_b": float(a_o[good_b].mean()),
        "rejected_repay": float(y[~prod].mean()),
        "challenger_only_repay": float(y[(a_o | a_l) & ~prod].mean()),
        "unreachable_good_b": float((good_b & ~(prod | a_o | a_l)).sum() / good_b.sum()),
        "agree": float((a_o & a_l).sum() / a_o.sum()),
    }
    r = np.random.default_rng(seed + 90_000)

    def summarise(
        eo: np.ndarray, el: np.ndarray, extra: np.ndarray, cost: float, tb: np.ndarray
    ) -> Dict[str, float]:
        d = eo - el
        return {
            "extra_loans": float(extra.mean()) * PER / n,
            "net_cost": cost * PER / n,
            "p_right": float(np.mean(d > 0)),
            "diff_ci": float(Z * np.nanstd(d)),
            "bias_oracle": float(np.nanmean(eo) - truth["oracle"]),
            "bias_label": float(np.nanmean(el) - truth["label"]),
            "tpr_b_bias": float(np.nanmean(tb) - out["true_tpr_b"]),
        }

    for e in UNIFORM:
        sl = r.random((reps, n)) < e
        extra = (sl & ~prod).sum(1)
        cost = -e * float(net(y[~prod]).sum())
        out[f"uniform_{e:g}"] = summarise(
            uniform_est(y, a_o, sl), uniform_est(y, a_l, sl), extra, cost, tpr_b(y, a_o, g, sl.astype(float))
        )
    pool = (a_o | a_l) & ~prod  # rows any challenger wants that production refused
    for q in TARGETED:
        sl = (r.random((reps, n)) < q) & pool
        wt = np.where(prod, 1.0, 0.0) + sl / q  # labeled weight: 1 for production rows, 1/q for sampled
        cost = -q * float(net(y[pool]).sum())
        out[f"targeted_{q:g}"] = summarise(
            targeted_est(y, a_o, prod, sl, q),
            targeted_est(y, a_l, prod, sl, q),
            sl.sum(1),
            cost,
            tpr_b(y, a_o, g, wt),
        )
    return out


def study(seeds: Optional[List[int]] = None, n: int = N, reps: int = REPS) -> Dict[str, object]:
    per = [one_seed(s, n, reps) for s in seeds or SEEDS]

    def tree(get) -> object:
        first = get(per[0])
        if isinstance(first, dict):
            return {k: tree(lambda p, k=k, get=get: get(p)[k]) for k in first}
        v = np.array([get(p) for p in per], float)
        return {"mean": float(v.mean()), "lo": float(v.min()), "hi": float(v.max())}

    return {"seeds": seeds or SEEDS, "n": n, "reps": reps, **tree(lambda p: p)}


def cheapest(r: Dict[str, object], design: str, target: float = 0.95) -> Optional[Dict[str, float]]:
    """Cheapest point of a design whose WORST seed still ranks the models right at `target` probability."""
    grid = UNIFORM if design == "uniform" else TARGETED
    for s in grid:
        p = r[f"{design}_{s:g}"]
        if p["p_right"]["lo"] >= target:
            return {"share": s, "net_cost": p["net_cost"]["mean"], "extra_loans": p["extra_loans"]["mean"]}
    return None
