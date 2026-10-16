"""The training set is not the population: a per-segment representativeness audit, and what a miss costs.

A declared, constructed world: loan applicants from two groups (B 30%) and two regions (rural 40%), independent, so
the deployment population has four cells: A-urban 42%, A-rural 28%, B-urban 18%, B-rural 12%. Rural B applicants
are "thin file": their recorded credit history understates their real track record by `thin_file` sd, so a model
can only score them well if it has seen enough of them to learn the offset.

The training set was assembled with QUOTAS ON THE MARGINALS: 30% B and 40% rural, exactly as in the population. The
cells were not controlled, and rural B ended up at 0.5% (100 rows) instead of 12%. A one-attribute-at-a-time audit passes; the
cell audit does not. The model (scikit-learn gradient boosting) is then fitted on that set and scored on the
deployment population, cell by cell, on the share of good payers it rejects.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

SEEDS = [199, 200, 201]
N_TRAIN = 20_000
N_POP = 60_000
FLAG_LO, FLAG_HI = 0.8, 1.25  # a segment is flagged when train share / population share leaves this band
Z = 2.576  # 99% Wilson interval on the training share
CELLS = ["A-urban", "A-rural", "B-urban", "B-rural"]
CURVE = [0, 50, 100, 200, 400, 900, 1900]  # rural-B rows added on top of the quota sample's 100

WORLD: Dict[str, float] = {
    "share_b": 0.30,
    "share_rural": 0.40,
    "thin_file": 1.5,  # how far rural B's recorded history understates the real one (sd)
    "noise": 0.8,
}
# the training set's cell shares: marginals match the population (B 30%, rural 40%), the B-rural cell does not
QUOTA: Dict[str, float] = {"A-urban": 0.305, "A-rural": 0.395, "B-urban": 0.295, "B-rural": 0.005}
EMPTY: Dict[str, float] = {"A-urban": 0.30, "A-rural": 0.40, "B-urban": 0.30, "B-rural": 0.0}


def pop_shares(w: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    w = w or WORLD
    b, r = w["share_b"], w["share_rural"]
    return {"A-urban": (1 - b) * (1 - r), "A-rural": (1 - b) * r, "B-urban": b * (1 - r), "B-rural": b * r}


def draw_cell(
    cell: str, n: int, rng: np.random.Generator, w: Optional[Dict[str, float]] = None
) -> Dict[str, np.ndarray]:
    w = w or WORLD
    g, rural = int(cell[0] == "B"), int(cell.endswith("rural"))
    income = rng.normal(0, 1, n)
    record = rng.normal(0, 1, n)  # real track record
    history = record - w["thin_file"] * g * rural  # what the bureau file shows
    y = (0.9 * income + 0.9 * record + rng.normal(0, w["noise"], n) > -0.3).astype(int)
    return {
        "income": income,
        "history": history,
        "g": np.full(n, g),
        "rural": np.full(n, rural),
        "y": y,
        "cell": np.full(n, CELLS.index(cell)),
    }


def sample(
    shares: Dict[str, float],
    n: int,
    seed: int,
    w: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """Exact cell counts (round(share * n)) plus optional extra rows per cell."""
    rng = np.random.default_rng(seed)
    parts = [draw_cell(c, int(round(shares[c] * n)) + (extra or {}).get(c, 0), rng, w) for c in CELLS]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def wilson(k: int, n: int, z: float = Z) -> List[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def audit(train: Dict[str, np.ndarray], target: Dict[str, float]) -> Dict[str, Dict[str, object]]:
    """Representation ratio = training share / population share for every marginal AND every cell. Flag when the
    ratio leaves [0.8, 1.25] AND the 99% Wilson interval on the training share excludes the population share, so a
    small segment is not flagged on noise alone."""
    n = len(train["y"])
    segs = {
        "group B": (train["g"] == 1, target["B-urban"] + target["B-rural"]),
        "region rural": (train["rural"] == 1, target["A-rural"] + target["B-rural"]),
    }
    for i, c in enumerate(CELLS):
        segs[c] = (train["cell"] == i, target[c])
    out = {}
    for name, (m, ps) in segs.items():
        k = int(m.sum())
        lo, hi = wilson(k, n)
        ratio = (k / n) / ps
        out[name] = {
            "rows": k,
            "train_share": k / n,
            "pop_share": ps,
            "ratio": ratio,
            "flag": bool((ratio < FLAG_LO or ratio > FLAG_HI) and not lo <= ps <= hi),
            "kind": "cell" if name in CELLS else "marginal",
        }
    return out


def fit(train: Dict[str, np.ndarray], weight: Optional[np.ndarray] = None) -> HistGradientBoostingClassifier:
    X = np.column_stack([train[c] for c in ("income", "history", "g", "rural")])
    return HistGradientBoostingClassifier(random_state=0).fit(X, train["y"], sample_weight=weight)


def score(model: HistGradientBoostingClassifier, d: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Per cell and overall: accuracy, and the share of good payers rejected (approve when p >= 0.5)."""
    X = np.column_stack([d[c] for c in ("income", "history", "g", "rural")])
    approve = model.predict_proba(X)[:, 1] >= 0.5
    out = {}
    for name, m in [("overall", np.ones(len(d["y"]), bool))] + [
        (c, d["cell"] == i) for i, c in enumerate(CELLS)
    ]:
        y, a = d["y"][m], approve[m]
        out[name] = {"acc": float((y == a).mean()), "good_rejected": float((~a[y == 1]).mean())}
    return out


def cell_weights(train: Dict[str, np.ndarray], target: Dict[str, float]) -> np.ndarray:
    n = len(train["y"])
    w = np.empty(n)
    for i, c in enumerate(CELLS):
        m = train["cell"] == i
        if m.any():  # an empty cell has no rows to weight: reweighting cannot reach it
            w[m] = target[c] / (m.sum() / n)
    return w


def one_seed(
    seed: int,
    w: Optional[Dict[str, float]] = None,
    quota: Optional[Dict[str, float]] = None,
    n_train: int = N_TRAIN,
    n_pop: int = N_POP,
    with_curve: bool = True,
) -> Dict[str, object]:
    ps = pop_shares(w)
    q = quota or QUOTA
    pop = sample(ps, n_pop, seed + 10_000, w)
    quota_set = sample(q, n_train, seed, w)
    prop_set = sample(ps, n_train, seed + 20_000, w)
    held = sample(
        q, n_pop // 4, seed + 30_000, w
    )  # the team's own test split: same process as the training set
    m_quota = fit(quota_set)
    res = {
        "audit": audit(quota_set, ps),
        "audit_control": audit(prop_set, ps),
        "quota_on_own_test": score(m_quota, held),
        "quota": score(m_quota, pop),
        "proportional": score(fit(prop_set), pop),
        "reweighted": score(fit(quota_set, cell_weights(quota_set, ps)), pop),
    }
    empty = sample(EMPTY, n_train, seed, w)
    res["empty"] = score(fit(empty), pop)
    res["empty_reweighted"] = score(fit(empty, cell_weights(empty, ps)), pop)
    if with_curve:
        res["curve"] = [
            score(fit(sample(q, n_train, seed, w, {"B-rural": k})), pop)["B-rural"]["good_rejected"]
            for k in CURVE
        ]
    return res


def study(seeds: Optional[List[int]] = None, w: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    seeds = seeds or SEEDS
    per = [one_seed(s, w) for s in seeds]

    def agg(vals) -> Dict[str, float]:
        v = np.array(vals, float)
        return {"mean": float(v.mean()), "lo": float(v.min()), "hi": float(v.max())}

    def sc(key: str) -> Dict[str, Dict[str, Dict[str, float]]]:
        return {
            c: {m: agg([p[key][c][m] for p in per]) for m in ("acc", "good_rejected")} for c in per[0][key]
        }

    return {
        "seeds": seeds,
        "n_train": N_TRAIN,
        "n_pop": N_POP,
        "pop_shares": pop_shares(w),
        "quota_shares": QUOTA,
        "audit": per[0]["audit"],
        "audit_control": per[0]["audit_control"],
        "quota_on_own_test": sc("quota_on_own_test"),
        "quota": sc("quota"),
        "proportional": sc("proportional"),
        "reweighted": sc("reweighted"),
        "empty": sc("empty"),
        "empty_reweighted": sc("empty_reweighted"),
        "curve": [agg([p["curve"][i] for p in per]) for i in range(len(CURVE))],
    }


def parse_shares(text: str) -> Dict[str, float]:
    """`cell=share` overrides of QUOTA (e.g. "B-rural=0.06, A-rural=0.34"). Shares must sum to 1."""
    q = dict(QUOTA)
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "=" not in part:
            raise ValueError(f"expected cell=share, got {part!r}")
        k, v = [x.strip() for x in part.split("=", 1)]
        if k not in QUOTA:
            raise ValueError(f"unknown cell {k!r}; known: {', '.join(CELLS)}")
        try:
            q[k] = float(v)
        except ValueError:
            raise ValueError(f"{k} needs a number, got {v!r}") from None
    if any(v < 0 for v in q.values()) or abs(sum(q.values()) - 1) > 1e-6:
        raise ValueError(f"cell shares must be non-negative and sum to 1 (they sum to {sum(q.values()):.3f})")
    return q
