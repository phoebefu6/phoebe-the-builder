"""What does fairness cost? The measured trade-off curve, and how much of the price is the ruler.

A declared, constructed world: applicants from two groups, B 30% of them. Unlike `proxy-variable-finder`, the groups
are NOT identical: B's true skill is lower by `skill_gap` sd (a real difference in the outcome, e.g. from unequal
schooling upstream). On top of that, past decisions, which are the training labels, held B to a higher bar
(`label_bias`). Both labels use the same noise draw, so they differ ONLY by the bias.

A model is fitted on the biased history (scikit-learn gradient boosting). The company selects 30% of applicants. The
frontier is drawn by group-specific cut-offs: for each target selection ratio r = rate(B) / rate(A), select the top of
each group so the total stays 30% and the ratio is exactly r, then measure the hit rate (share of the selected who
are positive) against the biased history and against true merit. Within a group the top-k by score is the best
selection for a given k, so this curve is the frontier for this score.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

SEEDS = [198, 199, 200]
N = 60_000
SELECT = 0.30
RATIOS = [round(0.05 + 0.01 * i, 2) for i in range(121)]  # 0.05 .. 1.25
AUDIT_N = [2_000, 20_000]
AUDIT_REPS = 400
BIAS_SWEEP = [0.0, 0.35, 0.7, 1.05]
READ_AT = [0.25, 0.5, 0.8, 1.0, 1.25]

WORLD: Dict[str, float] = {
    "share_b": 0.30,  # share of applicants in group B
    "skill_gap": 0.25,  # REAL difference in true skill, B lower (sd)
    "label_bias": 0.7,  # extra bar past decisions held B to (sd)
    "label_noise": 0.6,
    "cert_skill": 0.8,  # certification reflects skill
    "exp_skill": 0.6,  # experience reflects skill ...
    "exp_gap": 0.4,  # ... and is depressed for B by access
    "zip_gap": 1.3,  # neighbourhood index, a pure proxy
}

MODELS: Dict[str, Dict[str, object]] = {
    "aware (group as a feature)": {"cols": ["cert", "exp", "zip"], "with_g": True},
    "unaware (group removed)": {"cols": ["cert", "exp", "zip"]},
    "drop zip": {"cols": ["cert", "exp"]},
    "reweighed (Kamiran-Calders)": {"cols": ["cert", "exp", "zip"], "reweigh": True},
}
FRONTIER_MODEL = "aware (group as a feature)"


def generate(n: int = N, seed: int = 198, world: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    w = world or WORLD
    rng = np.random.default_rng(seed)
    g = (rng.random(n) < w["share_b"]).astype(int)
    skill = rng.normal(0, 1, n) - w["skill_gap"] * g
    e = rng.normal(0, w["label_noise"], n)
    d = {"g": g, "skill": skill}
    d["cert"] = w["cert_skill"] * skill + rng.normal(0, 0.6, n)
    d["exp"] = w["exp_skill"] * skill - w["exp_gap"] * g + rng.normal(0, 0.7, n)
    d["zip"] = w["zip_gap"] * g + rng.normal(0, 1, n)
    d["y_true"] = (skill + e > 0.5).astype(int)
    d["y_hist"] = (skill - w["label_bias"] * g + e > 0.5).astype(int)
    return d


def fit_score(d: Dict[str, np.ndarray], spec: Dict[str, object]) -> np.ndarray:
    """Train on the first half (biased labels); return scores for the second half."""
    n = len(d["g"])
    tr, te = slice(0, n // 2), slice(n // 2, n)
    X = np.column_stack([d[c] for c in spec["cols"]] + ([d["g"]] if spec.get("with_g") else []))
    y, g = d["y_hist"][tr], d["g"][tr]
    wt = None
    if spec.get("reweigh"):
        # weight each (group, label) cell so group and label are independent in the training data
        wt = np.empty(len(y))
        for gv in (0, 1):
            for yv in (0, 1):
                m = (g == gv) & (y == yv)
                wt[m] = (g == gv).mean() * (y == yv).mean() / max(m.mean(), 1e-12)
    m = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y, sample_weight=wt)
    return m.predict_proba(X[te])[:, 1]


def _top(s: np.ndarray, k: int) -> np.ndarray:
    sel = np.zeros(len(s), bool)
    if k > 0:
        sel[np.argsort(-s, kind="stable")[:k]] = True
    return sel


def select_at_ratio(s: np.ndarray, g: np.ndarray, r: float, rate: float = SELECT) -> np.ndarray:
    """Group-specific cut-offs: total selected = rate * n, rate(B) / rate(A) = r (as near as whole people allow)."""
    na, nb = int((g == 0).sum()), int((g == 1).sum())
    total = int(round(rate * len(s)))
    sa = total / (na + nb * r)
    kb = min(nb, int(round(sa * r * nb)))
    ka = total - kb
    sel = np.zeros(len(s), bool)
    sel[g == 0] = _top(s[g == 0], ka)
    sel[g == 1] = _top(s[g == 1], kb)
    return sel


def outcome(sel: np.ndarray, g: np.ndarray, yh: np.ndarray, yt: np.ndarray) -> Dict[str, float]:
    sa, sb = sel[g == 0].mean(), sel[g == 1].mean()
    return {"ratio": float(sb / sa), "hit_hist": float(yh[sel].mean()), "hit_true": float(yt[sel].mean())}


def frontier(s: np.ndarray, g: np.ndarray, yh: np.ndarray, yt: np.ndarray) -> Dict[str, List[float]]:
    rows = [outcome(select_at_ratio(s, g, r), g, yh, yt) for r in RATIOS]
    return {"ratio": [x["ratio"] for x in rows], "hit_hist": [x["hit_hist"] for x in rows],
            "hit_true": [x["hit_true"] for x in rows]}


def single_threshold(s: np.ndarray, g: np.ndarray, yh: np.ndarray, yt: np.ndarray) -> Dict[str, float]:
    return outcome(_top(s, int(round(SELECT * len(s)))), g, yh, yt)


def at(fr: Dict[str, List[float]], r: float, key: str) -> float:
    return float(fr[key][RATIOS.index(round(r, 2))])


def peak(fr: Dict[str, List[float]], key: str) -> Dict[str, float]:
    i = int(np.argmax(fr[key]))
    return {"ratio": RATIOS[i], "hit": float(fr[key][i])}


def prices(fr: Dict[str, List[float]], start: float) -> Dict[str, float]:
    """What moving from ratio `start` (the unconstrained choice) to 0.8 and to 1.0 does to each hit rate, in points.
    Positive = the hit rate fell (a cost); negative = it rose."""
    out = {}
    for tgt, name in ((0.8, "fourfifths"), (1.0, "parity")):
        for key in ("hit_hist", "hit_true"):
            out[f"{name}_{key}"] = 100 * (at(fr, start, key) - at(fr, tgt, key))
    return out


def audit(s: np.ndarray, g: np.ndarray, yh: np.ndarray, sizes: List[int], reps: int, seed: int) -> Dict[str, Dict[str, float]]:
    """The price of four-fifths against history as a team would ESTIMATE it on a validation sample of a given size:
    best single-threshold hit rate minus the hit rate at ratio 0.8, both measured on the same sample."""
    rng = np.random.default_rng(seed)
    out = {}
    for m in sizes:
        v = []
        for _ in range(reps):
            i = rng.choice(len(s), m, replace=False)
            ss, gg, yy = s[i], g[i], yh[i]
            base = yy[_top(ss, int(round(SELECT * m)))].mean()
            v.append(100 * (base - yy[select_at_ratio(ss, gg, 0.8)].mean()))
        v = np.array(v)
        out[str(m)] = {"p05": float(np.percentile(v, 5)), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                       "share_le_0": float((v <= 0).mean())}
    return out


def one_seed(seed: int, n: int = N, world: Optional[Dict[str, float]] = None, with_audit: bool = True) -> Dict[str, object]:
    d = generate(n, seed, world)
    te = slice(n // 2, n)
    g, yh, yt = d["g"][te], d["y_hist"][te], d["y_true"][te]
    scores = {k: fit_score(d, v) for k, v in MODELS.items()}
    s = scores[FRONTIER_MODEL]
    fr = frontier(s, g, yh, yt)
    points = {k: single_threshold(v, g, yh, yt) for k, v in scores.items()}
    oracle = outcome(_top(d["skill"][te], int(round(SELECT * (n - n // 2)))), g, yh, yt)
    start = round(points[FRONTIER_MODEL]["ratio"], 2)
    res = {"frontier": fr, "points": points, "peak_hist": peak(fr, "hit_hist"), "peak_true": peak(fr, "hit_true"),
           "start": start, "prices": prices(fr, start), "oracle_ratio": oracle["ratio"],
           "gap_below": {k: 100 * (at(fr, round(p["ratio"], 2), "hit_hist") - p["hit_hist"]) for k, p in points.items()
                         if 0.05 <= p["ratio"] <= 1.25}}
    if with_audit:
        res["audit"] = audit(s, g, yh, AUDIT_N, AUDIT_REPS, seed)
    return res


def bias_sweep(seed: int, n: int = N, world: Optional[Dict[str, float]] = None) -> List[Dict[str, float]]:
    """Price of parity measured both ways as the history gets more biased. label_bias = 0 is the control: no
    yardstick error, so whatever parity costs there is a real cost of the true skill gap."""
    w0 = dict(world or WORLD)
    rows = []
    for b in BIAS_SWEEP:
        w = dict(w0, label_bias=b)
        r = one_seed(seed, n, w, with_audit=False)
        rows.append({"label_bias": b, "start": r["start"], "peak_true": r["peak_true"]["ratio"], **r["prices"],
                     **{f"true_at_{x}": 100 * at(r["frontier"], x, "hit_true") for x in READ_AT}})
    return rows


def study(seeds: Optional[List[int]] = None, n: int = N, world: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    seeds = seeds or SEEDS
    per = [one_seed(sd, n, world) for sd in seeds]
    sweeps = [bias_sweep(sd, n, world) for sd in seeds]

    def agg(vals) -> Dict[str, float]:
        v = np.array(vals, float)
        return {"mean": float(v.mean()), "lo": float(v.min()), "hi": float(v.max())}

    fr = {k: list(np.mean([p["frontier"][k] for p in per], axis=0)) for k in ("hit_hist", "hit_true")}
    fr["ratio"] = RATIOS
    return {
        "seeds": seeds, "n": n,
        "frontier": fr,
        "points": {k: {m: agg([p["points"][k][m] for p in per]) for m in ("ratio", "hit_hist", "hit_true")} for k in MODELS},
        "gap_below": {k: agg([p["gap_below"][k] for p in per]) for k in per[0]["gap_below"]},
        "peak_hist": {m: agg([p["peak_hist"][m] for p in per]) for m in ("ratio", "hit")},
        "peak_true": {m: agg([p["peak_true"][m] for p in per]) for m in ("ratio", "hit")},
        "start": agg([p["start"] for p in per]),
        "oracle_ratio": agg([p["oracle_ratio"] for p in per]),
        "prices": {k: agg([p["prices"][k] for p in per]) for k in per[0]["prices"]},
        "audit": {m: {q: agg([p["audit"][m][q] for p in per]) for q in per[0]["audit"][m]} for m in per[0]["audit"]},
        "sweep": [{k: agg([s[i][k] for s in sweeps]) for k in sweeps[0][i]} for i in range(len(BIAS_SWEEP))],
    }


def parse_world(text: str) -> Dict[str, float]:
    """`key=value` overrides of WORLD. Unknown keys and bad numbers raise ValueError."""
    w = dict(WORLD)
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "=" not in part:
            raise ValueError(f"expected key=value, got {part!r}")
        k, v = [x.strip() for x in part.split("=", 1)]
        if k not in WORLD:
            raise ValueError(f"unknown parameter {k!r}; known: {', '.join(WORLD)}")
        try:
            w[k] = float(v)
        except ValueError:
            raise ValueError(f"{k} needs a number, got {v!r}") from None
    if not 0 < w["share_b"] < 1:
        raise ValueError("share_b must be in (0, 1)")
    return w
