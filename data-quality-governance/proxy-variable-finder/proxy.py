"""We removed the attribute and it still discriminates. Finding the features that stand in for it.

A declared, constructed world: applicants from two groups have the SAME distribution of true skill. Past hiring
decisions, which are the training labels, held group B to a higher bar. Five features reach the model:

- `zip`      neighbourhood index - no information about skill, strongly tied to group (a classic proxy)
- `exp`      years of experience - carries skill, and is depressed for B by historical access (proxy AND signal)
- `cert`     certification score - carries skill, unrelated to group
- `commute`  and `shift` - each unrelated to group on its own; TOGETHER they identify it (a joint proxy that a
             one-feature-at-a-time screen cannot see)

Every model is real (scikit-learn gradient boosting), trained on the biased historical labels, and judged two ways:
against those labels (what the team can measure) and against true skill (what they actually want). Selection is the
top 30% of scores. Results are averaged over several seeds, and the spread is reported.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEEDS = [197, 198, 199]
N = 80_000
SELECT = 0.30
FEATURES = ["zip", "exp", "cert", "commute", "shift"]

WORLD: Dict[str, float] = {
    "share_b": 0.30,  # share of applicants in group B
    "zip_gap": 1.4,  # zip index shift for B (no skill content)
    "exp_skill": 0.7,  # how much experience reflects skill
    "exp_gap": 0.5,  # experience lost by B to historical access
    "cert_skill": 0.6,  # how much the certification reflects skill
    "pair_link": 0.85,  # how strongly commute x shift identifies group
    "label_bias": 0.9,  # extra bar past decisions held B to
    "label_noise": 0.6,
}


def generate(n: int = N, seed: int = 197, world: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    w = world or WORLD
    rng = np.random.default_rng(seed)
    g = (rng.random(n) < w["share_b"]).astype(int)
    skill = rng.normal(0, 1, n)
    d = {"g": g, "skill": skill}
    d["zip"] = w["zip_gap"] * g + rng.normal(0, 1, n)
    d["exp"] = w["exp_skill"] * skill - w["exp_gap"] * g + rng.normal(0, 0.7, n)
    d["cert"] = w["cert_skill"] * skill + rng.normal(0, 0.8, n)
    c = rng.normal(0, 1, n)
    s = np.abs(rng.normal(0, 1, n))
    linked = rng.random(n) < w["pair_link"]
    # group B: shift has the same sign as commute; group A: opposite sign. Unlinked rows get a random sign.
    sign = np.where(linked, np.where(g == 1, np.sign(c), -np.sign(c)), rng.choice([-1.0, 1.0], n))
    d["commute"], d["shift"] = c, sign * s
    d["y_hist"] = (skill - w["label_bias"] * g + rng.normal(0, w["label_noise"], n) > 0.4).astype(int)
    d["y_true"] = (skill + rng.normal(0, w["label_noise"], n) > 0.4).astype(int)
    return d


def _auc_either(y: np.ndarray, s: np.ndarray) -> float:
    """How well s separates y, direction ignored (a proxy can point either way)."""
    a = roc_auc_score(y, s)
    return float(max(a, 1 - a))


def _fit_auc(d: Dict[str, np.ndarray], cols: List[str], target: str = "g") -> float:
    n = len(d["g"])
    tr, te = slice(0, n // 2), slice(n // 2, n)
    X = np.column_stack([d[c] for c in cols])
    m = HistGradientBoostingClassifier(random_state=0).fit(X[tr], d[target][tr])
    return _auc_either(d[target][te], m.predict_proba(X[te])[:, 1])


def screen(d: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Three proxy screens, from the one most teams run to the one that sees joint proxies."""
    n = len(d["g"])
    te = slice(n // 2, n)
    single = {f: _auc_either(d["g"][te], d[f][te]) for f in FEATURES}
    pairs = {f"{a}+{b}": _fit_auc(d, [a, b]) for a, b in combinations(FEATURES, 2)}
    joint = _fit_auc(d, FEATURES)
    drop_one = {f: joint - _fit_auc(d, [x for x in FEATURES if x != f]) for f in FEATURES}
    return {"single": single, "pairs": pairs, "joint": joint, "drop_one": drop_one}


def _repair(d: Dict[str, np.ndarray], cols: List[str]) -> Dict[str, np.ndarray]:
    """Remove the part of each column that the group predicts in the MEAN (subtract the group mean). This is the
    linear repair most toolkits offer; it cannot touch a proxy that lives in an interaction."""
    out = dict(d)
    for c in cols:
        x = d[c].astype(float).copy()
        for gv in (0, 1):
            m = d["g"] == gv
            x[m] = x[m] - x[m].mean()
        out[c] = x
    return out


MODELS: Dict[str, Dict[str, object]] = {
    "aware (group as a feature)": {"cols": FEATURES, "with_g": True},
    "unaware (group removed)": {"cols": FEATURES},
    "drop zip": {"cols": ["exp", "cert", "commute", "shift"]},
    "drop zip + the commute/shift pair": {"cols": ["exp", "cert"]},
    "linear repair of every feature": {"cols": FEATURES, "repair": FEATURES},
    "drop zip + pair, repair exp": {"cols": ["exp", "cert"], "repair": ["exp"]},
    "cert only": {"cols": ["cert"]},
}


def evaluate(d: Dict[str, np.ndarray], spec: Dict[str, object]) -> Dict[str, float]:
    n = len(d["g"])
    tr, te = slice(0, n // 2), slice(n // 2, n)
    dd = _repair(d, spec.get("repair", [])) if spec.get("repair") else d
    cols = [dd[c] for c in spec["cols"]] + ([dd["g"]] if spec.get("with_g") else [])
    X = np.column_stack(cols)
    m = HistGradientBoostingClassifier(random_state=0).fit(X[tr], d["y_hist"][tr])
    s = m.predict_proba(X[te])[:, 1]
    g, yt, yh = d["g"][te], d["y_true"][te], d["y_hist"][te]
    sel = s >= np.quantile(s, 1 - SELECT)
    sa, sb = sel[g == 0].mean(), sel[g == 1].mean()
    qa, qb = sel[(g == 0) & (yt == 1)].mean(), sel[(g == 1) & (yt == 1)].mean()
    return {"sel_a": float(sa), "sel_b": float(sb), "ratio": float(sb / sa), "auc_hist": float(roc_auc_score(yh, s)),
            "auc_true": float(roc_auc_score(yt, s)), "group_from_score": _auc_either(g, s),
            "qual_sel_a": float(qa), "qual_sel_b": float(qb)}


def study(seeds: Optional[List[int]] = None, n: int = N, world: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """Every model and the screen on each seed; returns the mean and the min-max spread of every number."""
    seeds = seeds or SEEDS
    per_seed = []
    for sd in seeds:
        d = generate(n, sd, world)
        per_seed.append({"screen": screen(d), "models": {k: evaluate(d, v) for k, v in MODELS.items()}})

    def agg(get) -> Dict[str, float]:
        v = np.array([get(r) for r in per_seed])
        return {"mean": float(v.mean()), "lo": float(v.min()), "hi": float(v.max())}

    models = {k: {m: agg(lambda r, k=k, m=m: r["models"][k][m]) for m in per_seed[0]["models"][k]} for k in MODELS}
    sc0 = per_seed[0]["screen"]
    screen_agg = {
        "single": {f: agg(lambda r, f=f: r["screen"]["single"][f]) for f in sc0["single"]},
        "pairs": {p: agg(lambda r, p=p: r["screen"]["pairs"][p]) for p in sc0["pairs"]},
        "joint": agg(lambda r: r["screen"]["joint"]),
        "drop_one": {f: agg(lambda r, f=f: r["screen"]["drop_one"][f]) for f in sc0["drop_one"]},
    }
    base = generate(n, seeds[0], world)
    truth = {"skill_gap": float(base["skill"][base["g"] == 0].mean() - base["skill"][base["g"] == 1].mean()),
             "hist_rate_a": float(base["y_hist"][base["g"] == 0].mean()),
             "hist_rate_b": float(base["y_hist"][base["g"] == 1].mean()),
             "true_rate_a": float(base["y_true"][base["g"] == 0].mean()),
             "true_rate_b": float(base["y_true"][base["g"] == 1].mean())}
    return {"models": models, "screen": screen_agg, "truth": truth, "seeds": seeds, "n": n}


def flagged(screen_res: Dict[str, object], threshold: float = 0.60) -> Dict[str, List[str]]:
    """What each screen flags at a common bar: AUC for predicting the group above `threshold`."""
    single = [f for f, v in screen_res["single"].items() if v["mean"] > threshold]
    pairs = [p for p, v in screen_res["pairs"].items()
             if v["mean"] > threshold and not any(f in single for f in p.split("+"))]
    return {"single": single, "pairs_hidden_from_single": pairs}


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
    if not 0 < w["share_b"] < 1 or not 0 <= w["pair_link"] <= 1:
        raise ValueError("share_b must be in (0, 1) and pair_link in [0, 1]")
    return w
