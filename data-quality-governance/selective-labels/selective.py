"""We only know outcomes for the people we approved: selective labels, and what they do to evaluation and audit.

A declared, constructed world: loan applicants with a bureau score `x` (recorded), an interviewer's read `u`
(never recorded) and a group flag (B 30%). Repayment depends on x and u equally. Past loan officers approved 40%
of applicants on x AND u, and held group B to a higher bar (`bar_b` sd). Repayment is observed only for the people
they approved: that is the label set every later model is trained, evaluated and audited on.

A new model (scikit-learn gradient boosting on x and group; it cannot see u) approves the top 40%. Its repay rate
among approved and its per-group true positive rate (good payers approved) are measured four ways on a history
period and compared to the truth, which only a simulation can see:
  labeled     - the approved rows, as a dashboard would
  rejected=bad - classic reject inference: every unlabeled applicant counted as a default
  ipw         - inverse propensity weighting, propensity of past approval fitted on x and group
  explore     - a random slice of applicants approved regardless of score (the only unbiased label set)
The control world sets `sel_u` = 0 (officers used only recorded data): there IPW must recover the truth.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

SEEDS = [200, 201, 202]
N = 40_000  # applicants per history period (one to train on, one to evaluate on)
APPROVE = 0.40  # capacity: the share approved, then and now
EXPLORE = [0.01, 0.02, 0.05, 0.10]  # share of applicants approved at random in the evaluation period
Z = 1.96

WORLD: Dict[str, float] = {
    "share_b": 0.30,
    "sel_u": 1.0,  # how much officers leaned on the unrecorded read (0 = recorded data only)
    "bar_b": 0.6,  # the extra bar group B had to clear (sd)
    "officer_noise": 0.5,
    "noise": 0.8,
}
CONTROL: Dict[str, float] = dict(WORLD, sel_u=0.0)


def draw(n: int, rng: np.random.Generator, w: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    """One history period: applicants, the true outcome, and the officers' decision (label observed iff approved)."""
    w = w or WORLD
    g = (rng.random(n) < w["share_b"]).astype(int)
    x = rng.normal(0, 1, n)
    u = rng.normal(0, 1, n)
    y = (0.8 * x + 0.8 * u + rng.normal(0, w["noise"], n) > -0.4).astype(int)  # 1 = repays
    s = x + w["sel_u"] * u - w["bar_b"] * g + rng.normal(0, w["officer_noise"], n)
    approved = s >= np.quantile(s, 1 - APPROVE)
    return {"x": x, "u": u, "g": g, "y": y, "approved": approved}


def features(d: Dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([d["x"], d["g"]])


def fit(d: Dict[str, np.ndarray], mask: Optional[np.ndarray] = None) -> HistGradientBoostingClassifier:
    m = np.ones(len(d["y"]), bool) if mask is None else mask
    return HistGradientBoostingClassifier(random_state=0).fit(features(d)[m], d["y"][m])


def decide(model: HistGradientBoostingClassifier, d: Dict[str, np.ndarray]) -> np.ndarray:
    p = model.predict_proba(features(d))[:, 1]
    return p >= np.quantile(p, 1 - APPROVE)


def propensity(d: Dict[str, np.ndarray]) -> np.ndarray:
    """P(past approval | x, g), fitted on the recorded features: all an analyst has."""
    clf = HistGradientBoostingClassifier(random_state=0).fit(features(d), d["approved"].astype(int))
    return np.clip(clf.predict_proba(features(d))[:, 1], 0.02, 1.0)


def metrics(y: np.ndarray, a: np.ndarray, g: np.ndarray, wt: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Repay rate among the model's approvals, and true positive rate (good payers approved) per group."""
    wt = np.ones(len(y)) if wt is None else wt

    def rate(num: np.ndarray, den: np.ndarray) -> float:
        s = float((wt * den).sum())
        return float((wt * num * den).sum() / s) if s > 0 else float("nan")

    out = {"repay": rate(y, a)}
    for k, name in ((0, "A"), (1, "B")):
        out[f"tpr_{name}"] = rate(a, (y == 1) & (g == k))
    out["tpr_gap"] = out["tpr_A"] - out["tpr_B"]
    return out


def evaluate(
    model: HistGradientBoostingClassifier, d: Dict[str, np.ndarray], seed: int
) -> Dict[str, Dict[str, float]]:
    """The model's metrics on a history period, four ways plus the truth."""
    a = decide(model, d)
    y, g, lab = d["y"], d["g"], d["approved"]
    out = {
        "truth": metrics(y, a, g),
        "labeled": metrics(y[lab], a[lab], g[lab]),
        "rejected_bad": metrics(np.where(lab, y, 0), a, g),
        "ipw": metrics(y[lab], a[lab], g[lab], 1 / propensity(d)[lab]),
    }
    rng = np.random.default_rng(seed + 50_000)
    order = rng.permutation(len(y))
    for e in EXPLORE:
        sl = order[: int(e * len(y))]
        m = metrics(y[sl], a[sl], g[sl])
        k = int(a[sl].sum())
        m["n_approved"] = k
        m["ci_half"] = float(Z * np.sqrt(m["repay"] * (1 - m["repay"]) / k)) if k else float("nan")
        out[f"explore_{e:g}"] = m
    return out


def one_seed(seed: int, w: Optional[Dict[str, float]] = None, n: int = N) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    train, ev = draw(n, rng, w), draw(n, rng, w)
    m_lab = fit(train, train["approved"])  # what a team can actually train
    m_full = fit(train)  # the oracle: every outcome known
    a_lab, a_full = decide(m_lab, ev), decide(m_full, ev)
    b = ev["g"] == 1
    return {
        "labeled_share": float(train["approved"].mean()),
        "hist_repay_labeled": float(train["y"][train["approved"]].mean()),
        "pop_repay": float(train["y"].mean()),
        "b_labeled_repay": float(train["y"][train["approved"] & (train["g"] == 1)].mean()),
        "a_labeled_repay": float(train["y"][train["approved"] & (train["g"] == 0)].mean()),
        "eval_lab_model": evaluate(m_lab, ev, seed),
        "eval_full_model": evaluate(m_full, ev, seed),
        "share_b_approved": {"labeled": float(a_lab[b].mean()), "oracle": float(a_full[b].mean())},
    }


def study(seeds: Optional[List[int]] = None, w: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    seeds = seeds or SEEDS
    per = [one_seed(s, w) for s in seeds]

    def agg(vals) -> Dict[str, float]:
        v = np.array(vals, float)
        return {"mean": float(v.mean()), "lo": float(v.min()), "hi": float(v.max())}

    def tree(get) -> object:
        first = get(per[0])
        if isinstance(first, dict):
            return {k: tree(lambda p, k=k, get=get: get(p)[k]) for k in first}
        return agg([get(p) for p in per])

    return {"seeds": seeds, "n": N, **tree(lambda p: p)}


def parse_world(text: str) -> Dict[str, float]:
    """`key=value` overrides of WORLD (e.g. "sel_u=0, bar_b=1.0")."""
    w = dict(WORLD)
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "=" not in part:
            raise ValueError(f"expected key=value, got {part!r}")
        k, v = [x.strip() for x in part.split("=", 1)]
        if k not in WORLD:
            raise ValueError(f"unknown setting {k!r}; known: {', '.join(WORLD)}")
        try:
            w[k] = float(v)
        except ValueError:
            raise ValueError(f"{k} needs a number, got {v!r}") from None
    if not 0 < w["share_b"] < 1:
        raise ValueError("share_b must be between 0 and 1")
    return w
