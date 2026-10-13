"""The model is fine overall. Group fairness metrics on one screening model, and why three of them cannot all hold.

A declared, constructed world: a screening model scores applicants from two groups. The groups differ in the share
who are qualified (the base rate), and the score is LESS informative for the smaller group, because the training data
held fewer of them. Given a qualified/unqualified label, each group's score is normal, so every rate below is exact:
selection rate, true and false positive rate, precision, the EEOC four-fifths ratio, accuracy and AUC.

A policy is one threshold per group (plus, for equalized odds, a randomised mix between two thresholds). Five policies
are compared: one threshold for everyone, and four policies that each enforce one fairness definition.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

SEED = 196
MC_REPS = 200_000
GROUPS = ["A", "B"]

# group: share of applicants, base rate (share qualified), score mean if qualified / unqualified, score sd.
WORLD: Dict[str, Dict[str, float]] = {
    "A": {"share": 0.70, "base": 0.40, "mu_q": 1.20, "mu_u": -0.40, "sd": 1.0},
    "B": {"share": 0.30, "base": 0.25, "mu_q": 0.70, "mu_u": -0.30, "sd": 1.0},
}
FOUR_FIFTHS = 0.8
GRID = np.round(np.arange(-2.0, 3.0001, 0.01), 2)  # candidate thresholds


def rates(g: str, t: float, world: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
    """Exact rates for group g when everyone scoring >= t is selected."""
    w = (world or WORLD)[g]
    tpr = float(norm.sf(t, w["mu_q"], w["sd"]))
    fpr = float(norm.sf(t, w["mu_u"], w["sd"]))
    p = w["base"]
    sel = p * tpr + (1 - p) * fpr
    ppv = p * tpr / sel if sel > 0 else float("nan")
    acc = p * tpr + (1 - p) * (1 - fpr)
    return {"tpr": tpr, "fpr": fpr, "sel": sel, "ppv": ppv, "acc": acc}


def mix(a: Dict[str, float], b: Dict[str, float], q: float, base: float) -> Dict[str, float]:
    """Randomise: use threshold a with probability q, threshold b otherwise. TPR and FPR mix linearly."""
    tpr = q * a["tpr"] + (1 - q) * b["tpr"]
    fpr = q * a["fpr"] + (1 - q) * b["fpr"]
    sel = base * tpr + (1 - base) * fpr
    return {"tpr": tpr, "fpr": fpr, "sel": sel, "ppv": base * tpr / sel if sel else float("nan"),
            "acc": base * tpr + (1 - base) * (1 - fpr)}


def auc(g: str, world: Optional[Dict[str, Dict[str, float]]] = None) -> float:
    """Binormal AUC with equal variances: Phi(d / sqrt(2))."""
    w = (world or WORLD)[g]
    return float(norm.cdf((w["mu_q"] - w["mu_u"]) / (w["sd"] * math.sqrt(2))))


def pooled_auc(world: Optional[Dict[str, Dict[str, float]]] = None) -> float:
    """AUC of the score over everyone: P(random qualified scores above random unqualified), mixtures by weight."""
    wd = world or WORLD
    wq = {g: wd[g]["share"] * wd[g]["base"] for g in GROUPS}
    wu = {g: wd[g]["share"] * (1 - wd[g]["base"]) for g in GROUPS}
    sq, su = sum(wq.values()), sum(wu.values())
    tot = 0.0
    for gq in GROUPS:
        for gu in GROUPS:
            d = wd[gq]["mu_q"] - wd[gu]["mu_u"]
            s = math.hypot(wd[gq]["sd"], wd[gu]["sd"])
            tot += (wq[gq] / sq) * (wu[gu] / su) * norm.cdf(d / s)
    return float(tot)


def overall(per: Dict[str, Dict[str, float]], world: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
    wd = world or WORLD
    acc = sum(wd[g]["share"] * per[g]["acc"] for g in GROUPS)
    sel = sum(wd[g]["share"] * per[g]["sel"] for g in GROUPS)
    return {"acc": acc, "sel": sel, "impact_ratio": min(per[g]["sel"] for g in GROUPS) / max(per[g]["sel"] for g in GROUPS)}


def _best(pairs, world) -> Tuple[float, float]:
    """Among threshold pairs, the one with the highest overall accuracy."""
    wd = world
    best, best_acc = None, -1.0
    for ta, tb in pairs:
        a = wd["A"]["share"] * rates("A", ta, wd)["acc"] + wd["B"]["share"] * rates("B", tb, wd)["acc"]
        if a > best_acc:
            best, best_acc = (ta, tb), a
    return best


def _match(metric: str, world) -> List[Tuple[float, float]]:
    """For each threshold of A, the threshold of B that makes `metric` closest to equal."""
    rb = {t: rates("B", t, world)[metric] for t in GRID}
    out = []
    for ta in GRID:
        va = rates("A", ta, world)[metric]
        if va != va:
            continue
        tb = min(GRID, key=lambda t: abs(rb[t] - va) if rb[t] == rb[t] else 9)
        out.append((float(ta), float(tb)))
    return out


def equalized_odds(world: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, object]:
    """Hardt, Price & Srebro (2016): equal TPR AND FPR. B's ROC curve lies under A's, so A is randomised down onto
    B's curve: pick B's threshold, then find the mix of two A thresholds hitting the same (FPR, TPR) point.
    Search over B's threshold and A's lower threshold, maximising overall accuracy."""
    wd = world or WORLD
    best = None
    ra = {t: rates("A", t, wd) for t in GRID}
    for tb in GRID[::2]:
        b = rates("B", tb, wd)
        # A point on the chord between A thresholds t1 < t2 hits (fpr_b, tpr_b) only if the chord passes through it.
        for t1 in GRID[::4]:
            a1 = ra[t1]
            if a1["fpr"] < b["fpr"]:
                continue
            # choose t2 so that the chord from a1 through b exits A's curve: solve mix weight on TPR and FPR
            for t2 in GRID[::4]:
                if t2 <= t1:
                    continue
                a2 = ra[t2]
                if a2["fpr"] > b["fpr"]:
                    continue
                q = (b["fpr"] - a2["fpr"]) / (a1["fpr"] - a2["fpr"]) if a1["fpr"] != a2["fpr"] else 0.0
                m = mix(a1, a2, q, wd["A"]["base"])
                if abs(m["tpr"] - b["tpr"]) > 0.003:
                    continue
                acc = wd["A"]["share"] * m["acc"] + wd["B"]["share"] * b["acc"]
                if best is None or acc > best[0]:
                    best = (acc, float(t1), float(t2), float(q), float(tb), m, b)
    _, t1, t2, q, tb, m, b = best
    return {"tA": (t1, t2, q), "tB": tb, "per": {"A": m, "B": b}}


def policies(world: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, object]]:
    wd = world or WORLD
    out: Dict[str, Dict[str, object]] = {}
    one = _best([(float(t), float(t)) for t in GRID], wd)
    out["one threshold (max accuracy)"] = {"tA": one[0], "tB": one[1]}
    out["demographic parity"] = dict(zip(("tA", "tB"), _best(_match("sel", wd), wd)))
    out["equal opportunity (TPR)"] = dict(zip(("tA", "tB"), _best(_match("tpr", wd), wd)))
    out["predictive parity (PPV)"] = dict(zip(("tA", "tB"), _best(_match("ppv", wd), wd)))
    for k, v in out.items():
        v["per"] = {"A": rates("A", v["tA"], wd), "B": rates("B", v["tB"], wd)}
    out["equalized odds (TPR+FPR)"] = equalized_odds(wd)
    for v in out.values():
        v["overall"] = overall(v["per"], wd)
        p = v["per"]
        v["gaps"] = {m: p["A"][m] - p["B"][m] for m in ("sel", "tpr", "fpr", "ppv")}
    return out


def impossibility(world: Optional[Dict[str, Dict[str, float]]] = None, tol: float = 0.02) -> Dict[str, object]:
    """Chouldechova (2017): FPR = p / (1 - p) * (1 - PPV) / PPV * TPR within a group. With different base rates p,
    equal PPV and equal TPR force unequal FPR. Check it on every deterministic threshold pair on the grid."""
    wd = world or WORLD
    ra = {t: rates("A", t, wd) for t in GRID}
    rb = {t: rates("B", t, wd) for t in GRID}
    n_pairs = n_two = 0
    n_all = 0
    closest = None
    for ta in GRID[::2]:
        a = ra[ta]
        for tb in GRID[::2]:
            b = rb[tb]
            if a["sel"] < 0.01 or b["sel"] < 0.01:
                continue
            n_pairs += 1
            g = [abs(a["ppv"] - b["ppv"]), abs(a["tpr"] - b["tpr"]), abs(a["fpr"] - b["fpr"])]
            if sum(x <= tol for x in g) >= 2:
                n_two += 1
            if all(x <= tol for x in g):
                n_all += 1
            worst = max(g)
            if closest is None or worst < closest[0]:
                closest = (worst, float(ta), float(tb), g)
    for g in GROUPS:  # the identity itself, at one threshold
        r = rates(g, 0.5, wd)
        p = wd[g]["base"]
        assert abs(r["fpr"] - p / (1 - p) * (1 - r["ppv"]) / r["ppv"] * r["tpr"]) < 1e-12
    return {"pairs": n_pairs, "two_of_three": n_two, "all_three": n_all, "tol": tol,
            "closest": {"worst_gap": closest[0], "tA": closest[1], "tB": closest[2],
                        "ppv_gap": closest[3][0], "tpr_gap": closest[3][1], "fpr_gap": closest[3][2]}}


def audit_power(t: float, n_b_list: List[int], world: Optional[Dict[str, Dict[str, float]]] = None,
                reps: int = 4000, seed: int = SEED) -> List[Dict[str, float]]:
    """An auditor sees n_B applicants from B (and A in proportion) and computes the four-fifths ratio.
    How often does the sample verdict (ratio < 0.8) agree with the population verdict?"""
    wd = world or WORLD
    rng = np.random.default_rng(seed)
    sa, sb = rates("A", t, wd)["sel"], rates("B", t, wd)["sel"]
    truth = min(sa, sb) / max(sa, sb) < FOUR_FIFTHS
    out = []
    for nb in n_b_list:
        na = int(round(nb * wd["A"]["share"] / wd["B"]["share"]))
        ka = rng.binomial(na, sa, reps)
        kb = rng.binomial(nb, sb, reps)
        pa, pb = ka / na, kb / nb
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.minimum(pa, pb) / np.maximum(pa, pb)
        ratio = np.nan_to_num(ratio, nan=1.0)
        verdict = ratio < FOUR_FIFTHS
        lo, hi = np.percentile(ratio, [5, 95])
        out.append({"n_B": nb, "agree": float((verdict == truth).mean()), "ratio_p5": float(lo), "ratio_p95": float(hi)})
    return out


def simulate(policy: Dict[str, object], n: int = MC_REPS, seed: int = SEED,
             world: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """Draw applicants, score them, apply the policy (randomising where it says so). Counts for Wilson checks."""
    wd = world or WORLD
    rng = np.random.default_rng(seed)
    out = {}
    for g in GROUPS:
        m = rng.binomial(n, wd[g]["share"])
        q = rng.random(m) < wd[g]["base"]
        s = np.where(q, rng.normal(wd[g]["mu_q"], wd[g]["sd"], m), rng.normal(wd[g]["mu_u"], wd[g]["sd"], m))
        t = policy["tA"] if g == "A" else policy["tB"]
        if isinstance(t, tuple):
            t1, t2, w = t
            use1 = rng.random(m) < w
            sel = np.where(use1, s >= t1, s >= t2)
        else:
            sel = s >= t
        out[g] = {"sel": (int(sel.sum()), m), "tpr": (int((sel & q).sum()), int(q.sum())),
                  "fpr": (int((sel & ~q).sum()), int((~q).sum())), "ppv": (int((sel & q).sum()), int(sel.sum()))}
    return out


def wilson(k: int, n: int, z: float = 2.5758) -> Tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def calibrate(world: Optional[Dict[str, Dict[str, float]]] = None, n: int = MC_REPS) -> List[Dict[str, object]]:
    rows = []
    for name, pol in policies(world).items():
        mc = simulate(pol, n=n, world=world)
        for g in GROUPS:
            for m in ("sel", "tpr", "fpr", "ppv"):
                k, nn = mc[g][m]
                lo, hi = wilson(k, nn)
                ex = pol["per"][g][m]
                rows.append({"policy": name, "group": g, "metric": m, "exact": ex, "mc": k / nn,
                             "inside": bool(lo <= ex <= hi)})
    return rows


def parse_world(text: str) -> Dict[str, Dict[str, float]]:
    """`B.base=0.3, B.mu_q=1.0` overrides. Unknown keys or bad numbers raise ValueError."""
    wd = {g: dict(v) for g, v in WORLD.items()}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "=" not in part or "." not in part.split("=")[0]:
            raise ValueError(f"expected group.key=value, got {part!r}")
        key, val = [x.strip() for x in part.split("=", 1)]
        g, k = key.split(".", 1)
        if g not in wd or k not in wd[g]:
            raise ValueError(f"unknown parameter {key!r}; groups {GROUPS}, keys {list(WORLD['A'])}")
        try:
            wd[g][k] = float(val)
        except ValueError:
            raise ValueError(f"{key} needs a number, got {val!r}") from None
    if abs(sum(wd[g]["share"] for g in GROUPS) - 1) > 1e-9:
        raise ValueError("group shares must sum to 1")
    for g in GROUPS:
        if not 0 < wd[g]["base"] < 1 or wd[g]["sd"] <= 0:
            raise ValueError("base rates must be in (0, 1) and sd positive")
    return wd


def data_fix(world: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, object]:
    """The remedy no threshold can provide: make B's score as informative as A's (same qualified-unqualified gap),
    base rates unchanged. Returns the fixed world's policies plus the gap it set."""
    wd = {g: dict(v) for g, v in (world or WORLD).items()}
    gap = wd["A"]["mu_q"] - wd["A"]["mu_u"]
    wd["B"]["mu_q"] = wd["B"]["mu_u"] + gap
    return {"world": wd, "auc_B": auc("B", wd), "policies": policies(wd)}
