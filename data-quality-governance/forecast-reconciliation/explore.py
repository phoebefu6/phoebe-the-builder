"""forecast-reconciliation: the regions do not sum to the forecast.

A fictional retailer: 1 total, 4 regions, 12 stores (3 per region), monthly sales. Each of the 17 series is
forecast on its own (trend + month-of-year regression), so the store forecasts do not add up to the region
forecasts and the regions do not add up to the national number. Seven ways to make them add up are scored,
per level, against the 12 months that actually happened. Seeded numpy Generator(211); every store fictional.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

SEED = 211
T, H = 48, 12
REGIONS = ["North", "South", "East", "West"]
STORES_PER = 3
LEVELS = ["total", "region", "store"]
METHODS = ["base", "bottom_up", "top_down_hist", "prorate", "ols", "wls", "mint_shrink"]
N_SEEDS = 30


def summing_matrix() -> np.ndarray:
    """Rows: total, 4 regions, 12 stores. Columns: 12 stores."""
    nb = len(REGIONS) * STORES_PER
    S = [np.ones(nb)]
    for r in range(len(REGIONS)):
        S.append(np.array([1.0 if j // STORES_PER == r else 0.0 for j in range(nb)]))
    S.extend(np.eye(nb))
    return np.array(S)


def level_rows() -> Dict[str, List[int]]:
    return {"total": [0], "region": list(range(1, 1 + len(REGIONS))),
            "store": list(range(1 + len(REGIONS), 1 + len(REGIONS) + len(REGIONS) * STORES_PER))}


def make_world(seed: int = SEED, shock: float = 0.0, t_train: int = T) -> dict:
    """Store sales = level x growth x season x noise. Noise = national + regional + store shocks (correlated).
    Growth differs by store, so the mix drifts. `shock` cuts one region's stores in the 12 test months only."""
    g = np.random.default_rng(seed)
    nb = len(REGIONS) * STORES_PER
    n = t_train + H
    t = np.arange(n)
    level = g.uniform(80, 220, nb)
    growth = g.normal(0.004, 0.008, nb)
    amp = g.uniform(0.05, 0.25, len(REGIONS))
    phase = g.uniform(0, 12, len(REGIONS))
    nat = g.normal(0, 0.03, n)
    reg = g.normal(0, 0.05, (len(REGIONS), n))
    bottom = np.zeros((nb, n))
    for j in range(nb):
        r = j // STORES_PER
        season = 1 + amp[r] * np.sin(2 * np.pi * (t - phase[r]) / 12)
        noise = nat + reg[r] + g.normal(0, 0.10, n)
        bottom[j] = level[j] * np.exp(growth[j] * t) * season * np.exp(noise)
    hit = None
    if shock:
        hit = int(g.integers(len(REGIONS)))
        for j in range(hit * STORES_PER, (hit + 1) * STORES_PER):
            bottom[j, t_train:] *= 1 - shock
    S = summing_matrix()
    return {"S": S, "y": S @ bottom, "bottom": bottom, "T": t_train, "shock_region": hit}


def design(n: int) -> np.ndarray:
    t = np.arange(n)
    return np.column_stack([np.ones_like(t), t] + [(t % 12 == m).astype(float) for m in range(1, 12)]).astype(float)


def base_forecasts(w: dict, log: bool = True) -> tuple:
    """Each series on its own: regression on intercept + trend + 11 month dummies, fitted to log sales (growth is
    multiplicative). Returns (forecast n x H, in-sample residuals n x T). log=False fits the raw scale instead."""
    Tn = w["T"]
    X = design(Tn + H)
    Xtr, Xte = X[:Tn], X[Tn:]
    y = w["y"][:, :Tn]
    tr = np.log if log else (lambda a: a)
    inv = np.exp if log else (lambda a: a)
    beta, *_ = np.linalg.lstsq(Xtr, tr(y).T, rcond=None)
    return inv(Xte @ beta).T, y - inv(Xtr @ beta).T


def shrink_cov(res: np.ndarray) -> np.ndarray:
    """Schafer-Strimmer shrinkage of the residual covariance toward its diagonal (as in MinT-shrink)."""
    e = res.T - res.T.mean(0)
    n = e.shape[0]
    cov = e.T @ e / n
    sd = np.sqrt(np.diag(cov))
    z = e / sd
    cor = z.T @ z / n
    wv = np.einsum("ti,tj->tij", z, z)
    var_cor = n / (n - 1) ** 3 * ((wv - wv.mean(0)) ** 2).sum(0)
    off = ~np.eye(cor.shape[0], dtype=bool)
    lam = float(np.clip(var_cor[off].sum() / (cor[off] ** 2).sum(), 0, 1))
    shrunk = lam * np.eye(cor.shape[0]) + (1 - lam) * cor
    np.fill_diagonal(shrunk, 1.0)
    return shrunk * np.outer(sd, sd)


def reconcile(w: dict, yhat: np.ndarray, res: np.ndarray, method: str) -> np.ndarray:
    S, rows = w["S"], level_rows()
    reg, sto = rows["region"], rows["store"]
    if method == "base":
        return yhat.copy()
    if method == "bottom_up":
        return S @ yhat[sto]
    if method == "top_down_hist":  # average historical share of each store in the total
        share = (w["y"][sto, : w["T"]] / w["y"][0, : w["T"]]).mean(1)
        return S @ np.outer(share, yhat[0])
    if method == "prorate":  # scale the pieces to the agreed number above them, level by level
        r = yhat[0] * yhat[reg] / yhat[reg].sum(0)
        b = np.zeros((len(sto), H))
        for i in range(len(reg)):
            kids = [k for k in range(len(sto)) if S[reg[i], k]]
            b[kids] = r[i] * yhat[[sto[k] for k in kids]] / yhat[[sto[k] for k in kids]].sum(0)
        return S @ b
    if method == "ols":
        Wi = np.eye(S.shape[0])
    elif method == "wls":  # structural scaling: variance proportional to the number of stores underneath
        Wi = np.diag(1 / S.sum(1))
    elif method == "mint_shrink":
        Wi = np.linalg.inv(shrink_cov(res))
    else:
        raise ValueError(method)
    G = np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)
    return S @ (G @ yhat)


def rmse_by_level(w: dict, yt: np.ndarray) -> Dict[str, float]:
    """RMSE per series over the 12 test months, averaged within each level."""
    act = w["y"][:, w["T"]:]
    e = np.sqrt(((yt - act) ** 2).mean(1))
    return {lv: float(e[ix].mean()) for lv, ix in level_rows().items()}


def coherence_gap(w: dict, yt: np.ndarray) -> float:
    """Largest |parent - sum of children| as a share of the total forecast, over the horizon."""
    S, rows = w["S"], level_rows()
    implied = S @ yt[rows["store"]]
    return float(np.abs(yt - implied).max() / yt[0].mean())


def run(w: dict) -> dict:
    yhat, res = base_forecasts(w)
    out = {}
    for m in METHODS:
        yt = reconcile(w, yhat, res, m)
        out[m] = {"rmse": rmse_by_level(w, yt), "gap": coherence_gap(w, yt), "fc": yt}
    base = out["base"]["rmse"]
    for m in METHODS:
        out[m]["vs_base"] = {lv: out[m]["rmse"][lv] / base[lv] - 1 for lv in LEVELS}
    return out


def study(seed: int = SEED) -> dict:
    w = make_world(seed)
    r = run(w)
    rows = level_rows()
    yhat, _ = base_forecasts(w)
    reg_sum = yhat[rows["region"]].sum(0)
    sto_sum = yhat[rows["store"]].sum(0)
    return {
        "methods": {m: {"rmse": v["rmse"], "vs_base": v["vs_base"], "gap": v["gap"]} for m, v in r.items()},
        "raw_scale_gap": coherence_gap(w, base_forecasts(w, log=False)[0]),
        "base_regions_vs_total": float(np.abs(reg_sum - yhat[0]).max() / yhat[0].mean()),
        "base_stores_vs_total": float(np.abs(sto_sum - yhat[0]).max() / yhat[0].mean()),
        "share_drift": float(np.abs(w["bottom"][:, -H:].sum(1) / w["bottom"][:, -H:].sum()
                                    - w["bottom"][:, :T].sum(1) / w["bottom"][:, :T].sum()).sum() / 2),
        "total_actual": float(w["y"][0, T:].sum()),
        "regions_actual": [float(x) for x in w["y"][level_rows()["region"], T:].sum(1)],
        "regions_fc": {m: [float(x) for x in v["fc"][level_rows()["region"]].sum(1)] for m, v in r.items()},
    }


def seeds(n: int = N_SEEDS, shock: float = 0.0, t_train: int = T) -> dict:
    """Across n worlds: mean change vs base per level, and how often each method beats base at EVERY level."""
    acc = {m: {lv: [] for lv in LEVELS} for m in METHODS}
    allwin = {m: 0 for m in METHODS}
    best = {lv: {m: 0 for m in METHODS} for lv in LEVELS}
    bias = {"base": [], "bottom_up": []}
    for s in range(1, n + 1):
        w = make_world(1000 + s, shock=shock, t_train=t_train)
        r = run(w)
        act = w["y"][0, w["T"]:].sum()
        for m in bias:
            bias[m].append(float(r[m]["fc"][0].sum() / act - 1))
        for m in METHODS:
            for lv in LEVELS:
                acc[m][lv].append(r[m]["vs_base"][lv])
            allwin[m] += all(r[m]["vs_base"][lv] < 0 for lv in LEVELS)
        for lv in LEVELS:
            best[lv][min(METHODS, key=lambda m: r[m]["rmse"][lv])] += 1
    return {"n": n, "mean": {m: {lv: float(np.mean(v)) for lv, v in d.items()} for m, d in acc.items()},
            "all_levels_better": allwin, "best_count": best,
            "total_bias": {m: float(np.median(v)) for m, v in bias.items()}}


def shock_study(shock: float = 0.15, n: int = N_SEEDS) -> dict:
    """A cut nobody's history contains: how wrong is the shocked region under each method?"""
    errs = {m: [] for m in METHODS}
    for s in range(1, n + 1):
        w = make_world(1000 + s, shock=shock)
        r = run(w)
        i = level_rows()["region"][w["shock_region"]]
        act = w["y"][i, w["T"]:].sum()
        for m in METHODS:
            errs[m].append(float(r[m]["fc"][i].sum() / act - 1))
    return {"shock": shock, "n": n, "median_bias": {m: float(np.median(v)) for m, v in errs.items()},
            "min_bias": {m: float(np.min(v)) for m, v in errs.items()}}
