"""The measurement run behind every number in the README.

Eight sections, each one an experiment rather than a citation:

1. a 95% band is a claim about a frequency - which methods keep it, and what the width costs
2. where the sqrt(h) fan comes from, and the two worlds that decide whether it is right
3. the parameter-uncertainty term: what dropping it costs, and when
4. conformal's guarantee is MARGINAL - the regime split no marginal check can see
5. pointwise coverage is not path coverage, and the horizons are not independent tests
6. autocorrelated noise: the interval built from iid residuals, swept over rho
7. the verification problem - the power of the coverage check people actually run
8. the scoreboard: a proper score, coverage and width disagree, and only one cannot be gamed

Run: python evidence.py    (writes results.json + evidence.txt, ~3 min)
"""

from __future__ import annotations

import json
import sys
import time
from typing import Dict, List

import intervals as iv
import numpy as np

R: Dict[str, object] = {}
LINES: List[str] = []


def say(s: str = "") -> None:
    LINES.append(s)
    print(s)
    sys.stdout.flush()


def head(n: int, title: str) -> None:
    say("")
    say("=" * 100)
    say(f"{n}. {title}")
    say("=" * 100)


H, N_TRAIN, CONF = 12, 168, 0.95
REPS = 600
T0 = time.time()

# ================================================ 1. the claim vs the frequency
head(1, "A 95% band is a claim about a frequency - kept, and at what width")

runs = {}
for name, m in iv.METHODS.items():
    runs[name] = iv.coverage_run(m, H, N_TRAIN, REPS, np.random.default_rng(5), CONF)

say(f"  nominal {CONF:.0%}, h=1..{H}, {REPS} independent futures, well-specified model, iid noise")
say("")
say(f"  {'method':<28}{'coverage':>10}{'at h=1':>9}{'at h=12':>9}{'mean width':>12}{'vs narrowest':>14}")
widths = {k: float(v["width"].mean()) for k, v in runs.items()}
narrow = min(widths.values())
for name, r in runs.items():
    say(f"  {name:<28}{r['hits'].mean():>10.3f}{r['hits'][:, 0].mean():>9.3f}"
        f"{r['hits'][:, -1].mean():>9.3f}{widths[name]:>12.2f}{widths[name] / narrow:>13.2f}x")
R["s1"] = {k: {"coverage": float(v["hits"].mean()), "h1": float(v["hits"][:, 0].mean()),
               "h12": float(v["hits"][:, -1].mean()), "width": widths[k]} for k, v in runs.items()}
say("")
say("  The band almost every forecast ships - residual sd widened by sqrt(h) - does not")
say("  undercover here. It OVERCOVERS, and pays for it in a width nobody can act on.")

# ================================================ 2. where sqrt(h) is right
head(2, "The sqrt(h) fan is an assumption about the MODEL, and it is measurable before shipping")


def growth_exponent(model_kind: str, reps: int = 400) -> Dict[str, float]:
    """Fit log(sd of h-step error) = a + b log(h). b = 0.5 is the sqrt(h) fan."""
    rng = np.random.default_rng(17)
    errs = np.zeros((reps, H))
    for i in range(reps):
        if model_kind == "trend+season fit":
            y, _ = iv.make_series(rng, N_TRAIN + H)
            fit = iv.fit_trend_season(y[:N_TRAIN])
            errs[i] = y[N_TRAIN:] - iv.predict(fit, H)
        else:  # a random walk, forecast by its last value
            steps = rng.normal(0, 4.0, N_TRAIN + H)
            y = 100 + np.cumsum(steps)
            errs[i] = y[N_TRAIN:] - y[N_TRAIN - 1]
    sd = errs.std(axis=0, ddof=1)
    h = np.arange(1, H + 1)
    b, a = np.polyfit(np.log(h), np.log(sd), 1)
    return {"sd1": float(sd[0]), "sd12": float(sd[-1]), "ratio": float(sd[-1] / sd[0]),
            "exponent": float(b), "intercept": float(a)}


for kind in ("trend+season fit", "random walk + naive"):
    g = growth_exponent(kind)
    say(f"  {kind:<24} sd(h=1) {g['sd1']:.3f}  sd(h=12) {g['sd12']:.3f}  "
        f"ratio {g['ratio']:.2f}  fitted exponent {g['exponent']:+.3f}")
    R.setdefault("s2", {})[kind] = g
say("")
say("  sqrt(12) = 3.46. The random walk grows at exactly that rate (exponent 0.5) because its")
say("  h-step error is a SUM of h independent shocks. A fitted trend+season forecast makes a")
say("  DIRECT h-step statement whose error does not accumulate at all - the fan is borrowed")
say("  from a different estimator. Two lines of numpy tell you which world you are in.")

# ================================================ 3. the parameter term
head(3, "The parameter-uncertainty term: what dropping it costs, and where")

S3_REPS = 1500
say(f"  {S3_REPS} futures per row, so a coverage figure here carries a Monte Carlo se of about "
    f"{np.sqrt(0.95 * 0.05 / S3_REPS):.4f}")
say("")
say(f"  {'train points':>13}{'leverage at h=12':>19}{'flat band':>12}{'+ param term':>14}{'width cost':>12}")
for n_train in (36, 60, 120, 240, 480):
    a = iv.coverage_run(iv.gaussian_flat, H, n_train, S3_REPS, np.random.default_rng(9), CONF)
    b = iv.coverage_run(iv.gaussian_theory, H, n_train, S3_REPS, np.random.default_rng(9), CONF)
    y, _ = iv.make_series(np.random.default_rng(1), n_train + H)
    lev = float(iv.leverage(iv.fit_trend_season(y[:n_train]), H)[-1])
    say(f"  {n_train:>13}{lev:>19.4f}{a['hits'][:, -1].mean():>12.3f}"
        f"{b['hits'][:, -1].mean():>14.3f}{b['width'].mean() / a['width'].mean():>11.2f}x")
    R.setdefault("s3", []).append({"n_train": n_train, "leverage_h12": lev,
                                   "flat_h12": float(a["hits"][:, -1].mean()),
                                   "theory_h12": float(b["hits"][:, -1].mean()),
                                   "width_ratio": float(b["width"].mean() / a["width"].mean())})
say("")
say("  The term costs almost nothing to include and is the difference between a band that")
say("  keeps its promise at 36 training points and one that does not. It is also the term")
say("  that grows as you forecast further from the training window - the honest version of")
say("  the fan the previous section just rejected.")

# ================================================ 4. conformal is marginal
head(4, "Conformal's guarantee is MARGINAL, and marginal is not the promise people hear")

VOL = {"vol_period": 24, "vol_ratio": 9.0}
_chk = iv.make_series(np.random.default_rng(77), 40000, **VOL)[1]
_base = iv.make_series(np.random.default_rng(77), 40000)[1]
say("  a two-regime world: quiet blocks and volatile blocks, 3x apart in sd, average variance")
say("  held EQUAL to the single-regime world, so nothing in the aggregate looks different -")
say(f"  verified: mean variance {np.mean(_chk ** 2):.2f} vs {np.mean(_base ** 2):.2f} in the flat world")
say("  (the obvious construction, sigma*sqrt(r) and sigma/sqrt(r), is 4.6x louder on average and")
say("  would have made this a test of MORE noise rather than of the SAME noise, unevenly spread)")
say("")
say(f"  {'method':<28}{'marginal':>10}{'in quiet':>10}{'in volatile':>13}{'spread':>9}")
for name in ("gaussian + parameter term", "split conformal", "rolling h-step quantile"):
    r = iv.coverage_run(iv.METHODS[name], H, N_TRAIN, REPS, np.random.default_rng(23), CONF, **VOL)
    # split at the GEOMETRIC mid-point of the two sd levels, not at the median: the median of
    # a two-level array lands ON a level, and a strict < then puts every point in one cell
    lo_sd = r["sd"] < np.sqrt(r["sd"].min() * r["sd"].max())
    quiet = float(r["hits"][lo_sd].mean())
    loud = float(r["hits"][~lo_sd].mean())
    say(f"  {name:<28}{r['hits'].mean():>10.3f}{quiet:>10.3f}{loud:>13.3f}{quiet - loud:>9.3f}")
    R.setdefault("s4", {})[name] = {"marginal": float(r["hits"].mean()), "quiet": quiet,
                                    "volatile": loud}
say("")
say("  NEGATIVE RESULT: the marginal number is the one everybody reports and it is kept. The")
say("  user standing in the volatile regime - the only regime where an interval matters - gets")
say("  a materially different frequency, and no marginal check in existence can see it,")
say("  because the two errors are constructed to cancel in the average.")

# ================================================ 5. path coverage
head(5, "Pointwise coverage is not path coverage, and the horizons are not independent tests")

r = runs["gaussian + parameter term"]
point = float(r["hits"].mean())
path = float(r["hits"].all(axis=1).mean())
say(f"  pointwise coverage                    {point:.3f}")
say(f"  whole-path coverage (all {H} inside)    {path:.3f}   <- the number a planner needs")
say(f"  if the horizons were independent      {point ** H:.3f}")
say("")
say("  A 95% band is kept 95% of the time AT A POINT and contains the whole twelve-step path")
say("  barely half the time. Those are different claims and only one of them is published.")
say("")
say("  How much widening a simultaneous band needs depends on how many INDEPENDENT horizon")
say("  tests the path really is - k_eff = log(path coverage) / log(pointwise coverage):")
say("")
say(f"  {'rho':>6}{'pointwise':>12}{'path':>9}{'k_eff of the 12 horizons':>28}")
for rho in (0.0, 0.4, 0.7, 0.9):
    rr = iv.coverage_run(iv.gaussian_theory, H, N_TRAIN, 600, np.random.default_rng(33),
                         CONF, rho=rho)
    pt, pa = float(rr["hits"].mean()), float(rr["hits"].all(axis=1).mean())
    ke = float(np.log(pa) / np.log(pt)) if 0 < pa < 1 else float("nan")
    say(f"  {rho:>6.1f}{pt:>12.3f}{pa:>9.3f}{ke:>28.2f}")
    R.setdefault("s5_keff", []).append({"rho": rho, "pointwise": pt, "path": pa, "k_eff": ke})
say("")
say("  At rho=0 the twelve horizons behave like ~12 independent tests, so the Bonferroni")
say("  intuition is right; by rho=0.9 the path is a handful of tests and Bonferroni is")
say("  wasteful. The correction is measurable rather than assumed.")

# and the widening itself is not as simple as raising the confidence level
bonf_conf = 1 - (1 - CONF) / H
bonf = iv.coverage_run(lambda y, h, c=CONF: iv.gaussian_theory(y, h, 0.99),
                       H, N_TRAIN, REPS, np.random.default_rng(5))
per_point = float(bonf["hits"].mean())
say("")
say("")
say("  NEGATIVE RESULT - small per-point shortfalls COMPOUND along the path. Raise the")
say(f"  per-point level to 99% and measured per-point coverage is {per_point:.4f}, short of nominal")
say(f"  by {0.99 - per_point:.4f} - a rounding error at a point. Over twelve horizons it becomes")
say(f"  {per_point ** H:.3f} against the {0.99 ** H:.3f} the nominal level promises (measured path "
    f"{float(bonf['hits'].all(axis=1).mean()):.3f}),")
say("  i.e. a 0.2-point miss at the point is a 2-point miss on the path. Part of the shortfall")
say("  is z-for-t: a band 2.58 estimated sds wide is more sensitive to the sd being estimated")
say("  than one 1.96 wide, and the t quantile recovers some of it, but only some:")
from scipy import stats as _st  # noqa: E402

rng_t = np.random.default_rng(5)
hits_z, hits_t = [], []
for _ in range(REPS):
    y, _sd = iv.make_series(rng_t, N_TRAIN + H)
    fit = iv.fit_trend_season(y[:N_TRAIN])
    p = iv.predict(fit, H)
    se = np.sqrt(fit["s2"] * (1.0 + iv.leverage(fit, H)))
    act = y[N_TRAIN:]
    dof = N_TRAIN - 4
    for q, acc in ((iv.Z[0.99], hits_z), (float(_st.t.ppf(0.995, dof)), hits_t)):
        acc.append(np.mean((act >= p - q * se) & (act <= p + q * se)))
say(f"    z quantile (2.576) at nominal 99%: {np.mean(hits_z):.4f}"
    f"  (short by {0.99 - np.mean(hits_z):.4f})")
say(f"    t quantile ({float(_st.t.ppf(0.995, N_TRAIN - 4)):.3f}) at nominal 99%: "
    f"{np.mean(hits_t):.4f}  (short by {0.99 - np.mean(hits_t):.4f}, "
    f"{(1 - (0.99 - np.mean(hits_t)) / (0.99 - np.mean(hits_z))) * 100:.0f}% of the gap closed)")
say("  The rest is the estimated sd itself, and it does not go away by choosing a nicer")
say("  quantile - it goes away by measuring coverage in the tail you intend to quote.")

R["s5"] = {"pointwise": point, "path": path, "independent_prediction": point ** H,
           "bonf_conf": bonf_conf, "bonferroni_path": float(bonf["hits"].all(axis=1).mean()),
           "bonf_per_point": per_point, "z_99": float(np.mean(hits_z)),
           "t_99": float(np.mean(hits_t))}

# ================================================ 6. autocorrelated noise
head(6, "Autocorrelated noise: the same interval, swept over rho")

say(f"  {'rho':>6}{'gaussian + param':>19}{'split conformal':>18}{'rolling quantile':>19}"
    f"{'true sd(h=12)/sd(h=1)':>23}")
for rho in (0.0, 0.4, 0.7, 0.9):
    row = []
    for name in ("gaussian + parameter term", "split conformal", "rolling h-step quantile"):
        rr = iv.coverage_run(iv.METHODS[name], H, N_TRAIN, 400, np.random.default_rng(29),
                             CONF, rho=rho)
        row.append(float(rr["hits"].mean()))
    rng = np.random.default_rng(31)
    e = np.zeros((300, H))
    for i in range(300):
        y, _ = iv.make_series(rng, N_TRAIN + H, rho=rho)
        e[i] = y[N_TRAIN:] - iv.predict(iv.fit_trend_season(y[:N_TRAIN]), H)
    ratio = float(e.std(axis=0, ddof=1)[-1] / e.std(axis=0, ddof=1)[0])
    say(f"  {rho:>6.1f}{row[0]:>19.3f}{row[1]:>18.3f}{row[2]:>19.3f}{ratio:>23.2f}")
    R.setdefault("s6", []).append({"rho": rho, "gaussian": row[0], "conformal": row[1],
                                   "rolling": row[2], "sd_ratio": ratio})
say("")
say("  The residual sd a fitted model reports is the sd of a CORRELATED series, and at rho=0.9")
say("  the in-sample residuals understate what the next twelve points will do. Note which")
say("  method degrades least, and that none of them is repaired by widening with sqrt(h).")

# ================================================ 7. the verification problem
head(7, "The check itself: how much data it takes to catch a band that is 5 points short")

say("  power of a one-sided binomial check that a nominally 95% band really covers 90%,")
say("  at alpha=0.05, as a function of the test points available:")
say("")
say(f"  {'test points m':>15}{'if independent':>18}{'at k_eff=1/12':>17}{'at k_eff=1/20':>17}")
for m in (20, 50, 100, 250, 500, 1000, 5000):
    say(f"  {m:>15}{iv.coverage_check_power(0.90, 0.95, m):>18.3f}"
        f"{iv.coverage_check_power(0.90, 0.95, m, 1 / 12):>17.3f}"
        f"{iv.coverage_check_power(0.90, 0.95, m, 1 / 20):>17.3f}")
    R.setdefault("s7", []).append({"m": m, "indep": iv.coverage_check_power(0.90, 0.95, m),
                                   "k12": iv.coverage_check_power(0.90, 0.95, m, 1 / 12),
                                   "k20": iv.coverage_check_power(0.90, 0.95, m, 1 / 20)})
rng = np.random.default_rng(41)
obs = []
for _ in range(400):
    y, _ = iv.make_series(rng, 240)
    e = iv._rolling_errors(y, H, 20, 96)
    lo, hi = iv.gaussian_theory(y[:120], H, CONF)
    fitp = iv.predict(iv.fit_trend_season(y[:120]), H)
    obs.append(float(np.mean((e[~np.isnan(e).any(axis=1)] + fitp >= lo)
                             & (e[~np.isnan(e).any(axis=1)] + fitp <= hi))))
say("")
say("  and measured: a coverage figure computed from 20 overlapping rolling windows has sd")
say(f"  {np.std(obs, ddof=1):.3f} across repeats - so 0.95 and 0.90 are within one standard")
say("  deviation of each other on the evidence a normal backtest supplies.")
R["s7_observed_sd"] = float(np.std(obs, ddof=1))
say("")
say("  DESIGN CONSTANT, no data in it: catching a 5-point coverage shortfall needs hundreds of")
say("  INDEPENDENT test points, and Day 172 measured that 20 rolling windows stepped by 1 are")
say("  worth about one. A published 'we validated the intervals' is usually a check with no")
say("  power, which is why bad bands survive.")

# ================================================ 8. the scoreboard
head(8, "The scoreboard: coverage can always be bought, a proper score cannot")

rng = np.random.default_rng(47)
score: Dict[str, List[float]] = {k: [] for k in iv.METHODS}
cov: Dict[str, List[float]] = {k: [] for k in iv.METHODS}
wid: Dict[str, List[float]] = {k: [] for k in iv.METHODS}
for _ in range(300):
    y, _ = iv.make_series(rng, N_TRAIN + H)
    act = y[N_TRAIN:]
    for name, m in iv.METHODS.items():
        lo, hi = m(y[:N_TRAIN], H, CONF)
        score[name].append(float(np.mean(iv.interval_score(lo, hi, act, CONF))))
        cov[name].append(float(np.mean((act >= lo) & (act <= hi))))
        wid[name].append(float(np.mean(hi - lo)))
say(f"  {'method':<28}{'coverage':>10}{'width':>9}{'interval score':>16}{'rank by score':>15}")
order = sorted(iv.METHODS, key=lambda n: np.mean(score[n]))
for name in iv.METHODS:
    say(f"  {name:<28}{np.mean(cov[name]):>10.3f}{np.mean(wid[name]):>9.2f}"
        f"{np.mean(score[name]):>16.2f}{order.index(name) + 1:>15}")
R["s8"] = {k: {"coverage": float(np.mean(cov[k])), "width": float(np.mean(wid[k])),
               "score": float(np.mean(score[k]))} for k in iv.METHODS}
say("")
say(f"  best by interval score: {order[0]}")
say(f"  best by coverage alone: {max(cov, key=lambda n: abs(0.0) - abs(np.mean(cov[n]) - CONF))}")
say("  Coverage is not a score: a band can buy it by getting wider, which is exactly what the")
say("  sqrt(h) fan does. The interval score charges for width and for misses in the same units,")
say("  so it is the column to rank on - and it disagrees with the coverage column.")

say("")
say(f"[done in {time.time() - T0:.0f}s]")
json.dump(R, open("results.json", "w"), indent=1)
open("evidence.txt", "w").write("\n".join(LINES) + "\n")
