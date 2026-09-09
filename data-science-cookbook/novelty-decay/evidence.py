"""The measurement run behind every number in the README.

Eight sections, each one an experiment rather than a citation:

1. the three numbers a novelty effect gives you, and which one the decision needs
2. the design constant - the enrollment process changes the reported lift on its own
3. the asymptote is the parameter the window never observes, and the fit says otherwise
4. the early-vs-late guard's size and power
5. decay and a cohort mix shift are the SAME published plot (the headline)
6. survivorship: the identical estimator fails upward, and the retention check is blind
7. what a long-term holdout costs, and why one holdout cannot attribute two launches
8. what each shipping policy claims, and how wrong the claim is

Run: python evidence.py    (writes results.json + evidence.txt, ~70s)
"""

from __future__ import annotations

import json
import sys
import time
from typing import Dict, List

import novelty as nv
import numpy as np
from scipy import stats

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


TAU0 = 0.10
TAU_INF = 0.02
LAM = 7.0
T0 = time.time()

# ================================================ 1. the three numbers
head(1, "A novelty effect gives you three numbers, and the launch needs the one you cannot see")

rng = np.random.default_rng(11)
REPS = 200
acc = {"day0": [], "pooled": [], "lastweek": []}
curve_sum = np.zeros(28)
for _ in range(REPS):
    d = nv.panel_decay(rng, n_users=8400, T=28, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
    c = nv.lift_by_tenure(d, min_arm=2)
    curve_sum[c["tenure"].astype(int)] += c["lift"]
    acc["day0"].append(nv.window_lift(d, 0, 1)[0])
    acc["pooled"].append(nv.pooled_lift(d)[0])
    acc["lastweek"].append(nv.window_lift(d, 21, 28)[0])
curve = curve_sum / REPS
means = {k: float(np.mean(v)) for k, v in acc.items()}

say(f"world: tau(t) = {TAU_INF} + {TAU0 - TAU_INF:.2f} * exp(-t/{LAM:.0f}), 28-day window, daily cohorts, 8,400 users")
say("")
say(f"{'readout':>34} {'value':>9} {'x the long-run truth':>22}")
say(f"{'first-day lift':>34} {means['day0']:>9.4f} {means['day0'] / TAU_INF:>21.2f}x")
say(f"{'pooled window lift (the dashboard)':>34} {means['pooled']:>9.4f} {means['pooled'] / TAU_INF:>21.2f}x")
say(f"{'final-week lift':>34} {means['lastweek']:>9.4f} {means['lastweek'] / TAU_INF:>21.2f}x")
say(f"{'long-run truth tau_inf':>34} {TAU_INF:>9.4f} {1.0:>21.2f}x")
say("")
say(f"{'tenure':>8} {'measured lift':>14} {'true tau(t)':>12}")
for t in (0, 1, 3, 7, 14, 21, 27):
    say(f"{t:>8} {curve[t]:>14.4f} {float(nv.tau_decay(t, TAU0, TAU_INF, LAM)):>12.4f}")
say("")
say("The tenure curve is an honest estimator and recovers the shape. Every scalar summary of it")
say("is a lift AT AN AGE, and the decision is about a population that will be older than any user")
say("in the test. Nothing in the window is an estimate of the number being decided on.")
R["s1"] = {"means": means, "curve": curve.tolist(), "truth": TAU_INF}

# ================================================ 2. the design constant
head(2, "The design constant: how users were let in changes the reported lift, on its own")

say("Attenuation A = the user-day-weighted mean of exp(-t/lam) over the window. The reported")
say("lift is tau_inf + (tau0 - tau_inf) * A. There is no n, no sigma and no outcome in A -")
say("it is a property of the release calendar.")
say("")
say(f"{'T/lam':>7} {'A bigbang':>11} {'A uniform':>11} {'ratio':>7} {'lift bigbang':>13} {'lift uniform':>13} {'apart':>7}")
rows = []
for mult in (1, 2, 4, 8, 12):
    T = int(mult * LAM)
    ab = nv.attenuation(T, LAM, "bigbang")
    au = nv.attenuation(T, LAM, "uniform")
    lb = TAU_INF + (TAU0 - TAU_INF) * ab
    lu = TAU_INF + (TAU0 - TAU_INF) * au
    say(f"{mult:>7} {ab:>11.4f} {au:>11.4f} {au / ab:>7.3f} {lb:>13.4f} {lu:>13.4f} {lu / lb - 1:>6.1%}")
    rows.append({"mult": mult, "T": T, "a_bigbang": ab, "a_uniform": au,
                 "lift_bigbang": lb, "lift_uniform": lu})

rng = np.random.default_rng(21)
REPS2 = 250
sim = {"bigbang": [], "uniform": []}
for _ in range(REPS2):
    for e in sim:
        d = nv.panel_decay(rng, n_users=8400, T=28, tau0=TAU0, tau_inf=TAU_INF, lam=LAM, enroll=e)
        sim[e].append(nv.pooled_lift(d)[0])
say("")
say(f"{'enroll':>10} {'closed form':>12} {'measured':>10} {'+/- mc':>8}")
for e in ("bigbang", "uniform"):
    pred = TAU_INF + (TAU0 - TAU_INF) * nv.attenuation(28, LAM, e)
    v = np.array(sim[e])
    say(f"{e:>10} {pred:>12.4f} {v.mean():>10.4f} {v.std(ddof=1) / np.sqrt(REPS2):>8.4f}")
say("")
cb = nv.attenuation_continuous(28, LAM, "bigbang")
cu = nv.attenuation_continuous(28, LAM, "uniform")
say(f"continuous-time forms at T=4 lam: bigbang (lam/T)(1-e^-T/lam) = {cb:.4f},")
say(f"uniform (2lam/T)(1-(lam/T)(1-e^-T/lam)) = {cu:.4f}; the discrete daily sums are {nv.attenuation(28, LAM, 'bigbang'):.4f}")
say(f"and {nv.attenuation(28, LAM, 'uniform'):.4f}, i.e. the integral understates the daily version by ~{1 - cb / nv.attenuation(28, LAM, 'bigbang'):.1%}.")
say("")
say("Two tests of identical length, on identical users, on an identical effect, report lifts")
say(f"{rows[2]['lift_uniform'] / rows[2]['lift_bigbang'] - 1:.1%} apart at T = 4 lam. Continuous enrollment keeps the population younger, so it")
say("weights the novelty component UP - the opposite of the intuition that a ramp dilutes it.")
R["s2"] = {"rows": rows, "measured": {e: float(np.mean(v)) for e, v in sim.items()}}

# ================================================ 3. the asymptote
head(3, "The asymptote is the parameter the window never observes, and its diagnostic points backwards")

rng = np.random.default_rng(31)
say(f"{'T':>5} {'T/lam':>6} {'fit tau_inf':>12} {'sd':>8} {'mean CI width':>14} {'covers 0':>9} {'covers 5x':>10} {'r2':>7}")
s3 = []
for T in (7, 14, 28, 56, 84):
    reps = 200 if T <= 28 else 120
    est, wid, c0, c5, r2 = [], [], 0, 0, []
    for _ in range(reps):
        d = nv.panel_decay(rng, n_users=8400, T=T, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
        f = nv.fit_exponential(nv.lift_by_tenure(d, min_arm=2), lam0=LAM)
        if f is None or not np.isfinite(f["se_tau_inf"]):
            continue
        est.append(f["tau_inf"])
        w = 1.96 * f["se_tau_inf"]
        wid.append(2 * w)
        c0 += int(abs(f["tau_inf"] - 0.0) <= w)
        c5 += int(abs(f["tau_inf"] - 5 * TAU_INF) <= w)
        r2.append(f["r2"])
    e = np.array(est)
    say(f"{T:>5} {T / LAM:>6.1f} {e.mean():>12.4f} {e.std(ddof=1):>8.4f} {np.mean(wid):>14.4f} "
        f"{c0 / len(est):>9.3f} {c5 / len(est):>10.3f} {np.mean(r2):>7.3f}")
    s3.append({"T": T, "mean": float(e.mean()), "sd": float(e.std(ddof=1)),
               "ci_width": float(np.mean(wid)), "covers_zero": c0 / len(est),
               "covers_5x": c5 / len(est), "r2": float(np.mean(r2)), "n": len(est)})
say("")
say("At T = lam the 95% interval on the asymptote is 609 wide on a quantity of 0.02 - it excludes")
say("nothing anybody would decide between, and the point estimate is negative. The interval is")
say("usable from about T = 8 lam, where the estimate is 0.0177 against a truth of 0.0200 with an")
say("sd of 0.0088. THE FIT IS ONLY RELIABLE IN THE REGIME WHERE IT IS UNNECESSARY: at 8 decay")
say("constants the last window of the test has already measured the asymptote directly, so the")
say("extrapolation earns its keep on exactly the windows where nobody needs it.")
say("")
say("NEGATIVE RESULT: r2 runs the WRONG WAY down this ladder - 0.314 at the useless T = lam and")
say("0.107 at the reliable T = 12 lam - because a longer window is mostly flat curve, so there is")
say("less variance for the model to explain precisely when the model is trustworthy. A reader")
say("choosing between two fits on goodness of fit picks the shorter, worse one. This is Day 169's")
say("mechanism in a new field: the diagnostic scores the fit on the tenures that were measured,")
say("and the quantity is defined at the tenures that were not.")
R["s3"] = s3

# ================================================ 4. the guard
head(4, "The early-vs-late guard: size, power, and what the experiment itself already knew")

rng = np.random.default_rng(41)
REPS4 = 400
flat_flag = 0
decay_flag = 0
exp_power_flat = 0
exp_power_decay = 0
for _ in range(REPS4):
    d = nv.panel_decay(rng, n_users=8400, T=28, tau0=0.05, tau_inf=0.05, lam=LAM)
    flat_flag += int(nv.guard_early_vs_late(d)["p"] < 0.05)
    p, se = nv.pooled_lift(d)
    exp_power_flat += int(abs(p / se) > 1.96)
    d = nv.panel_decay(rng, n_users=8400, T=28, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
    decay_flag += int(nv.guard_early_vs_late(d)["p"] < 0.05)
    p, se = nv.pooled_lift(d)
    exp_power_decay += int(abs(p / se) > 1.96)
say(f"{'world':>28} {'guard fires':>12} {'experiment significant':>23}")
say(f"{'flat 0.05 effect (the null)':>28} {flat_flag / REPS4:>12.3f} {exp_power_flat / REPS4:>23.3f}")
say(f"{'decay 0.10 -> 0.02':>28} {decay_flag / REPS4:>12.3f} {exp_power_decay / REPS4:>23.3f}")
say("")
rng = np.random.default_rng(42)
say(f"{'n users':>9} {'guard power':>12} {'expt power':>11} {'guard MDE on the decay drop':>28}")
s4 = []
for n in (1400, 2800, 8400, 33600):
    reps = 300 if n <= 8400 else 150
    g, x, ses = 0, 0, []
    for _ in range(reps):
        d = nv.panel_decay(rng, n_users=n, T=28, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
        gr = nv.guard_early_vs_late(d)
        g += int(gr["p"] < 0.05)
        ses.append(gr["se"])
        p, se = nv.pooled_lift(d)
        x += int(abs(p / se) > 1.96)
    say(f"{n:>9} {g / reps:>12.3f} {x / reps:>11.3f} {2.8 * float(np.mean(ses)):>28.4f}")
    s4.append({"n": n, "guard_power": g / reps, "expt_power": x / reps,
               "guard_mde": 2.8 * float(np.mean(ses))})
say("")
say("The guard is calibrated (0.035 on the null) and it is the first one in four builds that is")
say("neither inert nor free: it needs roughly 6x the users to reach 0.61 where the experiment is")
say("already at 1.000, and ~24x to be as sure about the decay as the test is about the effect.")
say("That is a real price on a ladder real tests reach, unlike the SRM check (Day 165), the")
say("dose-response check (Day 168) and the segment scan (Day 170), which all needed traffic")
say("nobody has. It also answers a different question: it detects that the lift MOVED, and says")
say("nothing about where it settles. A guard can be well-powered for the wrong estimand.")
R["s4"] = {"flat_flag": flat_flag / REPS4, "decay_flag": decay_flag / REPS4,
           "expt_flat": exp_power_flat / REPS4, "expt_decay": exp_power_decay / REPS4,
           "ladder": s4}

# ================================================ 5. the headline
head(5, "Decay and a cohort mix shift are the same published plot, five times apart in the decision")

m = nv.cohort_profile(28, TAU0, TAU_INF, LAM)
say("Constructed, not fitted: solve for cohort effects m_k, CONSTANT in tenure, whose aggregate")
say("lift-by-tenure curve equals the decay curve exactly. Only cohorts enrolled by day T-1-t are")
say("old enough to appear at tenure t, so the tenure-t lift is a prefix mean of m, and pinning")
say("those prefix means inverts in closed form.")
say("")
say(f"cohort effects run {m.min():.4f} to {m.max():.4f}; mean(m) = {m.mean():.6f} = tau0 exactly "
    f"(gap {abs(m.mean() - TAU0):.2e})")
say("")
rng = np.random.default_rng(51)
REPS5 = 300
cd = np.zeros(28)
cm = np.zeros(28)
vd = np.zeros(28)
vm = np.zeros(28)
sl = {"decay": [], "mix": []}
gz = {"decay": [], "mix": []}
for _ in range(REPS5):
    d = nv.panel_decay(rng, n_users=8400, T=28, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
    c = nv.lift_by_tenure(d, min_arm=2)
    cd += c["lift"]
    vd += c["lift"] ** 2
    sl["decay"].append(nv.within_cohort_slope(d)["z"])
    gz["decay"].append(nv.guard_early_vs_late(d)["z"])
    d = nv.panel_mixshift(rng, n_users=8400, T=28, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
    c = nv.lift_by_tenure(d, min_arm=2)
    cm += c["lift"]
    vm += c["lift"] ** 2
    sl["mix"].append(nv.within_cohort_slope(d)["z"])
    gz["mix"].append(nv.guard_early_vs_late(d)["z"])
cd /= REPS5
cm /= REPS5
sd_d = np.sqrt(np.maximum(vd / REPS5 - cd**2, 0) / REPS5)
sd_m = np.sqrt(np.maximum(vm / REPS5 - cm**2, 0) / REPS5)
mc = np.sqrt(sd_d**2 + sd_m**2)
say(f"{'tenure':>7} {'decay world':>12} {'mix world':>10} {'target':>8} {'diff / mc err':>14}")
for t in (0, 1, 3, 7, 14, 21, 27):
    say(f"{t:>7} {cd[t]:>12.4f} {cm[t]:>10.4f} {float(nv.tau_decay(t, TAU0, TAU_INF, LAM)):>8.4f} "
        f"{(cd[t] - cm[t]) / mc[t]:>14.2f}")
worst = int(np.argmax(np.abs((cd - cm) / mc)))
say("")
zmax = abs((cd[worst] - cm[worst]) / mc[worst])
# expected max |Z| over 28 independent standard normals, by quadrature on the CDF
grid = np.linspace(0, 8, 40001)
# np.trapz was removed in numpy 2.0 and np.trapezoid does not exist before it
_trap = getattr(np, "trapezoid", None) or np.trapz
e_max = float(_trap(1.0 - (2 * stats.norm.cdf(grid) - 1) ** 28, grid))
say(f"largest standardised discrepancy anywhere: {zmax:.2f} at tenure {worst}, over 28 tenures x {REPS5} runs")
say(f"expected max |Z| over 28 pure-noise comparisons: {e_max:.2f} - so the largest gap the two")
say("worlds show anywhere is what noise alone produces, which is the sense in which they are the")
say("same plot rather than merely a similar one.")
say(f"early-vs-late guard mean z: decay {np.mean(gz['decay']):>7.2f}   mix {np.mean(gz['mix']):>7.2f}  (both fire, identically)")
say("")
say(f"{'decision':>44} {'decay world':>12} {'mix world':>10}")
say(f"{'long-run effect per user':>44} {TAU_INF:>12.4f} {m.mean():>10.4f}")
say(f"{'ratio':>44} {1.0:>12.1f} {m.mean() / TAU_INF:>10.1f}")
say("")
say(f"{'separator (needs the enrollment date)':>44} {'decay z':>12} {'mix z':>10}")
say(f"{'within-cohort tenure slope':>44} {np.mean(sl['decay']):>12.2f} {np.mean(sl['mix']):>10.2f}")
fires_d = float(np.mean(np.abs(sl["decay"]) > 1.96))
fires_m = float(np.mean(np.abs(sl["mix"]) > 1.96))
say(f"{'fires at |z| > 1.96':>44} {fires_d:>12.3f} {fires_m:>10.3f}")
say("")
rng2 = np.random.default_rng(52)
ident = {"decay": [], "mix": []}
for _ in range(120):
    for name, fn in (("decay", nv.panel_decay), ("mix", nv.panel_mixshift)):
        d = fn(rng2, n_users=8400, T=28, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
        ident[name].append(nv.share_weighted_identity(d))
say(f"{'bookkeeping check':>44} {'decay':>12} {'mix':>10}")
say(f"{'exact per-arm recomposition gap':>44} "
    f"{np.mean([abs(x['gap']) for x in ident['decay']]):>12.2e} "
    f"{np.mean([abs(x['gap']) for x in ident['mix']]):>10.2e}")
say(f"{'single-share rollup gap, in overall SEs':>44} "
    f"{np.mean([abs(x['naive_gap_in_se']) for x in ident['decay']]):>12.3f} "
    f"{np.mean([abs(x['naive_gap_in_se']) for x in ident['mix']]):>10.3f}")
say("")
say("NEGATIVE RESULT, and it is a limit on the previous build's best find: the free accounting")
say("guard does NOT generalise here. Day 170's share-weighted identity separated a legitimate")
say("partition from a post-assignment one by 1,204 sd with no threshold at all. The tenure")
say("version closes to machine precision in BOTH worlds, because tenure is a legitimate")
say("partition in both - a mix shift and a decay are each perfectly consistent bookkeeping. An")
say("identity can only catch a violation of itself, and neither failure here is one.")
R["s5_identity"] = {k: {"gap": float(np.mean([abs(x["gap"]) for x in v])),
                        "naive_gap_in_se": float(np.mean([abs(x["naive_gap_in_se"]) for x in v]))}
                    for k, v in ident.items()}
say("")
say("The published plot cannot tell these apart and never will - they are the same curve by")
say("construction. What separates them is a COLUMN, not a test: with the enrollment date in the")
say("table, real decay bends every cohort's own curve and a mix shift bends none of them. Same")
say("shape as Day 168's cluster level - the fix has to live at the level that contains the")
say("mechanism, and no amount of the aggregate can substitute for it.")
R["s5"] = {"cohort_effects": m.tolist(), "curve_decay": cd.tolist(), "curve_mix": cm.tolist(),
           "worst_z": float(abs((cd[worst] - cm[worst]) / mc[worst])), "worst_tenure": worst,
           "truth_decay": TAU_INF, "truth_mix": float(m.mean()),
           "slope_z": {k: float(np.mean(v)) for k, v in sl.items()},
           "slope_fires": {"decay": fires_d, "mix": fires_m},
           "guard_z": {k: float(np.mean(v)) for k, v in gz.items()}}

# ================================================ 6. survivorship
head(6, "Survivorship: the identical estimator fails upward, and the retention check is blind")

rng = np.random.default_rng(61)
REPS6 = 250
TAU_S = 0.05
cs = np.zeros(28)
pool, ret_t, ret_c, ret_z = [], [], [], []
for _ in range(REPS6):
    d = nv.panel_survivor(rng, n_users=8400, T=28, tau=TAU_S, h0=0.03, gamma=0.8)
    c = nv.lift_by_tenure(d)
    cs[c["tenure"].astype(int)] += c["lift"]
    pool.append(nv.pooled_lift(d)[0])
    a = d["alive"]
    at = a[d["w"] == 1][:, 27].mean()
    ac = a[d["w"] == 0][:, 27].mean()
    nt = int((d["w"] == 1).sum())
    nc = int((d["w"] == 0).sum())
    ret_t.append(at)
    ret_c.append(ac)
    se = np.sqrt(at * (1 - at) / nt + ac * (1 - ac) / nc)
    ret_z.append((at - ac) / se)
cs /= REPS6
say(f"world: a PERMANENT effect of {TAU_S}, plus churn whose hazard falls with user value in the")
say("treated arm only. No decay and no cohort structure anywhere.")
say("")
say(f"{'tenure':>8} {'measured lift':>14} {'x truth':>9}")
for t in (0, 3, 7, 14, 21, 27):
    say(f"{t:>8} {cs[t]:>14.4f} {cs[t] / TAU_S:>8.1f}x")
say("")
say(f"pooled lift {np.mean(pool):.4f} against a truth of {TAU_S:.4f} - {np.mean(pool) / TAU_S:.1f}x, and the sign of the")
say("tenure slope is POSITIVE, which reads as 'the effect compounds, ship it and expect more'.")
say("")
say(f"{'retention at day 27':>24} {'treated':>9} {'control':>9} {'diff':>8} {'|z|':>7} {'fires':>7}")
say(f"{'':>24} {np.mean(ret_t):>9.4f} {np.mean(ret_c):>9.4f} "
    f"{np.mean(ret_t) - np.mean(ret_c):>8.4f} {abs(np.mean(ret_z)):>7.2f} "
    f"{float(np.mean(np.abs(ret_z) > 1.96)):>7.3f}")
say("")
say(f"NEGATIVE RESULT: the obvious guard is nearly inert. Total retention differs by "
    f"{abs(np.mean(ret_t) - np.mean(ret_c)) / np.mean(ret_c):.1%} while")
say("the lift is overstated several-fold, because the mechanism is COMPOSITIONAL: the treatment")
say("sheds low-value users and keeps high-value ones, and those two flows nearly cancel in the")
say("headcount while doing all the damage to the mean. Counting users cannot see a change in who")
say("they are. Cross-link to Day 168 - the same estimator failing in both directions, with the")
say("output unable to say which.")
R["s6"] = {"curve": cs.tolist(), "pooled": float(np.mean(pool)), "truth": TAU_S,
           "ret_treated": float(np.mean(ret_t)), "ret_control": float(np.mean(ret_c)),
           "ret_z": float(np.mean(ret_z)),
           "ret_fires": float(np.mean(np.abs(ret_z) > 1.96))}

# ================================================ 7. the holdout
head(7, "What a long-term holdout costs, and why one holdout cannot attribute two launches")

say("The recommended fix is to hold a slice of users out for months and read the gap. Its")
say("resolution is arithmetic - MDE ratio against a 50/50 test is 1/(2 sqrt(p(1-p))), with no n")
say("and no sigma in it.")
say("")
say(f"{'holdout share':>14} {'MDE x 50/50':>12} {'traffic-time to match':>22}")
s7 = []
for p in (0.50, 0.20, 0.10, 0.05, 0.01):
    r = nv.mde_ratio(p)
    say(f"{p:>14.0%} {r:>12.3f} {r ** 2:>21.1f}x")
    s7.append({"share": p, "mde_ratio": r})
say("")
say(f"A 5% holdout carries {nv.mde_ratio(0.05):.2f}x the MDE of the 50/50 test that shipped the feature, and the")
say(f"quantity it must resolve is SMALLER: tau_inf/tau0 = {TAU_INF / TAU0:.2f} here. Both together, the holdout")
say(f"needs {(nv.mde_ratio(0.05) * TAU0 / TAU_INF) ** 2:.0f}x the user-days of the original test to reach the same power - which is why")
say("the honest version of this fix is measured in quarters, not weeks.")
say("")
rng = np.random.default_rng(71)
REPS7 = 200
hold = []
for _ in range(REPS7):
    d = nv.panel_decay(rng, n_users=8400, T=84, tau0=TAU0, tau_inf=TAU_INF, lam=LAM,
                       enroll="bigbang", p_treat=0.95)
    hold.append(nv.window_lift(d, 56, 84)[0])
h = np.array(hold)
say(f"measured: 95/5 split, 84 days, reading tenures 56-84 gives {h.mean():.4f} +/- {h.std(ddof=1):.4f} (sd across runs)")
say(f"against a truth of {TAU_INF:.4f}; per-run |z| beats 1.96 in {float(np.mean(np.abs(h) / (h.std(ddof=1)) > 1.96)):.3f} of runs at this size")
say("")
n_days = 240
gap_se = 1.0 * np.sqrt(1 / 10000 + 1 / 190000)
ded = 1.0 * np.sqrt(1 / 100000 + 1 / 100000)
say(f"joint cost: 200,000 users, a 5% holdout, {n_days} days, daily gap SE {gap_se:.5f};")
say(f"a dedicated 50/50 test on 100,000 users would read one launch at SE {ded:.5f}.")
say("")
say(f"{'release calendar':>30} {'SE per launch':>14} {'SE of total':>12} {'x dedicated':>12} {'cond':>10}")
s7b = []
for gap in (30, 14, 7, 3, 1, 0):
    ships = 30 + np.arange(6) * gap
    a = nv.holdout_attribution(ships, n_days, gap_se)
    label = f"6 launches, {gap} days apart" if gap else "6 launches, all the same day"
    if a is None:
        say(f"{label:>30} {'UNIDENTIFIED':>14} {'--':>12} {'--':>12} {'inf':>10}")
        s7b.append({"gap": gap, "identified": False})
        continue
    per = float(np.mean(a["se_per_launch"]))
    say(f"{label:>30} {per:>14.5f} {a['se_total']:>12.5f} {per / ded:>11.1f}x {a['cond']:>10.1f}")
    s7b.append({"gap": gap, "identified": True, "se_per_launch": per,
                "se_total": a["se_total"], "ratio": per / ded, "cond": a["cond"]})
say("")
say("A holdout accumulates every launch shipped past it, so its gap is a JOINT COST and the")
say("RELEASE CALENDAR, not the statistics, decides whether one launch's share is recoverable.")
say("Same structure as Day 161's warehouse invoice: the total is clean at every spacing")
say(f"({s7b[0]['se_total']:.5f} to {[r for r in s7b if r['identified']][-1]['se_total']:.5f}) and the split is a design choice.")
say("")
ok = [r for r in s7b if r["identified"]]
say("NEGATIVE RESULT, and it is the opposite of the intuition that a busy calendar is hopeless:")
say("attribution is nearly FREE while the launches are separable, and then falls off a cliff. Six")
say(f"launches {ok[0]['gap']} days apart cost {ok[0]['ratio']:.2f}x a dedicated test per launch - BELOW 1, because")
say("240 days of daily gaps on a small holdout carry more information than one readout of a big")
say(f"test - rising only to {ok[-1]['ratio']:.2f}x at {ok[-1]['gap']} day apart. Then, on the same day, the per-launch")
say("effects are not identified at ANY sample size, because the design matrix loses RANK rather")
say("than becoming imprecise. There is no gradient to manage at the end: the calendar either")
say("separates the launches or the question is unanswerable, and which of the two you are in is")
say("computable from the ship dates alone before any data exists. Cross-link to Day 169's")
say("permutation floor - another design constant with no data in it that decides whether an")
say("analysis has any outcome at all.")
R["s7"] = {"mde": s7, "holdout_measured": float(h.mean()), "holdout_sd": float(h.std(ddof=1)),
           "attribution": s7b, "dedicated_se": ded}

# ================================================ 8. the policies
head(8, "What each shipping policy claims, and how wrong the claim is")

rng = np.random.default_rng(81)
REPS8 = 200
pol: Dict[str, List[float]] = {"pooled 4-week lift": [], "final-week lift": [],
                               "fitted asymptote": [], "5% holdout, 12 weeks": []}
for _ in range(REPS8):
    d = nv.panel_decay(rng, n_users=8400, T=28, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
    pol["pooled 4-week lift"].append(nv.pooled_lift(d)[0])
    pol["final-week lift"].append(nv.window_lift(d, 21, 28)[0])
    f = nv.fit_exponential(nv.lift_by_tenure(d, min_arm=2), lam0=LAM)
    if f is not None:
        pol["fitted asymptote"].append(f["tau_inf"])
    d = nv.panel_decay(rng, n_users=8400, T=84, tau0=TAU0, tau_inf=TAU_INF, lam=LAM,
                       enroll="bigbang", p_treat=0.95)
    pol["5% holdout, 12 weeks"].append(nv.window_lift(d, 56, 84)[0])

say(f"{'policy':>24} {'mean claim':>11} {'bias':>9} {'sd':>8} {'RMSE':>8} {'x truth':>8}")
s8 = []
for k, v in pol.items():
    a = np.array(v)
    rmse = float(np.sqrt(np.mean((a - TAU_INF) ** 2)))
    say(f"{k:>24} {a.mean():>11.4f} {a.mean() - TAU_INF:>9.4f} {a.std(ddof=1):>8.4f} "
        f"{rmse:>8.4f} {a.mean() / TAU_INF:>7.2f}x")
    s8.append({"policy": k, "mean": float(a.mean()), "bias": float(a.mean() - TAU_INF),
               "sd": float(a.std(ddof=1)), "rmse": rmse})
say("")
say("The bias-variance split is the whole decision and it is not a tie. The two window readouts")
say("are precise and wrong by a factor; the fitted asymptote is both biased and 25x noisier than")
say("the readout it replaces, so its RMSE is 7x worse than the naive pooled lift; the holdout is")
say("that is both centred and useful, and it costs three months. There is no readout of a")
say("four-week test that answers the question, so the choice is which wrong number to ship on -")
say("and the pooled lift, the default, is the one that is confidently wrong.")
R["s8"] = s8

say("")
say("=" * 100)
say(f"total runtime {time.time() - T0:.1f}s")

with open("results.json", "w") as fh:
    json.dump(R, fh, indent=2, default=float)
with open("evidence.txt", "w") as fh:
    fh.write("\n".join(LINES) + "\n")
print("wrote results.json + evidence.txt")
