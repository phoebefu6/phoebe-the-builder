"""The measurement run behind every number in the README.

Eight sections, each one an experiment rather than a citation:

1. the two estimators, doing the job they are good at
2. the design constant - a segment scan's alarm rate is arithmetic, and my first closed
   form for it was wrong
3. the winner's curse - what the worst segment's number is an estimate of
4. the guard's power - a segment scan is blind by construction
5. correlated cuts - "we looked at 20 slices" is not "we ran 20 tests"
6. the CATE tree's own subgroup is a fitted object (the headline)
7. slicing on a variable measured after assignment, and the one guard in this build with
   no power problem, because it is algebra rather than a test
8. what the "no segment may be harmed" gate actually buys

Run: python evidence.py    (writes results.json + evidence.txt, ~80s)
"""

from __future__ import annotations

import json
import sys
import time
from typing import Dict, List

import hte
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


ALPHA = 0.05
ATE = 0.05
T0 = time.time()

# ============================================================ 1. the estimators work
head(1, "The two estimators, on the problems they are actually good at")

rng = np.random.default_rng(11)
REPS, N, K, DELTA = 400, 20000, 20, 0.30
caught, ate_sig = 0, 0
for _ in range(REPS):
    d = hte.trial_partition(rng, n=N, k=K, ate=ATE, delta=DELTA, harmed=0)
    res = hte.scan(d["y"], d["w"], hte.partition_matrix(d["seg"], K))
    caught += int(res["est"][0] < 0 and hte.bonferroni(res["p"])[0] < ALPHA)
    est, se = hte.diff_means(d["y"], d["w"])
    ate_sig += int(abs(est / se) > 1.96)
say(f"one truly harmed segment (tau = {ATE:.2f} - {DELTA:.2f} = {ATE - DELTA:+.2f}), n={N:,}, K={K}")
say(f"  scan finds it (Bonferroni)       {caught / REPS:.3f}")
say(f"  the overall test is significant  {ate_sig / REPS:.3f}   true ATE {ATE - DELTA / K:+.4f}")
R["s1_scan_power_big_harm"] = caught / REPS
R["s1_overall_power"] = ate_sig / REPS

rng = np.random.default_rng(12)
REPS_T, NT, BETA = 200, 8000, 0.20
r_tree, r_seg, r_const = [], [], []
for _ in range(REPS_T):
    c = hte.trial_continuous(rng, n=NT, ate=ATE, beta=BETA, k=K)
    h = hte.honest_fit(rng, c["x"], c["w"], c["y"], depth=2, min_leaf=150)
    b = h["b"]
    r_tree.append(hte.rmse(hte.tau_hat_rows(h["tree"], None, h["honest"], c["x"][b]), c["tau"][b]))
    res = hte.scan(c["y"], c["w"], hte.partition_matrix(c["seg"], K))
    r_seg.append(hte.rmse(res["est"][c["seg"][b]], c["tau"][b]))
    r_const.append(hte.rmse(np.full(b.size, hte.diff_means(c["y"], c["w"])[0]), c["tau"][b]))
say("")
say(f"heterogeneity along a covariate: tau(x) = {ATE:.2f} + {BETA:.2f}*x0, sliced on an unrelated segment")
say(f"  RMSE, honest CATE tree           {np.mean(r_tree):.4f}")
say(f"  RMSE, the {K}-segment scan        {np.mean(r_seg):.4f}")
say(f"  RMSE, one number for everybody   {np.mean(r_const):.4f}   (sd of true tau {BETA:.4f})")
R["s1_rmse_tree"] = float(np.mean(r_tree))
R["s1_rmse_scan"] = float(np.mean(r_seg))
R["s1_rmse_constant"] = float(np.mean(r_const))
say(f"  the scan carries {np.mean(r_seg) / np.mean(r_tree):.2f}x the tree's error and {np.mean(r_seg) / np.mean(r_const):.2f}x the")
say("  do-nothing baseline's: slicing in the wrong place is WORSE than not slicing at all,")
say("  because it adds 20 segments' worth of noise and no signal.")

# ============================================================ 2. the design constant
head(2, "THE DESIGN CONSTANT: the alarm rate is arithmetic - and 1-(1-0.025)^K is the wrong arithmetic")

say("Homogeneous world: every segment has exactly the same true effect, so any segment the scan")
say("calls HARMED is a false alarm. The number everybody quotes for this is 1-(1-0.025)^K = 39.7%")
say("at K=20. That expression is this build's world at a true effect of ZERO. With a real effect")
say("the whole family shifts and the alarm rate falls - so the closed form needs the effect in it:")
say("")
say("    P(harm flag) = 1 - (1 - Phi(-1.96 - ate/se_seg))^K,   se_seg = sigma*sqrt(4K/n)")
say("")
say(f"{'K':>4} {'true ate':>9} {'se_seg':>7} {'measured':>9} {'naive cf':>9} {'this cf':>8} {'bonf':>7} {'bh':>7}")
rng = np.random.default_rng(21)
REPS2 = 2000
rows2 = []
for k, ate in ((20, 0.0), (20, 0.05), (20, 0.10), (5, 0.05), (50, 0.05)):
    h_ = bo = bhf = 0
    for _ in range(REPS2):
        d = hte.trial_partition(rng, n=8000, k=k, ate=ate)
        res = hte.scan(d["y"], d["w"], hte.partition_matrix(d["seg"], k))
        h_ += int(hte.harm_flag(res))
        bo += int(hte.harm_flag(res, method="bonferroni"))
        bhf += int(hte.harm_flag(res, method="bh"))
    naive = hte.fwer_independent(k, ALPHA / 2)
    mine = hte.harm_alarm_closed_form(k, 8000, ate)
    say(
        f"{k:>4} {ate:>9.2f} {hte.segment_se(8000, k):>7.3f} {h_ / REPS2:>9.3f} {naive:>9.3f}"
        f" {mine:>8.3f} {bo / REPS2:>7.3f} {bhf / REPS2:>7.3f}"
    )
    rows2.append(
        {"k": k, "ate": ate, "measured": h_ / REPS2, "naive_cf": naive, "cf": mine, "bonf": bo / REPS2, "bh": bhf / REPS2}
    )
R["s2_table"] = rows2
say("")
say("The naive form is out by a factor of 3 at K=20 on a launch that worked (0.397 vs 0.130). The")
say("form with the effect in it matches to MC error everywhere except K=50, where a segment holds")
say("80 users per arm and the normal approximation inside the test is itself liberal.")
say("")
say("The uncomfortable reading: a scan is MOST likely to invent a harmed segment exactly when the")
say("launch did nothing at all, and a strong launch is partly self-protecting. The story 'it")
say("worked overall but hurt segment 7' is least trustworthy in the cases people tell it about.")

say("")
say("NEGATIVE RESULT - the house pattern does NOT hold here. Days 165/167/168 all found a bias or")
say("a size flat in n while the experiment's power rose. This one MOVES, because the reference is")
say("not a zero effect: ate/se_seg grows with sqrt(n), so a bigger test is a safer scan.")
say(f"{'n':>9} {'measured':>9} {'closed form':>12} {'overall power':>14}")
rng = np.random.default_rng(22)
rows2b = []
for n in (2000, 8000, 32000, 128000):
    h_, pw = 0, 0
    for _ in range(600):
        d = hte.trial_partition(rng, n=n, k=20, ate=ATE)
        res = hte.scan(d["y"], d["w"], hte.partition_matrix(d["seg"], 20))
        h_ += int(hte.harm_flag(res))
        est, se = hte.diff_means(d["y"], d["w"])
        pw += int(est / se > 1.96)
    cf = hte.harm_alarm_closed_form(20, n, ATE)
    say(f"{n:>9,} {h_ / 600:>9.3f} {cf:>12.3f} {pw / 600:>14.3f}")
    rows2b.append({"n": n, "measured": h_ / 600, "cf": cf, "power": pw / 600})
R["s2_scaling_in_n"] = rows2b

# ============================================================ 3. winner's curse
head(3, "The worst segment's number is not an estimate of that segment's effect")

rng = np.random.default_rng(31)
REPS3 = 1500
worst, worst_repl, worst_flagged, se_mean = [], [], [], []
for _ in range(REPS3):
    d = hte.trial_partition(rng, n=8000, k=20, ate=ATE)
    res = hte.scan(d["y"], d["w"], hte.partition_matrix(d["seg"], 20))
    j = int(np.argmin(res["est"]))
    worst.append(res["est"][j])
    se_mean.append(res["se"][j])
    worst_flagged.append(res["p"][j] < ALPHA and res["est"][j] < 0)
    d2 = hte.trial_partition(rng, n=8000, k=20, ate=ATE)
    m = d2["seg"] == j
    worst_repl.append(hte.diff_means(d2["y"][m], d2["w"][m])[0])
w_mean, se_bar = float(np.mean(worst)), float(np.mean(se_mean))
pred = ATE + se_bar * hte.expected_extreme_z(20, "min")
say(f"K=20, n=8,000, true effect {ATE:+.2f} in EVERY segment. The segment that came out worst:")
say(f"  reported effect                 {w_mean:+.4f}  +/- {np.std(worst, ddof=1) / np.sqrt(REPS3):.4f}")
say(f"  closed form ate + se*E[min Z20] {pred:+.4f}   (E[min Z20] = {hte.expected_extreme_z(20, 'min'):+.4f})")
say(f"  same segment, fresh sample      {np.mean(worst_repl):+.4f}  <- the truth is {ATE:+.2f}")
say(f"  and it is called significant    {np.mean(worst_flagged):.3f} of the time")
say("")
say(f"The reported harm is {abs(w_mean - ATE) / se_bar:.2f} standard errors of pure selection. Nothing about that")
say("segment produced it; the MINIMUM operator produced it, and the quadrature knows the number")
say(f"before the experiment runs (predicted {pred:+.4f}, measured {w_mean:+.4f}, gap {abs(pred - w_mean):.4f}).")
say("Re-running the same segment returns the truth, which is why 'it did not replicate' is the")
say("usual epilogue and why nobody learns anything from it.")
R["s3_worst_reported"] = w_mean
R["s3_worst_closed_form"] = float(pred)
R["s3_worst_replication"] = float(np.mean(worst_repl))
R["s3_worst_flag_rate"] = float(np.mean(worst_flagged))

rng = np.random.default_rng(32)
repl_hit = 0
REPS3B = 600
for _ in range(REPS3B):
    d = hte.trial_partition(rng, n=8000, k=20, ate=ATE, delta=0.30, harmed=0)
    res = hte.scan(d["y"], d["w"], hte.partition_matrix(d["seg"], 20))
    repl_hit += int(int(np.argmin(res["est"])) == 0)
say("")
say(f"With a real harmed segment (tau {ATE - 0.30:+.2f} in segment 0) the worst segment IS segment 0")
say(f"{repl_hit / REPS3B:.1%} of the time - the RANKING carries information even where the number does not.")
R["s3_worst_is_true_harmed"] = repl_hit / REPS3B

# ============================================================ 4. the guard is blind
head(4, "NEGATIVE RESULT: a segment scan is blind by construction, because a segment is 1/K of the data")

n_arm = 8000 / 2
say(f"n=8,000, K=20. The experiment's own MDE is {hte.mde(n_arm):.4f}; a segment holds 400 users, so")
say(f"its MDE is {hte.mde(n_arm / 20):.4f} - {np.sqrt(20):.2f}x larger, exactly sqrt(K), and a Bonferroni-corrected")
say("segment test is larger again. That is arithmetic. The simulation only confirms it.")
say("")
say(f"{'harm delta':>11} {'true seg tau':>13} {'pre-reg 1':>10} {'scan 20 bonf':>13} {'scan 20 bh':>11} {'overall +ve':>12}")
rng = np.random.default_rng(41)
rows4 = []
for delta in (0.10, 0.30, 0.60, 1.00, 2.00):
    raw = bo = bhh = ov = 0
    for _ in range(500):
        d = hte.trial_partition(rng, n=8000, k=20, ate=ATE, delta=delta, harmed=0)
        res = hte.scan(d["y"], d["w"], hte.partition_matrix(d["seg"], 20))
        raw += int(res["est"][0] < 0 and res["p"][0] < ALPHA)
        bo += int(res["est"][0] < 0 and hte.bonferroni(res["p"])[0] < ALPHA)
        bhh += int(res["est"][0] < 0 and hte.bh(res["p"])[0] < ALPHA)
        est, se = hte.diff_means(d["y"], d["w"])
        ov += int(est / se > 1.96)
    say(
        f"{delta:>11.2f} {ATE - delta:>13.2f} {raw / 500:>10.3f} {bo / 500:>13.3f} {bhh / 500:>11.3f} {ov / 500:>12.3f}"
    )
    rows4.append({"delta": delta, "prereg": raw / 500, "bonf": bo / 500, "bh": bhh / 500, "overall": ov / 500})
R["s4_power"] = rows4
say("")
say("'pre-reg 1' is the same test at alpha 0.05 on ONE named segment: naming the segment in advance")
say("costs nothing in power and deletes section 2 entirely. The gap pre-reg -> bonf is the price of")
say("the other 19 looks - 0.652 to 0.266 at delta 0.30, i.e. scanning costs 59% of the detections.")
say("Also visible: the overall column FALLS as the harm grows, because one segment's harm is diluted")
say("1/K. At delta 1.00 the launch is a wash overall (ATE +0.000) while a twentieth of users lose")
say("0.95 - the aggregate test is not a weak version of the segment question, it is a different one.")

rng = np.random.default_rng(42)
rows4b = []
say("")
say("How much traffic does the guard need to reach 0.80 on a harm the size of the ATE itself?")
say(f"{'n':>10} {'P(catch, bonf)':>15} {'overall power':>14}")
for n in (8000, 40000, 200000, 1000000):
    hit, ov = 0, 0
    reps = 300 if n <= 200000 else 120
    for _ in range(reps):
        d = hte.trial_partition(rng, n=n, k=20, ate=ATE, delta=2 * ATE, harmed=0)
        res = hte.scan(d["y"], d["w"], hte.partition_matrix(d["seg"], 20))
        hit += int(res["est"][0] < 0 and hte.bonferroni(res["p"])[0] < ALPHA)
        est, se = hte.diff_means(d["y"], d["w"])
        ov += int(est / se > 1.96)
    say(f"{n:>10,} {hit / reps:>15.3f} {ov / reps:>14.3f}")
    rows4b.append({"n": n, "catch": hit / reps, "overall": ov / reps})
R["s4_traffic"] = rows4b
say("")
say("The experiment is done at 40,000. The guard protecting it is at 0.027 there and needs roughly")
say("25x that traffic to be worth reading - the same shape as Day 165's SRM guard and Day 168's")
say("dose-response guard: calibrated, cheap to run, and blind at the sample size you have.")

# ============================================================ 5. correlated cuts
head(5, "'We looked at 20 slices' is not 'we ran 20 tests' - and the gap is measurable")

say("20 overlapping binary cuts (mobile / new / EU / ...), each about half the users, sharing one")
say("latent factor with loading rho. True effect 0 everywhere, so every flag is a false alarm and")
say("k-effective is read straight off the alarm rate.")
say("")
say(f"{'rho':>5} {'P(harm flag)':>13} {'k effective':>12} {'bonf FWER':>10} {'raw power':>10} {'bonf power':>11}")
rng = np.random.default_rng(51)
rows5 = []
for rho in (0.0, 0.3, 0.6, 0.9, 0.99):
    h_, bo = 0, 0
    for _ in range(700):
        d = hte.trial_slices(rng, n=8000, n_slices=20, rho=rho, ate=0.0)
        res = hte.scan(d["y"], d["w"], d["slices"])
        h_ += int(hte.harm_flag(res))
        bo += int(hte.harm_flag(res, method="bonferroni"))
    keff = hte.k_effective(h_ / 700, ALPHA / 2)
    pw_raw = pw_bo = 0
    for _ in range(400):
        d = hte.trial_slices(rng, n=8000, n_slices=20, rho=rho, ate=0.0)
        y = d["y"] + d["w"] * np.where(d["slices"][:, 0], -0.08, 0.0)
        res = hte.scan(y, d["w"], d["slices"])
        pw_raw += int(res["est"][0] < 0 and res["p"][0] < ALPHA)
        pw_bo += int(res["est"][0] < 0 and hte.bonferroni(res["p"])[0] < ALPHA)
    say(
        f"{rho:>5.2f} {h_ / 700:>13.3f} {keff:>12.1f} {bo / 700:>10.3f} {pw_raw / 400:>10.3f} {pw_bo / 400:>11.3f}"
    )
    rows5.append(
        {"rho": rho, "harm": h_ / 700, "k_eff": keff, "bonf_fwer": bo / 700, "raw_power": pw_raw / 400, "bonf_power": pw_bo / 400}
    )
R["s5_table"] = rows5
say("")
say("Even at rho=0 the 20 cuts behave like ~10 tests, not 20 - independent DEFINITIONS still make")
say("dependent ESTIMATES, because two half-sample slices share a quarter of the users, giving a")
say("correlation of 0.5 between their effect estimates before any correlation in the cuts")
say("themselves. So Bonferroni over overlapping slices is already 2x conservative at rho=0, and")
say("by rho=0.99 it is 7x: k-effective 2.8, family rate 0.069, alpha divided by 20 anyway, and")
say("power more than halved (0.708 raw -> 0.300). Both numbers are computable from the slice")
say("memberships alone, before any outcome is looked at - and neither is the number 20.")

# ============================================================ 6. the headline
head(6, "NEGATIVE RESULT: the CATE tree's subgroup is a FITTED object, and its effect is its objective")

say("True world: tau = 0.05 for EVERY user, five covariates, none of them doing anything. There is")
say("no heterogeneity to find. Fit a depth-2 transformed-outcome tree, then read the worst leaf's")
say("effect two ways - from the data that chose the split, and from a held-out half.")
say("")
rng = np.random.default_rng(61)
REPS6 = 300
ins, hon, cov_i, cov_h, sig_i, sig_h, nl = [], [], [], [], [], [], []
for _ in range(REPS6):
    c = hte.trial_continuous(rng, n=4000, ate=ATE, beta=0.0, k=20)
    h = hte.honest_fit(rng, c["x"], c["w"], c["y"], depth=2, min_leaf=100)
    j = int(np.argmin(h["in_sample"]["est"]))
    ins.append(h["in_sample"]["est"][j])
    hon.append(h["honest"]["est"][j])
    cov_i.append(hte.ci_covers(h["in_sample"]["est"][j], h["in_sample"]["se"][j], ATE))
    cov_h.append(hte.ci_covers(h["honest"]["est"][j], h["honest"]["se"][j], ATE))
    sig_i.append(h["in_sample"]["est"][j] < 0 and h["in_sample"]["p"][j] < ALPHA)
    sig_h.append(h["honest"]["est"][j] < 0 and h["honest"]["p"][j] < ALPHA)
    nl.append(hte.n_leaves(h["tree"]))
say(f"leaves found                       {np.mean(nl):.2f}")
say(f"worst leaf, in-sample effect       {np.mean(ins):+.4f}   truth {ATE:+.2f}")
say(f"worst leaf, honest effect          {np.mean(hon):+.4f}")
say(f"in-sample CI covers the truth      {np.mean(cov_i):.3f}   <- nominal 0.95")
say(f"honest CI covers the truth         {np.mean(cov_h):.3f}")
say(f"in-sample calls it harmed (p<.05)  {np.mean(sig_i):.3f}")
say(f"honest calls it harmed (p<.05)     {np.mean(sig_h):.3f}")
R["s6_null"] = {
    "leaves": float(np.mean(nl)),
    "in_sample": float(np.mean(ins)),
    "honest": float(np.mean(hon)),
    "cov_in": float(np.mean(cov_i)),
    "cov_honest": float(np.mean(cov_h)),
    "sig_in": float(np.mean(sig_i)),
    "sig_honest": float(np.mean(sig_h)),
}
say("")
say("A subgroup that does not exist is reported as harmed by half of these runs, with an interval")
say("that covers the truth 0.30 of the time. The tree is not misbehaving - it was asked for the")
say("split that maximises the difference in the effect signal, and reading the effect off the same")
say("rows is reading the objective function back out. Day 169 found this in a synthetic control's")
say("pre-period fit; it is the same failure with a different fitted object, and the fix is the same")
say("shape: score the thing on data that did not choose it.")

say("")
say("With REAL heterogeneity the honest half is not free - it costs half the rows:")
rng = np.random.default_rng(62)
rows6 = []
for beta in (0.0, 0.10, 0.20, 0.40):
    i_err, h_err, i_cov, h_cov = [], [], [], []
    for _ in range(200):
        c = hte.trial_continuous(rng, n=4000, ate=ATE, beta=beta, k=20)
        h = hte.honest_fit(rng, c["x"], c["w"], c["y"], depth=2, min_leaf=100)
        for key, errs, covs in (("in_sample", i_err, i_cov), ("honest", h_err, h_cov)):
            idx = h["b"] if key == "honest" else h["a"]
            lid = hte.leaf_of(h["tree"], c["x"][idx])
            truth = np.array(
                [c["tau"][idx][lid == j].mean() if np.any(lid == j) else np.nan for j in range(hte.n_leaves(h["tree"]))]
            )
            est, se = h[key]["est"], h[key]["se"]
            ok = np.isfinite(truth) & np.isfinite(est) & np.isfinite(se)
            errs.append(float(np.mean(np.abs(est[ok] - truth[ok]))))
            covs.append(float(np.mean(np.abs(est[ok] - truth[ok]) <= 1.96 * se[ok])))
    say(
        f"  beta={beta:.2f}  leaf |error| in-sample {np.mean(i_err):.4f} honest {np.mean(h_err):.4f}"
        f"   coverage {np.mean(i_cov):.3f} / {np.mean(h_cov):.3f}"
    )
    rows6.append(
        {
            "beta": beta,
            "err_in": float(np.mean(i_err)),
            "err_honest": float(np.mean(h_err)),
            "cov_in": float(np.mean(i_cov)),
            "cov_honest": float(np.mean(h_cov)),
        }
    )
R["s6_beta"] = rows6
say("The in-sample leaf estimate is worse at EVERY beta, including beta=0.40 where the")
say("heterogeneity is loud - selection bias does not wash out once the signal is real, it just")
say("stops being the largest term. Honest coverage sits at nominal across the whole range.")

# ============================================================ 7. post-assignment slice
head(7, "NEGATIVE RESULT: a post-assignment segment breaks the parts, and the check that sees it is algebra")

rng = np.random.default_rng(71)
REPS7 = 400
eng, non, overall7, srm_fire, gaps_post = [], [], [], 0, []
for _ in range(REPS7):
    d = hte.trial_post_gate(rng, n=20000, ate=ATE, gate_effect=0.30)
    e = d["engaged"]
    eng.append(hte.diff_means(d["y"][e], d["w"][e])[0])
    non.append(hte.diff_means(d["y"][~e], d["w"][~e])[0])
    overall7.append(hte.diff_means(d["y"], d["w"])[0])
    n1, n0 = int(d["w"][e].sum()), int((1 - d["w"][e]).sum())
    srm_fire += int(abs((n1 - n0) / np.sqrt(n1 + n0)) > 3.4808)
    gaps_post.append(hte.consistency_gap(d["y"], d["w"], np.stack([e, ~e], axis=1)))
say(f"True effect is {ATE:+.2f} for everybody; treatment also makes engagement more likely.")
say(f"  overall effect (correct)            {np.mean(overall7):+.4f}")
say(f"  'engaged users' segment             {np.mean(eng):+.4f}")
say(f"  'not engaged' segment               {np.mean(non):+.4f}")
say(f"  per-segment count check fires       {srm_fire / REPS7:.3f}  (SRM test at 0.0005, Day 165)")
say("")
say("BOTH parts say the launch hurt while the whole says it helped. That is arithmetically")
say("impossible for a real partition, and it is the tell: conditioning on a variable treatment")
say("moved makes each segment a different population in each arm.")
rng = np.random.default_rng(72)
gaps_legit = []
for _ in range(400):
    d = hte.trial_partition(rng, n=20000, k=2, ate=ATE)
    gaps_legit.append(hte.consistency_gap(d["y"], d["w"], hte.partition_matrix(d["seg"], 2)))
say("")
say("So make it a check. Share-weighted mean of the segment effects, minus the overall effect, in")
say("units of the overall SE - no counterfactual, no assumption, and nobody runs it:")
say(f"  pre-assignment partition   {np.mean(gaps_legit):+.5f} sd {np.std(gaps_legit, ddof=1):.5f}   (max |gap| {np.max(np.abs(gaps_legit)):.4f})")
say(f"  post-assignment gate       {np.mean(gaps_post):+.5f} sd {np.std(gaps_post, ddof=1):.5f}")
say(f"  separation                 {abs(np.mean(gaps_post)) / np.std(gaps_legit, ddof=1):.0f} legit-sd")
R["s7"] = {
    "engaged": float(np.mean(eng)),
    "not_engaged": float(np.mean(non)),
    "overall": float(np.mean(overall7)),
    "srm_fire": srm_fire / REPS7,
    "gap_legit_sd": float(np.std(gaps_legit, ddof=1)),
    "gap_post_mean": float(np.mean(gaps_post)),
}
say("")
say("This is the only guard in the build without a power problem, and the reason is that it is not")
say("a test: for a partition fixed before assignment the identity holds up to O(1/n), so the")
say("legit distribution has almost no width. Days 165, 168 and 169 all ended with a calibrated")
say("guard that could not see the failure it was built for. The one that works here works because")
say("it checks an accounting identity instead of estimating a parameter.")

# ============================================================ 8. the gate
head(8, "What the 'no segment may be harmed' gate actually buys")

rng = np.random.default_rng(81)
REPS8 = 600
good_blocked = {"raw": 0, "bonferroni": 0, "bh": 0}
bad_caught = {"raw": 0, "bonferroni": 0, "bh": 0}
for _ in range(REPS8):
    d = hte.trial_partition(rng, n=8000, k=20, ate=ATE)
    res = hte.scan(d["y"], d["w"], hte.partition_matrix(d["seg"], 20))
    for m in good_blocked:
        good_blocked[m] += int(hte.harm_flag(res, method=m))
    d = hte.trial_partition(rng, n=8000, k=20, ate=ATE, delta=0.30, harmed=0)
    res = hte.scan(d["y"], d["w"], hte.partition_matrix(d["seg"], 20))
    for m in bad_caught:
        bad_caught[m] += int(hte.harm_flag(res, method=m))
say(f"{'gate':>12} {'blocks a launch that helped everyone':>38} {'catches a real -0.25 segment':>30}")
for m in ("raw", "bonferroni", "bh"):
    say(f"{m:>12} {good_blocked[m] / REPS8:>38.3f} {bad_caught[m] / REPS8:>30.3f}")
R["s8"] = {
    "good_blocked": {m: good_blocked[m] / REPS8 for m in good_blocked},
    "bad_caught": {m: bad_caught[m] / REPS8 for m in bad_caught},
}
say("")
say("Read the two columns together before choosing. The uncorrected gate is the one people run and")
say("it is the only one that ever blocks a good launch. Correcting it makes the false alarms")
say("vanish and leaves detection where section 4 put it - which means the gate is honest about")
say("being weak rather than dishonest about being strong. If a segment genuinely matters, the way")
say("to protect it is to name it before the test and size for it, not to scan for it afterwards.")

say("")
say("=" * 100)
say(f"total runtime {time.time() - T0:.1f}s")

with open("results.json", "w") as fh:
    json.dump(R, fh, indent=2, default=float)
with open("evidence.txt", "w") as fh:
    fh.write("\n".join(LINES) + "\n")
print("wrote results.json + evidence.txt")
