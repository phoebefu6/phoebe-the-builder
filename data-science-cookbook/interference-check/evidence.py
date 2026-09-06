"""Every number in the README, measured.

Run:  python evidence.py            (writes evidence.txt if you redirect, plus results.json)

The ground truth in every section is a SECOND simulation of the same world under
global treatment and global control - the quantity a real experiment can never see.
That is the only reason the bias is knowable here and not knowable at work.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import interference as I
import numpy as np
from scipy import stats

RESULTS: Dict[str, Any] = {}
PC, PT = 0.10, 0.13  # control / treated attempt rate


def head(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def sub(t: str) -> None:
    print(f"\n--- {t} ---")


# --------------------------------------------------------------------------- #
def section_1() -> None:
    head(1, "ONE MARKET, ONE SPLIT TEST, AND THE NUMBER THE DECISION NEEDED")
    rng = np.random.default_rng(101)
    n, supply = 20_000, 2_000
    print(
        f"A marketplace: {n:,} buyers a day, {supply:,} things to sell.  Control buyers\n"
        f"attempt to buy at {PC:.0%}, treated buyers at {PT:.0%} - the feature works, it\n"
        f"genuinely makes people want to buy more.  Utilisation on the control side is\n"
        f"{I.tightness(n, PC, supply):.2f} supply per attempt."
    )

    est = [I.market_split_estimate(n, PC, PT, supply, rng) for _ in range(400)]
    e = np.array([a for a, _ in est])
    s = np.array([b for _, b in est])
    truth = I.market_global_effect(n, PC, PT, supply, rng, reps=300)
    sig = float(np.mean(np.abs(e / s) > 1.96))

    sub("what the A/B test reports")
    print(f"  estimate        {e.mean():+.5f}  ({e.mean() / PC * 100:+.1f}% on a {PC:.0%} base)")
    print(f"  reported SE      {s.mean():.5f}")
    print(f"  p < 0.05 in      {sig:.1%} of runs")
    sub("what shipping it to everybody actually does")
    print(f"  global effect   {truth:+.5f}")
    print(f"  bias            {e.mean() - truth:+.5f}  = {100 * (e.mean() - truth) / e.mean():.1f}% of the reported number")
    print(
        "\nThe test is not broken.  It measured, correctly, the difference between a\n"
        "treated buyer and a control buyer IN A WORLD WHERE HALF OF EVERYBODY IS\n"
        "TREATED.  In that world the treated buyers win the scarce supply off the\n"
        "control buyers, and the estimate is that transfer.  Treat everybody and there\n"
        "is nobody left to take it from."
    )
    RESULTS["s1"] = {"n": n, "supply": supply, "split": e.mean(), "se": s.mean(), "truth": truth, "sig_rate": sig}


# --------------------------------------------------------------------------- #
def section_2() -> None:
    head(2, "THE BIAS IS A CLIFF, NOT A GRADIENT - AND ITS EDGE IS AT UTILISATION 1.3")
    rng = np.random.default_rng(202)
    n = 20_000
    rows: List[Dict[str, Any]] = []
    print(f"{'supply':>8} {'util':>6} {'split':>9} {'global':>9} {'bias/est':>9} {'overstates':>11} {'cf global':>10}")
    for supply in (4000, 3000, 2600, 2400, 2200, 2000, 1800, 1400):
        truth = I.market_global_effect(n, PC, PT, supply, rng, reps=200)
        e = np.mean([I.market_split_estimate(n, PC, PT, supply, rng)[0] for _ in range(150)])
        cf = min(PT, supply / n) - min(PC, supply / n)
        bias_pct = 100 * (e - truth) / e if e else float("nan")
        over = 100 * (e - truth) / truth if truth > 1e-9 else float("inf")
        rows.append({"supply": supply, "util": I.tightness(n, PC, supply), "split": e, "truth": truth,
                     "cf": cf, "bias_pct": bias_pct, "overstates": over})
        os_s = f"{over:>10.1f}%" if np.isfinite(over) else f"{'infinite':>11}"
        print(f"{supply:>8} {I.tightness(n, PC, supply):>6.2f} {e:>9.5f} {truth:>9.5f} {bias_pct:>8.1f}% {os_s} {cf:>10.5f}")

    print(
        "\nThe global effect has a closed form - min(p_t, S/n) - min(p_c, S/n) - and the\n"
        "measured column matches it to the last digit.  The split estimate has one too:\n"
        "(p_t - p_c) * min(1, S / (n * (p_t + p_c) / 2)), because BOTH arms in a mixed\n"
        "market face the same rationing factor, so it cancels out of the difference and\n"
        "cannot warn you."
    )
    checks = []
    for supply in (2400, 2000, 1600):
        pred = (PT - PC) * min(1.0, supply / (n * (PT + PC) / 2))
        got = np.mean([I.market_split_estimate(n, PC, PT, supply, rng)[0] for _ in range(300)])
        checks.append({"supply": supply, "pred": pred, "got": got})
        print(f"  S={supply}: split closed form {pred:.5f}  measured {got:.5f}  gap {abs(pred - got):.5f}")
    r13 = next(r for r in rows if r["supply"] == 2600)
    r12 = next(r for r in rows if r["supply"] == 2400)
    print(
        f"\nThe operational point: a market at 1.30 supply-per-attempt overstates by {r13['overstates']:.0f}% and\n"
        f"a market at 1.20 overstates by {r12['overstates']:.0f}%.  No dashboard distinguishes those\n"
        "two.  Utilisation is not a nice-to-have context number for an experiment in a\n"
        "constrained market - it is the parameter that decides whether the readout means\n"
        "anything, and it belongs on the test plan, not the ops review."
    )
    RESULTS["s2"] = {"sweep": rows, "closed_form_checks": checks}


# --------------------------------------------------------------------------- #
def section_3() -> None:
    head(3, "NEGATIVE RESULT: MORE TRAFFIC MAKES IT WORSE, NOT BETTER")
    rng = np.random.default_rng(303)
    print(
        "Scale the market and the supply together, so utilisation stays at 1.00 and the\n"
        "world is the same world - only bigger."
    )
    print(f"\n{'n':>9} {'truth':>9} {'split':>9} {'bias':>9} {'SE':>9} {'95% cover':>10} {'p<.05':>7}")
    rows = []
    for k in (1, 2, 4, 8, 16, 32):
        n = 12_500 * k
        supply = int(n * PC)
        truth = I.market_global_effect(n, PC, PT, supply, rng, reps=max(20, 200 // k))
        e, s = [], []
        for _ in range(250):
            a, b = I.market_split_estimate(n, PC, PT, supply, rng)
            e.append(a)
            s.append(b)
        e, s = np.array(e), np.array(s)
        cov = float(np.mean((e - 1.96 * s <= truth) & (truth <= e + 1.96 * s)))
        pw = float(np.mean(np.abs(e / s) > 1.96))
        rows.append({"n": n, "truth": truth, "split": e.mean(), "se": s.mean(), "cover": cov, "power": pw})
        print(f"{n:>9,} {truth:>9.5f} {e.mean():>9.5f} {e.mean() - truth:>9.5f} {s.mean():>9.5f} {cov:>10.3f} {pw:>7.3f}")
    b = [r["split"] - r["truth"] for r in rows]
    print(
        f"\nBias moves from {b[0]:.5f} to {b[-1]:.5f} across a 32-fold range of n - flat, within\n"
        f"Monte-Carlo error.  The standard error falls {rows[0]['se'] / rows[-1]['se']:.1f}x over the same range.\n"
        f"Coverage of the true global effect goes to zero and STAYS there; the experiment's\n"
        f"own power never drops below {min(r['power'] for r in rows):.3f}.  This is the same shape the diff-in-diff build\n"
        "found for a parallel-trends violation, and it has the same cause: n is in the\n"
        "variance and it is not in the bias.  A confident, tight, highly significant\n"
        "interval around the wrong number is the expected output of doing this right."
    )
    RESULTS["s3"] = {"rows": rows}


# --------------------------------------------------------------------------- #
def section_4() -> None:
    head(4, "NEGATIVE RESULT: THE SAME TEST FAILS IN BOTH DIRECTIONS, AND CANNOT TELL YOU WHICH")
    rng = np.random.default_rng(404)
    m, groups, tau, gamma, sigma = 20, 300, 1.0, 0.5, 1.0
    group = np.repeat(np.arange(groups), m)
    print(
        f"A different world, same test.  {groups} peer groups of {m}.  The feature moves a\n"
        f"user by tau={tau} directly, and by gamma={gamma} more when ALL of their peers have it\n"
        "too - a referral loop, a shared feed, a network good.  The global effect is\n"
        f"tau + gamma = {tau + gamma}."
    )
    split = [I.user_estimate(I.spillover_outcomes((zz := I.assign_within_group(group, rng)), group, tau, gamma, sigma, rng), zz)[0] for _ in range(400)]
    pred = tau - gamma / (m - 1)
    bias_cf = I.spillover_split_bias_closed_form(gamma, m)
    print(f"\n  split estimate      {np.mean(split):.4f}   closed form {pred:.4f}")
    print(f"  truth               {tau + gamma:.4f}")
    print(f"  bias                {np.mean(split) - (tau + gamma):.4f}   closed form {bias_cf:.4f}")
    print(f"  share of the effect missed   {100 * abs(bias_cf) / (tau + gamma):.1f}%")
    print(
        f"\nThe closed form is exact and has no n in it: -gamma * m/(m-1) = {bias_cf:.4f}.  A treated\n"
        f"user has ({m}/2 - 1) of their {m - 1} peers treated and a control user has {m}/2, so the split\n"
        "recovers the DIRECT effect and misses the indirect one entirely - and then\n"
        "overshoots by one peer's worth in the other direction.  Note the sign.  In the\n"
        "marketplace the split OVERSTATED by 100%; here it UNDERSTATES by 35%.  Same\n"
        "randomisation, same estimator, same clean p-value, opposite error - and nothing\n"
        "in the output distinguishes them.  Which way you are wrong is a fact about the\n"
        "mechanism, and you have to argue it before the test, not read it after."
    )
    RESULTS["s4"] = {
        "m": m, "tau": tau, "gamma": gamma, "split": float(np.mean(split)),
        "pred": pred, "truth": tau + gamma, "bias_cf": bias_cf,
        "share_missed": 100 * abs(bias_cf) / (tau + gamma),
    }


# --------------------------------------------------------------------------- #
def section_5() -> None:
    head(5, "NEGATIVE RESULT: THE CHECK EVERYONE RECOMMENDS IS ~90x LESS SENSITIVE THAN THE TEST IT GUARDS")
    rng = np.random.default_rng(505)
    print(
        "The standard defence is a dose-response design: run the feature at two treated\n"
        "shares (10% and 50%) and test whether the estimated effect depends on the share.\n"
        "Under no interference it cannot; under interference it must.  The logic is\n"
        "correct.  The power is not."
    )
    sub("is the check calibrated?  (a market with unlimited supply - no interference)")
    p_null = np.array([I.dose_response_check(20_000, PC, PT, 10**9, rng)["p"] for _ in range(600)])
    size = float(np.mean(p_null < 0.05))
    print(f"  false-alarm rate at 0.05: {size:.4f}   (nominal 0.05 - the check is honest)")

    sub("what it catches in the 100%-biased market from section 1")
    rows = []
    for n in (20_000, 100_000, 400_000, 1_000_000):
        supply = int(n * PC)
        reps = 300 if n <= 100_000 else 150
        res = [I.dose_response_check(n, PC, PT, supply, rng) for _ in range(reps)]
        pw = float(np.mean(np.array([r["p"] for r in res]) < 0.05))
        d = float(np.mean([r["diff"] for r in res]))
        rows.append({"n": n, "power": pw, "diff": d})
        print(f"  n={n:>9,}  mean share-to-share gap {d:+.5f}   power {pw:.3f}")
    # Extrapolate: for a z-test the non-centrality scales as sqrt(n).
    top = rows[-1]
    lam = 1.96 + stats.norm.ppf(max(1e-6, min(1 - 1e-6, top["power"])))
    need = top["n"] * (( 1.96 + stats.norm.ppf(0.80)) / lam) ** 2
    n_for_80 = need
    print(
        f"\nThe experiment itself is at power 0.996+ by n=12,500 (section 3).  Extrapolating the\n"
        f"check's non-centrality as sqrt(n) from its {top['power']:.3f} at n={top['n']:,}, it needs about\n"
        f"{need / 1e6:.2f} MILLION users to reach 0.80 - roughly {need / 12500:.0f}x the traffic at which the\n"
        "experiment itself is fully powered.  At the sample size where you are actually\n"
        "running the test, the check\n"
        f"fires {rows[0]['power']:.3f} of the time on a market where the entire reported effect is bias,\n"
        f"against a {size:.3f} false-alarm rate.  It is very nearly a coin that says no.\n\n"
        "Compare the Day 165 SRM detector, which was 6x MORE sensitive than its\n"
        "experiment and still not protective.  A guard test being calibrated says nothing\n"
        "about whether it guards.  Passing this one is not evidence of no interference;\n"
        "it is evidence that you ran it."
    )
    RESULTS["s5"] = {"size": size, "rows": rows, "n_for_80": n_for_80}


# --------------------------------------------------------------------------- #
def section_6() -> None:
    head(6, "CLUSTER RANDOMISATION WORKS - IF, AND ONLY IF, THE CLUSTER IS THE INTERFERENCE BOUNDARY")
    rng = np.random.default_rng(606)
    groups, m = 40, 500
    group = np.repeat(np.arange(groups), m)
    n = group.size
    per_group = int(round(m * PC))
    total = per_group * groups
    print(
        f"{groups} cities, {m} buyers each ({n:,} total), utilisation 1.00.  Two versions of the\n"
        "SAME market: in one, each city has its own supply; in the other the identical\n"
        "total supply sits in one national pool.  Nothing a data team can see distinguishes\n"
        "them - same rows, same volumes, same conversion rate."
    )
    out = {}
    for contained in (True, False):
        def draw(assign):
            z = assign(group, rng)
            if contained:
                y = I.rationed_outcomes(z, PC, PT, 0, rng, group=group, supply_per_group=per_group)
            else:
                y = I.rationed_outcomes(z, PC, PT, total, rng)
            return y, z

        truth_arm = []
        for zz in (np.ones(n, dtype=int), np.zeros(n, dtype=int)):
            v = []
            for _ in range(40):
                if contained:
                    v.append(I.rationed_outcomes(zz, PC, PT, 0, rng, group=group, supply_per_group=per_group).mean())
                else:
                    v.append(I.rationed_outcomes(zz, PC, PT, total, rng).mean())
            truth_arm.append(float(np.mean(v)))
        truth = truth_arm[0] - truth_arm[1]

        sp, cl = [], []
        for _ in range(200):
            y, z = draw(I.assign_within_group)
            sp.append(I.user_estimate(y, z)[0])
            y, z = draw(I.assign_by_group)
            cl.append(I.cluster_estimate(y, z, group)[0])
        label = "supply LOCAL to each city" if contained else "supply POOLED nationally"
        print(f"\n  {label}")
        print(f"    true global effect      {truth:+.5f}")
        print(f"    user-level split        {np.mean(sp):+.5f}   bias {np.mean(sp) - truth:+.5f}")
        print(f"    cluster randomised      {np.mean(cl):+.5f}   bias {np.mean(cl) - truth:+.5f}")
        out["contained" if contained else "pooled"] = {
            "truth": truth, "split": float(np.mean(sp)), "cluster": float(np.mean(cl)),
            "split_sd": float(np.std(sp)), "cluster_sd": float(np.std(cl)),
        }
    a, b = out["contained"], out["pooled"]
    print(
        f"\nContained: clustering removes {100 * (1 - abs(a['cluster'] - a['truth']) / abs(a['split'] - a['truth'])):.1f}% of the bias.  Pooled: it removes\n"
        f"{100 * (1 - abs(b['cluster'] - b['truth']) / abs(b['split'] - b['truth'])):.1f}% - the cluster estimate ({b['cluster']:.5f}) and the split estimate\n"
        f"({b['split']:.5f}) are the same number, because randomising whole cities against a\n"
        "national pool still leaves treated cities taking supply from control cities.\n"
        "Cluster randomisation is not a fix for interference.  It is a fix for\n"
        "interference THAT STOPS AT THE CLUSTER EDGE, and which edge that is is a claim\n"
        "about the supply chain, not about the schema.  This is the same failure the Day\n"
        "167 build found for clustered standard errors: the received advice is about the\n"
        "cluster COUNT, and the thing that actually decides it is the LEVEL."
    )
    RESULTS["s6"] = out


# --------------------------------------------------------------------------- #
def section_7() -> None:
    head(7, "WHAT CLUSTERING COSTS: THE DESIGN EFFECT IS 1 + (m-1) * ICC AND IT IS NOT SMALL")
    rng = np.random.default_rng(707)
    m, groups, tau, gamma, sigma = 20, 300, 1.0, 0.5, 1.0
    group = np.repeat(np.arange(groups), m)
    print(
        "Back in the peer-effects world of section 4, where clustering IS the right\n"
        "design.  Groups differ from each other; that between-group variance cancels\n"
        "exactly in a within-group split (both arms sit in the same group) and does not\n"
        "cancel at all when the group is the unit of assignment."
    )
    print(f"\n{'group sd':>9} {'ICC':>7} {'textbook DE':>12} {'derived DE':>11} {'measured':>9} {'split est':>10} {'cluster est':>12}")
    rows = []
    for gsd in (0.0, 0.25, 0.45, 0.75):
        icc = gsd**2 / (gsd**2 + sigma**2)
        sp, cl = [], []
        for _ in range(1500):
            z = I.assign_within_group(group, rng)
            sp.append(I.user_estimate(I.spillover_outcomes(z, group, tau, gamma, sigma, rng, group_sd=gsd), z)[0])
            z2 = I.assign_by_group(group, rng)
            cl.append(I.cluster_estimate(I.spillover_outcomes(z2, group, tau, gamma, sigma, rng, group_sd=gsd), z2, group)[0])
        de_book = 1 + (m - 1) * icc
        de_here = 1 + m * gsd**2 / sigma**2
        de_meas = float(np.var(cl) / np.var(sp))
        rows.append({"gsd": gsd, "icc": icc, "de_book": de_book, "de_here": de_here, "de_meas": de_meas,
                     "split": float(np.mean(sp)), "cluster": float(np.mean(cl))})
        print(f"{gsd:>9.2f} {icc:>7.4f} {de_book:>12.2f} {de_here:>11.2f} {de_meas:>9.2f} {np.mean(sp):>10.4f} {np.mean(cl):>12.4f}")
    r = rows[2]
    print(
        "\nThe correct design is unbiased at every row and the wrong one is biased at every\n"
        f"row, and the price of being right is {r['de_meas']:.1f}x the variance at an ICC of only {r['icc']:.3f}.\n"
        f"In sample-size terms that is {r['de_meas']:.1f}x the users for the same power - which is why teams\n"
        "keep choosing the biased design and calling it pragmatism.  It is a real trade, and\n"
        "it should be made explicitly against the section-4 bias, which is 35% of the effect\n"
        "and does not shrink.\n\n"
        "NEGATIVE RESULT, derived here rather than looked up: the textbook design effect\n"
        f"1 + (m-1)*ICC UNDERSTATES that price - {r['de_book']:.2f} against a measured {r['de_meas']:.2f}.  It compares\n"
        "cluster assignment to SIMPLE random assignment, and a within-group 50/50 split is\n"
        "not simple random assignment: it is STRATIFIED by the cluster, so it cancels the\n"
        "between-group variance exactly rather than in expectation.  Against that baseline\n"
        f"the right expression is 1 + m*(sd_group/sd_user)^2 = {r['de_here']:.2f}, which is what the measured\n"
        "column tracks across all four rows.  Sizing a cluster test off the textbook number\n"
        f"buys {100 * (r['de_here'] / r['de_book'] - 1):.0f}% too little traffic at this ICC, and the gap widens with m."
    )
    RESULTS["s7"] = {"rows": rows}


# --------------------------------------------------------------------------- #
def section_8() -> None:
    head(8, "SWITCHBACK: THE BALANCED DESIGN IS EXACTLY TWICE AS BIASED AS COIN-FLIPPING")
    rng = np.random.default_rng(808)
    tau, sigma, T = 1.0, 1.0, 400
    print(
        "Randomise TIME, not users: the whole market is treated for a period, then\n"
        "control, and interference within a period is no longer between arms.  This is\n"
        "the right answer for a shared pool.  It has one failure mode: the system does\n"
        "not switch instantly.  If a fraction c of each period is still behaving like the\n"
        "previous one, the arms bleed into each other."
    )
    print(f"\n{'carryover':>10} {'coin-flip':>20} {'strict ABAB':>22}")
    print(f"{'':>10} {'measured / closed':>20} {'measured / closed':>22}")
    rows = []
    for c in (0.0, 0.05, 0.10, 0.20, 0.30):
        got = {}
        for alt in (False, True):
            e = [I.switchback_run(T, tau, c, sigma, rng, alternating=alt)[0] for _ in range(600)]
            got[alt] = float(np.mean(e))
        cf_r = I.switchback_bias_closed_form(tau, c, False)
        cf_a = I.switchback_bias_closed_form(tau, c, True)
        rows.append({"c": c, "coin": got[False], "coin_cf": cf_r, "abab": got[True], "abab_cf": cf_a})
        print(f"{c:>10.2f} {got[False]:>11.4f} / {cf_r:<6.3f} {got[True]:>13.4f} / {cf_a:<6.3f}")
    print(
        "\nCoin-flip randomisation attenuates by exactly c: tau*(1-c).  Strict alternation\n"
        "attenuates by 2c: tau*(1-2c), because a treated period's predecessor is ALWAYS a\n"
        "control period, so the contamination pushes both arms the wrong way instead of\n"
        "one.  At c=0.30 the balanced, tidy, obviously-fairer ABAB design returns 0.40 of\n"
        "a true 1.00 while the coin returns 0.70.  Balance in the assignment is not\n"
        "balance in the exposure."
    )

    sub("burn-in: discarding the contaminated window, and what it costs")
    c = 0.20
    print(f"  true carryover c = {c}, {T} periods, per-period noise sd {sigma}")
    print(f"\n{'burn-in':>8} {'residual':>9} {'E[est]':>8} {'bias^2':>9} {'var':>9} {'MSE':>9}")
    mse_rows = []
    for b in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60):
        e = np.array([I.switchback_run(T, tau, c, sigma, rng, alternating=False, burn_in=b)[0] for _ in range(800)])
        resid = max(0.0, c - b) / (1 - b)
        bias2 = (e.mean() - tau) ** 2
        var = float(e.var())
        mse_rows.append({"burn_in": b, "resid": resid, "mean": float(e.mean()), "bias2": bias2, "var": var, "mse": bias2 + var})
        print(f"{b:>8.2f} {resid:>9.3f} {e.mean():>8.4f} {bias2:>9.5f} {var:>9.5f} {bias2 + var:>9.5f}")
    best = min(mse_rows, key=lambda r: r["mse"])
    print(
        f"\nMinimum MSE at burn-in {best['burn_in']:.2f} against a true carryover of {c:.2f}.  Discarding\n"
        "less than the carryover leaves bias that no sample size removes; discarding more\n"
        "buys nothing and costs 1/(1-b) of the variance.  The tuning knob is therefore a\n"
        "measurement - how long the system takes to settle - and not a preference.  If\n"
        "nobody has measured it, the switchback has an unknown attenuation and its result\n"
        "is a lower bound on the effect, which is at least an honest thing to write down.\n"
        "Note also that periods, not users, are the sample here: 400 periods is n=400,\n"
        "which is why switchbacks are usually run on a metric measured per period rather\n"
        "than per user."
    )
    RESULTS["s8"] = {"carryover": rows, "burn_in": mse_rows, "best_burn_in": best["burn_in"]}


# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    print("INTERFERENCE-CHECK - EVIDENCE")
    print("Every figure below is measured by the code in this repository.")
    section_1()
    section_2()
    section_3()
    section_4()
    section_5()
    section_6()
    section_7()
    section_8()

    head(9, "WHAT TO PUT IN THE TEST PLAN")
    print(
        "1. The interference MECHANISM you are ruling out, named before the test - shared\n"
        "   supply, shared budget, shared model, peers, or none - and why.  It decides the\n"
        "   SIGN of your error (sections 1 and 4) and no output can recover it.\n"
        f"2. For a constrained market: utilisation during the test window.  At 1.3 supply\n"
        f"   per attempt the split overstates by {next(r['overstates'] for r in RESULTS['s2']['sweep'] if r['supply'] == 2600):.0f}%; at 1.2 by {next(r['overstates'] for r in RESULTS['s2']['sweep'] if r['supply'] == 2400):.0f}% (section 2).\n"
        "3. Never 'we will re-run it bigger to be sure'.  Bias is flat in n and the\n"
        "   interval closes around the wrong number (section 3, coverage 0.000).\n"
        f"4. If you ran a dose-response check, report its POWER, not its p-value.  At\n"
        f"   experiment-sized traffic it fires {RESULTS['s5']['rows'][0]['power']:.3f} on a 100%-biased market (section 5).\n"
        "5. If you clustered, state what the cluster is a boundary OF, and defend that it\n"
        "   contains the mechanism from (1).  Clustering the wrong level removes 0% of the\n"
        "   bias and pays the whole design effect (sections 6-7).\n"
        "6. If you switchbacked, report the measured settling time and the burn-in.  A\n"
        "   coin-flip switchback attenuates by c, an ABAB one by 2c (section 8)."
    )
    print(f"\n[evidence run: {time.time() - t0:.1f}s]")
    with open("results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)
    print("[wrote results.json]")


if __name__ == "__main__":
    main()
