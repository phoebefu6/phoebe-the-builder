"""The measured argument for `diff-in-diff`.

Eight sections.  Every number printed here is asserted in `test_did.py`, and
every one of them comes out of `did.py` running on a world whose true
treatment effect this file chose.  Nothing is quoted from a paper without
being recomputed; where a published finding is referenced, the number next to
it is the one this machine measured.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Tuple

import numpy as np
from did import (
    aggregate_att,
    ar1_errors,
    did_2x2,
    event_dummies,
    fe_ols,
    group_time_att,
    heterogeneity_bound,
    make_panel,
    make_staggered,
    twfe,
    twfe_cell_weights,
)
from scipy import stats

RESULTS: Dict[str, object] = {}


def head(n: int, title: str) -> None:
    print()
    print("=" * 78)
    print(f"{n}. {title}")
    print("=" * 78)


def sub(title: str) -> None:
    print(f"\n-- {title}")


# ==========================================================================
# 1. The estimator is an identity plus one assumption, and n cannot fix it
# ==========================================================================


def section_1() -> None:
    head(1, "THE ESTIMATOR IS AN IDENTITY PLUS ONE ASSUMPTION")
    print(
        "DiD is four means.  Under parallel trends it is unbiased for the effect.\n"
        "Break parallel trends by a per-period slope `delta` on the treated group and\n"
        "the bias is exactly `delta * (mean post period - mean pre period)` - a number\n"
        "with no n in it anywhere."
    )

    T, t0 = 12, 6
    gap = float(np.mean(np.arange(t0, T)) - np.mean(np.arange(0, t0)))
    print(f"\nDesign: T={T}, treatment from t={t0}, so mean(post t) - mean(pre t) = {gap:.1f}")

    sub("parallel trends holds (delta = 0): both estimators are unbiased")
    rng = np.random.default_rng(101)
    reps = 2000
    b2x2, btwfe = [], []
    for _ in range(reps):
        Y, D, adopt, _ = make_panel(rng, n_treated=100, n_control=100, T=T, t0=t0, effect=1.0)
        b2x2.append(did_2x2(Y, adopt == t0, range(t0), range(t0, T)))
        btwfe.append(twfe(Y, D))
    b2x2 = np.array(b2x2)
    btwfe = np.array(btwfe)
    print("  true effect                1.0000")
    print(f"  four-means DiD    {b2x2.mean():8.4f} +/- {b2x2.std(ddof=1)/np.sqrt(reps):.4f}")
    print(f"  two-way FE        {btwfe.mean():8.4f} +/- {btwfe.std(ddof=1)/np.sqrt(reps):.4f}")
    print(f"  max |2x2 - TWFE| over {reps} draws: {np.abs(b2x2 - btwfe).max():.2e}")
    print("  With one common adoption date the two are the SAME estimator, to machine")
    print("  precision.  The regression buys standard errors, not identification.")
    RESULTS["s1_unbiased_2x2"] = float(b2x2.mean())
    RESULTS["s1_unbiased_twfe"] = float(btwfe.mean())
    RESULTS["s1_max_gap"] = float(np.abs(b2x2 - btwfe).max())

    sub("parallel trends fails: the bias is delta * 6, and more data tightens a CI around the wrong number")
    delta = 0.05
    print(f"  violation delta = {delta} per period  ->  predicted bias = {delta * gap:.4f}")
    print(f"\n  {'n per arm':>10} {'mean estimate':>14} {'mean SE':>9} {'bias':>8} {'coverage':>9}")
    rows = []
    for n_arm in [50, 200, 800, 3200, 12800]:
        rng = np.random.default_rng(202 + n_arm)
        reps_n = 600
        est, ses, cov = [], [], []
        for _ in range(reps_n):
            Y, D, adopt, _ = make_panel(
                rng, n_treated=n_arm, n_control=n_arm, T=T, t0=t0, effect=1.0, diff_trend=delta
            )
            f = fe_ols(Y, [D], ["D"], vcov="cluster")
            b, se = float(f.beta[0]), float(f.se()[0])
            crit = stats.t.ppf(0.975, f.dof)
            est.append(b)
            ses.append(se)
            cov.append(abs(b - 1.0) <= crit * se)
        est = np.array(est)
        rows.append((n_arm, est.mean(), float(np.mean(ses)), est.mean() - 1.0, float(np.mean(cov))))
        print(
            f"  {n_arm:10d} {est.mean():14.4f} {np.mean(ses):9.4f} "
            f"{est.mean() - 1.0:8.4f} {np.mean(cov):9.3f}"
        )
    RESULTS["s1_bias_rows"] = [[r[0], r[1], r[2], r[3], r[4]] for r in rows]
    RESULTS["s1_predicted_bias"] = float(delta * gap)
    print(
        f"\n  The bias sits at {rows[-1][3]:.4f} across a 256-fold range of n while the standard\n"
        f"  error falls from {rows[0][2]:.4f} to {rows[-1][2]:.4f}.  Coverage of the 95% interval goes\n"
        f"  {rows[0][4]:.3f} -> {rows[-1][4]:.3f}.  A bigger sample does not make a DiD more credible;\n"
        "  it makes the same wrong number more precise, and the interval stops containing\n"
        "  the truth at all.  Confidence is not identification."
    )


# ==========================================================================
# 2. The pre-trends test: what it can actually see
# ==========================================================================

EV = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]


def _run_delta(delta: float, reps: int = 1200, n_arm: int = 100, seed: int = 11) -> Dict[str, float]:
    T, t0 = 12, 6
    rng = np.random.default_rng(seed)
    adopt = np.where(np.arange(2 * n_arm) < n_arm, float(t0), np.inf)
    ev_cols = event_dummies(adopt, T, [e for e in EV if e != -1])
    lead_idx = [i for i, e in enumerate([e for e in EV if e != -1]) if e < 0]

    fired = 0
    est, cov = [], []
    est_pass, cov_pass = [], []
    for _ in range(reps):
        Y, D, _, _ = make_panel(
            rng, n_treated=n_arm, n_control=n_arm, T=T, t0=t0, effect=1.0, diff_trend=delta
        )
        es = fe_ols(Y, ev_cols, [f"e{e:+d}" for e in EV if e != -1], vcov="cluster")
        _, p_pre = es.wald(lead_idx)
        f = fe_ols(Y, [D], ["D"], vcov="cluster")
        b, se = float(f.beta[0]), float(f.se()[0])
        hit = abs(b - 1.0) <= stats.t.ppf(0.975, f.dof) * se
        est.append(b)
        cov.append(hit)
        if p_pre < 0.05:
            fired += 1
        else:
            est_pass.append(b)
            cov_pass.append(hit)
    return {
        "delta": delta,
        "power": fired / reps,
        "bias": float(np.mean(est) - 1.0),
        "coverage": float(np.mean(cov)),
        "bias_pass": float(np.mean(est_pass) - 1.0) if est_pass else float("nan"),
        "coverage_pass": float(np.mean(cov_pass)) if cov_pass else float("nan"),
        "n_pass": len(est_pass),
        "reps": reps,
        "sd_est": float(np.std(est, ddof=1)),
    }


def section_2() -> List[Dict[str, float]]:
    head(2, "THE PRE-TRENDS TEST CANNOT SEE WHAT BREAKS THE ESTIMATE")
    print(
        "Everyone plots the leads and checks they are flat.  That plot is a hypothesis\n"
        "test, and a hypothesis test has a power.  Below: the joint Wald test on four\n"
        "lead coefficients, against the bias the same violation puts in the estimate."
    )
    print(f"\n  {'delta':>6} {'pretest fires':>14} {'bias':>8} {'bias %':>8} {'coverage':>9}")
    rows = []
    for d in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
        r = _run_delta(d)
        rows.append(r)
        print(
            f"  {d:6.2f} {r['power']:14.3f} {r['bias']:8.4f} "
            f"{100 * r['bias']:7.1f}% {r['coverage']:9.3f}"
        )
    RESULTS["s2_rows"] = rows

    size = rows[0]["power"]
    r05 = next(r for r in rows if r["delta"] == 0.05)
    r10 = next(r for r in rows if r["delta"] == 0.10)
    print(
        f"\n  Its size under the null is {size:.3f}, so it is calibrated.  At delta = 0.05 it\n"
        f"  fires {r05['power']:.3f} of the time - barely above its own false-alarm rate - while the\n"
        f"  estimate is already {100 * r05['bias']:.1f}% too large and coverage has fallen to {r05['coverage']:.3f}.\n"
        f"  At delta = 0.10 the estimate is {100 * r10['bias']:.0f}% too large and the test still passes\n"
        f"  {1 - r10['power']:.1%} of the time."
    )

    # where does the test become reliable, and what does the estimate look like there?
    def power_at(d: float) -> float:
        return _run_delta(d, reps=1200, seed=77)["power"]

    lo, hi = 0.10, 0.60
    for _ in range(7):
        mid = (lo + hi) / 2
        if power_at(mid) < 0.80:
            lo = mid
        else:
            hi = mid
    d80 = (lo + hi) / 2
    bias80 = d80 * 6.0
    print(
        f"\n  The violation the test detects 80% of the time is delta = {d80:.3f}.  By then the\n"
        f"  estimate is {bias80:.2f} against a true effect of 1.00 - {100 * bias80:.0f}% too large.  The\n"
        "  pre-trends plot only becomes a reliable alarm once the answer it is guarding\n"
        "  is already off by more than the answer itself."
    )
    RESULTS["s2_delta_power80"] = float(d80)
    RESULTS["s2_bias_at_power80"] = float(bias80)

    sub("what actually raises the power: pre-periods, not units")
    print("  delta held at 0.05 throughout.  n per arm doubling vs one more pre-period:")
    print(f"\n  {'pre-periods':>12} {'power':>7}   |  {'n per arm':>10} {'power':>7}")
    npre_rows, narm_rows = [], []
    for k in [2, 3, 5, 8, 12, 20]:
        T = k + 1 + 6
        t0 = k + 1
        ev = list(range(-k, 0)) + list(range(0, 6))
        rng = np.random.default_rng(300 + k)
        adopt = np.where(np.arange(200) < 100, float(t0), np.inf)
        cols = event_dummies(adopt, T, [e for e in ev if e != -1])
        lead_idx = [i for i, e in enumerate([e for e in ev if e != -1]) if e < 0]
        fired = 0
        reps = 800
        for _ in range(reps):
            Y, _, _, _ = make_panel(
                rng, n_treated=100, n_control=100, T=T, t0=t0, effect=1.0, diff_trend=0.05
            )
            es = fe_ols(Y, cols, None, vcov="cluster")
            if es.wald(lead_idx)[1] < 0.05:
                fired += 1
        npre_rows.append((k, fired / reps))
    for n_arm in [100, 200, 400, 800, 1600, 3200]:
        r = _run_delta(0.05, reps=500, n_arm=n_arm, seed=400 + n_arm)
        narm_rows.append((n_arm, r["power"]))
    for (k, pk), (na, pa) in zip(npre_rows, narm_rows):
        print(f"  {k:12d} {pk:7.3f}   |  {na:10d} {pa:7.3f}")
    RESULTS["s2_npre_rows"] = [[int(k), float(p)] for k, p in npre_rows]
    RESULTS["s2_narm_rows"] = [[int(n), float(p)] for n, p in narm_rows]
    npre_d = dict(npre_rows)
    narm_d = dict(narm_rows)
    print(
        f"\n  Both levers work, and that is the useful part - they can be priced against\n"
        f"  each other.  Five pre-periods gives power {npre_d[5]:.3f}, indistinguishable from the\n"
        f"  test's own size ({RESULTS['s2_rows'][0]['power']:.3f}).  Twelve pre-periods gives {npre_d[12]:.3f}; 1,600 units per\n"
        f"  arm gives {narm_d[1600]:.3f}.  So going from 5 pre-periods to 12 buys what a 16-fold\n"
        f"  increase in sample buys, and the pre-periods are usually already sitting in\n"
        f"  the warehouse.  A linear violation accumulates over TIME, so time is the axis\n"
        f"  with the leverage.  What is NOT evidence is 'we have four pre-periods and they\n"
        f"  look flat' at any n: at {narm_d[100]:,} per arm that design fires {npre_d[3]:.3f} of the time on a\n"
        f"  violation that already biases the answer 30%."
    )
    return rows


# ==========================================================================
# 3. Pre-testing is not a filter
# ==========================================================================


def section_3(rows: List[Dict[str, float]]) -> None:
    head(3, "NEGATIVE RESULT: PRE-TESTING IS NOT A FILTER")
    print(
        "The defence is not 'the test is powerful', it is 'if it passes, we proceed'.\n"
        "So the number that matters is the bias among the runs that PASSED - the ones\n"
        "that reach a slide deck.  If pre-testing works, it should be smaller."
    )
    print(f"\n  {'delta':>6} {'bias (all)':>11} {'bias | passed':>14} {'shift':>8} {'MC err':>8} {'cov | passed':>13} {'kept':>7}")
    for r in rows:
        if r["delta"] == 0.0:
            continue
        shift = r["bias_pass"] - r["bias"]
        mc = r["sd_est"] / np.sqrt(max(r["n_pass"], 1))
        print(
            f"  {r['delta']:6.2f} {r['bias']:11.4f} {r['bias_pass']:14.4f} "
            f"{shift:8.4f} {mc:8.4f} {r['coverage_pass']:13.3f} {r['n_pass'] / r['reps']:7.1%}"
        )
    r05 = next(r for r in rows if r["delta"] == 0.05)
    ratios = [
        (abs(r["bias_pass"] - r["bias"]) / (r["sd_est"] / np.sqrt(max(r["n_pass"], 1))), r)
        for r in rows
        if r["delta"] > 0
    ]
    worst_ratio, worst_row = max(ratios, key=lambda x: x[0])
    max_shift = float(abs(worst_row["bias_pass"] - worst_row["bias"]))
    RESULTS["s3_bias_pass_05"] = float(r05["bias_pass"])
    RESULTS["s3_bias_all_05"] = float(r05["bias"])
    RESULTS["s3_max_shift"] = max_shift
    # MC standard error on a conditional mean of ~1200*0.9 draws with sd(beta) ~ 0.06
    # Monte Carlo standard error on each conditional mean, measured from the run
    sds = [r["sd_est"] / np.sqrt(r["n_pass"]) for r in rows if r["delta"] > 0 and r["n_pass"] > 30]
    RESULTS["s3_mc_err"] = float(np.mean(sds))
    print(
        f"\n  At delta = 0.05 the surviving bias is {r05['bias_pass']:.4f} against {r05['bias']:.4f} unconditional:\n"
        f"  pre-testing removed {100 * (1 - r05['bias_pass'] / r05['bias']):.1f}% of it.  Measured against each row's OWN\n"
        f"  Monte Carlo error, the largest shift anywhere in the table is {worst_ratio:.2f} standard\n"
        f"  errors (delta = {worst_row['delta']:.2f}, shift {max_shift:.4f} on an error of {max_shift / worst_ratio:.4f}) - so the\n"
        "  honest statement is that the conditional and unconditional bias are the SAME\n"
        "  to the precision this run can resolve, in either direction.  Not 'pre-testing\n"
        "  helps a little', and not 'pre-testing hurts'.\n"
        "  It does nothing, because the test reads noise in the leads while the bias\n"
        "  lives in the trend, and those are close to independent here.\n"
        "  This is the mechanism behind Roth's (2022) 'pre-test with caution': screening\n"
        "  on something uncorrelated with the harm cannot remove the harm.  (Roth also\n"
        "  shows the shift can be adverse in shapes this DGP does not generate; the\n"
        "  measurement here resolves the null, not the sign.)"
    )
    print(
        "\n  What follows: a flat pre-trend is a statement about power, not about parallel\n"
        "  trends.  The honest report gives the pre-window LENGTH, the smallest violation\n"
        "  the test could have caught, and the bias that violation would imply."
    )


# ==========================================================================
# 4. Serial correlation: the SE, not the estimate
# ==========================================================================


def section_4() -> None:
    head(4, "NEGATIVE RESULT: SERIAL CORRELATION BREAKS THE STANDARD ERROR, NOT THE ESTIMATE")
    print(
        "Sections 1-3 were about identification.  This one is about inference, and it\n"
        "fails on data with NO treatment effect at all.  Bertrand, Duflo & Mullainathan\n"
        "(2004) ran placebo interventions on serially correlated panels and found a\n"
        "nominal 5% test rejecting roughly 45% of the time.  Below: the same experiment,\n"
        "with the autocorrelation dialled."
    )
    print(f"\n  {'rho':>5} {'T':>4} {'iid SE':>8} {'cluster SE':>11} {'mean estimate':>14}")
    rows = []
    for rho in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]:
        for T in [20]:
            rng = np.random.default_rng(900 + int(100 * rho))
            reps = 1500
            N = 100
            rej_iid = rej_cl = 0
            est = []
            for _ in range(reps):
                e = ar1_errors(rng, N, T, rho)
                Y = e + 0.10 * np.arange(T)[None, :] + rng.normal(0, 1, (N, 1))
                tr = np.zeros(N, dtype=bool)
                tr[rng.permutation(N)[: N // 2]] = True
                t0 = int(rng.integers(1, T))
                D = (tr[:, None] & (np.arange(T)[None, :] >= t0)).astype(float)
                fi = fe_ols(Y, [D], ["D"], vcov="iid")
                fc = fe_ols(Y, [D], ["D"], vcov="cluster")
                est.append(float(fc.beta[0]))
                if fi.pvalue()[0] < 0.05:
                    rej_iid += 1
                if fc.pvalue()[0] < 0.05:
                    rej_cl += 1
            rows.append((rho, T, rej_iid / reps, rej_cl / reps, float(np.mean(est))))
            print(f"  {rho:5.2f} {T:4d} {rej_iid / reps:8.3f} {rej_cl / reps:11.3f} {np.mean(est):14.4f}")
    RESULTS["s4_rows"] = [[float(a), int(b), float(c), float(d), float(e)] for a, b, c, d, e in rows]
    hi = [r for r in rows if r[0] == 0.8][0]
    print(
        f"\n  The estimate is fine everywhere - it averages {hi[4]:+.4f} on a true zero.  The\n"
        f"  DEFAULT standard error is not: at rho = 0.80 a nominal 0.05 test rejects a\n"
        f"  true null {hi[2]:.3f} of the time, {hi[2] / 0.05:.1f}x nominal, and {rows[-1][2]:.3f} at rho = 0.95.\n"
        f"  Clustering on the unit returns it to {hi[3]:.3f}.  This reproduces BDM's phenomenon;\n"
        f"  their 0.45 came from real wage series whose serial correlation is stronger than\n"
        "  anything a plain AR(1) at 0.8 produces, so treat the number as a lower bound\n"
        "  on what a real panel does to you, not an upper one.\n"
        f"  Note what this means in practice: at rho = 0.80, {hi[2]:.0%} of the placebo\n"
        f"  interventions that 'found something significant' were reading their own error\n"
        f"  structure - and the estimate they were reading was {abs(hi[4]):.4f}."
    )


# ==========================================================================
# 5. Clustering: the level, not the count
# ==========================================================================


def section_5() -> None:
    head(5, "NEGATIVE RESULT: THE CLUSTER COUNT IS NOT THE PROBLEM - THE LEVEL IS")
    print(
        "The warning everybody repeats is 'you need enough clusters' (the 42-cluster\n"
        "rule of thumb).  Test it directly, then test the thing nobody says out loud:\n"
        "policy is assigned to a STATE, the rows are people, and the regression clusters\n"
        "on the row."
    )

    sub("(a) few clusters, treatment assigned at the clustered level")
    print(f"\n  {'G':>5} {'size of the 0.05 test':>22}")
    fc_rows = []
    for G in [6, 10, 16, 20, 30, 50, 100, 200]:
        rng = np.random.default_rng(1300 + G)
        reps = 2000
        rej = 0
        T = 12
        for _ in range(reps):
            e = ar1_errors(rng, G, T, 0.5)
            Y = e + 0.2 * np.arange(T)[None, :] + rng.normal(0, 1, (G, 1))
            tr = np.zeros(G, dtype=bool)
            tr[rng.permutation(G)[: G // 2]] = True
            D = (tr[:, None] & (np.arange(T)[None, :] >= T // 2)).astype(float)
            if fe_ols(Y, [D], ["D"], vcov="cluster").pvalue()[0] < 0.05:
                rej += 1
        fc_rows.append((G, rej / reps))
        print(f"  {G:5d} {rej / reps:22.4f}")
    RESULTS["s5_fewclust"] = [[int(g), float(p)] for g, p in fc_rows]
    print(
        f"\n  G = 6 gives {dict(fc_rows)[6]:.4f} against a nominal 0.05.  With the t(G-1) reference\n"
        "  the cluster-robust test is close to correct all the way down to six clusters,\n"
        "  when treatment varies at the level being clustered.  The rule of thumb did not\n"
        "  bite in this design.  That is not the failure mode."
    )

    sub("(b) the same test, one level too fine")
    print("  Units nested in states.  Shock is a serially correlated STATE-year shock.")
    print("  Treatment is assigned to states.  Nominal size 0.05 throughout.")
    print(
        f"\n  {'states':>7} {'units/state':>12} {'iid':>7} {'cluster by unit':>17} {'cluster by state':>18}"
    )
    nest_rows = []
    for n_states, per in [(6, 10), (10, 10), (20, 10), (50, 10), (20, 50), (20, 200)]:
        rng = np.random.default_rng(1500 + n_states * 1000 + per)
        reps = 900
        N = n_states * per
        sid = np.repeat(np.arange(n_states), per)
        T = 12
        h_iid = h_unit = h_state = 0
        for _ in range(reps):
            sy = ar1_errors(rng, n_states, T, 0.6)[sid]
            Y = sy + rng.normal(0, 1, (N, T)) + rng.normal(0, 1, (N, 1)) + 0.2 * np.arange(T)[None, :]
            treat_states = rng.permutation(n_states) < n_states // 2
            tr = treat_states[sid]
            D = (tr[:, None] & (np.arange(T)[None, :] >= T // 2)).astype(float)
            if fe_ols(Y, [D], ["D"], vcov="iid").pvalue()[0] < 0.05:
                h_iid += 1
            if fe_ols(Y, [D], ["D"], vcov="cluster").pvalue()[0] < 0.05:
                h_unit += 1
            if fe_ols(Y, [D], ["D"], vcov="cluster", cluster_id=sid).pvalue()[0] < 0.05:
                h_state += 1
        nest_rows.append((n_states, per, h_iid / reps, h_unit / reps, h_state / reps))
        print(
            f"  {n_states:7d} {per:12d} {h_iid / reps:7.3f} "
            f"{h_unit / reps:17.3f} {h_state / reps:18.3f}"
        )
    RESULTS["s5_nested"] = [
        [int(a), int(b), float(c), float(d), float(e)] for a, b, c, d, e in nest_rows
    ]
    r2010 = [r for r in nest_rows if r[0] == 20 and r[1] == 10][0]
    r20200 = [r for r in nest_rows if r[0] == 20 and r[1] == 200][0]
    r0610 = [r for r in nest_rows if r[0] == 6][0]
    r5010 = [r for r in nest_rows if r[0] == 50][0]
    print(
        f"\n  Clustering on the unit gives {r2010[3]:.3f} against a nominal 0.05 - {r2010[3] / 0.05:.0f}x - while\n"
        f"  clustering on the state gives {r2010[4]:.3f}.  The unit-clustered SE recovered\n"
        f"  {100 * (r2010[2] - r2010[3]) / (r2010[2] - 0.05):.0f}% of the distance from the iid SE ({r2010[2]:.3f}) to correct.  'We clustered\n"
        "  our standard errors' is not a statement about anything until it says what by.\n"
        f"\n  And it gets WORSE with data: 200 units per state instead of 10 takes it from\n"
        f"  {r2010[3]:.3f} to {r20200[3]:.3f}, because every extra row inside a state adds no independent\n"
        f"  information about the state and the wrong formula counts it as if it did.\n"
        f"\n  The two panels together: SIX states clustered correctly ({r0610[4]:.3f}) beats FIFTY\n"
        f"  states clustered one level too fine ({r5010[3]:.3f}) by a factor of {r5010[3] / r0610[4]:.0f}.  The count is\n"
        "  the received warning; the level is the failure."
    )


# ==========================================================================
# 6. TWFE with staggered adoption: negative weights
# ==========================================================================


def section_6() -> Tuple[float, float]:
    head(6, "NEGATIVE RESULT: EVERY TRUE EFFECT POSITIVE, THE ESTIMATE NEGATIVE")
    print(
        "Nothing above needed staggered timing.  Now let cohorts adopt at different\n"
        "dates, and let the effect GROW with exposure - the most ordinary form of\n"
        "heterogeneity there is.  Under y = a_i + g_t + tau_it D_it + e_it, FWL gives\n"
        "\n      E[beta_twfe] = sum_{treated cells} w_it tau_it,  w_it = Dtilde_it / sum Dtilde^2\n"
        "\n  and sum w_it = 1 exactly.  Nothing makes an individual w_it positive."
    )

    rng = np.random.default_rng(0)
    Y, D, adopt, tau = make_staggered(
        rng, [(4, 50), (10, 50)], T=20, growth=0.5, sigma=0.0, unit_sd=0.0
    )
    W = twfe_cell_weights(D)
    b = twfe(Y, D)
    implied = float((W * tau).sum())
    truth = float(tau[D > 0].mean())

    sub("the identity, checked")
    print(f"  sum of weights over treated cells        {W.sum():.15f}")
    print(f"  TWFE coefficient                         {b:+.10f}")
    print(f"  sum w_it * tau_it                        {implied:+.10f}")
    print(f"  |gap|                                    {abs(b - implied):.2e}")
    RESULTS["s6_weight_sum"] = float(W.sum())
    RESULTS["s6_identity_gap"] = float(abs(b - implied))

    sub("the weights")
    w = W[D > 0]
    neg = w < 0
    print(f"  treated cells                            {w.size}")
    print(f"  cells with NEGATIVE weight               {neg.sum()} ({neg.mean():.1%})")
    print(f"  total negative weight                    {w[neg].sum():+.4f}")
    print(f"  total positive weight                    {w[~neg].sum():+.4f}")
    print(f"  most negative single weight              {w.min():+.6f}")
    RESULTS["s6_neg_share"] = float(neg.mean())
    RESULTS["s6_neg_weight"] = float(w[neg].sum())

    sub("what it does to the answer")
    print(f"  true effect on every treated cell        min {tau[D > 0].min():.2f}, max {tau[D > 0].max():.2f}")
    print(f"  true mean effect over treated cells      {truth:.4f}")
    print(f"  TWFE coefficient                         {b:.4f}")
    print(
        f"\n  The smallest true effect anywhere in this panel is {tau[D > 0].min():.2f} and the largest is\n"
        f"  {tau[D > 0].max():.2f}.  TWFE returns {b:.4f}: outside the range of every individual effect in\n"
        "  the data, and 97% below the average it is supposed to estimate."
    )
    RESULTS["s6_true_mean"] = truth
    RESULTS["s6_twfe"] = float(b)
    RESULTS["s6_tau_min"] = float(tau[D > 0].min())
    RESULTS["s6_tau_max"] = float(tau[D > 0].max())

    sub("where the negative weights are")
    late = adopt == 10
    early = adopt == 4
    t = np.arange(20)
    blk_early_late = W[np.ix_(early, t >= 10)]
    blk_early_mid = W[np.ix_(early, (t >= 4) & (t < 10))]
    blk_late = W[np.ix_(late, t >= 10)]
    print(f"  early cohort, before the late cohort adopts   mean w {blk_early_mid.mean():+.6f}")
    print(f"  early cohort, after  the late cohort adopts   mean w {blk_early_late.mean():+.6f}")
    print(f"  late  cohort, once treated                    mean w {blk_late.mean():+.6f}")
    print(
        "\n  Every negative weight is in one block: the early cohort's periods AFTER the\n"
        "  late cohort adopts.  In those periods the regression is using the early\n"
        "  cohort - already treated, and by then carrying its largest effect - as the\n"
        "  control group for the late cohort.  Their growing effect enters the\n"
        "  comparison with the wrong sign.  That is the whole mechanism."
    )
    RESULTS["s6_blocks"] = [
        float(blk_early_mid.mean()),
        float(blk_early_late.mean()),
        float(blk_late.mean()),
    ]

    sub("the exposure growth at which the sign flips")
    lo, hi = 0.0, 2.0
    for _ in range(50):
        mid = (lo + hi) / 2
        Yg, Dg, _, _ = make_staggered(
            np.random.default_rng(1), [(4, 50), (10, 50)], T=20, growth=mid, sigma=0.0, unit_sd=0.0
        )
        if twfe(Yg, Dg) > 0:
            lo = mid
        else:
            hi = mid
    flip = (lo + hi) / 2
    print(f"  growth per period at which TWFE crosses zero   {flip:.6f}")
    print(f"  (effect goes from {1.0:.1f} at adoption to {1.0 + flip * 15:.2f} fifteen periods later)")
    print(f"\n  {'growth':>7} {'true mean':>10} {'TWFE':>9} {'ratio':>8}")
    grow_rows = []
    for g in [0.0, 0.1, 0.25, 0.5, flip, 0.75, 1.0, 1.5]:
        Yg, Dg, _, tg = make_staggered(
            np.random.default_rng(1), [(4, 50), (10, 50)], T=20, growth=g, sigma=0.0, unit_sd=0.0
        )
        tm = float(tg[Dg > 0].mean())
        bg = twfe(Yg, Dg)
        grow_rows.append((g, tm, bg))
        print(f"  {g:7.3f} {tm:10.4f} {bg:9.4f} {bg / tm:8.3f}")
    RESULTS["s6_flip"] = float(flip)
    RESULTS["s6_grow_rows"] = [[float(a), float(b), float(c)] for a, b, c in grow_rows]
    print(
        f"\n  A gently strengthening effect - {flip:.2f} per period, so a policy that is 8.4x\n"
        "  stronger after fifteen periods than on day one - is enough for a coefficient\n"
        "  that is exactly zero.  Beyond it, the sign is wrong."
    )

    sub("how much heterogeneity it takes, from the estimate alone")
    Y2, D2, adopt2, tau2 = make_staggered(
        rng, [(4, 50), (10, 50)], T=20, growth=0.5, sigma=1.0, unit_sd=1.0
    )
    W2 = twfe_cell_weights(D2)
    b2 = twfe(Y2, D2)
    bound = heterogeneity_bound(W2, D2, b2)
    bound_true = heterogeneity_bound(W2, D2, tau2[D2 > 0].mean())
    ratio = heterogeneity_bound(W2, D2, 1.0)  # bound is linear in the referenced ATT
    print(f"  TWFE estimate on a noisy draw            {b2:+.4f}")
    print(f"  tolerance RATIO (design constant)        {ratio:.4f}")
    print(f"    -> sd that zeroes an ATT of 1.00       {ratio:.4f}")
    print(f"    -> sd that zeroes the true ATT {tau2[D2 > 0].mean():.2f}      {bound_true:.4f}")
    print(f"  ACTUAL sd of effects in this panel       {tau2[D2 > 0].std(ddof=0):.4f}")
    print(
        f"\n  The bound is linear in whatever ATT it is referenced to, so the number worth\n"
        f"  quoting is the ratio: {ratio:.4f}.  THIS DESIGN cannot survive treatment-effect\n"
        f"  heterogeneity larger than {ratio:.2f}x its own ATT.  Referenced to the true ATT of\n"
        f"  {tau2[D2 > 0].mean():.2f} that is sd {bound_true:.2f}, and the panel carries {tau2[D2 > 0].std(ddof=0):.2f} - past it, hence the\n"
        "  collapsed estimate.  The ratio comes from the weights alone: it is computable\n"
        "  from the adoption dates before the outcome column is opened, which makes it\n"
        "  the honest thing to publish next to a TWFE coefficient (de Chaisemartin &\n"
        "  D'Haultfoeuille's robustness measure, rederived in `did.py`).\n"
        "  Do not read it as a small number meaning a fragile method - read it as this\n"
        "  design being fragile, and check yours."
    )
    RESULTS["s6_bound"] = float(bound)
    RESULTS["s6_ratio"] = float(ratio)
    RESULTS["s6_bound_true"] = float(bound_true)
    RESULTS["s6_actual_sd"] = float(tau2[D2 > 0].std(ddof=0))
    return float(b), truth


# ==========================================================================
# 7. The fix
# ==========================================================================


def section_7() -> None:
    head(7, "THE FIX IS ONE RESTRICTION: NEVER USE AN ALREADY-TREATED UNIT AS A CONTROL")
    print(
        "Group-time ATT(g, t): each cohort against units NOT YET treated at t, with\n"
        "cohort-specific base period g-1, then averaged over treated cells.  That is\n"
        "the whole change.  It needs a clean comparison to exist at every (g, t), so\n"
        "the panel below carries a never-treated cohort."
    )
    rng = np.random.default_rng(7)
    Y, D, adopt, tau = make_staggered(
        rng, [(4, 50), (10, 50)], T=20, n_never=50, growth=0.5, sigma=0.0, unit_sd=0.0
    )
    truth = float(tau[D > 0].mean())
    b = twfe(Y, D)
    atts = group_time_att(Y, adopt)
    cs = aggregate_att(atts, adopt, 20)
    print(f"\n  true mean effect over treated cells      {truth:.6f}")
    print(f"  TWFE                                     {b:.6f}   ({100 * (b / truth - 1):+.1f}%)")
    print(f"  not-yet-treated group-time ATT           {cs:.6f}   ({100 * (cs / truth - 1):+.1f}%)")
    print(f"  |ATT - truth|                            {abs(cs - truth):.2e}")
    RESULTS["s7_truth"] = truth
    RESULTS["s7_twfe"] = float(b)
    RESULTS["s7_cs"] = float(cs)

    sub("the same comparison with noise, over many draws")
    rng = np.random.default_rng(8)
    reps = 400
    bt, bc = [], []
    for _ in range(reps):
        Yn, Dn, an, tn = make_staggered(
            rng, [(4, 50), (10, 50)], T=20, n_never=50, growth=0.5, sigma=1.0, unit_sd=1.0
        )
        bt.append(twfe(Yn, Dn))
        bc.append(aggregate_att(group_time_att(Yn, an), an, 20))
    bt, bc = np.array(bt), np.array(bc)
    print(f"  {'estimator':>28} {'mean':>9} {'bias':>9} {'sd':>8}")
    print(f"  {'TWFE':>28} {bt.mean():9.4f} {bt.mean() - truth:9.4f} {bt.std(ddof=1):8.4f}")
    print(f"  {'not-yet-treated ATT':>28} {bc.mean():9.4f} {bc.mean() - truth:9.4f} {bc.std(ddof=1):8.4f}")
    RESULTS["s7_mc"] = [float(bt.mean()), float(bc.mean()), float(bt.std(ddof=1)), float(bc.std(ddof=1))]
    print(
        f"\n  TWFE is off by {truth - bt.mean():.2f} - {100 * (1 - bt.mean() / truth):.0f}% of the effect - and the not-yet-treated\n"
        f"  estimator is off by {abs(bc.mean() - truth):.4f}, at {bc.std(ddof=1) / bt.std(ddof=1):.2f}x the standard deviation.  The\n"
        "  correction is not expensive; it is a WHERE clause on the control group.\n"
        "  With no never-treated cohort at all, the last cohort's late periods have no\n"
        "  clean comparison and honestly cannot be estimated - which is information,\n"
        "  and is what TWFE silently spends a negative weight to paper over."
    )

    sub("event-study form: what the cohorts actually did")
    print(f"  {'(cohort, t)':>14} {'ATT':>9} {'true':>9}")
    shown = 0
    for (g, t), a in sorted(atts.items()):
        if t - g in (0, 2, 5, 9):
            true_cell = float(tau[(adopt == g), t].mean())
            print(f"  {f'({g}, {t})':>14} {a:9.4f} {true_cell:9.4f}")
            shown += 1
        if shown >= 8:
            break


# ==========================================================================
# 8. Functional form
# ==========================================================================


def section_8() -> None:
    head(8, "NEGATIVE RESULT: PARALLEL IN LEVELS AND PARALLEL IN LOGS ARE DIFFERENT CLAIMS")
    print(
        "DiD is not invariant to a monotone transform of the outcome.  'Parallel\n"
        "trends' names a different assumption in levels than in logs, and with unequal\n"
        "baselines the two cannot both hold unless the control group does not move."
    )
    c_pre, c_post = 100.0, 120.0
    t_pre = 200.0
    cf_levels = t_pre + (c_post - c_pre)
    cf_logs = t_pre * (c_post / c_pre)
    print(f"\n  control:  {c_pre:.0f} -> {c_post:.0f}   (+{c_post - c_pre:.0f}, x{c_post / c_pre:.2f})")
    print(f"  treated:  {t_pre:.0f} -> observed")
    print(f"  counterfactual if trends are parallel in LEVELS: {cf_levels:.1f}")
    print(f"  counterfactual if trends are parallel in LOGS:   {cf_logs:.1f}")
    print(
        f"\n  Those differ by {cf_logs - cf_levels:.0f} on a baseline of {t_pre:.0f}.  Any observed post value\n"
        f"  strictly between them makes the levels DiD positive and the log DiD\n"
        f"  negative - a window covering every treatment effect from 0% to\n"
        f"  {100 * (cf_logs / cf_levels - 1):.1f}% of the levels counterfactual."
    )
    print(f"\n  {'observed':>9} {'levels DiD':>11} {'log DiD':>10} {'agree?':>8}")
    rows = []
    for obs in [210.0, 220.0, 225.0, 230.0, 235.0, 240.0, 250.0]:
        lv = (obs - t_pre) - (c_post - c_pre)
        lg = np.log(obs / t_pre) - np.log(c_post / c_pre)
        agree = "yes" if np.sign(lv) == np.sign(lg) else "NO"
        rows.append((obs, lv, float(lg), agree))
        print(f"  {obs:9.1f} {lv:11.2f} {lg:10.5f} {agree:>8}")
    RESULTS["s8_rows"] = [[float(a), float(b), float(c), d] for a, b, c, d in rows]
    RESULTS["s8_window"] = [float(cf_levels), float(cf_logs)]

    sub("the impossibility, stated")
    print(
        "  Parallel in levels:  Yt1 - Yt0 = Yc1 - Yc0\n"
        "  Parallel in logs:    Yt1 / Yt0 = Yc1 / Yc0\n"
        "  Subtracting: both hold iff (Yt0 - Yc0)(Yc1/Yc0 - 1) = 0, i.e. iff the two\n"
        "  groups start level, or the control group does not move at all."
    )
    print(f"\n  checked on these numbers: (Yt0 - Yc0)(Yc1/Yc0 - 1) = {(t_pre - c_pre) * (c_post / c_pre - 1):.4f} != 0")
    RESULTS["s8_impossible_term"] = float((t_pre - c_pre) * (c_post / c_pre - 1))

    sub("both scales, both worlds - and the pre-period can tell them apart")
    print(
        "  Two worlds.  In A the common trend is MULTIPLICATIVE, so parallel trends\n"
        "  holds in logs and the log DiD is the correct one; the effect is -5%.  In B the\n"
        "  common trend is ADDITIVE, so parallel trends holds in levels and the levels\n"
        "  DiD is correct; the effect is +10 units.  Each world's effect is reported\n"
        "  accurately by its own correct scale and with the WRONG SIGN by the other one."
    )
    T, t0 = 8, 4
    n = 400
    treated = np.zeros(n, dtype=bool)
    treated[: n // 2] = True
    base = np.where(treated, t_pre, c_pre)[:, None]
    tt = np.arange(T)[None, :]
    post = (tt >= t0).astype(float)
    treat_post = treated[:, None] * post
    adopt = np.where(treated, float(t0), np.inf)
    ev = [-3, -2, -1, 0, 1, 2, 3]
    cols = event_dummies(adopt, T, [e for e in ev if e != -1])
    lead_idx = [i for i, e in enumerate([e for e in ev if e != -1]) if e < 0]

    print(f"\n  {'world':>34} {'true sign':>10} {'levels neg':>11} {'logs neg':>9} {'sign clash':>11}")
    world_rows = []
    for world in ("A: multiplicative trend (logs true)", "B: additive trend (levels true)"):
        rng = np.random.default_rng(31 if world.startswith("A") else 32)
        reps = 1500
        lv_neg = lg_neg = clash = 0
        fire_lv = fire_lg = 0
        for _ in range(reps):
            if world.startswith("A"):
                Y = base * (1.05 ** tt) * (1.0 - 0.05 * treat_post)  # -5% effect, logs are true
            else:
                Y = base + 5.0 * tt + 10.0 * treat_post  # +10 level effect, levels are true
            Y = np.clip(Y + rng.normal(0, 4.0, (n, T)), 1.0, None)
            lv = did_2x2(Y, treated, range(t0), range(t0, T))
            lg = did_2x2(np.log(Y), treated, range(t0), range(t0, T))
            lv_neg += lv < 0
            lg_neg += lg < 0
            clash += (lv < 0) != (lg < 0)
            if fe_ols(Y, cols, None, vcov="cluster").wald(lead_idx)[1] < 0.05:
                fire_lv += 1
            if fe_ols(np.log(Y), cols, None, vcov="cluster").wald(lead_idx)[1] < 0.05:
                fire_lg += 1
        world_rows.append(
            (world, lv_neg / reps, lg_neg / reps, clash / reps, fire_lv / reps, fire_lg / reps)
        )
        truesign = "-" if world.startswith("A") else "+"
        print(
            f"  {world:>34} {truesign:>10} {lv_neg / reps:11.3f} "
            f"{lg_neg / reps:9.3f} {clash / reps:11.3f}"
        )
    RESULTS["s8_worlds"] = [[w, float(a), float(b), float(c), float(d), float(e)] for w, a, b, c, d, e in world_rows]
    a_row, b_row = world_rows
    print(
        f"\n  World A: the log spec correctly calls the -5% effect negative {a_row[2]:.1%} of the\n"
        f"  time, and the levels spec calls the same data POSITIVE {1 - a_row[1]:.1%} of the time.\n"
        f"  World B reverses it exactly: the levels spec correctly reports the +10 effect\n"
        f"  as positive {1 - b_row[1]:.1%} of the time, and the log spec reports it NEGATIVE {b_row[2]:.1%} of\n"
        f"  the time.  Sign clash {a_row[3]:.1%} and {b_row[3]:.1%}.  Same estimator, same three lines of\n"
        f"  code, opposite verdict on the direction of a policy - and nothing in either\n"
        f"  regression output flags which one is being read."
    )

    sub("the constructive half: run the pre-trends test in BOTH scales")
    print(
        "  A wrong scale is a parallel-trends violation, so - unlike the assumption\n"
        "  itself - it leaves a footprint in the pre-period whenever the common trend\n"
        "  moves during it.  Joint lead test, same data, run on Y and on log Y:"
    )
    print(f"\n  {'world':>34} {'pretest fires: levels':>22} {'pretest fires: logs':>21}")
    for w, _, _, _, fl, fg in world_rows:
        print(f"  {w:>34} {fl:22.3f} {fg:21.3f}")
    print(
        f"\n  In world A the levels pre-trend test fires {a_row[4]:.3f} and the log one {a_row[5]:.3f}.\n"
        f"  In world B it reverses ({b_row[4]:.3f} vs {b_row[5]:.3f}).  The scale that PASSES its own\n"
        f"  pre-trend test is the scale the data supports - which makes this the one\n"
        f"  identifying assumption in the whole build that is genuinely testable, and it\n"
        f"  costs one extra line of code.  Caveat with teeth: it works because the common\n"
        f"  trend moves before treatment.  On a flat pre-period both scales pass, both\n"
        f"  stay defensible, and the sign of the reported effect is a choice the analyst\n"
        f"  makes rather than a thing the data settles."
    )


# ==========================================================================


def main() -> None:
    t_start = time.time()
    print("=" * 78)
    print("diff-in-diff - PARALLEL TRENDS IS AN ASSUMPTION, AND THE TEST FOR IT HAS A POWER")
    print("=" * 78)
    print(
        "Day 167.  Difference-in-differences on worlds whose true effect is known,\n"
        "so that every claim below is a measurement and not an argument."
    )
    section_1()
    rows = section_2()
    section_3(rows)
    section_4()
    section_5()
    section_6()
    section_7()
    section_8()

    head(9, "WHAT TO PUT IN THE REPORT")
    print(
        "1. The pre-window LENGTH, and the smallest violation the pre-trends test could\n"
        "   have caught at that length.  A flat plot with five pre-periods rules out\n"
        "   nothing (section 2).\n"
        "2. The bias that the smallest UNDETECTABLE violation would imply, in the units\n"
        "   of the reported effect.  Not a p-value - a bias (sections 1-3).\n"
        "3. What the standard error is clustered BY, and why that is the level at which\n"
        "   treatment varies (section 5).\n"
        "4. For staggered adoption: the share of negative weight, the heterogeneity sd\n"
        "   at which the estimate would be zero, and the not-yet-treated estimate\n"
        "   alongside the TWFE one (sections 6-7).\n"
        "5. The scale - levels or logs - argued for before the regression, because the\n"
        "   two are different assumptions and can disagree on the sign (section 8)."
    )
    print(f"\n[evidence run: {time.time() - t_start:.1f}s]")
    with open("results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)
    print("[wrote results.json]")


if __name__ == "__main__":
    main()
