"""The measured argument.  Every number in the README and the notebook is printed here.

Run: python evidence.py   (~60s)
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np
from synth import (
    attenuation_prediction,
    fit_synth,
    hull_excess,
    in_donor_band,
    make_panel,
    naive_estimates,
    placebo_pvalue,
    pvalue_floor,
    rmse,
)

RESULTS: Dict[str, object] = {}
EFFECT = 5.0


def head(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n {n}. {title}\n{'=' * 78}")


def mcerr(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


# ============================================================ 1
def section_1() -> None:
    head(1, "THE INSTRUMENT WORKS: one treated unit, no control group, a fitted one")
    print(
        """A single market got the intervention.  There is no control group, so the three things
people reach for are the closest-looking single market, the average of the others, and a
difference-in-differences against that average.  Synthetic control instead asks: which
WEIGHTED average of the others tracked this market before the intervention?

The panel below is an interactive fixed-effects model - each unit loads on three common
time factors - and the treated unit's loadings are a lopsided convex combination of the
donors', so it is genuinely an unusual member of its own pool.  A convex combination that
reproduces it exists; the estimators differ in whether they can find it."""
    )
    n_sims = 600
    acc: Dict[str, List[float]] = {k: [] for k in ("synth", "best_donor", "donor_mean", "did")}
    weights_used, top_weight = [], []
    for s in range(n_sims):
        rng = np.random.default_rng(1000 + s)
        Y = make_panel(rng, effect=EFFECT)
        f = fit_synth(Y, 0, 30)
        acc["synth"].append(f.att)
        weights_used.append(int((f.weights > 1e-4).sum()))
        top_weight.append(float(f.weights.max()))
        for k, v in naive_estimates(Y, 0, 30).items():
            acc[k].append(v)

    print(f"\n  {n_sims} panels, 20 donors, 30 pre-periods, 12 post, true effect {EFFECT}\n")
    print(f"  {'estimator':<16}{'mean':>9}{'bias':>9}{'rmse':>9}{'x synth rmse':>15}")
    base = rmse(np.array(acc["synth"]), EFFECT)
    for k, label in [
        ("synth", "synthetic"),
        ("did", "diff-in-diff"),
        ("donor_mean", "donor average"),
        ("best_donor", "closest donor"),
    ]:
        a = np.array(acc[k])
        r = rmse(a, EFFECT)
        print(f"  {label:<16}{a.mean():>9.3f}{a.mean() - EFFECT:>9.3f}{r:>9.3f}{r / base:>15.2f}")
    print(
        f"\n  The fit is sparse by construction: {np.mean(weights_used):.1f} donors carry non-zero"
        f" weight out of 20,\n  the largest one carries {np.mean(top_weight):.2f} of the total."
        " That sparsity is the simplex constraint,\n  not a penalty anyone chose - it is what"
        " section 6 removes."
    )
    RESULTS["s1"] = {
        k: {"mean": float(np.mean(v)), "rmse": rmse(np.array(v), EFFECT)} for k, v in acc.items()
    }
    RESULTS["s1"]["sparsity"] = {"donors_used": float(np.mean(weights_used)), "top": float(np.mean(top_weight))}


# ============================================================ 2
def section_2() -> None:
    head(2, "THE P-VALUE HAS A FLOOR, AND THE FLOOR IS THE DESIGN, NOT THE DATA")
    print(
        """Inference is a permutation: refit the estimator with each donor playing the treated
unit, and rank the real one.  With N units there are N ranks, so the smallest p-value the
procedure can return is 1/N = 1/(donors+1).  No effect size changes this.  No sample size
changes it either - it is arithmetic on the donor count, available before any data."""
    )
    print(f"\n  {'donors':>8}{'floor 1/(J+1)':>16}{'0.05 reachable':>17}{'0.10 reachable':>16}")
    floors = {}
    for j in (5, 9, 10, 15, 19, 20, 39, 50):
        f = pvalue_floor(j)
        floors[j] = f
        print(f"  {j:>8}{f:>16.4f}{'yes' if f <= 0.05 else 'NO':>17}{'yes' if f <= 0.10 else 'NO':>16}")
    print(
        "\n  19 donors is the smallest pool that can ever produce p <= 0.05, and 9 the smallest"
        "\n  that can produce p <= 0.10.  A study of 12 US states cannot report a 5% result."
    )

    n_sims = 400
    print(f"\n  Power of the placebo test, {n_sims} panels each (rejection at the nominal level):\n")
    print(f"  {'donors':>8}{'effect':>9}{'p<=0.05':>10}{'p<=0.10':>10}{'median p':>11}")
    power: Dict[str, float] = {}
    for j in (10, 20, 40):
        for eff in (0.0, 2.0, 5.0, 10.0):
            ps = []
            for s in range(n_sims):
                rng = np.random.default_rng(20000 + s)
                Y = make_panel(rng, n_donors=j, effect=eff)
                p, _ = placebo_pvalue(Y, 0, 30)
                ps.append(p)
            ps_a = np.array(ps)
            r05, r10 = float((ps_a <= 0.05).mean()), float((ps_a <= 0.10).mean())
            power[f"j{j}_e{eff}"] = r05
            power[f"j{j}_e{eff}_10"] = r10
            print(f"  {j:>8}{eff:>9.1f}{r05:>10.3f}{r10:>10.3f}{np.median(ps_a):>11.3f}")
    print(
        "\n  The 0.05 column is exactly 0.000 for 10 donors at EVERY effect size, including one"
        "\n  twice the size that 40 donors detects with certainty.  That is not low power, it is"
        "\n  zero power: the test has no outcome that rejects.  Under the null (effect 0.0) the"
        "\n  0.10 column sits at the floor's granularity, which is what calibration looks like"
        "\n  when the p-value can only take J+1 values."
    )
    RESULTS["s2"] = {"floors": floors, "power": power}


# ============================================================ 3
def section_3() -> None:
    head(3, "NEGATIVE: THE PRE-PERIOD FIT IS THE THING EVERYONE SHOWS, AND IT IS NOT EVIDENCE")
    print(
        """Every synthetic control paper opens with the pre-period plot: two lines lying on top of
each other.  It is offered as the credential.  But the weights were CHOSEN to make those
lines coincide, so the fit is a measure of how much freedom the optimiser had, and the
freedom grows with the donor count while the accuracy does not."""
    )
    t_pre = 10
    n_sims = 500
    print(f"\n  {t_pre} pre-periods held fixed, donor pool grown, {n_sims} panels each:\n")
    print(f"  {'donors':>8}{'pre-RMSPE':>12}{'att rmse':>11}{'corr(pre,|err|)':>18}{'p of corr':>11}")
    rows = []
    for j in (5, 10, 20, 40, 80):
        pre, err = [], []
        for s in range(n_sims):
            rng = np.random.default_rng(30000 + s)
            Y = make_panel(rng, n_donors=j, t_pre=t_pre, effect=EFFECT)
            f = fit_synth(Y, 0, t_pre)
            pre.append(f.pre_rmspe)
            err.append(f.att - EFFECT)
        pre_a, err_a = np.array(pre), np.abs(np.array(err))
        c = float(np.corrcoef(pre_a, err_a)[0, 1])
        # two-sided p for the correlation, t = r sqrt((n-2)/(1-r^2))
        from scipy import stats as st

        tstat = c * np.sqrt((n_sims - 2) / max(1e-12, 1 - c**2))
        pv = float(2 * st.t.sf(abs(tstat), n_sims - 2))
        rows.append((j, float(pre_a.mean()), rmse(np.array(err) + EFFECT, EFFECT), c, pv))
        print(f"  {j:>8}{pre_a.mean():>12.3f}{rows[-1][2]:>11.3f}{c:>18.3f}{pv:>11.4f}")
    RESULTS["s3"] = {"rows": rows}
    best, worst = rows[-1], rows[0]
    print(
        f"\n  Going from {worst[0]} donors to {best[0]}, the pre-period fit improves"
        f" {worst[1] / best[1]:.1f}x - the picture in the\n  paper gets strictly more convincing -"
        f" while the error on the actual estimate goes\n  {worst[2]:.3f} -> {best[2]:.3f}"
        f" ({'better' if best[2] < worst[2] else 'WORSE'}).  Within a fixed pool the correlation"
        "\n  between how good the fit looks and how wrong the answer is stays near zero."
    )


# ============================================================ 4
def section_4() -> None:
    head(4, "NEGATIVE: THE CONVEX HULL IS A HARD WALL, AND THE LEVER THAT MOVES IT IS NOT THE OBVIOUS ONE")
    print(
        """The weights are non-negative and sum to one, so the synthetic unit is a convex
combination and cannot leave the donor range: never above the highest donor, never below the
lowest, in any period.  That is a structural guarantee, checked exactly below.  It means a
treated unit sitting above its whole pool by a constant h simply cannot be matched.

The obvious guess is that the estimator then returns tau + h.  It does not.  The optimiser
buys back part of h by tilting the weights toward the highest-level donors and paying for it
in factor fit - so the bias is a FRACTION of h, and the fraction is not a constant."""
    )
    n_sims = 400
    print(f"\n  {'shift h':>9}{'mean att':>11}{'bias':>9}{'bias/h':>9}{'hull excess':>14}{'pre-RMSPE':>12}{'in band':>10}")
    rows = []
    for h in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0):
        atts, pres, exc, band = [], [], [], []
        for s in range(n_sims):
            rng = np.random.default_rng(40000 + s)
            Y = make_panel(rng, effect=EFFECT, hull_shift=h)
            f = fit_synth(Y, 0, 30)
            atts.append(f.att)
            pres.append(f.pre_rmspe)
            exc.append(hull_excess(Y, 0, 30)["post"])
            band.append(in_donor_band(Y, 0, f))
        m, b = float(np.mean(atts)), float(np.mean(atts)) - EFFECT
        rows.append((h, m, b, b / h if h else 0.0, float(np.mean(exc)), float(np.mean(pres)), float(np.mean(band))))
        print(
            f"  {h:>9.1f}{m:>11.3f}{b:>9.3f}{b / h if h else 0.0:>9.3f}"
            f"{np.mean(exc):>14.3f}{np.mean(pres):>12.3f}{np.mean(band):>10.3f}"
        )
    print(
        "\n  The 'in band' column is 1.000 everywhere: the guarantee holds exactly, in every one"
        f"\n  of {n_sims * len(rows)} fits.  'hull excess' is the part of the treated path lying outside the"
        "\n  donor range - computable from the data alone, with no counterfactual in it - and it is"
        "\n  an honest lower bound on the damage rather than the damage itself."
    )

    print("\n  Now the two levers, separately, at h = 5.0 (bias to remove: the h=5 row above):\n")
    print(f"  {'held fixed':>28}{'grown':>10}{'from':>10}{'to':>10}{'bias from':>12}{'bias to':>10}")
    lever = {}
    for label, fixed, grow_key, lo, hi in (
        ("30 pre-periods", {"t_pre": 30}, "n_donors", 5, 80),
        ("20 donors", {"n_donors": 20}, "t_pre", 15, 240),
    ):
        out = []
        for v in (lo, hi):
            atts = []
            for s in range(n_sims):
                rng = np.random.default_rng(41000 + s)
                kw = dict(fixed)
                kw[grow_key] = v
                Y = make_panel(rng, effect=EFFECT, hull_shift=5.0, **kw)
                atts.append(fit_synth(Y, 0, kw.get("t_pre", 30)).att)
            out.append(float(np.mean(atts)) - EFFECT)
        lever[grow_key] = (lo, hi, out[0], out[1])
        print(f"  {label:>28}{grow_key:>10}{lo:>10}{hi:>10}{out[0]:>12.3f}{out[1]:>10.3f}")
    d_lo, d_hi, d_b0, d_b1 = lever["n_donors"]
    t_lo, t_hi, t_b0, t_b1 = lever["t_pre"]
    print(
        f"\n  Widening the pool {d_hi // d_lo}x removes {(1 - d_b1 / d_b0) * 100:.0f}% of the bias;"
        f" lengthening the history {t_hi // t_lo}x removes {(1 - t_b1 / t_b0) * 100:.0f}%."
        "\n  The donor column is the part worth noticing.  In section 9 - the same estimator, the"
        " same\n  panel, but a NOISE problem rather than a coverage problem - growing the pool 16x"
        " moved the\n  error by 3% and was indistinguishable from doing nothing.  Here it removes"
        f" {(1 - d_b1 / d_b0) * 100:.0f}% of a real\n  bias.  Donors are not a sample-size lever and"
        " they are not useless; they buy hull coverage\n  and nothing else.  Which of the two"
        " problems you have is not readable off the pre-period\n  plot everybody publishes - but"
        " it IS readable off the hull excess above, which nobody\n  publishes."
    )
    RESULTS["s4"] = {"shift_rows": rows, "levers": lever}


# ============================================================ 5
def section_5() -> None:
    head(5, "THE RECOMMENDED TEST STATISTIC EARNS ITS KEEP (the one place in this build it does)")
    print(
        """Abadie ranks units by the RATIO of post-period to pre-period fit error, not by the raw
gap.  The reason is section 4: a unit the donors cannot fit has a large gap in every period,
before and after, and a raw-gap ranking reads that as an effect.  Dividing by the pre-period
error cancels it.  Below, the null is true - there is no effect at all - and the treated unit
is simply un-fittable (h = 5)."""
    )
    n_sims = 500
    print(f"\n  {'h':>5}{'true effect':>13}{'stat':>10}{'p<=0.10':>10}{'p<=0.20':>10}{'median p':>11}")
    rows = []
    for h, eff in ((0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0)):
        for stat in ("gap", "ratio"):
            ps = []
            for s in range(n_sims):
                rng = np.random.default_rng(50000 + s)
                Y = make_panel(rng, effect=eff, hull_shift=h)
                p, _ = placebo_pvalue(Y, 0, 30, statistic=stat)
                ps.append(p)
            pa = np.array(ps)
            rows.append((h, eff, stat, float((pa <= 0.10).mean()), float((pa <= 0.20).mean()), float(np.median(pa))))
            print(
                f"  {h:>5.1f}{eff:>13.1f}{stat:>10}{(pa <= 0.10).mean():>10.3f}"
                f"{(pa <= 0.20).mean():>10.3f}{np.median(pa):>11.3f}"
            )
    d = {(r[0], r[1], r[2]): r for r in rows}
    print(
        f"""
  Three things in that table, in order of how much they matter.

  1. On a TRUE NULL with an un-fittable treated unit the raw-gap statistic fires
     {d[(5.0, 0.0, "gap")][3]:.3f} of the time at a nominal 0.10 - it is reading section 4's
     level shift as an effect - while the ratio fires {d[(5.0, 0.0, "ratio")][3]:.3f}.  This is
     the one guard in this build that does the job it is sold for.

  2. It is paid for.  With a real effect on an un-fittable unit the ratio finds it
     {d[(5.0, 5.0, "ratio")][3]:.3f} of the time against the raw gap's {d[(5.0, 5.0, "gap")][3]:.3f} -
     it has thrown away the same signal it correctly refused in row 2, because from inside
     the statistic those two situations are identical.

  3. The raw gap is not merely mis-sized, it is inert on a clean null: {d[(0.0, 0.0, "gap")][3]:.3f}
     at a nominal 0.10, median p {d[(0.0, 0.0, "gap")][5]:.3f}.  The placebo units are not
     exchangeable with the treated one - a donor is an extreme point of its own pool and is
     fitted badly by the units left over, while the treated unit is interior and fits well - so
     the reference distribution is drawn from harder problems than the one being tested.
     Dividing by each unit's own pre-period error is what puts them back on the same scale
     ({d[(0.0, 0.0, "ratio")][3]:.3f} at 0.10, median {d[(0.0, 0.0, "ratio")][5]:.3f}).  The ratio's
     real job is normalisation, not robustness.

  None of this repairs the ESTIMATE.  Section 4's bias is still in the number; the test has
  only stopped calling it an effect."""
    )
    RESULTS["s5"] = {"rows": rows}


# ============================================================ 6
def section_6() -> None:
    head(6, "NEGATIVE: DROPPING THE SIMPLEX CONSTRAINT BUYS A BETTER FIT AND A WORSE ANSWER")
    print(
        """The constraint w >= 0, sum w = 1 is the whole method.  Regression on the donors with no
constraint is strictly better at the thing the pre-period plot measures, and that is exactly
why it is worse: it extrapolates."""
    )
    n_sims = 500
    print(f"\n  {'pre-periods':>13}{'solver':>10}{'pre-RMSPE':>12}{'att rmse':>11}{'weight sum':>12}{'neg wts':>9}")
    rows = []
    for tp in (12, 30, 60):
        for solver in ("simplex", "ols"):
            pre, atts, wsum, negs = [], [], [], []
            for s in range(n_sims):
                rng = np.random.default_rng(60000 + s)
                Y = make_panel(rng, t_pre=tp, effect=EFFECT)
                f = fit_synth(Y, 0, tp, solver=solver)
                pre.append(f.pre_rmspe)
                atts.append(f.att)
                wsum.append(float(f.weights.sum()))
                negs.append(int((f.weights < -1e-6).sum()))
            r = (tp, solver, float(np.mean(pre)), rmse(np.array(atts), EFFECT), float(np.mean(wsum)), float(np.mean(negs)))
            rows.append(r)
            print(f"  {tp:>13}{solver:>10}{r[2]:>12.3f}{r[3]:>11.3f}{r[4]:>12.3f}{r[5]:>9.1f}")
    by = {(r[0], r[1]): r for r in rows}
    print("")
    for tp in (12, 30, 60):
        s_, o_ = by[(tp, "simplex")], by[(tp, "ols")]
        tighter = "EXACTLY (interpolates)" if o_[2] < 1e-9 else f"{s_[2] / o_[2]:.2f}x tighter"
        print(f"  {tp:>3} pre-periods: OLS fits {tighter:>22} and estimates {o_[3] / s_[3]:>5.2f}x worse")
    print(
        "\n  With fewer pre-periods than donors the unconstrained fit passes through every"
        "\n  pre-period point exactly - a pre-period RMSPE of 0.000, the most convincing"
        "\n  version of the picture in section 3 - and is the worst estimator in this build."
        "\n  The gap closes as history lengthens, which is the honest version of the claim: the"
        "\n  constraint is a small-sample device, and it is also a device against being fooled"
        "\n  by your own diagnostic."
    )
    RESULTS["s6"] = {"rows": rows}


# ============================================================ 7
def section_7() -> None:
    head(7, "NEGATIVE: A DONOR THAT ALSO GOT TREATED ATTENUATES BY EXACTLY ITS OWN WEIGHT")
    print(
        """The donor pool is assumed untouched by the intervention.  When a neighbouring market
picks up some share gamma of the effect and carries weight w in the fit, that share is
subtracted from the estimate: the synthetic unit rises with the real one.  The prediction
is tau * (1 - sum_j w_j gamma_j), using the weights the optimiser actually chose - which
are not known in advance, so this is not a correction anyone can apply up front.

This is Day 168's finding at a different level: there the contamination was between arms of
one experiment, here it is between the treated unit and its own control group."""
    )
    n_sims = 400
    print(f"\n  {'gamma':>7}{'donors hit':>12}{'mean att':>11}{'measured ratio':>16}{'predicted':>12}{'gap':>9}")
    rows = []
    for gamma, n_hit in ((0.0, 1), (0.5, 1), (1.0, 1), (1.0, 3), (0.5, 5)):
        ratios, preds = [], []
        for s in range(n_sims):
            rng = np.random.default_rng(70000 + s)
            base = make_panel(rng, effect=EFFECT)
            f0 = fit_synth(base, 0, 30)
            hit = list(np.argsort(f0.weights)[::-1][:n_hit])
            rng2 = np.random.default_rng(70000 + s)
            Y = make_panel(rng2, effect=EFFECT, contaminated={int(i): gamma for i in hit})
            f = fit_synth(Y, 0, 30)
            ratios.append(f.att / EFFECT)
            preds.append(attenuation_prediction(f.weights, {int(i): gamma for i in hit}))
        mr, mp = float(np.mean(ratios)), float(np.mean(preds))
        rows.append((gamma, n_hit, mr * EFFECT, mr, mp, mr - mp))
        print(f"  {gamma:>7.1f}{n_hit:>12}{mr * EFFECT:>11.3f}{mr:>16.4f}{mp:>12.4f}{mr - mp:>9.4f}")
    clean = [r[5] for r in rows if r[0] == 0.0][0]
    spread = max(abs(r[5] - clean) for r in rows)
    print(
        f"\n  The gap column is {clean:.4f} in the uncontaminated row and stays there:"
        f" it never moves by more\n  than {spread:.5f} across contaminations that destroy"
        " two thirds of the effect.  That constant is\n  the estimator's own clean bias, not"
        " an error in the closed form - the attenuation is fully\n  explained by the weights"
        " the optimiser happened to choose."
        "\n  Nothing in the output flags this.  The pre-period fit is untouched - the"
        " contamination is\n  post-period by construction - so the credential in section 3"
        " is not merely uninformative\n  here, it is perfect while the answer is wrong."
    )
    RESULTS["s7"] = {"rows": rows}


# ============================================================ 8
def section_8() -> None:
    head(8, "ANTICIPATION IS THE ONE FAILURE THE PRE-PERIOD PLOT CAN SEE - WHEN IT IS LOUD")
    print(
        """If the market reacted before the official date - an announcement, a leak, firms moving
early - then the last few pre-periods already contain part of the effect, and the optimiser
fits the donors to a partly-treated series.  The received advice is to check the pre-period
fit and to back the date off if it looks wrong.  Below, the screen is given the best possible
form - reject when pre-RMSPE exceeds the 90th percentile of its own clean distribution, a
threshold no real analyst has - and its POWER is measured against the bias it must catch."""
    )
    n_sims = 500
    clean = []
    for s in range(n_sims):
        rng = np.random.default_rng(80000 + s)
        clean.append(fit_synth(make_panel(rng, effect=EFFECT), 0, 30).pre_rmspe)
    thresh = float(np.quantile(clean, 0.90))
    print(f"\n  Clean pre-RMSPE 90th percentile (the screen's threshold): {thresh:.4f}\n")
    print(f"  {'periods':>9}{'share':>8}{'mean att':>11}{'bias':>9}{'% of effect':>13}{'screen power':>14}")
    rows = []
    for a, share in ((0, 1.0), (2, 1.0), (4, 1.0), (6, 1.0), (10, 1.0), (4, 0.5), (4, 0.25), (10, 0.25)):
        atts, flags = [], []
        for s in range(n_sims):
            rng = np.random.default_rng(80000 + s)
            Y = make_panel(rng, effect=EFFECT, anticipation=a, anticipation_share=share)
            f = fit_synth(Y, 0, 30)
            atts.append(f.att)
            flags.append(f.pre_rmspe > thresh)
        m = float(np.mean(atts))
        rows.append((a, share, m, m - EFFECT, (m - EFFECT) / EFFECT * 100, float(np.mean(flags))))
        print(
            f"  {a:>9}{share:>8.2f}{m:>11.3f}{m - EFFECT:>9.3f}"
            f"{(m - EFFECT) / EFFECT * 100:>13.1f}{np.mean(flags):>14.3f}"
        )
    RESULTS["s8"] = {"threshold": thresh, "rows": rows}
    full = {r[0]: r for r in rows if r[1] == 1.0}
    quiet = [r for r in rows if r[1] == 0.25]
    print(
        f"""
  This is the one failure in the build the standard screen actually catches, and the reason
  is structural: anticipation is the only contamination that lands in the pre-period, which
  is the only place the diagnostic can see.  Two periods of it costs {full[2][4]:+.1f}% of the
  effect and gets flagged {full[2][5]:.3f} of the time.

  It is caught because it is LOUD, not because the screen is good.  At a quarter of the
  effect leaking early, {quiet[0][0]} periods costs {quiet[0][4]:+.1f}% and is flagged {quiet[0][5]:.3f} -
  barely above the screen's own {0.10:.2f} false-alarm rate - and {quiet[1][0]} periods costs
  {quiet[1][4]:+.1f}% at {quiet[1][5]:.3f}.  Same shape as Day 167's pre-trends test and Day 168's
  dose-response guard: the diagnostic's power arrives after the damage, it just arrives
  sooner here than it did there.

  Note the sign.  Every bias in this build so far ran positive; this one runs NEGATIVE -
  fitting donors to an already-treated pre-period drags the counterfactual up, and the
  estimate down.  A researcher who suspects anticipation and cannot rule it out does not
  know which direction they are wrong in without knowing which mechanism they have."""
    )


# ============================================================ 9
def section_9() -> None:
    head(9, "WHICH LEVER: ONE MORE PRE-PERIOD IS WORTH MORE THAN ONE MORE DONOR, UP TO A POINT")
    print(
        """Both are things a researcher can go and get: dig up more history, or widen the pool of
comparison markets.  They are not interchangeable, and the exchange rate is measurable."""
    )
    n_sims = 500
    donors = (5, 10, 20, 40, 80)
    pres = (8, 15, 30, 60, 120)
    print("\n  RMSE of the estimate (true effect 5.0):\n")
    print("  " + "pre \\ donors".ljust(14) + "".join(f"{j:>10}" for j in donors))
    grid = {}
    for tp in pres:
        line = f"  {tp:<14}"
        for j in donors:
            atts = []
            for s in range(n_sims):
                rng = np.random.default_rng(90000 + s)
                Y = make_panel(rng, n_donors=j, t_pre=tp, effect=EFFECT)
                atts.append(fit_synth(Y, 0, tp).att)
            r = rmse(np.array(atts), EFFECT)
            grid[(tp, j)] = r
            line += f"{r:>10.3f}"
        print(line)
    RESULTS["s9"] = {f"{tp}x{j}": v for (tp, j), v in grid.items()}
    print(
        f"\n  Reading the corners: 8 pre-periods and 80 donors gives {grid[(8, 80)]:.3f};"
        f" 120 pre-periods and 5\n  donors gives {grid[(120, 5)]:.3f}."
        f"  Along the top row, growing the pool 16x moves the error"
        f"\n  {grid[(8, 5)]:.3f} -> {grid[(8, 80)]:.3f}"
        f" ({(1 - grid[(8, 80)] / grid[(8, 5)]) * 100:+.0f}%); down the left column, growing the"
        f" history 15x moves it\n  {grid[(8, 5)]:.3f} -> {grid[(120, 5)]:.3f}"
        f" ({(1 - grid[(120, 5)] / grid[(8, 5)]) * 100:+.0f}%).  The pool has a second job though -"
        "\n  section 2's p-value floor is 1/(J+1), and no amount of history buys a single"
        " rank of it."
    )


def main() -> None:
    t0 = time.time()
    print("SYNTHETIC CONTROL: one market, one intervention, a control group that has to be built")
    print("Day 169 - data-science-cookbook - phoebe-the-builder")
    for fn in (section_1, section_2, section_3, section_4, section_5, section_6, section_7, section_8, section_9):
        fn()
    print(f"\n{'=' * 78}\n  total runtime {time.time() - t0:.1f}s\n{'=' * 78}")
    with open("results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
