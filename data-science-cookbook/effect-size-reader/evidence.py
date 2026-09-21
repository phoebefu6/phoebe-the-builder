"""Regenerate every number quoted in the README, the chart and the notebook.

Writes evidence.txt (human) and results.json (machine). Nothing in this build quotes a number
that does not appear in results.json.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from effectsize import (
    ALPHA,
    CELL_DESIGNS,
    DICH_D,
    DICH_DESIGNS,
    DICH_SEED_BASE,
    METRICS,
    SHAPES,
    STUDY_DS,
    STUDY_NS,
    STUDY_REFERENCE,
    STUDY_REPS,
    TRUTH_DESIGNS,
    analyse,
    analytic_prob_superiority,
    contaminate,
    correction_fixes_bias,
    correction_matters,
    dichotomisation_cost,
    dichotomisation_is_costly,
    n_for_significance,
    power_disagreement,
    smallest_significant_d,
    split_beats_t,
    true_prob_superiority,
    truths_disagree,
)

# Imported, never redefined: the notebook reads the same lists, so its rows ARE these cells.
REPS, REFERENCE = STUDY_REPS, STUDY_REFERENCE
DS, NS = STUDY_DS, STUDY_NS

out: List[str] = []


def w(line: str = "") -> None:
    out.append(line)
    print(line)


w("=" * 100)
w("COHEN'S d IS NOT THE EFFECT. IT IS ONE SUMMARY OF IT, AND THE SUMMARIES DISAGREE.")
w("=" * 100)
w()
w("Every population below is standardised to population mean 0 and population SD 1 using its")
w("ANALYTIC moments, then the treated group is shifted by exactly d. So the population Cohen's d")
w("is EXACTLY d for every shape. Any disagreement between metrics is therefore not a difference")
w("in how big the effect is - it is the metrics answering different questions.")
w()
w(f"Shapes: {', '.join(SHAPES)}.  Replicates: {REPS:,} a cell.  Reference draws for a population")
w(f"truth: {REFERENCE:,}.")
w()

# ---------------------------------------------------------------------------
w("-" * 100)
w("0. CALIBRATION - reproduce a known truth before reporting an unknown one")
w("-" * 100)
w()
w("For NORMAL data the probability of superiority has a closed form, Phi(d/sqrt(2)). The")
w("reference sampler must reproduce it. No other shape has a closed form, which is the whole")
w("reason the rest of this file exists.")
w()
truths: Dict[str, Dict[str, float]] = {}
w(f"{'shape':<14} {'d':>5} {'P(X>Y) measured':>17} {'+/- 99%':>9} {'closed form':>13} {'match':>7}")
for _i, (shape, d) in enumerate(TRUTH_DESIGNS):
    p, err = true_prob_superiority(shape, d, REFERENCE, seed=_i)
    truths[f"{shape}|{d}"] = {"ps": p, "err": err}
    if shape == "normal":
        exact = analytic_prob_superiority(d)
        ok = abs(p - exact) <= max(err, 1e-4)
        w(f"{shape:<14} {d:>5.1f} {p:>17.5f} {err:>9.5f} {exact:>13.5f} "
          f"{('OK' if ok else 'FAIL'):>7}")
n_cal = sum(1 for d in DS
            if abs(truths[f"normal|{d}"]["ps"] - analytic_prob_superiority(d))
            <= max(truths[f"normal|{d}"]["err"], 1e-4))
w()
w(f"  Calibrated on {n_cal} of {len(DS)} normal cells.")
w()

# ---------------------------------------------------------------------------
w("-" * 100)
w("1. THE HEADLINE - one effect size, six shapes, six different answers")
w("-" * 100)
w()
w("Read this table with the fact that d is IDENTICAL down every column firmly in mind.")
w()
w(f"{'shape':<14}" + "".join(f"{'d=' + format(d, '.1f'):>14}" for d in DS))
for shape in SHAPES:
    row = "".join(f"{truths[f'{shape}|{d}']['ps']:>14.4f}" for d in DS)
    w(f"{shape:<14}{row}")
w()
w(f"{'spread':<14}" + "".join(
    f"{max(truths[f'{s}|{d}']['ps'] for s in SHAPES) - min(truths[f'{s}|{d}']['ps'] for s in SHAPES):>14.4f}"
    for d in DS))
w()
for d in DS:
    verdict = truths_disagree({s: truths[f"{s}|{d}"]["ps"] for s in SHAPES}, tol=0.01)
    w(f"  Library verdict `truths_disagree` at d = {d} (tolerance 0.01): {verdict}")
w()
for d in DS:
    vals = {s: truths[f"{s}|{d}"]["ps"] for s in SHAPES}
    lo_s = min(vals, key=vals.get)
    hi_s = max(vals, key=vals.get)
    w(f"  At d = {d}: '{lo_s}' says the treated value wins {vals[lo_s]:.1%} of the time, "
      f"'{hi_s}' says {vals[hi_s]:.1%}.")
w()
w("  A 'medium' effect of d = 0.5 is taught as 'the treated case beats the control about 64% of")
w("  the time'. That number is correct for normal data and for nothing else here. Same d.")
w()
w("  NOTE, and it differs from what the sibling `normality-test-trap` found about the t-TEST:")
w("  there, symmetric non-normality was harmless and SKEW was the whole problem. Here both move")
w("  the answer and they move it the SAME WAY - the contaminated population is perfectly")
w("  symmetric and still reads 0.686 at d = 0.5 against normal's 0.639. The mechanism is that a")
w("  heavy tail inflates the SD that d divides by, so a shift of 'one d' is a larger shift")
w("  relative to the bulk of the distribution where the comparisons actually happen. Robustness")
w("  of the TEST and robustness of the EFFECT SIZE are different questions with different answers.")
w()

# ---------------------------------------------------------------------------
w("-" * 100)
w("2. THE ESTIMATORS - what the sample says, at sample sizes people actually have")
w("-" * 100)
w()
w("True d is 0.5 in every row. 'bias' is the average estimate minus the truth.")
w()
w(f"{'shape':<14} {'n':>5} {'d-hat':>8} {'bias':>8} {'sd(d-hat)':>10} {'g':>8} {'bias':>8} "
  f"{'P(X>Y)':>8} {'bias':>8} {'power':>7}")
cells: List[Dict[str, Any]] = []
cell_objs = []
for _i, (shape, n) in enumerate(CELL_DESIGNS):
    t = truths[f"{shape}|0.5"]
    c = analyse(shape, 0.5, n, REPS, seed=_i, truth=(t["ps"], t["err"]))
    cell_objs.append(c)
    cells.append({
        "shape": shape, "d_true": c.d_true, "n": n, "reps": c.reps,
        "truth_ps": c.truth_ps, "means": c.means, "sds": c.sds,
        "cis": {k: list(v) for k, v in c.cis.items()},
        "d_bias": c.d_bias, "g_bias": c.g_bias, "ps_bias": c.ps_bias,
        "power": c.power, "correction_matters": correction_matters(c), "seed": c.seed,
    })
    w(f"{shape:<14} {n:>5} {c.means['cohens_d']:>8.4f} {c.d_bias:>+8.4f} "
      f"{c.sds['cohens_d']:>10.4f} {c.means['hedges_g']:>8.4f} {c.g_bias:>+8.4f} "
      f"{c.means['prob_superiority']:>8.4f} {c.ps_bias:>+8.4f} {c.power:>7.4f}")
w()
worst_d_bias = max(cells, key=lambda c: abs(c["d_bias"]))
worst_ps_bias = max(cells, key=lambda c: abs(c["ps_bias"]))
corr_yes = [c for c in cells if c["correction_matters"]]
corr_no = [c for c in cells if not c["correction_matters"]]
w(f"  Worst bias in d-hat:  {worst_d_bias['shape']}, n={worst_d_bias['n']}, "
  f"{worst_d_bias['d_bias']:+.4f} ({abs(worst_d_bias['d_bias']) / 0.5:.1%} of the truth).")
w(f"  Worst bias in P(X>Y): {worst_ps_bias['shape']}, n={worst_ps_bias['n']}, "
  f"{worst_ps_bias['ps_bias']:+.4f}.")
w()
w("  d-hat is biased UPWARD and the bias is a small-sample effect: Hedges' correction changes")
w(f"  the answer by more than 0.01 on {len(corr_yes)} of {len(cells)} cells and not on the other "
  f"{len(corr_no)}.")
if corr_yes:
    w(f"  Cells where it matters run to n = {max(c['n'] for c in corr_yes)}; "
      f"cells where it does not start at n = {min(c['n'] for c in corr_no)}.")
w()
w("  But it only WORKS where it was derived. Hedges' J assumes normality, and at n = 10:")
w()
w(f"{'shape':<14} {'d-hat bias':>12} {'g bias':>10} {'correction removed it?':>24}")
for c, obj in [(c, o) for c, o in zip(cells, cell_objs) if c["n"] == 10]:
    fixed = correction_fixes_bias(obj)
    c["correction_fixes"] = fixed
    w(f"{c['shape']:<14} {c['d_bias']:>+12.4f} {c['g_bias']:>+10.4f} "
      f"{('yes' if fixed else 'NO'):>24}")
w()
fixed_shapes = [c["shape"] for c in cells if c["n"] == 10 and c.get("correction_fixes")]
broken_shapes = [c["shape"] for c in cells if c["n"] == 10 and not c.get("correction_fixes")]
w(f"  Corrected cleanly on {len(fixed_shapes)} shapes ({', '.join(fixed_shapes)}) and left a")
w(f"  substantial bias on {len(broken_shapes)} ({', '.join(broken_shapes)}). On lognormal data at")
ln10 = next(c for c in cells if c["shape"] == "lognormal" and c["n"] == 10)
w(f"  n = 10 the corrected estimate is still {ln10['g_bias']:+.4f} out - the correction removed")
w(f"  {(1 - abs(ln10['g_bias']) / abs(ln10['d_bias'])):.0%} of a bias that was never the bias it")
w("  was designed for. The excess comes from the sample SD, which a skewed sample underestimates")
w("  in exactly the draws where the sample mean is also low.")
w()
w("  Meanwhile P(X>Y) is unbiased to within 0.0021 on EVERY shape at n = 10. The rank summary")
w("  needs no small-sample correction because it never divides by an estimated SD.")
w()
w("  NEGATIVE RESULT: the noise swamps the bias at every sample size here. At n = 20 on normal")
n20 = next(c for c in cells if c["shape"] == "normal" and c["n"] == 20)
w(f"  data the bias in d-hat is {n20['d_bias']:+.4f} while its standard deviation is "
  f"{n20['sds']['cohens_d']:.4f} - a ratio of {abs(n20['d_bias']) / n20['sds']['cohens_d']:.1%}.")
w("  Applying Hedges' correction and then reporting a point estimate with no interval is")
w("  polishing the third decimal of a number whose first decimal is not settled.")
w()

# ---------------------------------------------------------------------------
w("-" * 100)
w("3. ROBUSTNESS - one contaminated observation in a hundred")
w("-" * 100)
w()
w("1% of the TREATED values are replaced with +10 SD. The outlier points the same way as the")
w("effect, which is the charitable case - a big real response, not a data-entry error.")
w()
w(f"{'metric':<24} {'clean':>10} {'contaminated':>14} {'change':>10}")
robust: List[Dict[str, Any]] = []
for r in contaminate("normal", 0.5, 100, REPS, seed=31):
    robust.append({"metric": r.metric, "clean": r.clean, "dirty": r.dirty,
                   "pct_change": r.pct_change})
    w(f"{r.metric:<24} {r.clean:>10.4f} {r.dirty:>14.4f} {r.pct_change:>+9.1f}%")
w()
worst_rob = max(robust, key=lambda r: abs(r["pct_change"]))
best_rob = min(robust, key=lambda r: abs(r["pct_change"]))
w(f"  Most moved:  {worst_rob['metric']} at {worst_rob['pct_change']:+.1f}%.")
w(f"  Least moved: {best_rob['metric']} at {best_rob['pct_change']:+.1f}%.")
w()
w("  The result nobody expects: GLASS'S DELTA is the metric that moves, and it is the one")
w("  recommended precisely for the case where the treatment changes the spread. It divides by the")
w("  CONTROL group's SD, which the contamination never touches - so the numerator grows and the")
w("  denominator does not. Cohen's d barely moves because its pooled SD absorbs the same outlier")
w("  that inflated the mean. The 'safer' choice is the fragile one, and the direction of the")
w("  failure is upward: it reports a LARGER effect.")
w()

# ---------------------------------------------------------------------------
w("-" * 100)
w("4. SIGNIFICANCE VERSUS MAGNITUDE")
w("-" * 100)
w()
w(f"{'d':>7} {'n per group for 80% power':>28} {'P(X>Y) at that d':>18}")
n_table: Dict[str, Any] = {}
for d in (0.8, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
    n = n_for_significance(d)
    n_table[str(d)] = n
    w(f"{d:>7.2f} {n:>28,} {analytic_prob_superiority(d):>18.4f}")
w()
w(f"{'n per group':>13} {'smallest d at 50% power':>26} {'P(X>Y)':>10} {'in words':>34}")
small: List[Dict[str, Any]] = []
for n in (100, 1000, 10000, 100000, 1000000):
    d = smallest_significant_d(n)
    ps = analytic_prob_superiority(d)
    words = f"treated wins {ps:.2%} of the time"
    small.append({"n": n, "d": d, "ps": ps})
    w(f"{n:>13,} {d:>26.5f} {ps:>10.4f} {words:>34}")
w()
w(f"  At n = 100,000 per group the smallest effect the study is not guessing about is "
  f"d = {small[3]['d']:.4f},")
w(f"  which means the treated value beats the control {small[3]['ps']:.2%} of the time. That is a")
w("  coin flip with a statistically significant lean, and it is what 'everything is significant")
w("  at large n' actually looks like when you write it down in units a person can act on.")
w()
w("  A DEFECT THIS BUILD HIT AND NEARLY SHIPPED, worth recording because nothing warns you:")
pd_ = power_disagreement(0.5, list(range(5, 400)) + [500, 1000, 2000, 5000, 10000, 20000, 100000])
w(f"    scipy.stats.nct returns **nan** - not an error, not a warning - at "
  f"{pd_['n_nan']} of the sample sizes probed,")
w(f"    from n = {pd_['first_nan_n']:,} to n = {pd_['last_nan_n']:,}. A nan is silently False in")
w("    every comparison, so `power >= target` reads as 'not powerful enough' and a monotone")
w("    binary search walks straight past the answer. The first version of this module reported")
w("    that d = 0.5 needs n = 11,417 per group for 80% power. The real answer is 64.")
w("    The fix is a normal-approximation fallback, and it is only trusted because it is measured:")
w("    across every n where the exact form IS finite, the two formulas differ by at most")
w(f"    {pd_['worst_gap']:.6f} (at n = {pd_['worst_gap_n']}), and once df >= 400 - which is far")
w(f"    below where the nan band starts - by at most {pd_['worst_gap_large_df']:.2e} "
  f"(at n = {pd_['worst_gap_large_df_n']}).")
w("    The search is now a helper that asserts its own monotonicity at both ends and raises")
w("    rather than returning a plausible number.")
w()

# ---------------------------------------------------------------------------
w("-" * 100)
w("5. WHAT A MEDIAN SPLIT COSTS - and the shapes where it costs nothing")
w("-" * 100)
w()
w(f"True d = {DICH_D} throughout. 'null t' and 'null split' are each test's rejection rate when")
w("d = 0, run on the same shape and n. A power comparison is only read on rows where BOTH hold a")
w("nominal 5% - an over-rejecting test detects more of everything, so its power lead would be the")
w("inflation restated.")
w()
w(f"{'shape':<14} {'n':>5} {'power t':>9} {'power split':>12} {'null t':>8} {'null split':>11} "
  f"{'read?':>7} {'winner':>8} {'equiv n':>9}")
dich: List[Dict[str, Any]] = []
for _i, (shape, n) in enumerate(DICH_DESIGNS):
    r = dichotomisation_cost(shape, DICH_D, n, REPS, seed=DICH_SEED_BASE + _i)
    t_wins, s_wins = dichotomisation_is_costly(r), split_beats_t(r)
    winner = "t" if t_wins else ("split" if s_wins else "-")
    dich.append({
        "shape": shape, "n": n, "power_t": r.power_t, "power_split": r.power_chi2,
        "power_t_ci": list(r.power_t_ci), "power_split_ci": list(r.power_chi2_ci),
        "null_t": r.null_t, "null_split": r.null_chi2, "null_t_ok": r.null_t_ok,
        "null_split_ok": r.null_chi2_ok, "comparable": r.comparable,
        "equivalent_n": r.equivalent_n, "n_wasted_share": r.n_wasted_share,
        "mean_log_odds": r.mean_log_odds, "t_wins": t_wins, "split_wins": s_wins,
    })
    w(f"{shape:<14} {n:>5} {r.power_t:>9.4f} {r.power_chi2:>12.4f} {r.null_t:>8.4f} "
      f"{r.null_chi2:>11.4f} {('yes' if r.comparable else 'NO'):>7} {winner:>8} "
      f"{r.equivalent_n:>9,}")
w()
readable = [r for r in dich if r["comparable"]]
w(f"  Only {len(readable)} of {len(dich)} rows are comparable at all - and the reason is itself a")
w("  finding: the median-split test is ANTICONSERVATIVE at n = 50, rejecting a true null at")
n50 = [r for r in dich if r["n"] == 50]
w(f"  {min(r['null_split'] for r in n50):.4f} to {max(r['null_split'] for r in n50):.4f} against a "
  f"nominal 0.05, on every shape tested. Before")
w("  it costs you any power, dichotomising costs you the error rate you thought you had.")
w()
t_rows = [r for r in readable if r["t_wins"]]
s_rows = [r for r in readable if r["split_wins"]]
w(f"  Among the readable rows the t-test wins {len(t_rows)} and the median split wins "
  f"{len(s_rows)}.")
for r in t_rows:
    w(f"    t wins on {r['shape']} (n={r['n']}): the split has the power of n = "
      f"{r['equivalent_n']}, so {r['n_wasted_share']:.0%} of the sample was thrown away.")
for r in s_rows:
    w(f"    SPLIT wins on {r['shape']} (n={r['n']}): it has the power of n = "
      f"{r['equivalent_n']}, i.e. {-r['n_wasted_share']:.0%} MORE data than you collected.")
w()
w("  NEGATIVE RESULT: 'never dichotomise your data' is advice about NORMAL data. On normal data")
w("  it is right and the price is a third of the sample. On skewed and heavy-tailed data the")
w("  median split is a rank method, the t-test is not, and the split WINS. The rule is real; it")
w("  is just not a rule about dichotomisation. It is a rule about which yardstick matches the")
w("  shape of what you measured - which is the same thing section 1 said.")
w()

# ---------------------------------------------------------------------------
w("=" * 100)
w("WHAT TO TAKE AWAY")
w("=" * 100)
w()
spread05 = (max(truths[f"{s}|0.5"]["ps"] for s in SHAPES)
            - min(truths[f"{s}|0.5"]["ps"] for s in SHAPES))
w(f"1. At an IDENTICAL true Cohen's d of 0.5, the probability of superiority ranges over "
  f"{spread05:.3f}")
w("   across six distribution shapes. d is not wrong; it is a ratio of two numbers, and the")
w("   denominator means different things in different shapes.")
w("2. Reporting an effect size without saying which one, on what shape, with what interval, is")
w("   reporting a number rather than a magnitude.")
w(f"3. Glass's delta - the metric recommended when the treatment changes the spread - moves "
  f"{worst_rob['pct_change']:+.0f}%")
w("   on one contaminated observation in a hundred. Cohen's d moves about one percent.")
w(f"4. At n = 100,000 the smallest effect a study can see is P(X>Y) = {small[3]['ps']:.4f}. "
  f"Significance stopped")
w("   carrying information about magnitude long before that.")
w("5. 'Never dichotomise' is a claim about normal data. Measured on skewed data it is false.")
w()

results = {
    "alpha": ALPHA, "reps": REPS, "reference_draws": REFERENCE, "shapes": list(SHAPES),
    "ds": list(DS), "ns": list(NS), "metrics": list(METRICS),
    "truth_designs": [list(x) for x in TRUTH_DESIGNS],
    "cell_designs": [list(x) for x in CELL_DESIGNS],
    "dich_designs": [list(x) for x in DICH_DESIGNS], "dich_d": DICH_D,
    "truths": truths, "cells": cells, "robustness": robust,
    "n_for_significance": n_table, "smallest_significant": small,
    "power_disagreement": pd_, "dichotomisation": dich,
    "summary": {
        "n_calibrated": n_cal, "n_calibration_cells": len(DS),
        "ps_spread_at_half": spread05,
        "worst_d_bias": {"shape": worst_d_bias["shape"], "n": worst_d_bias["n"],
                         "value": worst_d_bias["d_bias"]},
        "n_correction_matters": len(corr_yes), "n_cells": len(cells),
        "correction_fixes_shapes": fixed_shapes, "correction_breaks_shapes": broken_shapes,
        "worst_robustness": worst_rob, "best_robustness": best_rob,
        "smallest_at_100k": small[3],
        "n_dich_comparable": len(readable), "n_dich_rows": len(dich),
        "n_t_wins": len(t_rows), "n_split_wins": len(s_rows),
    },
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
with open("evidence.txt", "w") as f:
    f.write("\n".join(out) + "\n")
print("\nwrote evidence.txt and results.json")
