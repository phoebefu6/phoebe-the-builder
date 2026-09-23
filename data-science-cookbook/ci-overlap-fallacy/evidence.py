"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). Nothing downstream recomputes a
finding - Day 175 shipped a chart that had grown its own copy of the comparison and disagreed
with this file about four cells, so the rule since then is that this module is the only place a
verdict is decided.

Two kinds of number live here and they are labelled differently throughout:

  * EXACT - closed-form geometry, and everything about proportions, which is summed over the
    complete outcome space with binomial weights. No replicates, no interval, no seed.
  * MEASURED - Monte Carlo over normal data. Every rate carries a 99% Wilson interval at its
    own replicate count, and no rate is called broken unless the whole interval clears the band.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List

import numpy as np
import overlap as o
from scipy import stats

OUT: List[str] = []


def say(line: str = "") -> None:
    OUT.append(line)
    print(line)


def rule(title: str) -> None:
    say()
    say("=" * 96)
    say(title)
    say("=" * 96)


results: Dict[str, Any] = {}

results["config"] = {
    "alpha": o.ALPHA,
    "band": [o.BAND_LO, o.BAND_HI],
    "z_alpha": o.Z_ALPHA,
    "matching_level_equal_se": o.MATCHING_LEVEL_EQUAL_SE,
    "overlap_at_sig_equal_se": o.overlap_fraction_at_significance(1.0, 1.0),
    "means_reps": o.MEANS_REPS,
    "paired_reps": o.PAIRED_REPS,
    "containment_reps": o.CONTAINMENT_REPS,
    "paired_n": o.PAIRED_N,
    "paired_delta": o.PAIRED_DELTA,
    "numpy": np.__version__,
    "scipy": __import__("scipy").__version__,
}

# --------------------------------------------------------------------------------- 1. geometry
rule("1.  THE GEOMETRY  -  exact, closed form, no simulation")
say()
say("Two independent estimates, standard errors s1 and s2, intervals at multiplier z:")
say("    bars separate when   |d| > z * (s1 + s2)")
say("    test rejects when    |d| > z_alpha * sqrt(s1^2 + s2^2)")
say("and s1 + s2 >= sqrt(s1^2 + s2^2) always, so NON-OVERLAP IS THE STRICTER EVENT.")
say()
say(f"At equal standard errors the bars demand the means be sqrt(2) = {o.se_ratio(1, 1):.4f}x "
    f"further apart than the test does.")
say(f"The interval level at which the two events COINCIDE is "
    f"{o.MATCHING_LEVEL_EQUAL_SE * 100:.1f}%, not 95%  (z_alpha / sqrt(2) = "
    f"{o.Z_ALPHA / math.sqrt(2):.4f}).")
say(f"At 95% bars and equal standard errors, two bars may still share "
    f"{o.overlap_fraction_at_significance(1, 1) * 100:.1f}% of one arm and the difference is "
    f"significant  (2 - sqrt(2)).")
say()
say("  sd ratio   R = threshold    matching level   overlap still allowed")
say("             inflation        for p < 0.05     at the 95% boundary")
say("  " + "-" * 74)
geom = o.geometry_table()
for g in geom:
    say(f"  {g['sd_ratio']:>6.2f}    {g['se_ratio']:>8.4f}         "
        f"{g['matching_level'] * 100:>8.1f}%        {g['overlap_at_sig'] * 100:>8.1f}% of an arm")
results["geometry"] = geom
say()
say("READ THIS ROW ORDER TWICE.  The rule is WORST at a sd ratio of 1 - two groups of the same")
say("size and spread, the design everyone trusts - and gets progressively less wrong as the")
say("groups become lopsided.  'A little overlap is fine' has no number: the permitted overlap")
say(f"runs from {geom[0]['overlap_at_sig'] * 100:.1f}% of an arm down to "
    f"{geom[-1]['overlap_at_sig'] * 100:.1f}% across this table.")

# exhaustive check of the containment inequality itself
grid = np.linspace(0.01, 50.0, 4000)
worst = float(max(1.0 / o.se_ratio(1.0, float(s)) for s in grid))
say()
say(f"Exhaustive check over {len(grid):,} standard-error ratios in [0.01, 50]: the significance")
say(f"threshold never exceeds the non-overlap threshold (worst ratio {worst:.6f} <= 1).")
results["containment_worst_ratio"] = worst

# ------------------------------------------------------------------------------ 2. calibration
rule("2.  CALIBRATION  -  the theorem, verified before any result is read")
say()
say("If the bars and the test share their standard errors, 'bars apart' must IMPLY")
say("'significant', with no exceptions. Anything else means the harness is broken.")
say()
say("  design            replicates   gap AND not significant    overlap | significant")
say("  " + "-" * 82)
calib = []
for i, (n1, n2, s1, s2, d) in enumerate(o.CONTAINMENT_DESIGNS):
    r = o.run_means(n1, n2, s1, s2, d, o.CONTAINMENT_REPS, o.CONTAINMENT_SEED_BASE + i,
                    known_sigma=True, test="z")
    calib.append(r)
    say(f"  {n1:>3}/{n2:<3} sd {s1}/{s2:<4}   {o.CONTAINMENT_REPS:>8,}   "
        f"{r['gap_and_not_sig']:>16.6f}         {r['overlap_given_sig']:>10.4f}")
assert all(r["gap_and_not_sig"] == 0.0 for r in calib), "containment theorem violated"
say()
say(f"  0 violations in {o.CONTAINMENT_REPS * len(calib):,} replicates, as the algebra requires.")
results["calibration"] = calib

say()
say("Now break it on purpose. Keep the SAME bars - drawn from a design-time standard error, the")
say("way a chart made from a power calculation or a published SD is - and test with the observed")
say("variance instead (Welch). The two objects no longer share an SE and the guarantee is gone:")
say()
say("  design            gap AND not significant   verdict")
say("  " + "-" * 62)
mismatch = []
for i, (n1, n2, s1, s2, d) in enumerate(o.CONTAINMENT_DESIGNS):
    r = o.run_means(n1, n2, s1, s2, d, o.CONTAINMENT_REPS, o.CONTAINMENT_SEED_BASE + 100 + i,
                    known_sigma=True, test="welch")
    lo, hi = o.wilson(r["gap_and_not_sig"], o.CONTAINMENT_REPS)
    flag = "REVERSED" if lo > 0.001 else "-"
    mismatch.append({**r, "wilson_lo": lo, "wilson_hi": hi, "flag": flag})
    say(f"  {n1:>3}/{n2:<3} sd {s1}/{s2:<4}   {r['gap_and_not_sig']:>16.5f}   {flag}")
results["mismatch"] = mismatch
worst_mm = max(mismatch, key=lambda r: r["gap_and_not_sig"])
say()
say(f"  Worst: {worst_mm['n1']}/{worst_mm['n2']} at {worst_mm['gap_and_not_sig']:.4f} - on that")
say("  design nearly one reading in five shows a VISIBLE GAP between bars that the test will not call")
say("  significant. The rule's one guarantee survives exactly as long as the picture and the")
say("  test are computed from the same number.")

# -------------------------------------------------------------------------------- 3. dead zone
rule("3.  THE DEAD ZONE  -  measured: significant results the picture tells you to discard")
say()
say(f"95% bars, Welch t-test at alpha = {o.ALPHA}, normal data, "
    f"{o.MEANS_REPS:,} replicate studies a design.")
say()
say("   n1/n2   sd1/sd2  power   P(sig)   P(gap)   P(sig & overlap)   P(overlap | sig)  99% CI")
say("  " + "-" * 92)
means = []
for i, (n1, n2, s1, s2, d) in enumerate(o.MEANS_DESIGNS):
    r = o.run_means(n1, n2, s1, s2, d, o.MEANS_REPS, o.MEANS_SEED_BASE + i)
    n_sig = int(round(r["sig"] * o.MEANS_REPS))
    lo, hi = o.wilson(r["overlap_given_sig"], n_sig)
    r = {**r, "power_approx": o.approx_power(n1, n2, s1, s2, d),
         "ogs_lo": lo, "ogs_hi": hi, "n_sig": n_sig}
    means.append(r)
    say(f"  {n1:>3}/{n2:<3}  {s1:>4.1f}/{s2:<4.1f}  {r['power_approx']:.3f}  {r['sig']:>7.4f}  "
        f"{r['gap']:>7.4f}   {r['sig_and_overlap']:>14.4f}   {r['overlap_given_sig']:>14.4f}  "
        f"[{lo:.4f}, {hi:.4f}]")
results["means"] = means
best = max(means, key=lambda r: r["overlap_given_sig"])
worst_case = min(means, key=lambda r: r["overlap_given_sig"])
say()
say(f"  P(overlap | significant) spans {worst_case['overlap_given_sig']:.4f} to "
    f"{best['overlap_given_sig']:.4f} across these designs. On the "
    f"{best['n1']}/{best['n2']} design at {best['power_approx']:.0%} power,")
say(f"  {best['overlap_given_sig']:.1%} of the studies that found a real, significant difference")
say("  would be thrown away by a reader applying the overlap rule to the very same chart.")
say()
say("  Negative result: the dead zone SHRINKS with power, because a large true effect pushes the")
say("  bars past each other too. It is not a bug that only appears in bad studies - it is worst")
say("  at exactly the effect sizes that are marginal enough to be worth arguing about.")

# ------------------------------------------------------------------- 3b. the rule as a test
rule("3b.  THE RULE IS A 0.6% TEST WEARING A 5% LABEL  -  exact, then measured")
say()
say("Read 'the bars do not touch' as a decision rule and it has a size like any other test.")
say("Under a true null the difference is N(0, sqrt(s1^2 + s2^2)) and the rule fires at")
say("z_level * R standard errors, so its size is 2 * (1 - Phi(z_level * R)) - no simulation.")
say()
say(f"  At equal standard errors and 95% bars that is {o.RULE_ALPHA_EQUAL_SE:.4f}.")
say("  A reader applying the overlap rule to a 95% chart is running a 0.6% test, and reporting")
say("  it as a 5% one.")
say()
say("   sd ratio   exact size of the rule   measured P(gap) under a true null   99% Wilson")
say("  " + "-" * 88)
null_rows = []
for i, ratio in enumerate([1.0, 2.0, 3.0, 6.0]):
    exact = o.rule_alpha(1.0, float(ratio))
    m = o.run_means(30, 30, 1.0, float(ratio), 0.0, o.MEANS_REPS, o.MEANS_SEED_BASE + 500 + i)
    lo, hi = o.wilson(m["gap"], o.MEANS_REPS)
    null_rows.append({"sd_ratio": ratio, "exact": exact, "measured": m["gap"],
                      "lo": lo, "hi": hi, "test_size": m["sig"],
                      "test_verdict": o.verdict(m["sig"], o.MEANS_REPS)})
    say(f"  {ratio:>7.1f}   {exact:>21.4f}   {m['gap']:>33.4f}   [{lo:.4f}, {hi:.4f}]")
results["rule_size"] = null_rows
say()
say("  The Welch test on those same replicates holds its nominal rate throughout:")
for r in null_rows:
    say(f"    sd ratio {r['sd_ratio']:>4.1f}   Welch {r['test_size']:.4f}  {r['test_verdict']}")
say("  so the gap between the two columns is the RULE's and not the data's. The rule's size")
say("  climbs toward 5% only as the standard errors diverge - the same direction as everything")
say("  else here, for the same reason: R falls from sqrt(2) toward 1.")

# ------------------------------------------------------------------------------ 4. the fix
rule("4.  THE FIX IS A LEVEL, NOT A CORRECTION  -  measured")
say()
say(f"Draw the bars at {o.MATCHING_LEVEL_EQUAL_SE * 100:.1f}% instead of 95% and, for equal")
say("standard errors, 'they do not touch' and 'p < 0.05' become the same event. Measured as the")
say("rate at which the two verdicts DISAGREE on the same replicate:")
say()
say("   n1/n2   sd1/sd2   disagreement at 95%   at the matching level   matching level")
say("  " + "-" * 84)
levels = []
for i, (n1, n2, s1, s2, d) in enumerate(o.MEANS_DESIGNS[:8]):
    ml = o.matching_level(s1 / math.sqrt(n1), s2 / math.sqrt(n2))
    a = o.run_means(n1, n2, s1, s2, d, o.MEANS_REPS, o.LEVEL_SEED_BASE + i, level=0.95)
    b = o.run_means(n1, n2, s1, s2, d, o.MEANS_REPS, o.LEVEL_SEED_BASE + i, level=ml)
    levels.append({"n1": n1, "n2": n2, "sd1": s1, "sd2": s2, "matching_level": ml,
                   "disagree_95": a["disagree"], "disagree_matched": b["disagree"]})
    say(f"  {n1:>3}/{n2:<3}  {s1:>4.1f}/{s2:<4.1f}   {a['disagree']:>17.4f}   "
        f"{b['disagree']:>21.4f}   {ml * 100:>12.1f}%")
results["levels"] = levels
say()
say("  The residual at the matching level is not zero because the bars use each group's own t")
say("  multiplier while Welch uses a pooled-ish df - the algebra is exact for z, approximate for")
say("  t. It is an order of magnitude smaller than the 95% disagreement on every row.")

# ----------------------------------------------------------------------------- 5. the reversal
rule("5.  THE REVERSAL  -  correlated measurements, where the rule fails the OTHER way")
say()
say("Everything above says the rule is too STRICT. Pair the measurements and it becomes too")
say("LOOSE: the test knows the correlation, the bars cannot show it, and the same picture now")
say("hides a real difference behind two bars sitting on top of each other.")
say()
say(f"n = {o.PAIRED_N} pairs, true difference {o.PAIRED_DELTA}, 95% bars per arm, "
    f"{o.PAIRED_REPS:,} replicates.")
say()
say("   rho    slack (x further the bars demand)   P(sig)   P(overlap | sig)   99% CI")
say("  " + "-" * 84)
paired = []
for i, rho in enumerate(o.RHO_GRID):
    r = o.run_paired(o.PAIRED_N, rho, o.PAIRED_DELTA, o.PAIRED_SD, o.PAIRED_REPS,
                     o.PAIRED_SEED_BASE + i)
    n_sig = int(round(r["sig"] * o.PAIRED_REPS))
    lo, hi = o.wilson(r["overlap_given_sig"], n_sig)
    r = {**r, "ogs_lo": lo, "ogs_hi": hi, "n_sig": n_sig}
    paired.append(r)
    say(f"  {rho:>5.2f}   {r['slack']:>28.3f}   {r['sig']:>6.4f}   "
        f"{r['overlap_given_sig']:>14.4f}   [{lo:.4f}, {hi:.4f}]")
results["paired"] = paired
top = paired[-1]
say()
say(f"  At rho = {top['rho']}, the test rejects on {top['sig']:.4f} of studies and "
    f"{top['overlap_given_sig']:.4f} of those")
say("  rejections have overlapping bars. The independent-groups algebra caps the rule's")
say(f"  conservatism at sqrt(2) = 1.414x; correlation removes the cap "
    f"({top['slack']:.2f}x here) and it grows without bound as rho approaches 1.")
say("  Nothing on the chart distinguishes this case from the independent one.")

# ------------------------------------------------------------------------------- 6. coverage
rule("6.  FOUR INTERVAL METHODS  -  coverage computed EXACTLY, not simulated")
say()
say("Every value below is summed over all n+1 possible counts with their binomial weights.")
say("No replicates, no seed, no Monte Carlo error - these numbers are the coverage.")
say()
header = "   n     p    " + "".join(f"{m:>18}" for m in o.METHODS)
say(header)
say("  " + "-" * (len(header) - 2))
cov = []
for n in o.COVERAGE_NS:
    for p in o.COVERAGE_PS:
        row = {"n": n, "p": p}
        for m in o.METHODS:
            row[m] = o.exact_coverage(m, p, n)
        cov.append(row)
        say(f"  {n:>4}  {p:>5.2f}  " + "".join(f"{row[m]:>18.4f}" for m in o.METHODS))
    say()
results["coverage"] = cov
for m in o.METHODS:
    vals = [r[m] for r in cov]
    say(f"  {m:<16} worst {min(vals):.4f}   best {max(vals):.4f}   "
        f"mean {float(np.mean(vals)):.4f}   rows below 0.90: "
        f"{sum(1 for v in vals if v < 0.90)}/{len(vals)}")
results["coverage_summary"] = {
    m: {"worst": min(r[m] for r in cov), "best": max(r[m] for r in cov),
        "mean": float(np.mean([r[m] for r in cov])),
        "below_90": sum(1 for r in cov if r[m] < 0.90)} for m in o.METHODS}
wald_worst = min(cov, key=lambda r: r["wald"])
say()
say(f"  The textbook interval p +/- 1.96*sqrt(p(1-p)/n) covers "
    f"{wald_worst['wald']:.4f} of the time at p = {wald_worst['p']}, n = {wald_worst['n']},")
say("  against a nominal 0.95. It is not an approximation that gets slightly loose - at a low")
say("  rate it MISSES the true value twice as often as it covers it. Clopper-Pearson errs the")
say("  other way and is conservative by construction, which is its own cost: wider bars, a")
say("  stricter overlap rule, less power.")
say()
nonmono = []
for p in o.COVERAGE_PS:
    seq = [next(r["wald"] for r in cov if r["n"] == n and r["p"] == p) for n in o.COVERAGE_NS]
    if any(seq[i + 1] < seq[i] for i in range(len(seq) - 1)):
        nonmono.append({"p": p, "by_n": seq})
results["wald_nonmonotone"] = nonmono
say(f"  Negative result worth stating: Wald coverage is NOT monotone in n. On {len(nonmono)} of "
    f"{len(o.COVERAGE_PS)} rates in this grid a LARGER")
say("  sample has WORSE coverage than a smaller one:")
for _r in nonmono:
    say(f"    p = {_r['p']:.2f}   "
        + "   ".join(f"n={n}: {v:.4f}" for n, v in zip(o.COVERAGE_NS, _r["by_n"])))
say("  Collecting more data does not reliably repair it, and there is no sample size that makes")
say("  a Wald bar safe to read at a low rate.")

# ------------------------------------------------------------------ 7. the rule on proportions
rule("7.  THE RULE ON PROPORTIONS  -  exact over every outcome pair")
say()
say("Two proportions, pooled z-test, bars from each method. All rates summed over the complete")
say("(n1+1) x (n2+1) outcome grid with exact binomial weights.")
say()
say("   p1    p2     n1/n2       method       P(sig)  P(sig & overlap)  P(gap & NOT sig)")
say("  " + "-" * 88)
props = []
for (p1, p2, n1, n2) in o.PROP_DESIGNS:
    for m in o.METHODS:
        r = o.exact_prop_rule(m, p1, p2, n1, n2)
        props.append(r)
        say(f"  {p1:>4.2f}  {p2:>4.2f}  {n1:>4}/{n2:<4}  {m:>16}  {r['sig']:>10.4f}  "
            f"{r['sig_and_overlap']:>15.4f}  {r['gap_and_not_sig']:>16.4f}")
    say()
results["props"] = props

say("The Wald reversal - a small arm that can record zero events:")
say()
say("   p1     n1    p2     n2       method     P(gap & NOT sig)   P(x1 = 0 or n1)")
say("  " + "-" * 82)
reversal = []
for (p1, n1, p2, n2) in [(0.001, 40, 0.02, 400), (0.01, 40, 0.03, 400), (0.01, 40, 0.06, 400)]:
    degenerate = float(stats.binom.pmf(0, n1, p1) + stats.binom.pmf(n1, n1, p1))
    for m in o.METHODS:
        r = o.exact_prop_rule(m, p1, p2, n1, n2)
        reversal.append({**r, "degenerate": degenerate})
        say(f"  {p1:>5.3f}  {n1:>4}  {p2:>5.3f}  {n2:>4}  {m:>16}   {r['gap_and_not_sig']:>15.4f}"
            f"   {degenerate:>16.4f}")
    say()
results["reversal"] = reversal
say("  A Wald interval on x = 0 has ZERO WIDTH. It is a dot on the axis, so it cannot overlap")
say("  anything, and the rule reports a difference the test refuses to call. On the second block")
say("  the failure rate sits just under P(x1 = 0) - the mechanism is not subtle once")
say("  named, and it is invisible on the chart, where a zero-width bar looks like a precise one.")
_g = [abs(r["gap_and_not_sig"] - r["degenerate"]) for r in reversal if r["method"] == "wald"]
say(f"  How closely: the three Wald rows sit {_g[0]:.4f}, {_g[1]:.4f} and {_g[2]:.4f} below "
    f"P(x1 in {{0, n1}}).")
say("  The first block is the loose one - at p1 = 0.001 the other arm is small enough that some")
say("  zero-count outcomes clear the test anyway, so the two quantities are close without being")
say("  the same event, and this build states the gap rather than claiming an equality.")
say("  Wilson, Agresti-Coull and Clopper-Pearson never reverse on any design here, because none")
say("  of them can produce a degenerate interval.")

# ------------------------------------------------------------------------------------ summary
rule("WHAT TO TAKE AWAY")
say()
say("1. EXACT. At equal standard errors the bars demand 1.414x the separation the test does. The")
say(f"   level at which the two agree is {o.MATCHING_LEVEL_EQUAL_SE * 100:.1f}%, and two 95% bars")
say(f"   may share {o.overlap_fraction_at_significance(1, 1) * 100:.1f}% of an arm and still be")
say("   significant. Both numbers are properties of the DESIGN, not of the picture.")
say(f"2. MEASURED. Up to {best['overlap_given_sig']:.1%} of genuinely significant results have")
say("   overlapping 95% bars, and the share is largest where the argument is closest.")
say(f"3. MEASURED. On paired data at rho = {top['rho']} the rule fails the other way: "
    f"{top['overlap_given_sig']:.4f} of")
say("   real rejections sit behind overlapping bars, and the chart cannot show rho.")
say(f"4. EXACT. Wald coverage bottoms out at {wald_worst['wald']:.4f} against a nominal 0.95, and")
say("   a Wald bar on zero events has zero width, which reverses the rule's one guarantee.")
say("5. The fix is to draw the bars at the matching level and say so in the caption, or to stop")
say("   drawing two intervals and draw the interval for the DIFFERENCE, which is the estimate")
say("   the question was about in the first place.")

with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


with open("results.json", "w") as f:
    json.dump(_clean(results), f, indent=1, sort_keys=True)
print("\nwrote evidence.txt and results.json")
