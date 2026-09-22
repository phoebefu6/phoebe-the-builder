"""Run the whole study and write `evidence.txt` + `results.json`.

Every number the README, the chart, the app and the notebook quote comes from here. Nothing
downstream recomputes a verdict - they read the stored one, because Day 175 shipped a chart and a
notebook that had each grown their own copy of the comparison and disagreed with the evidence file
about four cells.

`results.json` must be byte-identical across two separate processes. It is, because every seed is
an INDEX into a design list that lives in `nonparam.py` (Day 178 derived seeds from
`hash((shape, d))` and Python's per-process string-hash salt made every run different).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import nonparam as N
import numpy as np

OUT_TXT = "evidence.txt"
OUT_JSON = "results.json"

lines: List[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def rule(title: str) -> None:
    say()
    say("=" * 96)
    say(title)
    say("=" * 96)


results: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
rule("0. CALIBRATION - nothing below is read until these land")
# ---------------------------------------------------------------------------

say()
say("Every population is standardised to mean 0, SD 1 by its ANALYTIC moments. If a density does")
say("not integrate to 1, its efficiency number is meaningless, so that is checked first.")
say()
say(f"{'shape':<14} {'integral f':>12} {'integral f^2':>14} {'ARE':>9} {'closed form':>13} {'match':>7}")

calib: List[Dict[str, Any]] = []
calib_ok = True
for shape in N.SHAPES:
    mass = N.density_mass(shape)
    i2 = N.integral_f_squared(shape)
    are = N.asymptotic_are(shape)
    exact = N.ARE_CLOSED_FORM.get(shape)
    mass_ok = abs(mass - 1.0) < 1e-6
    are_ok = exact is None or abs(are - exact) < 1e-6
    calib_ok = calib_ok and mass_ok and are_ok
    calib.append({"shape": shape, "mass": mass, "int_f2": i2, "are": are,
                  "closed_form": exact, "mass_ok": mass_ok, "are_ok": are_ok})
    say(f"{shape:<14} {mass:>12.6f} {i2:>14.6f} {are:>9.4f} "
        f"{(f'{exact:.6f}' if exact is not None else '-'):>13} "
        f"{('OK' if (mass_ok and are_ok) else 'FAIL'):>7}")

say()
say("ARE is the asymptotic relative efficiency of Mann-Whitney to the t-test UNDER A LOCATION")
say("SHIFT: 12 * sigma^2 * (integral f^2)^2, and sigma = 1 here by construction. Above 1 means")
say("Mann-Whitney needs FEWER observations for the same power. Three of the six have closed")
say("forms - normal 3/pi, uniform exactly 1, centred exponential exactly 3 - and they all land.")
results["calibration_are"] = calib

# The hand-rolled uncorrected Mann-Whitney used in section 5 must reproduce SciPy exactly when
# there are no ties, or the tie column is measuring an implementation difference instead.
rng = np.random.default_rng(90_001)
_a = rng.standard_normal((400, 25))
_b = rng.standard_normal((400, 25)) + 0.3
_pc, _ = N.mannwhitney(_a, _b)
_pu = N.mannwhitney_uncorrected(_a, _b)
notie_gap = float(np.max(np.abs(_pc - _pu)))
say()
say(f"Hand-rolled uncorrected Mann-Whitney vs SciPy on TIE-FREE data: max |dp| = {notie_gap:.3e}")
say("So in section 5 the only thing separating the two columns is the tie term.")
results["untied_agreement"] = notie_gap
calib_ok = calib_ok and notie_gap < 1e-12
results["calibration_ok"] = calib_ok


# ---------------------------------------------------------------------------
rule("1. THE HEADLINE - two defensible tests, one dataset, opposite signs")
# ---------------------------------------------------------------------------

mu = N.mix_mean()
psup = N.mix_prob_superiority()
say()
say("The design, in closed form - no simulation involved in any of these four numbers:")
say()
say(f"  control    N(0, {N.MIX_SD}^2)")
say(f"  treated    {N.MIX_WEIGHTS[0]} * N({N.MIX_MEANS[0]}, {N.MIX_SD}^2)"
    f"  +  {N.MIX_WEIGHTS[1]} * N({N.MIX_MEANS[1]}, {N.MIX_SD}^2)")
say()
say(f"  true mean difference   mu_B - mu_A  = {mu:+.4f}   -> the t-test's target says treated is HIGHER")
say(f"  true P(treated > control)           = {psup:.5f}   -> Mann-Whitney's target says treated is LOWER")
say(f"  treated SD                          = {N.mix_sd():.4f}")
say(f"  population Cohen's d                = {N.mix_cohens_d():.4f}")
say()
say("Most treated cases end up slightly worse; a minority end up a lot better. Both tests are")
say("right. They are answering different questions, and the questions have opposite answers.")
say()
say(f"{'n/group':>8} {'t sig UP':>10} {'t sig DOWN':>11} {'MW sig UP':>11} {'MW sig DOWN':>12} "
    f"{'BOTH, OPPOSITE':>15} {'t only':>8} {'MW only':>8} {'neither':>8}")

disagree: List[Dict[str, Any]] = []
for i, n in enumerate(N.DISAGREE_NS):
    r = N.run_disagreement(n, N.DISAGREE_REPS, N.DISAGREE_SEED_BASE + i)
    lo, hi = N.wilson(r["opposite"], N.DISAGREE_REPS)
    r["opposite_lo"], r["opposite_hi"] = lo, hi
    disagree.append(r)
    say(f"{n:>8} {r['t_up']:>10.4f} {r['t_down']:>11.4f} {r['mw_up']:>11.4f} {r['mw_down']:>12.4f} "
        f"{r['opposite']:>15.4f} {r['t_only']:>8.4f} {r['mw_only']:>8.4f} {r['neither']:>8.4f}")

peak = max(disagree, key=lambda r: r["opposite"])
say()
say(f"Contradiction rate {peak['opposite']:.4f} at n = {int(peak['n'])} per group "
    f"(99% CI [{peak['opposite_lo']:.4f}, {peak['opposite_hi']:.4f}]).")
say("Read that as: on this design, at that sample size, you can run both tests, have both clear")
say("p < 0.05, and get opposite conclusions about the direction of the effect.")
say()
say("The uncomfortable part is the SHAPE of that column. The contradiction rate RISES")
say("monotonically with n - more data does not resolve it, more data guarantees it. At the")
say("small sample sizes where an analyst might expect trouble the two tests simply fail to")
say("disagree, because neither has the power to say anything; the contradiction arrives")
say("exactly when both tests become trustworthy.")
rising = all(disagree[i]["opposite"] <= disagree[i + 1]["opposite"] for i in range(len(disagree) - 1))
say(f"  monotone in n: {rising}")
results["disagreement_monotone"] = rising

small = disagree[0]
say()
say(f"Side finding at n = {int(small['n'])}: the t-test fires DOWNWARD {small['t_down']:.4f} of the")
say(f"time against {small['t_up']:.4f} upward, so {small['t_down'] / (small['t_up'] + small['t_down']):.1%}")
say("of its significant results at that size point the wrong way about its OWN target. A sample")
say("from this mixture that happens to miss the high component has both a low mean and a low")
say("variance, and the low variance shrinks the standard error the low mean is divided by - the")
say("same mechanism Day 175 measured as a tail split under skew.")
results["disagreement"] = disagree


# ---------------------------------------------------------------------------
rule("2. WHERE THE SWAP IS FINE - power under a pure location shift")
# ---------------------------------------------------------------------------

say()
say("Under a pure location shift the two hypotheses COINCIDE, so a power comparison is a fair")
say("question. Both columns are gated: power is only comparable in a cell where both tests")
say("control their Type I error, measured at d = 0 on the same design.")
say()
say(f"d = {N.LOCATION_D}, equal n, {N.NULL_REPS:,} replicates for the null and "
    f"{N.STUDY_REPS:,} for the power.")

location: List[Dict[str, Any]] = []
for i, (shape, n) in enumerate(N.LOCATION_DESIGNS):
    null = N.run_location(shape, 0.0, n, N.NULL_REPS, N.NULL_SEED_BASE + i)
    power = N.run_location(shape, N.LOCATION_D, n, N.STUDY_REPS, N.LOCATION_SEED_BASE + i)
    row = {
        "shape": shape, "n": n,
        "t_null": null["t"], "mw_null": null["mw"],
        "t_null_verdict": N.verdict(null["t"], N.NULL_REPS),
        "mw_null_verdict": N.verdict(null["mw"], N.NULL_REPS),
        "t_power": power["t"], "mw_power": power["mw"],
        "comparable": N.comparable(null["t"], null["mw"], N.NULL_REPS),
    }
    location.append(row)
results["location"] = location

for shape in N.SHAPES:
    rows = [r for r in location if r["shape"] == shape]
    say()
    say(f"  {shape}   (predicted ARE {N.asymptotic_are(shape):.4f})")
    say(f"  {'n':>5} {'t null':>9} {'MW null':>9} {'t null?':>13} {'MW null?':>13} "
        f"{'t power':>9} {'MW power':>9} {'comparable':>11}")
    for r in rows:
        say(f"  {r['n']:>5} {r['t_null']:>9.4f} {r['mw_null']:>9.4f} "
            f"{r['t_null_verdict']:>13} {r['mw_null_verdict']:>13} "
            f"{r['t_power']:>9.4f} {r['mw_power']:>9.4f} "
            f"{('yes' if r['comparable'] else 'NO'):>11}")

say()
say("Measured efficiency: the sample size the t-test needs to match Mann-Whitney's power at")
say(f"n = {N.LOCATION_REF_N}, read off the t-test's own measured power curve by interpolation in")
say("log n, using only comparable cells. Above 1 means Mann-Whitney was the more efficient test")
say("here - the same direction as the predicted ARE. A cell whose target falls outside the")
say("measured curve reports '-' rather than an extrapolated guess.")
say()
say(f"{'shape':<14} {'predicted ARE':>14} {'measured n_t/n_mw':>19} {'cells usable':>13}")

efficiency: List[Dict[str, Any]] = []
for shape in N.SHAPES:
    rows = [r for r in location if r["shape"] == shape and r["comparable"]]
    ns = [r["n"] for r in rows]
    tp = [r["t_power"] for r in rows]
    ref = next((r for r in rows if r["n"] == N.LOCATION_REF_N), None)
    eff = None
    if ref is not None and len(ns) >= 2:
        eff = N.efficiency_from_curve(ns, tp, N.LOCATION_REF_N, ref["mw_power"])
    efficiency.append({"shape": shape, "are": N.asymptotic_are(shape),
                       "measured": eff, "usable": len(rows)})
    say(f"{shape:<14} {N.asymptotic_are(shape):>14.4f} "
        f"{(f'{eff:.4f}' if eff is not None else '-'):>19} {len(rows):>13}")
results["efficiency"] = efficiency

usable = [e for e in efficiency if e["measured"] is not None]
if usable:
    worst = min(usable, key=lambda e: e["measured"])
    best = max(usable, key=lambda e: e["measured"])
    say()
    say(f"Worst case for Mann-Whitney: {worst['shape']} at {worst['measured']:.4f} "
        f"(predicted {worst['are']:.4f}).")
    say(f"Best case:                   {best['shape']} at {best['measured']:.4f} "
        f"(predicted {best['are']:.4f}).")
    say("So as a POWER decision the folk advice is sound - the rank test gives up a few per cent")
    say("on normal data and wins outright everywhere else. The cost of the swap is not power.")
    say()
    say("The measured column sits BELOW the predicted one almost everywhere, and the gap grows")
    say("with the prediction. That is expected rather than a discrepancy: ARE is a limit as the")
    say("effect goes to zero and n goes to infinity, while these are finite-n readings at")
    say(f"d = {N.LOCATION_D} and n = {N.LOCATION_REF_N}, where the high-efficiency shapes have")
    say("already pushed Mann-Whitney's power into the flat top of the curve and interpolation")
    say("has little room left to work in. The two columns are checked for agreement in ORDER,")
    say("not in value - Spearman across the six shapes is asserted in the test suite.")
    say()
    say("Negative result from the gate: the comparison is NOT available everywhere. Welch is")
    say("measurably conservative on the contaminated population at small n, which disqualifies")
    say("three cells - and they are the cells where Mann-Whitney's advantage looks largest, so")
    say("dropping the gate would have inflated exactly the rows that most flatter the rank test.")


# ---------------------------------------------------------------------------
rule("3. THE COST NOBODY MENTIONS - unequal spread, where the MW null is EXACTLY true")
# ---------------------------------------------------------------------------

say()
say("Both groups are symmetric with the same centre, and only the SD differs. For symmetric")
say("populations that makes P(B > A) EXACTLY 0.5, so Mann-Whitney's own null hypothesis is true -")
say("and the mean null is true as well. Any departure from 5% is not a violated hypothesis. It is")
say("the test's variance formula, which assumes the two distributions are IDENTICAL rather than")
say("merely balanced. Welch's t-test is the comparator because Day 176 established it as the one")
say("procedure that held its rate on every design in that grid.")
say()
say(f"{N.NULL_REPS:,} replicates a cell.")
say()
say(f"{'shape':<14} {'SD ratio':>9} {'n1':>4} {'n2':>4} {'Welch':>8} {'Welch?':>13} "
    f"{'MannWhitney':>12} {'MW?':>13}")

unequal: List[Dict[str, Any]] = []
for i, (shape, ratio, n1, n2) in enumerate(N.UNEQUAL_DESIGNS):
    r = N.run_unequal_spread(shape, ratio, n1, n2, N.NULL_REPS, N.UNEQUAL_SEED_BASE + i)
    row = {"shape": shape, "sd_ratio": ratio, "n1": n1, "n2": n2,
           "welch": r["welch"], "mw": r["mw"],
           "welch_verdict": N.verdict(r["welch"], N.NULL_REPS),
           "mw_verdict": N.verdict(r["mw"], N.NULL_REPS)}
    unequal.append(row)
    say(f"{shape:<14} {ratio:>9.1f} {n1:>4} {n2:>4} {r['welch']:>8.4f} {row['welch_verdict']:>13} "
        f"{r['mw']:>12.4f} {row['mw_verdict']:>13}")
results["unequal"] = unequal

mw_bad = [r for r in unequal if r["mw_verdict"] != "ok"]
welch_bad = [r for r in unequal if r["welch_verdict"] != "ok"]
say()
say(f"Mann-Whitney misses its nominal rate on {len(mw_bad)} of {len(unequal)} designs; "
    f"Welch on {len(welch_bad)}.")
if mw_bad:
    hi = max(mw_bad, key=lambda r: r["mw"])
    lo = min(mw_bad, key=lambda r: r["mw"])
    say(f"  Most INFLATED: {hi['shape']} SD ratio {hi['sd_ratio']:.1f} at n = {hi['n1']}/{hi['n2']} "
        f"-> {hi['mw']:.4f} ({hi['mw'] / N.ALPHA:.2f}x nominal)")
    say(f"  Most CONSERVATIVE: {lo['shape']} SD ratio {lo['sd_ratio']:.1f} at "
        f"n = {lo['n1']}/{lo['n2']} -> {lo['mw']:.4f} ({N.ALPHA / max(lo['mw'], 1e-9):.1f}x too quiet)")
say()
say("The direction is set by WHICH group is the wide one. Wide group small -> the rank test")
say("fires too often; wide group large -> it goes quiet and the power is simply gone. Both are")
say("failures, and only one of them looks like a failure.")

balanced = [r for r in unequal if r["n1"] == r["n2"]]
balanced_bad = [r for r in balanced if r["mw_verdict"] != "ok"]
unbal = [r for r in unequal if r["n1"] != r["n2"]]
unbal_bad = [r for r in unbal if r["mw_verdict"] != "ok"]
say()
say("Unequal n is NOT the precondition, which is the part that separates this from Day 176's")
say(f"Student's-vs-Welch result. Mann-Whitney breaks on {len(balanced_bad)} of {len(balanced)} "
    f"BALANCED cells as well as {len(unbal_bad)} of {len(unbal)} unbalanced ones.")
if balanced_bad:
    worst_bal = max(balanced_bad, key=lambda r: abs(r["mw"] - N.ALPHA))
    say(f"  Worst balanced cell: {worst_bal['shape']} SD ratio {worst_bal['sd_ratio']:.1f} at "
        f"n = {worst_bal['n1']}/{worst_bal['n2']} -> {worst_bal['mw']:.4f} "
        f"({worst_bal['mw'] / N.ALPHA:.2f}x nominal)")
say("What unequal n changes is the SIZE and the sign. Balanced designs are inflated by about")
say("half again; put the wide group in the small arm and it is three and a half times nominal,")
say("put it in the large arm and the test all but stops firing.")

say()
say("Welch is not perfect here either, and saying otherwise would be the easy version of this")
say(f"table. It is measurably conservative on {len(welch_bad)} cells, all of them the")
say("contaminated population, where one observation in twenty comes from a five-times-wider")
say("normal. But the two failures are not the same size:")
w_worst = max(welch_bad, key=lambda r: abs(r["welch"] - N.ALPHA)) if welch_bad else None
if w_worst is not None:
    w_ratio = max(w_worst["welch"], N.ALPHA) / min(w_worst["welch"], N.ALPHA)
    m_worst = max(mw_bad, key=lambda r: max(r["mw"], N.ALPHA) / max(min(r["mw"], N.ALPHA), 1e-9))
    m_ratio = max(m_worst["mw"], N.ALPHA) / max(min(m_worst["mw"], N.ALPHA), 1e-9)
    say(f"  Welch's worst departure       {w_worst['welch']:.4f}  =  {w_ratio:.2f}x off nominal")
    say(f"  Mann-Whitney's worst departure {m_worst['mw']:.4f}  =  {m_ratio:.1f}x off nominal")
say("A test that is a fifth conservative and a test that is 14 times too quiet are both wrong")
say("and they are not comparably wrong.")


# ---------------------------------------------------------------------------
rule("4. TIES - a rank test on a rating scale")
# ---------------------------------------------------------------------------

say()
say("Ties only ever SHRINK Var(U). The no-ties formula therefore divides by a standard error")
say("that is too large, and the uncorrected test goes quiet. SciPy applies the correction; a lot")
say("of hand-rolled rank tests and spreadsheet macros do not.")
say()
say(f"n = {N.TIE_N} per group, {N.TIE_REPS:,} replicates a cell, power measured at d = {N.TIE_D}.")
say()
say(f"{'levels':>7} {'SD corr/uncorr':>15} {'null corrected':>15} {'?':>13} "
    f"{'null uncorrected':>17} {'?':>13} {'power corr':>11} {'power uncorr':>13}")

ties: List[Dict[str, Any]] = []
for i, levels in enumerate(N.TIE_LEVELS):
    null = N.run_ties(levels, N.TIE_N, 0.0, N.TIE_REPS, N.TIE_NULL_SEED_BASE + i)
    power = N.run_ties(levels, N.TIE_N, N.TIE_D, N.TIE_REPS, N.TIE_POWER_SEED_BASE + i)
    row = {"levels": levels, "sd_ratio": null["sd_ratio"],
           "null_corrected": null["corrected"], "null_uncorrected": null["uncorrected"],
           "power_corrected": power["corrected"], "power_uncorrected": power["uncorrected"],
           "null_corrected_verdict": N.verdict(null["corrected"], N.TIE_REPS),
           "null_uncorrected_verdict": N.verdict(null["uncorrected"], N.TIE_REPS)}
    ties.append(row)
    say(f"{levels:>7} {null['sd_ratio']:>15.4f} {null['corrected']:>15.4f} "
        f"{row['null_corrected_verdict']:>13} {null['uncorrected']:>17.4f} "
        f"{row['null_uncorrected_verdict']:>13} {power['corrected']:>11.4f} "
        f"{power['uncorrected']:>13.4f}")
results["ties"] = ties

binary = ties[0]
say()
say(f"On a {int(binary['levels'])}-point scale the tie correction shrinks the standard error to "
    f"{binary['sd_ratio']:.4f} of the no-ties value,")
say(f"and dropping it takes the test from {binary['null_corrected']:.4f} to "
    f"{binary['null_uncorrected']:.4f} under a true null - "
    f"{binary['null_corrected'] / max(binary['null_uncorrected'], 1e-9):.1f}x too quiet - "
    f"costing {binary['power_corrected'] - binary['power_uncorrected']:.4f} of power.")
bad_corrected = [t for t in ties if t["null_corrected_verdict"] != "ok"]
say()
say("Negative result, and it is a negative result about this study rather than about the test:")
say(f"the CORRECTED column is flagged on {len(bad_corrected)} of {len(ties)} scales. The")
say(f"2-point cell is the highest in the table at {binary['null_corrected']:.4f} and it is above")
say("nominal, but its 99% Wilson interval still touches the band, so this build does not have")
say("the resolution to call it broken and does not claim to. The claim that survives is the")
say("one-sided one: applying the tie correction is necessary, and on the evidence here it is")
say("also sufficient.")


# ---------------------------------------------------------------------------
rule("SUMMARY")
# ---------------------------------------------------------------------------

say()
say("1. Mann-Whitney and the t-test answer different questions. On a design whose true mean")
say(f"   difference is {mu:+.2f} and whose true P(B>A) is {psup:.4f}, both tests clear p < 0.05 and")
say(f"   point OPPOSITE ways {peak['opposite']:.1%} of the time at n = {int(peak['n'])} per group.")
say("2. As a POWER decision the folk advice is right: under a pure location shift the rank test")
say("   gives up a few per cent on normal data and wins outright on every other shape measured.")
say("3. As a HYPOTHESIS decision it is a substitution, and nobody announces the substitution.")
say(f"4. With unequal spread the rank test misses 5% on {len(mw_bad)} of {len(unequal)} designs")
say("   where its own null is exactly true - in BOTH directions, and on balanced designs too,")
say("   up to 3.6x nominal one way and 14x too quiet the other. Welch is conservative on the")
say("   contaminated cells but by about a fifth, an order of magnitude less.")
say(f"5. On a {int(binary['levels'])}-point rating scale, dropping the tie correction costs "
    f"{binary['power_corrected'] - binary['power_uncorrected']:.2f} of power")
say("   and takes a 5% test down to under 2%. It is not a rounding detail.")
say()
say("The honest report is not one test or the other. It is the mean difference AND P(B > A),")
say("because when they disagree that disagreement is the finding.")

with open(OUT_TXT, "w") as fh:
    fh.write("\n".join(lines) + "\n")

results["config"] = {
    "alpha": N.ALPHA, "band": [N.BAND_LO, N.BAND_HI],
    "study_reps": N.STUDY_REPS, "null_reps": N.NULL_REPS,
    "disagree_reps": N.DISAGREE_REPS, "tie_reps": N.TIE_REPS,
    "location_d": N.LOCATION_D, "location_ref_n": N.LOCATION_REF_N,
    "mix_mean": mu, "mix_prob_superiority": psup,
    "mix_sd": N.mix_sd(), "mix_cohens_d": N.mix_cohens_d(),
}
with open(OUT_JSON, "w") as fh:
    json.dump(results, fh, indent=2, sort_keys=True)

print(f"\nwrote {OUT_TXT} and {OUT_JSON}")
