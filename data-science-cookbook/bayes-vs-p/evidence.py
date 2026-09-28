"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). The study numbers are exact (closed-form normal
integrals and quadrature); the Monte Carlo section only checks them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import bayes as B
import numpy as np
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


results: Dict[str, Any] = {"config": {
    "alpha": B.ALPHA, "tau": B.TAU, "r_jzs": B.R_JZS, "ns": list(B.NS), "taus": list(B.TAUS),
    "mc_reps": B.MC_REPS, "numpy": np.__version__, "scipy": __import__("scipy").__version__,
}}

rule("1.  CALIBRATION  -  every Bayes factor computed two ways, and one of them by somebody else")
cal = B.calibrate()
results["calibration"] = cal
say(f"  normal-prior BF: closed form vs quadrature over the prior (18 cases)   max |log gap| {cal['max_log_gap_closed_vs_quadrature']:.1e}")
say(f"  JZS BF: quadrature over g vs quadrature over delta (9 cases)          max |log gap| {cal['max_log_gap_jzs_g_vs_delta']:.1e}")
say(f"  JZS BF vs pingouin.bayesfactor_ttest (9 cases)                        max |log gap| {cal['max_log_gap_jzs_vs_pingouin']:.1e}")
say(f"  t-test decisions vs scipy.stats.ttest_1samp                           {cal['scipy_decision_mismatches']} "
    f"mismatches in {cal['decisions_compared']}")
say(f"  max BF10 over normal priors at z = 1.96: closed form vs search        gap {cal['max_bf10_closed_vs_search_gap']:.1e}")
mc = B.monte_carlo()
results["monte_carlo"] = mc
say(f"\n  raw N(delta, 1) data, {B.MC_REPS:,} reps a design; exact inside the 99% Wilson interval?")
for r in mc:
    say(f"    n={r['n']:>5} delta={r['delta']:<5}  "
        + "  ".join(f"{k}: {r[k]['exact']:.4f}/{r[k]['mc']:.4f} {'in' if r[k]['inside_99'] else 'OUT'}"
                    for k in ("p_significant", "bf01_over_3", "sig_and_bf01_over_1")))

rule("2.  LINDLEY'S CURVE  -  hold the p-value at exactly 0.05 and grow n.  BF01 > 1 means evidence FOR H0")
lind = []
say(f"  {'n':>9}   normal prior tau=1   JZS r=0.707 (t-test)   normal prior tau=0.1")
for n in B.NS:
    t = float(stats.t.isf(B.ALPHA / 2, n - 1))
    row = {"n": n, "bf01_tau1": B.bf01_at_p(0.05, n, 1.0), "bf01_jzs": 1 / B.bf10_jzs(t, n),
           "bf01_tau01": B.bf01_at_p(0.05, n, 0.1)}
    lind.append(row)
    say(f"  {n:>9}   {row['bf01_tau1']:>18.3f}   {row['bf01_jzs']:>20.3f}   {row['bf01_tau01']:>20.3f}")
results["lindley"] = lind
flip = {f"p={p}_k={k}": B.n_where_p_supports_null(p, k) for p in (0.05, 0.01, 0.005) for k in (1.0, 3.0, 10.0)}
results["flip_n"] = flip
say("\n  smallest n at which a result at exactly this p is evidence FOR the null (tau = 1):")
for p in (0.05, 0.01, 0.005):
    say(f"    p = {p:<5}  BF01 >= 1 from n = {flip[f'p={p}_k=1.0']:>8.1f}   >= 3 from n = {flip[f'p={p}_k=3.0']:>8.1f}"
        f"   >= 10 from n = {flip[f'p={p}_k=10.0']:>9.1f}")

rule("3.  THE CEILING  -  the most any normal alternative can favour H1 at p = 0.05")
mx = B.max_bf10_over_normal_priors(B.Z_CRIT)
results["ceiling"] = {**mx, "sbb_bound_p05": B.sbb_bound(0.05), "sbb_bound_p005": B.sbb_bound(0.005),
                      "max_bf10_p005": B.max_bf10_over_normal_priors(float(stats.norm.isf(0.0025)))["closed_form"]}
say(f"  best normal prior at z = 1.96: BF10 = {mx['closed_form']:.3f} (tau * sqrt(n) = {mx['best_tau_sqrt_n']:.3f})")
say(f"  Sellke-Bayarri-Berger bound 1/(-e p ln p): p = 0.05 -> {B.sbb_bound(0.05):.3f},  p = 0.005 -> {B.sbb_bound(0.005):.3f}")
say(f"  best normal prior at p = 0.005: BF10 = {results['ceiling']['max_bf10_p005']:.3f}")
say("  A p of 0.05 can never be better than ~2:1 evidence for an effect, whatever prior you choose.")

rule("4.  EXACT VERDICT PROBABILITIES WHEN THE NULL IS TRUE (delta = 0, tau = 1)")
say(f"  {'n':>9}   P(p < .05)   P(BF10 > 3)   P(BF01 > 3)   P(p < .05 AND BF01 > 1)   share of 'findings' the BF calls null")
null_rows = []
for n in B.NS:
    v = B.verdicts(n, 0.0)
    v["share_of_sig_bf_null"] = v["sig_and_bf01_over_1"] / v["p_significant"]
    null_rows.append(v)
    say(f"  {n:>9}   {v['p_significant']:>10.4f}   {v['bf10_over_3']:>11.4f}   {v['bf01_over_3']:>11.4f}"
        f"   {v['sig_and_bf01_over_1']:>23.4f}   {v['share_of_sig_bf_null']:>10.1%}")
results["null"] = null_rows
say("\n  The p-value's false-positive rate is 0.05 at every n. The Bayes factor's falls toward zero.")

rule("5.  NEGATIVE RESULT  -  a real but tiny effect (delta = 0.02 SD).  The Bayes factor is not a p-value fix")
tiny = []
say(f"  {'n':>9}   P(p < .05)   P(BF10 > 3)   P(BF01 > 3, i.e. 'evidence of no effect')")
for n in (1000, 5000, 10000, 20000, 50000, 100000, 1000000):
    v = B.verdicts(n, 0.02)
    tiny.append(v)
    say(f"  {n:>9}   {v['p_significant']:>10.4f}   {v['bf10_over_3']:>11.4f}   {v['bf01_over_3']:>11.4f}")
results["tiny_effect"] = tiny
v20 = next(v for v in tiny if v["n"] == 20000)
say(f"\n  At n = 20,000 the p-value finds this effect {v20['p_significant']:.0%} of the time; the Bayes factor calls it")
say(f"  'no effect' {v20['bf01_over_3']:.0%} of the time. It is not wrong - 0.02 SD IS close to zero on a tau = 1 scale - but it has swapped")
say("  one error for another, not removed it.")
pw = []
for d in (0.5, 0.2, 0.1, 0.05):
    a, b = B.n_for_power(d, "p_significant"), B.n_for_power(d, "bf10_over_3")
    pw.append({"delta": d, "n_p": a, "n_bf": b, "ratio": b / a})
say("\n  n for 80% chance of a verdict for the effect:  p < .05  vs  BF10 > 3")
for r in pw:
    say(f"    delta = {r['delta']:<5}  {r['n_p']:>6}  vs  {r['n_bf']:>6}   ({r['ratio']:.2f}x)")
results["power"] = pw

rule("6.  PRIOR SENSITIVITY  -  one dataset, one p, six Bayes factors")
ex = B.exemplar()
results["exemplar"] = ex
say(f"  seed {ex['seed']}: n = {ex['n']:,} observations, true delta {ex['true_delta']} SD, observed d = {ex['d']:.4f}")
say(f"  one-sample t = {ex['t']:.3f}   p = {ex['p']:.4f}   ->  'significant'")
say(f"  JZS BF01 (r = 0.707, the software default) = {ex['bf01']:.2f}   ->  'strong evidence for no effect'")
for tau, v in ex["bf01_by_tau"].items():
    say(f"    normal prior tau = {tau:<5}  BF01 = {v:>7.3f}")
say("  The verdict turns on tau: a prior expecting effects of ~0.05 SD calls the same data weak evidence FOR an effect.")

with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
with open("results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
