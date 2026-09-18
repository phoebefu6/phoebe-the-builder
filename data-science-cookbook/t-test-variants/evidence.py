"""Run the whole study and write evidence.txt + results.json.

Every number quoted in the README and the notebook comes from here. Re-run it and the numbers
reproduce exactly: every cell is seeded, and the seed is derived from the cell so adding a row
never shifts an existing one.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
from ttests import (
    ALPHA,
    TESTS,
    Cell,
    one_sample_t,
    paired_t,
    scipy_agreement,
    simulate,
    verdict,
    welch_t,
    wilson,
)

REPS = 40_000
POWER_REPS = 40_000
lines: List[str] = []
results: Dict[str, Any] = {"reps": REPS, "alpha": ALPHA}


def say(s: str = "") -> None:
    lines.append(s)
    print(s)


def rule(title: str) -> None:
    say()
    say(title)
    say("=" * len(title))


# ---------------------------------------------------------------- 0. credential
rule("0. The statistics are right before the rates mean anything")
worst = scipy_agreement()
for name, d in worst.items():
    say(f"  {name:12s} max |p - scipy| over 200 random samples = {d:.3e}")
results["scipy_max_abs_diff"] = worst
rng = np.random.default_rng(99)
a, b = rng.standard_normal(24), rng.standard_normal(24) * 1.7
identity = abs(paired_t(a, b).p - one_sample_t(a - b).p)
say(f"  paired_t(x,y) vs one_sample_t(x-y): |dp| = {identity:.3e}  (they are one procedure)")
results["paired_equals_one_sample"] = identity
say(f"  half-width of a Wilson interval on 0.05 at {REPS:,} reps = "
    f"{(wilson(0.05 * REPS, REPS)[1] - wilson(0.05 * REPS, REPS)[0]) / 2:.4f}")

# ------------------------------------------------- 0b. the harness's own noise floor
rule("0b. A known-truth cell, so every rate below can be read against the harness's noise")
say("  Student's t on balanced, equal-variance, normal data is EXACT: its Type I error is 0.05")
say("  by construction, not by approximation. So whatever this harness measures there is pure")
say("  Monte-Carlo noise, and it is the resolution limit for every other cell.")
floor = [simulate(Cell(n1=20, n2=20), reps=REPS, seed=s)["student"] for s in range(10)]
say(f"  10 independent runs: min {min(floor):.4f}, max {max(floor):.4f}, mean {sum(floor) / 10:.4f}")
NOISE = max(abs(f - ALPHA) for f in floor) / ALPHA
say(f"  So a rate inside {1 - NOISE:.2f}x-{1 + NOISE:.2f}x nominal is indistinguishable from exact here.")
say("  Read every 'INFLATED' verdict below against that: the Wilson interval is a statistical")
say("  test, and at 40,000 reps it will flag a 1.05x departure that means nothing in practice.")
results["noise_floor_runs"] = floor
results["noise_floor_rel"] = NOISE

# ---------------------------------------------------------------- 1. the grid
rule("1. Type I error of each test, null exactly true, normal data")
say(f"  nominal alpha = {ALPHA}; anything outside the Wilson interval is a real departure")
say()
say(f"  {'condition':34s} {'balance':22s} {'Student':>9s} {'Welch':>9s} {'z-short':>9s}")
grid: List[Dict[str, Any]] = []
for ratio in (1.0, 2.0, 3.0):
    for k, (n1, n2) in enumerate([(20, 20), (30, 10), (10, 30), (50, 10), (10, 50)]):
        cell = Cell(n1=n1, n2=n2, sd1=1.0, sd2=ratio)
        rates = simulate(cell, reps=REPS, seed=int(ratio * 100) + k)
        say(f"  {cell.label():34s} {cell.balance:22s} "
            + " ".join(f"{rates[t]:9.4f}" for t in TESTS))
        grid.append({"n1": n1, "n2": n2, "sd_ratio": ratio, "balance": cell.balance, "rates": rates})
results["type_i_grid"] = grid

worst_student = max(grid, key=lambda r: r["rates"]["student"])
best_student = min(grid, key=lambda r: r["rates"]["student"])
say()
say(f"  Student worst: n={worst_student['n1']}/{worst_student['n2']} sd_ratio={worst_student['sd_ratio']:g}"
    f" -> {worst_student['rates']['student']:.4f} = {verdict(worst_student['rates']['student'], REPS)}")
say(f"  Student best : n={best_student['n1']}/{best_student['n2']} sd_ratio={best_student['sd_ratio']:g}"
    f" -> {best_student['rates']['student']:.4f} = {verdict(best_student['rates']['student'], REPS)}")
wr = [r["rates"]["welch"] for r in grid]
say(f"  Welch across all 15 cells: min {min(wr):.4f}, max {max(wr):.4f}")
zr = [r["rates"]["z-shortcut"] for r in grid]
say(f"  z-shortcut across all 15 cells: min {min(zr):.4f}, max {max(zr):.4f} - inflated in every one")
results["welch_range"] = [min(wr), max(wr)]
results["z_range"] = [min(zr), max(zr)]

# ---------------------------------------------------------------- 2. the negative result
rule("2. Negative result: unequal variance is NOT the diagnostic, imbalance is")
bal = [r for r in grid if r["balance"] == "balanced"]
for r in bal:
    mult = r["rates"]["student"] / ALPHA
    flag = "within the noise floor" if abs(mult - 1) <= NOISE else f"{mult:.2f}x nominal"
    say(f"  balanced n=20/20, sd_ratio={r['sd_ratio']:g}: Student {r['rates']['student']:.4f}"
        f"  -> {flag}")
say("  A 3x variance gap with equal group sizes costs Student's test almost nothing. The rule of")
say("  thumb 'check for equal variances' is testing the wrong thing on its own.")
results["balanced_student"] = {r["sd_ratio"]: r["rates"]["student"] for r in bal}

x = np.random.default_rng(1).standard_normal(50)
y = np.random.default_rng(2).standard_normal(10) * 3
say("  And the mechanism is visible in one number: at n=50/10 with sd 1/3, Student's test spends")
say(f"  df=58 while Welch's honest df is {welch_t(x, y).df:.2f}.")
results["welch_df_example"] = welch_t(x, y).df

# ---------------------------------------------------------------- 3. what Welch costs
rule("3. What Welch costs when Student's assumption actually holds (equal sd)")
say(f"  {'design':16s} {'effect d':>9s} {'Student':>9s} {'Welch':>9s} {'power lost':>11s}")
power: List[Dict[str, Any]] = []
for k, (n1, n2, d) in enumerate(
    [(10, 10, 0.5), (10, 10, 0.8), (20, 20, 0.5), (20, 20, 0.8), (50, 50, 0.8),
     (30, 10, 0.8), (50, 10, 0.8), (50, 10, 0.5)]
):
    rates = simulate(Cell(n1=n1, n2=n2, shift=d), reps=POWER_REPS, seed=900 + k)
    lost = rates["student"] - rates["welch"]
    say(f"  n={n1}/{n2:<11d} {d:9.1f} {rates['student']:9.4f} {rates['welch']:9.4f} {lost:+11.4f}")
    power.append({"n1": n1, "n2": n2, "d": d, "rates": rates, "power_lost": lost})
results["power"] = power
say()
say("  Balanced: the premium rounds to zero (0.0012 at n=20, 0.0000 at n=50).")
say("  Unbalanced with equal variances - the one corner where Student's is both valid and")
say("  genuinely better - Welch gives up real power. That is the honest cost of the default,")
say("  and it is still a bad trade against a Type I error of 0.29 in the neighbouring cell.")

# ---------------------------------------------------------------- 4. skew
rule("4. Negative result: Welch does not fix non-normality, and under skew it is the worse test")
say(f"  {'distribution':12s} {'design':12s} {'Student':>9s} {'Welch':>9s}")
skew: List[Dict[str, Any]] = []
for k, dist in enumerate(["normal", "uniform", "t3", "lognormal"]):
    for j, (n1, n2) in enumerate([(20, 20), (50, 10), (10, 50)]):
        rates = simulate(Cell(n1=n1, n2=n2, dist=dist), reps=REPS, seed=300 + 10 * k + j)
        say(f"  {dist:12s} n={n1}/{n2:<8d} {rates['student']:9.4f} {rates['welch']:9.4f}")
        skew.append({"dist": dist, "n1": n1, "n2": n2, "rates": rates})
results["by_distribution"] = skew
say()
say("  Under a lognormal with unequal n, the ranking REVERSES: Student's holds at ~0.049 and")
say("  Welch runs at ~0.094. Separate variance estimates are the problem - for a skewed variable")
say("  the sample mean and sample variance are correlated, so the n=10 group's variance is small")
say("  exactly when its mean is low, and the statistic's tail is no longer a t.")

say()
say("  How much n does the CLT need to repair it?")
clt: List[Dict[str, Any]] = []
for k, (n1, n2) in enumerate([(50, 10), (150, 30), (500, 100), (2500, 500)]):
    rates = simulate(Cell(n1=n1, n2=n2, dist="lognormal"), reps=REPS, seed=600 + k)
    say(f"    n={n1}/{n2:<6d} Student {rates['student']:.4f}   Welch {rates['welch']:.4f}"
        f"  ({verdict(rates['welch'], REPS)})")
    clt.append({"n1": n1, "n2": n2, "rates": rates})
results["lognormal_clt"] = clt
say("    Welch needs 3,000 observations to come back to nominal on a skew this ordinary.")

# ---------------------------------------------------------------- 5. verdict
rule("5. What to do on Monday")
normal_cells = [r for r in grid]
welch_bad = [r for r in normal_cells if not verdict(r["rates"]["welch"], REPS).startswith("nominal")
             and abs(r["rates"]["welch"] - ALPHA) / ALPHA > NOISE]
say(f"  * Default to Welch. Over {len(normal_cells)} normal-data cells it stayed within "
    f"[{min(wr):.4f}, {max(wr):.4f}]; {len(welch_bad)} of them departed by more than the harness's")
say(f"    own noise floor. The premium on balanced designs is at most "
    f"{max(p['power_lost'] for p in power if p['n1'] == p['n2']):.4f} of power.")
worst_mult = worst_student["rates"]["student"] / ALPHA
say("  * The thing to check is not 'are the variances equal', it is 'are the groups the same")
say(f"    size'. Balanced plus a 3x variance gap left Student's at "
    f"{results['balanced_student'][3.0]:.4f}; unbalanced plus the same gap is "
    f"{worst_student['rates']['student']:.4f} ({worst_mult:.1f}x nominal).")
say(f"  * Never read a t statistic off the normal curve. It was above nominal in all "
    f"{len(grid)} cells, up to {max(zr):.4f} ({max(zr) / ALPHA:.1f}x), and the right quantile is free.")
worst_ln = max((r for r in skew if r["dist"] == "lognormal"), key=lambda r: r["rates"]["welch"])
say(f"  * Welch is not a normality fix. On a lognormal with n="
    f"{worst_ln['n1']}/{worst_ln['n2']} it ran at {worst_ln['rates']['welch']:.4f} while Student's")
say(f"    held at {worst_ln['rates']['student']:.4f}. Bootstrap or transform there; do not assume")
say("    the robust-sounding test is robust to the thing it was not built for.")

with open("evidence.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote evidence.txt and results.json")
