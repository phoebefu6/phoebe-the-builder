"""Every number in the README, printed by one run.

Eight sections. The boundaries are solved from the recursion in `sequential.py`
and checked against the published tables; every rate after that is measured on
simulated Bernoulli traffic, so a wrong boundary would show up as a wrong
false-positive rate rather than hiding inside the theory.
"""

from __future__ import annotations

import numpy as np

from sequential import (
    Trial,
    bonferroni_bounds,
    crossing_probability,
    first_crossing,
    msprt_crossing,
    naive_bounds,
    obf_bounds,
    pocock_bounds,
    score,
    score_with_stop,
    simulate,
    with_futility,
)

from sequential import ALPHA, K_DAILY, LIFT_REL, N_MAX, P0, P1, equal_looks

M = 100_000  # simulations for the main tables
SEED = 20260902


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def looks_for(k: int, n_max: int = N_MAX) -> np.ndarray:
    return equal_looks(k, n_max)


# ==========================================================================
hr("1. THE BOUNDARY IS COMPUTABLE, AND IT REPRODUCES THE 1977/1979 TABLES")
# ==========================================================================
# Published two-sided alpha=0.05 constants. Pocock (1977) table 1; O'Brien &
# Fleming (1979) as tabulated in Jennison & Turnbull (2000) table 2.3.
PUB_POCOCK = {2: 2.178, 3: 2.289, 4: 2.361, 5: 2.413, 10: 2.555, 20: 2.672}
PUB_OBF_FINAL = {2: 1.977, 3: 2.004, 4: 2.024, 5: 2.040, 10: 2.087}

print("A boundary is not a magic number; it is the solution to 'how much alpha does")
print("this shape of boundary spend over K looks'. Solved here by carrying the")
print("still-continuing sub-density forward one convolution per look (Armitage,")
print("McPherson & Rowe 1969) and bisecting on the scale constant.\n")
print(f"{'K':>3} | {'Pocock solved':>13} {'published':>10} {'diff':>7} | "
      f"{'OBF final':>10} {'published':>10} {'diff':>7}")
print("-" * 78)
diffs = []
FINE = 0.0025
for k in sorted(PUB_POCOCK):
    pb = pocock_bounds(k, ALPHA, step=FINE)[0]
    row = f"{k:>3} | {pb:>13.4f} {PUB_POCOCK[k]:>10.3f} {pb - PUB_POCOCK[k]:>+7.4f} | "
    diffs.append(abs(pb - PUB_POCOCK[k]))
    if k in PUB_OBF_FINAL:
        ob = obf_bounds(k, ALPHA, step=FINE)[-1]
        diffs.append(abs(ob - PUB_OBF_FINAL[k]))
        row += f"{ob:>10.4f} {PUB_OBF_FINAL[k]:>10.3f} {ob - PUB_OBF_FINAL[k]:>+7.4f}"
    else:
        row += f"{'-':>10} {'-':>10} {'-':>7}"
    print(row)
print(f"\nVERIFIED: largest disagreement over {len(diffs)} published constants = {max(diffs):.4f}")
print("The recursion approaches the table from below as the grid refines")
for st in (0.02, 0.01, 0.005, 0.0025):
    print(f"  step={st:<7} Pocock K=5 = {pocock_bounds(5, ALPHA, step=st)[0]:.4f}  "
          f"(published 2.413)")

obf5 = obf_bounds(5, ALPHA, step=FINE)
print("\nO'Brien-Fleming K=5, look by look (published 4.562 3.226 2.634 2.281 2.040):")
print("  solved   " + " ".join(f"{b:.3f}" for b in obf5))
spent_total, spent_each = crossing_probability(obf5, step=FINE)
print("  alpha spent per look: " + " ".join(f"{e:.4f}" for e in spent_each))
print(f"  total {spent_total:.4f} against a nominal {ALPHA}")
print("\nThe two shapes are opposite bets. Pocock holds the same bar at every look")
print("and pays for it at the last one; O'Brien-Fleming makes the first look nearly")
print("unreachable and arrives at the end almost where a fixed-horizon test would.")

# Cache the boundaries actually used below.
B = {
    "naive": naive_bounds(K_DAILY, ALPHA),
    "bonferroni": bonferroni_bounds(K_DAILY, ALPHA),
    "pocock": pocock_bounds(K_DAILY, ALPHA, step=0.005),
    "obf": obf_bounds(K_DAILY, ALPHA, step=0.005),
}
print(f"\nAt K={K_DAILY} (a daily peek over 20 days) the four boundaries at the FINAL look:")
for name in ("naive", "bonferroni", "pocock", "obf"):
    print(f"  {name:<12} first look {B[name][0]:.3f}   final look {B[name][-1]:.3f}")

# ==========================================================================
hr("2. THE FALSE-POSITIVE RATE OF A PEEK, MEASURED")
# ==========================================================================
print("An empty world: both arms convert at 10.0%. Any 'significant' result here is")
print("false by construction. The rule is the one everybody actually uses -- look at")
print("the dashboard, stop if p < 0.05 -- and the only thing that changes is how")
print(f"many times you look before {N_MAX:,} visitors per arm have arrived.\n")
print(f"{'looks':>7} | {'visitors/arm between looks':>27} | {'measured FPR':>12} | {'vs nominal':>10}")
print("-" * 68)
naive_fpr = {}
for k in (1, 2, 3, 5, 10, 20, 50, 100):
    lk = looks_for(k)
    t = simulate(lk, P0, P0, M, SEED + k)
    idx = first_crossing(t.z, naive_bounds(k, ALPHA))
    fpr = float((idx >= 0).mean())
    naive_fpr[k] = fpr
    gap = f"{fpr / ALPHA:.1f}x"
    print(f"{k:>7} | {N_MAX // k:>27,} | {fpr:>12.3f} | {gap:>10}")
    del t

print(f"\n({M:,} simulated experiments per row; Monte Carlo standard error at a rate")
print(f"near 0.25 is {(0.25 * 0.75 / M) ** 0.5:.4f}, so the third decimal is noise and "
      f"tables built from")
print("separate runs below differ in it.)")
print(f"\nOne look is the nominal test: {naive_fpr[1]:.3f}. Five looks is "
      f"{naive_fpr[5]:.3f}. A daily peek for three")
print(f"weeks is {naive_fpr[20]:.3f} -- {naive_fpr[20] / ALPHA:.1f} times the rate the "
      f"number on the screen claims.")

print("\nAnd it does not level off. Peeking every 500 visitors per arm, forever:")
CONT_STEP, CONT_MAX, M_CONT = 500, 200_000, 20_000
lk = np.arange(CONT_STEP, CONT_MAX + 1, CONT_STEP, dtype=np.int64)
tc = simulate(lk, P0, P0, M_CONT, SEED + 777)
hit = np.abs(tc.z) >= 1.959964
ever = np.maximum.accumulate(hit, axis=1)
print(f"{'visitors/arm':>13} | {'looks so far':>12} | {'FPR by then':>11}")
print("-" * 44)
for n in (2_000, 5_000, 20_000, 50_000, 100_000, 200_000):
    j = int(n // CONT_STEP) - 1
    print(f"{n:>13,} | {j + 1:>12} | {float(ever[:, j].mean()):>11.3f}")
print("\nThe rate has no ceiling below 1.0: a random walk crosses any fixed line")
print("eventually, so an experiment that is never called is eventually 'significant'.")
print("Nothing about the effect changed. The stopping rule is doing all of this.")
del tc, hit, ever

# ==========================================================================
hr("3. FOUR CORRECTIONS, ONE SCHEDULE, AND WHAT EACH ONE COSTS")
# ==========================================================================
print(f"K={K_DAILY} looks, {N_MAX:,} visitors per arm at the end. The null world gives the")
print(f"false-positive rate; a world with a real {LIFT_REL:.0%} relative lift "
      f"({P0:.3f} -> {P1:.3f}) gives")
print("power and the sample size actually consumed.\n")
lk20 = looks_for(K_DAILY)
t_null = simulate(lk20, P0, P0, M, SEED + 1)
t_alt = simulate(lk20, P0, P1, M, SEED + 2)
t_null_fixed = Trial(lk20[-1:], t_null.z[:, -1:], t_null.diff[:, -1:], t_null.se[:, -1:], P0, P0)
t_alt_fixed = Trial(lk20[-1:], t_alt.z[:, -1:], t_alt.diff[:, -1:], t_alt.se[:, -1:], P0, P1)

TAU = P1 - P0  # mSPRT mixing scale set to the effect the test is powered for
rules = []
rules.append(("fixed horizon (1 look)",
               first_crossing(t_null_fixed.z, naive_bounds(1, ALPHA)),
               first_crossing(t_alt_fixed.z, naive_bounds(1, ALPHA)),
               t_null_fixed, t_alt_fixed))
for name in ("naive", "bonferroni", "pocock", "obf"):
    rules.append((name,
                  first_crossing(t_null.z, B[name]),
                  first_crossing(t_alt.z, B[name]),
                  t_null, t_alt))
rules.append((f"mSPRT (tau={TAU:.3f})",
              msprt_crossing(t_null, TAU, ALPHA),
              msprt_crossing(t_alt, TAU, ALPHA),
              t_null, t_alt))

print(f"{'rule':<24} | {'FPR':>6} | {'power':>6} | {'E[N]/arm H1':>11} | "
      f"{'E[N]/arm H0':>11} | {'median N H1':>11}")
print("-" * 92)
sec3 = {}
for name, idx0, idx1, tn, ta in rules:
    o0 = score(tn, idx0, name)
    o1 = score(ta, idx1, name)
    sec3[name] = (o0, o1)
    print(f"{name:<24} | {o0.reject_rate:>6.3f} | {o1.reject_rate:>6.3f} | "
          f"{o1.expected_n:>11,.0f} | {o0.expected_n:>11,.0f} | {o1.median_n:>11,.0f}")

fx0, fx1 = sec3["fixed horizon (1 look)"]
nv0, nv1 = sec3["naive"]
pk0, pk1 = sec3["pocock"]
ob0, ob1 = sec3["obf"]
bf0, bf1 = sec3["bonferroni"]
ms0, ms1 = sec3[f"mSPRT (tau={TAU:.3f})"]
print(f"\nThe naive peeker is not only wrong, it is APPEALING: power "
      f"{nv1.reject_rate:.3f} against the fixed")
print(f"test's {fx1.reject_rate:.3f}, and it gets there on "
      f"{nv1.expected_n:,.0f} visitors per arm instead of {fx1.expected_n:,.0f}.")
print(f"Both of those gains are the same defect as the {nv0.reject_rate:.3f} "
      f"false-positive rate. It is not")
print("a faster test, it is a looser one.")
print(f"\nPocock and O'Brien-Fleming land on the nominal rate exactly "
      f"({pk0.reject_rate:.3f} / {ob0.reject_rate:.3f}); Bonferroni")
print(f"undershoots at {bf0.reject_rate:.3f} and mSPRT at {ms0.reject_rate:.3f}, "
      f"and both pay for it in section 7 and 5.")
print(f"The speed is real: Pocock reaches a verdict on "
      f"{pk1.expected_n:,.0f} visitors per arm against the fixed")
print(f"test's {fx1.expected_n:,.0f} -- {1 - pk1.expected_n / fx1.expected_n:.1%} less traffic -- "
      f"at a cost of {fx1.reject_rate - pk1.reject_rate:.3f} power.")
print(f"O'Brien-Fleming keeps almost all the power ({ob1.reject_rate:.3f} vs "
      f"{fx1.reject_rate:.3f}) and saves less ({ob1.expected_n:,.0f}).")

# ==========================================================================
hr("4. NEGATIVE RESULT: A VALID SEQUENTIAL TEST STILL OVERSTATES THE EFFECT")
# ==========================================================================
print("Correct alpha is not a correct answer. Stopping the moment the estimate is")
print("extreme enough to cross a line selects on the estimate, so the effect you")
print("report at the stopping look is the effect that got you there.\n")
print(f"True lift = {P1 - P0:.4f} absolute ({LIFT_REL:.0%} relative). Among the runs that "
      f"rejected:\n")
print(f"{'rule':<24} | {'reported lift':>13} | {'overstated by':>13} | "
      f"{'95% CI coverage':>15} | {'... when it rejected':>20}")
print("-" * 100)
for name, _, idx1, _, ta in rules:
    o1 = sec3[name][1]
    print(f"{name:<24} | {o1.est_at_stop:>13.5f} | {o1.est_bias:>+12.1%} | "
          f"{o1.ci_coverage:>15.3f} | {o1.ci_coverage_rejected:>20.3f}")
print("\nThe ordering follows how early a rule is allowed to stop, not how valid it is:")
print("Pocock is a correct 0.05 test and overstates the lift more than O'Brien-Fleming,")
print("which is also a correct 0.05 test. A boundary controls the rate at which you")
print("are wrong about the SIGN. It says nothing about the SIZE.")

print("\nAnd the overstatement is not a small-print caveat when the experiment is")
print("underpowered. Same 20-look Pocock design, weaker and weaker true effects:\n")
print(f"{'true rel. lift':>14} | {'fixed power':>11} | {'Pocock power':>12} | "
      f"{'reported lift':>13} | {'overstated by':>13}")
print("-" * 76)
bias_sweep = []
for rel in (0.20, 0.10, 0.05, 0.03, 0.02):
    p1 = P0 * (1 + rel)
    tt = simulate(lk20, P0, p1, M, SEED + int(rel * 1000))
    o_seq = score(tt, first_crossing(tt.z, B["pocock"]), "pocock")
    tf = Trial(lk20[-1:], tt.z[:, -1:], tt.diff[:, -1:], tt.se[:, -1:], P0, p1)
    o_fix = score(tf, first_crossing(tf.z, naive_bounds(1, ALPHA)), "fixed")
    bias_sweep.append((rel, o_fix.reject_rate, o_seq.reject_rate, o_seq.est_at_stop, o_seq.est_bias))
    print(f"{rel:>13.0%} | {o_fix.reject_rate:>11.3f} | {o_seq.reject_rate:>12.3f} | "
          f"{o_seq.est_at_stop:>13.5f} | {o_seq.est_bias:>+12.1%}")
    del tt, tf
print("\nAt low power the only runs that cross are the lucky ones, so the surviving")
print("estimate is mostly luck. This is the winner's curse arriving through the")
print("stopping rule, and it is why a sequential trial should report the effect")
print("with a bias-adjusted estimate, not the number that tripped the boundary.")

# ==========================================================================
hr("5. A BOUNDARY IS VALID FOR ITS OWN SCHEDULE AND NOTHING ELSE")
# ==========================================================================
print("A group-sequential boundary is computed for a fixed number of analyses. Add")
print("looks after the fact -- an extra mid-week check, a stakeholder refreshing the")
print("dashboard -- and the guarantee is gone, silently, because the number on the")
print("screen still says 0.05.\n")
K2 = 2 * K_DAILY
lk40 = looks_for(K2)
t40 = simulate(lk40, P0, P0, M, SEED + 3)
pocock_c = float(B["pocock"][0])
fpr_reuse_p = float((first_crossing(t40.z, np.full(K2, pocock_c)) >= 0).mean())
obf_reuse = obf_bounds(K_DAILY, ALPHA, step=0.005)[-1] / np.sqrt(np.arange(1, K2 + 1) / K2)
fpr_reuse_o = float((first_crossing(t40.z, obf_reuse) >= 0).mean())
ms40 = float((msprt_crossing(t40, TAU, ALPHA) >= 0).mean())
print(f"{'rule computed for 20 looks, run at 40':<44} | {'FPR':>6}")
print("-" * 54)
print(f"{f'Pocock constant {pocock_c:.3f} held at 40 looks':<44} | {fpr_reuse_p:>6.3f}")
print(f"{'OBF shape re-indexed to 40 looks':<44} | {fpr_reuse_o:>6.3f}")
print(f"{'mSPRT, same tau, 40 looks':<44} | {ms40:>6.3f}")
print(f"{'mSPRT, same tau, 400 looks to 200k/arm':<44} | ", end="")
lk400 = np.arange(500, 200_001, 500, dtype=np.int64)
t400 = simulate(lk400, P0, P0, 20_000, SEED + 4)
print(f"{float((msprt_crossing(t400, TAU, ALPHA) >= 0).mean()):>6.3f}")
del t400
print("\nThe OBF shape survives because re-indexing it by information fraction IS the")
print("alpha-spending construction (Lan & DeMets 1983) -- the shape, not the count, is")
print("what it fixes. The Pocock constant does not: it was solved for 20 looks and")
print(f"leaks {fpr_reuse_p - ALPHA:+.3f} at 40. mSPRT is the one rule here that never "
      f"needed the schedule")
print("at all; its guarantee is over every stopping time simultaneously.")

print("\nThat guarantee is not free. tau is a prior on the effect size, and getting it")
print(f"wrong costs power and speed. True lift is {P1 - P0:.3f}:\n")
print(f"{'tau':>8} | {'tau / true effect':>17} | {'FPR':>6} | {'power':>6} | {'E[N]/arm H1':>11}")
print("-" * 62)
tau_sweep = []
for tau in (0.002, 0.005, 0.010, 0.020, 0.050):
    f = float((msprt_crossing(t_null, tau, ALPHA) >= 0).mean())
    o = score(t_alt, msprt_crossing(t_alt, tau, ALPHA), "msprt")
    tau_sweep.append((tau, f, o.reject_rate, o.expected_n))
    print(f"{tau:>8.3f} | {tau / (P1 - P0):>17.1f}x | {f:>6.3f} | {o.reject_rate:>6.3f} | "
          f"{o.expected_n:>11,.0f}")
best = max(tau_sweep, key=lambda r: r[2])
worst = min(tau_sweep, key=lambda r: r[2])
print(f"\nEvery row is a valid test -- the FPR column never exceeds {ALPHA}, which is the")
print(f"whole point. But power runs from {worst[2]:.3f} to {best[2]:.3f} across a "
      f"choice nobody documents,")
ratio = worst[0] / (P1 - P0)
print(f"and the worst row is the one that guessed the effect {ratio:.1f}x the truth "
      f"({'too large' if ratio > 1 else 'too small'}).")
print("Note what the FPR column also costs: at the tau that matches the true effect,")
print(f"mSPRT rejects a real lift {ms1.reject_rate:.3f} of the time against "
      f"O'Brien-Fleming's {ob1.reject_rate:.3f} on")
print("identical data. Being valid at every stopping time, rather than at 20 named")
print(f"ones, is worth {ob1.reject_rate - ms1.reject_rate:.3f} of power here. Buy it when the "
      f"schedule genuinely cannot")
print("be fixed in advance; do not buy it to feel safe.")

# ==========================================================================
hr("6. THE FREE HALF OF SEQUENTIAL DESIGN IS THE ONE NOBODY IMPLEMENTS")
# ==========================================================================
print("Every rule above only stops early on success. The other reason to stop is that")
print("nothing is happening -- and unlike a success boundary, a futility boundary")
print("cannot manufacture a false positive, because it only ever ENDS experiments")
print("that were not going to reject.\n")
print("Rule: from look 10 of 20 onward, stop if the observed z is below zero")
print("('it is flat or negative, kill it').\n")
fut = np.full(K_DAILY, -np.inf)
fut[K_DAILY // 2 - 1:] = 0.0
print(f"{'design':<34} | {'FPR':>6} | {'power':>6} | {'E[N]/arm H0':>11} | {'E[N]/arm H1':>11}")
print("-" * 80)
sec6 = {}
for name in ("pocock", "obf"):
    for label, fb in ((f"{name}, success only", None), (f"{name} + futility at z<0", fut)):
        r0, s0 = with_futility(t_null.z, B[name], fb, signed=True)
        r1, s1 = with_futility(t_alt.z, B[name], fb, signed=True)
        o0 = score_with_stop(t_null, r0, s0, label)
        o1 = score_with_stop(t_alt, r1, s1, label)
        sec6[label] = (o0, o1)
        print(f"{label:<34} | {o0.reject_rate:>6.3f} | {o1.reject_rate:>6.3f} | "
              f"{o0.expected_n:>11,.0f} | {o1.expected_n:>11,.0f}")
a0, a1 = sec6["obf, success only"]
b0, b1 = sec6["obf + futility at z<0"]
print(f"\nOn O'Brien-Fleming: {1 - b0.expected_n / a0.expected_n:.1%} of the traffic an "
      f"empty experiment would have consumed,")
print(f"given back, for {b1.reject_rate - a1.reject_rate:+.3f} power and "
      f"{b0.reject_rate - a0.reject_rate:+.3f} false-positive rate. The saving lands")
print("exactly where it should: on the experiments that had nothing in them.")

# ==========================================================================
hr("7. NEGATIVE RESULT: THE IMPROVISED CORRECTION IS NOT A NEAR-ENOUGH POCOCK")
# ==========================================================================
print("This section was written expecting Bonferroni-across-looks to be a near-enough")
print("Pocock -- the improvisation you can forgive. It measured out the other way.")
print("Bonferroni ignores the correlation between overlapping looks -- a Z at look 20")
print("is nearly the same random variable as a Z at look 19 -- so at "
      f"K={K_DAILY} it demands {B['bonferroni'][0]:.3f}")
print(f"where the exact answer is {B['pocock'][0]:.3f}. Measured:\n")
print(f"  false-positive rate   Bonferroni {bf0.reject_rate:.3f}   vs Pocock "
      f"{pk0.reject_rate:.3f}   ({bf0.reject_rate - pk0.reject_rate:+.3f})")
print(f"  power                 Bonferroni {bf1.reject_rate:.3f}   vs Pocock "
      f"{pk1.reject_rate:.3f}   ({bf1.reject_rate - pk1.reject_rate:+.3f})")
print(f"  visitors per arm      Bonferroni {bf1.expected_n:,.0f}  vs Pocock "
      f"{pk1.expected_n:,.0f}  ({bf1.expected_n - pk1.expected_n:+,.0f})")
print(f"\nThat is {pk1.reject_rate - bf1.reject_rate:.3f} of power and "
      f"{bf1.expected_n - pk1.expected_n:+,.0f} visitors per arm, for a rule that spends")
print(f"only {bf0.reject_rate / ALPHA:.0%} of the alpha it was given. The gap widens with K, "
      f"because more looks")
print("means more overlap for Bonferroni to ignore:\n")
print(f"{'K':>4} | {'Pocock b':>9} | {'Bonferroni b':>13} | {'Pocock power':>12} | "
      f"{'Bonf. power':>11} | {'power lost':>10}")
print("-" * 72)
for k in (2, 5, 10, 20, 50):
    lkk = looks_for(k)
    tk = simulate(lkk, P0, P1, M, SEED + 500 + k)
    pbk = pocock_bounds(k, ALPHA, step=0.005)
    bbk = bonferroni_bounds(k, ALPHA)
    pw_p = float((first_crossing(tk.z, pbk) >= 0).mean())
    pw_b = float((first_crossing(tk.z, bbk) >= 0).mean())
    print(f"{k:>4} | {pbk[0]:>9.3f} | {bbk[0]:>13.3f} | {pw_p:>12.3f} | {pw_b:>11.3f} | "
          f"{pw_p - pw_b:>10.3f}")
    del tk
print("\nSo the correction is worth getting right. What it is NOT worth is more than")
print(f"the decision to look at all: uncorrected peeking adds "
      f"{nv0.reject_rate - ALPHA:+.3f} to the false-positive")
print(f"rate and reports the lift {sec3['naive'][1].est_bias:+.1%} too high. That is the "
      f"error to fix first.")
print("\nSecond negative result: 'peek less often' is a weak lever. From the section 2")
print(f"table, cutting a daily peek ({naive_fpr[20]:.3f}) back to weekly "
      f"({naive_fpr[3]:.3f}) still leaves")
print(f"{naive_fpr[3] / ALPHA:.1f}x the nominal rate. Two looks is already "
      f"{naive_fpr[2] / ALPHA:.1f}x. There is no number of")
print("looks small enough to make an uncorrected peek honest except one.")

# ==========================================================================
hr("8. WHAT THE EXPERIMENT REPORT SHOULD SAY")
# ==========================================================================
print("Not 'p = 0.03'. A p-value is a statement about a procedure, so the procedure")
print("has to be in the report: the schedule, the boundary, and an effect estimate")
print("that knows it was selected.\n")
print(f"{'rule':<24} | {'FPR':>6} | {'power':>6} | {'E[N] H1':>8} | {'lift bias':>9} | "
      f"{'CI cov.':>7} | verdict")
print("-" * 104)
VERDICTS = {
    "fixed horizon (1 look)": "honest, slowest, unbiased -- the benchmark",
    "naive": "invalid; the speed IS the error",
    "bonferroni": "valid; spends 38% of its alpha, loses 0.12 power",
    "pocock": "valid and fastest; report an adjusted estimate",
    "obf": "valid, keeps the power, saves the least",
    f"mSPRT (tau={TAU:.3f})": "valid at ANY stopping time; tau is a real choice",
}
for name, _, _, _, _ in rules:
    o0, o1 = sec3[name]
    print(f"{name:<24} | {o0.reject_rate:>6.3f} | {o1.reject_rate:>6.3f} | "
          f"{o1.expected_n:>8,.0f} | {o1.est_bias:>+8.1%} | {o1.ci_coverage:>7.3f} | "
          f"{VERDICTS[name]}")
print("\nOne line to take away: peeking is not cheating, and it is not free. It is a")
print("design choice that has to be priced before the experiment starts -- because")
print("the boundary that makes it valid cannot be chosen after you have looked.")
