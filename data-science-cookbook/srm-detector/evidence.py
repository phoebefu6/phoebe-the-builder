"""Sample ratio mismatch, measured.

Eight sections. Every number printed here is asserted in ``test_srm.py`` and
every one is produced from numpy and scipy on a world whose true effect and
true bias are known in closed form - there is no data file and no network.

Run: ``python evidence.py``
"""

from __future__ import annotations

import numpy as np
import srm

RNG_SEED = 20260903
TRIALS = 6000
EXACT_TRIALS = 600
DISAGREE_SEEDS = 8
DISAGREE_TRIALS = 500

W = srm.World()
A_REFLEX = srm.ALPHA_REFLEX
A_PLATFORM = srm.ALPHA_PLATFORM


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("SAMPLE RATIO MISMATCH")
    print(f"World: {W.per_arm:,} per arm intended, base rate {W.base_rate:.4f}, "
          f"true relative lift {W.true_rel_lift:.1%}")
    print(f"       {W.low_share:.0%} of users are low-intent (convert at {W.p_low:.1%}); "
          f"the rest convert at {W.p_high:.4%}")
    print("Thresholds in circulation: 0.05 (the reflex) and 0.0005 (what large "
          "platforms publish)")

    # ---------------------------------------------------------------- 1
    rule("1. THE TEST IS NOT THE DECISION - five of them give the same answer")

    print(f"chi-square critical value at 0.05   = {srm.chi2_critical(A_REFLEX):.4f}  (published 3.8415)")
    print(f"chi-square critical value at 0.0005 = {srm.chi2_critical(A_PLATFORM):.4f}  (published 12.1157)")

    n_check, dev_check = 200_000, 0.003
    a_check = int(round(n_check * (0.5 + dev_check)))
    z = (a_check / n_check - 0.5) / np.sqrt(0.25 / n_check)
    identity_gap = abs(srm.chi2_stat(a_check, n_check - a_check) - z * z)
    print(f"z^2 == chi-square statistic, exactly: gap = {identity_gap:.2e}  "
          "(so 'z-test vs chi-square' is not a choice)")

    worst_log = 0.0
    for n in (100, 1_000, 10_000, 200_000):
        for dev in (0.0, 0.002, 0.005, 0.01, 0.02):
            a = int(round(n * (0.5 + dev)))
            pc = max(srm.p_chi2(a, n - a), 1e-300)
            pe = max(srm.p_binom_exact(a, n - a), 1e-300)
            worst_log = max(worst_log, abs(np.log10(pc) - np.log10(pe)))
    print(f"chi-square vs EXACT binomial, worst |log10 p ratio| over that grid = {worst_log:.4f}")
    print(f"  -> p-values within a factor of {10 ** worst_log:.2f} everywhere; the exact test costs "
          "O(n) per call and buys nothing")

    d_healthy = srm.simulate(W, "healthy", 0.0, TRIALS, rng)
    d_broken = srm.simulate(W, "mcar_loss", 0.015, TRIALS, rng)
    print()
    print(f"{'detector':<20} {'fires on healthy':>17} {'fires on 1.5% loss':>19}   (alpha = 0.0005)")
    both = list(srm.DETECTORS.items()) + list(srm.EXACT.items())
    fire = {}
    for name, fn in both:
        limit = EXACT_TRIALS if name in srm.EXACT else None
        f0 = srm.flag_rate(d_healthy["n_ctrl"], d_healthy["n_trt"], fn, A_PLATFORM, limit=limit)
        f1 = srm.flag_rate(d_broken["n_ctrl"], d_broken["n_trt"], fn, A_PLATFORM, limit=limit)
        fire[name] = (f0, f1)
        print(f"{name:<20} {f0:>17.4f} {f1:>19.4f}")
    # Comparing flag RATES across different trial counts would measure Monte Carlo
    # error and call it a difference between tests. Compare the verdicts instead,
    # trial by trial, and count the trials on which any two of the five disagree.
    disagree = 0
    checked = 0
    for seed in range(DISAGREE_SEEDS):
        rd = np.random.default_rng(500 + seed)
        dd = srm.simulate(W, "mcar_loss", 0.015, DISAGREE_TRIALS, rd)
        for i in range(DISAGREE_TRIALS):
            a, b = int(dd["n_ctrl"][i]), int(dd["n_trt"][i])
            verdicts = {
                srm.p_chi2(a, b) < A_PLATFORM,
                srm.p_chi2_yates(a, b) < A_PLATFORM,
                srm.p_g_test(a, b) < A_PLATFORM,
                srm.p_normal_z(a, b) < A_PLATFORM,
                srm.p_binom_exact(a, b) < A_PLATFORM,
            }
            checked += 1
            if len(verdicts) > 1:
                disagree += 1
    spread = disagree / checked
    print()
    print(f"NEGATIVE RESULT: over {checked:,} trials, any two of the five statistical tests reach")
    print(f"  a different verdict on {disagree} of them - {spread:.4%}, and only ever where the")
    print(f"  p-value is already within a factor of {10 ** worst_log:.2f} of the threshold. Yates, the")
    print("  G-test and the O(n) exact binomial are the plain chi-square with extra steps.")
    print("  Which SRM test to use is not a question worth a meeting. Which THRESHOLD, and")
    print("  what it is pointed at, is the whole thing.")

    # ---------------------------------------------------------------- 2
    rule("2. '49.3 / 50.7' IS NOT A FINDING - it is a finding only with n attached")

    print(f"{'n':>12} {'chi-square p':>15} {'exact p':>15}  verdict at 0.0005")
    for n in (1_000, 10_000, 100_000, 1_000_000, 10_000_000):
        a = int(round(n * 0.493))
        pc = srm.p_chi2(a, n - a)
        # The exact test is O(n) per call, so it is skipped where it would cost a
        # second and agree to two significant figures anyway (see section 1).
        pe = f"{srm.p_binom_exact(a, n - a):.3e}" if n <= 1_000_000 else "(skipped)"
        verdict = "MISMATCH" if pc < A_PLATFORM else "consistent"
        print(f"{n:>12,} {pc:>15.3e} {pe:>15}  {verdict}")

    cross = {}
    for alpha in (A_REFLEX, A_PLATFORM):
        lo, hi = 100, 50_000_000
        for _ in range(60):
            mid = (lo + hi) // 2
            a = int(round(mid * 0.493))
            if srm.p_chi2(a, mid - a) < alpha:
                hi = mid
            else:
                lo = mid
        cross[alpha] = hi
    print()
    print(f"The SAME split, 49.3/50.7, crosses 0.05 at n = {cross[A_REFLEX]:,} and "
          f"0.0005 at n = {cross[A_PLATFORM]:,}.")
    print("Below that it is the healthiest thing in the report; above it, the experiment is void.")
    print("The percentage on the dashboard carries none of that. It is a ratio, not a test.")

    # ---------------------------------------------------------------- 3
    rule("3. 'WITHIN 1%' NAMES TWO RULES - one inert, one uncontrolled")

    dev_ratio = 1.01 / 2.01 - 0.5
    print(f"'share within 1 point of 50%'      -> tolerates a share deviation of {0.01:.5f}")
    print(f"'arm ratio within 1% of 1.00'      -> tolerates a share deviation of {dev_ratio:.5f}"
          f"   ({0.01 / dev_ratio:.1f}x tighter)")
    f_abs_h = srm.flag_rate(d_healthy["n_ctrl"], d_healthy["n_trt"], srm.eyeball_abs, 0.5)
    f_rat_h = srm.flag_rate(d_healthy["n_ctrl"], d_healthy["n_trt"], srm.eyeball_ratio, 0.5)
    f_abs_b = srm.flag_rate(d_broken["n_ctrl"], d_broken["n_trt"], srm.eyeball_abs, 0.5)
    f_rat_b = srm.flag_rate(d_broken["n_ctrl"], d_broken["n_trt"], srm.eyeball_ratio, 0.5)
    print()
    print(f"{'rule':<24} {'fires on healthy':>17} {'fires on 1.5% loss':>19}")
    print(f"{'outside 49/51 (share)':<24} {f_abs_h:>17.4f} {f_abs_b:>19.4f}")
    print(f"{'ratio outside 0.99-1.01':<24} {f_rat_h:>17.4f} {f_rat_b:>19.4f}")
    print(f"{'chi-square at 0.0005':<24} {fire['chi2'][0]:>17.4f} {fire['chi2'][1]:>19.4f}")
    print()
    print(f"The share rule fires {f_abs_b:.0%} of the time on a mismatch the chi-square test sees "
          f"{fire['chi2'][1]:.0%} of the time:")
    print("  at 100k per arm it is inert, and it stays inert at 10,000,000 where the p-value is 0.")
    print(f"The ratio rule is a real detector with a {f_rat_h:.2%} false-alarm rate - "
          f"{f_rat_h / A_PLATFORM:,.0f}x the")
    print("  0.0005 it is standing in for. Two rules a team would describe with the same")
    print("  sentence, failing in opposite directions.")

    # ---------------------------------------------------------------- 4
    rule("4. THE HEALTH CHECK IS 6x MORE SENSITIVE THAN THE EXPERIMENT IT PROTECTS")

    print(f"{'per arm':>10} {'min detectable lift':>20} {'min detectable split dev':>26} {'ratio':>8}")
    ratios = {}
    for m in (5_000, 25_000, 100_000, 1_000_000):
        mde = srm.mde_rel_lift(m, W.base_rate, A_REFLEX)
        dev = srm.mdd_share(2 * m, A_REFLEX)
        rel_dev = dev / 0.5
        ratios[m] = mde / rel_dev
        print(f"{m:>10,} {mde:>19.2%} {rel_dev:>25.3%} {ratios[m]:>7.2f}x")
    span = max(ratios.values()) - min(ratios.values())
    print()
    print(f"Both are 80%-power thresholds at alpha 0.05. The ratio is {ratios[100_000]:.2f}x and it "
          f"moves {span:.3f}")
    print("across a 200-fold change in sample size - it is a constant of the design, not a")
    print("property of the data, because both thresholds are the same multiple of 1/sqrt(n).")
    print("At the 0.0005 threshold the ratio is "
          f"{srm.mde_rel_lift(100_000, W.base_rate, A_REFLEX) / (srm.mdd_share(200_000, A_PLATFORM) / 0.5):.2f}x, "
          "so even a 100x stricter alpha leaves the")
    print("health check the more sensitive of the two instruments by a wide margin.")

    # ---------------------------------------------------------------- 5
    rule("5. NEGATIVE RESULT: 6x MORE SENSITIVE, AND STILL NOT SENSITIVE ENOUGH")

    print("The detectable mismatch shrinks with n. The BIAS a mismatch causes does not -")
    print("it is a property of who went missing, not of how many users the test had.")
    print()
    print(f"{'per arm':>10} {'alpha':>8} {'min detectable loss':>20} {'bias it already carries':>25}")
    band = {}
    for m in (5_000, 25_000, 100_000, 1_000_000):
        for alpha in (A_REFLEX, A_PLATFORM):
            dev = srm.mdd_share(2 * m, alpha)
            loss = srm.loss_for_share_deviation(dev)
            rate = min(loss / W.low_share, 1.0)
            bias = (srm.analytic_est_lift(W, "selective_loss", rate) - W.true_rel_lift) / W.true_rel_lift
            band[(m, alpha)] = (loss, bias)
            print(f"{m:>10,} {alpha:>8} {loss:>19.2%} {bias:>24.1%}")
    print()
    loss25, bias25 = band[(25_000, A_PLATFORM)]
    loss1m, bias1m = band[(1_000_000, A_PLATFORM)]
    print("At 25,000 per arm - an ordinary test - the smallest mismatch the platform threshold")
    print(f"can reliably catch is {loss25:.2%} of one arm, and a selective loss that size already")
    print(f"overstates the effect by {bias25:.0%}. Everything below it is invisible, and invisible")
    print("is not the same as harmless.")
    print(f"The check only becomes protective around 1,000,000 per arm ({loss1m:.2%} loss, "
          f"{bias1m:.0%} bias),")
    print("which is not the size of most experiments.")

    # ---------------------------------------------------------------- 6
    rule("6. NEGATIVE RESULT: A PASSING CHECK IS NOT EVIDENCE THE ARMS ARE COMPARABLE")

    print("Same selective loss in treatment, plus the identical NUMBER of users removed from")
    print("control at random - the sort of thing a bot filter or a dedup step does. The counts")
    print("come out even. Nothing else does.")
    print()
    print(f"{'mechanism':<22} {'count loss':>11} {'flags @0.05':>12} {'flags @0.0005':>14} "
          f"{'reported lift':>14} {'bias':>8}")
    rows = {}
    for mech, rate in (("healthy", 0.0), ("mcar_loss", 0.015), ("selective_loss", 0.05),
                       ("balanced_selective", 0.05)):
        d = srm.simulate(W, mech, rate, TRIALS, rng)
        p = srm.vector_p_chi2(d["n_ctrl"], d["n_trt"])
        est = float(d["est_rel_lift"].mean())
        bias = (est - W.true_rel_lift) / W.true_rel_lift
        cl = srm.count_loss_of(W, mech, rate)
        rows[mech] = {"f05": float((p < A_REFLEX).mean()), "f0005": float((p < A_PLATFORM).mean()),
                      "est": est, "bias": bias, "loss": cl}
        print(f"{mech:<22} {cl:>10.2%} {rows[mech]['f05']:>12.4f} {rows[mech]['f0005']:>14.4f} "
              f"{est:>13.3%} {bias:>+8.1%}")
    print()
    bs = rows["balanced_selective"]
    print(f"The balanced case fires {bs['f05']:.4f} of the time at 0.05 - that is the NULL "
          f"({A_REFLEX}), not a")
    print(f"weak signal - while the reported lift is {bs['est']:.3%} against a true "
          f"{W.true_rel_lift:.1%}, an overstatement")
    print(f"of {bs['bias']:.0%}. An SRM test is a test of COUNTS. Exchangeability is what the")
    print("experiment needs, and equal counts are consistent with any amount of its absence.")

    # ---------------------------------------------------------------- 7
    rule("7. THE VERDICT CARRIES NO INFORMATION ABOUT THE HARM")

    mc, sl = rows["mcar_loss"], rows["selective_loss"]
    print(f"Two mechanisms, the same {mc['loss']:.2%} of the treatment arm missing:")
    print(f"  records dropped at random   flags @0.05 {mc['f05']:.4f}   bias {mc['bias']:+.1%}")
    print(f"  low-intent users bounced    flags @0.05 {sl['f05']:.4f}   bias {sl['bias']:+.1%}")
    print()
    print(f"The detector cannot tell them apart - {abs(mc['f05'] - sl['f05']):.4f} apart in flag rate,")
    print("because a detector sees two integers and a mechanism is not two integers. One of")
    print(f"these experiments is fine after reweighting; the other overstates its effect by "
          f"{sl['bias']:.0%}.")
    print("So the p-value is a trigger to go and find out WHO is missing. It is not a")
    print("severity score, and it must not be read as one.")

    # ---------------------------------------------------------------- 8
    rule("8. WHERE TO POINT IT: SEGMENTS, AND WHY THE THRESHOLD IS 0.0005")

    segs = srm.DEFAULT_SEGMENTS
    print("One broken segment (safari, "
          f"{segs[2].share:.0%} of traffic), loss confined to it:")
    print()
    print(f"{'segment loss':>13} {'aggregate @0.05':>17} {'aggregate @0.0005':>19} "
          f"{'per-segment, Bonferroni @0.0005':>33}")
    seg_rows = {}
    for L in (0.02, 0.04, 0.06, 0.10, 0.20):
        d = srm.simulate_segmented(W.per_arm, segs, "safari", L, TRIALS // 2, rng)
        agg = srm.vector_p_chi2(d["n_ctrl"], d["n_trt"])
        ps = np.vstack([srm.vector_p_chi2(d["n_ctrl_seg"][i], d["n_trt_seg"][i])
                        for i in range(len(segs))])
        seg_rows[L] = {
            "agg05": float((agg < A_REFLEX).mean()),
            "agg0005": float((agg < A_PLATFORM).mean()),
            "bonf": float((ps.min(axis=0) < A_PLATFORM / len(segs)).mean()),
        }
        r = seg_rows[L]
        print(f"{L:>12.0%} {r['agg05']:>17.3f} {r['agg0005']:>19.3f} {r['bonf']:>33.3f}")

    d0 = srm.simulate_segmented(W.per_arm, segs, None, 0.0, TRIALS, rng)
    ps0 = np.vstack([srm.vector_p_chi2(d0["n_ctrl_seg"][i], d0["n_trt_seg"][i])
                     for i in range(len(segs))])
    fp_any = float((ps0.min(axis=0) < A_REFLEX).mean())
    fp_bonf = float((ps0.min(axis=0) < A_PLATFORM / len(segs)).mean())
    r6 = seg_rows[0.06]
    print()
    print(f"At a 6% loss inside one segment the aggregate check sees it {r6['agg0005']:.3f} of the time")
    print(f"at 0.0005 and {r6['agg05']:.3f} at 0.05; splitting by segment and paying Bonferroni sees it")
    print(f"{r6['bonf']:.3f} - a {r6['bonf'] / max(r6['agg0005'], 1e-9):.0f}x improvement for three extra "
          "chi-square calls. And the")
    print("correction is not the expensive part: on a healthy world the Bonferroni-corrected")
    print(f"segment sweep false-alarms {fp_bonf:.4f} of the time, against {fp_any:.4f} for the same")
    print("three tests at an uncorrected 0.05.")

    print()
    print("Two independent reasons the published threshold is 0.0005 and not 0.05:")
    print()
    print(f"  (a) volume. {fp_any:.4f} of healthy tests false-alarm on a 3-segment sweep at 0.05.")
    n_tests, n_slices = 500, 20
    print(f"      At {n_tests} experiments a year x {n_slices} slices that is "
          f"{n_tests * n_slices * A_REFLEX:,.0f} false alarms at 0.05")
    print(f"      and {n_tests * n_slices * A_PLATFORM:,.1f} at 0.0005. A health check nobody believes "
          "is not a health check.")
    print()
    print("  (b) the stopping rule. Day 164 priced optional stopping for the effect test; the")
    print("      identical arithmetic applies to the health check, which is looked at daily and")
    print("      corrected never:")
    print()
    print(f"{'looks':>7} {'realized FPR @0.05':>20} {'realized FPR @0.0005':>22}")
    seq = {}
    for looks in (1, 5, 20):
        f05 = srm.sequential_srm_fpr(2 * W.per_arm, looks, A_REFLEX, TRIALS, rng)
        f0005 = srm.sequential_srm_fpr(2 * W.per_arm, looks, A_PLATFORM, TRIALS, rng)
        seq[looks] = (f05, f0005)
        print(f"{looks:>7} {f05:>20.4f} {f0005:>22.4f}")
    print()
    print(f"      A daily SRM check for three weeks is a {seq[20][0]:.3f} test at 0.05 - "
          f"{seq[20][0] / A_REFLEX:.1f}x nominal, and")
    print(f"      the same shape Day 164 found for the effect. At 0.0005 it lands at "
          f"{seq[20][1]:.4f}:")
    print(f"      still {seq[20][1] / A_PLATFORM:.0f}x its nominal alpha, but {seq[20][0] / seq[20][1]:.0f}x "
          "fewer false alarms in absolute terms.")
    print("      The strict threshold is partly just an unstated correction for peeking.")

    # ---------------------------------------------------------------- close
    rule("WHAT THIS MEANS FOR A REPORT")
    print("1. Never report the split as a percentage. Report the p-value, the n, and the")
    print("   threshold - the same three numbers make 49.3/50.7 fine and fatal.")
    print("2. Test once, at the end, at 0.0005, or use a boundary if you must look daily.")
    print(f"   Test choice is free (five of them disagree on {spread:.2%} of trials); threshold is not.")
    print("3. Sweep the segments and pay the correction. It is three chi-square calls and it")
    print(f"   found {r6['bonf'] / max(r6['agg0005'], 1e-9):.0f}x more of a segment-confined break "
          "at a lower false-alarm rate.")
    print("4. A flag means go and find out WHO is missing, then reweight or re-run. It does")
    print("   not say how bad it is.")
    print("5. A pass means the counts are even. It is not evidence of comparability, and on a")
    print(f"   balanced selective loss it is the null exactly while the effect is {bs['bias']:.0%} high.")
    print("   The only reliable guard against that is a pre-experiment A/A on the same")
    print("   pipeline, where the true effect is known to be zero.")


if __name__ == "__main__":
    main()
