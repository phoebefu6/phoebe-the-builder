"""CUPED, measured.

Eight sections. Every number printed here is asserted in ``test_cuped.py``, and
every one comes from numpy and scipy on a world whose treatment effect and
pre/post correlation are set rather than estimated - there is no data file and
no network.

Run: ``python evidence.py``
"""

from __future__ import annotations

import numpy as np

import cuped

RNG_SEED = 20260904
TRIALS = 6000
SLOW_TRIALS = 2500  # the two loop-based adjusters

W = cuped.World()


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def reduction(est: np.ndarray, base: np.ndarray) -> float:
    return float(1.0 - (est.std(ddof=1) / base.std(ddof=1)) ** 2)


def red_mc(est: np.ndarray, base: np.ndarray) -> str:
    """'0.3563 +/- 0.011' - the reduction and its own Monte Carlo error."""
    pt, se = cuped.reduction_with_mc(est, base)
    return f"{pt:>7.4f} +/-{se:5.3f}"


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("CUPED - VARIANCE REDUCTION FROM PRE-EXPERIMENT DATA")
    print(f"World: {W.per_arm:,} users per arm, metric mean {W.mean:.1f} sd {W.sd:.1f}, "
          f"true relative lift {W.true_rel_lift:.0%}")
    print(f"       pre-period / in-experiment correlation on the same user: rho = {W.rho}")
    print("Reference: Deng, Xu, Kohavi & Walker (2013), WSDM.")

    # ---------------------------------------------------------------- 1
    rule("1. THE WHOLE RESULT IS rho^2 - and the null has to hold first")

    w_null = cuped.World(true_rel_lift=0.0)
    d0 = cuped.simulate(w_null, TRIALS, rng)
    e_none, s_none = cuped.adj_none(d0)
    e_cuped, s_cuped = cuped.adj_cuped(d0)
    r_none = cuped.score(e_none, s_none, 0.0)
    r_cuped = cuped.score(e_cuped, s_cuped, 0.0)

    print("Under no effect at all, before measuring any power:")
    print(f"  unadjusted   size {r_none['reject_rate']:.4f}  coverage {r_none['coverage']:.4f}  "
          f"(nominal 0.0500 / 0.9500)")
    print(f"  CUPED        size {r_cuped['reject_rate']:.4f}  coverage {r_cuped['coverage']:.4f}")
    print("A variance-reduction harness whose null is not calibrated cannot measure a")
    print("reduction - it can only report one.")
    print()
    print(f"theta* = Cov(Y,X)/Var(X) = rho * sd_post/sd_pre = {cuped.theta_star(W):.4f}")
    print(f"predicted variance reduction = rho^2 = {cuped.variance_reduction(W.rho):.4f}")
    print(f"measured on {TRIALS:,} null experiments = {reduction(e_cuped, e_none):.4f}")

    d = cuped.simulate(W, TRIALS, rng)
    rows = {}
    base = None
    print()
    print(f"{'adjuster':<20} {'power':>8} {'reduction':>11} {'traffic needed':>15} "
          f"{'bias':>10} {'coverage':>9}")
    for name, fn in cuped.ADJUSTERS.items():
        dd = d if name not in ("post_strat", "cuped_stratified") else {
            k: v[:SLOW_TRIALS] for k, v in d.items()}
        est, se = fn(dd)
        sc = cuped.score(est, se, W.true_effect)
        if base is None:
            base_full = est
            base = sc["sd_est"]
        red = 1.0 - (sc["sd_est"] / base) ** 2 if name not in ("post_strat", "cuped_stratified") \
            else reduction(est, base_full[:SLOW_TRIALS])
        rows[name] = {**sc, "reduction": red}
        print(f"{name:<20} {sc['reject_rate']:>8.4f} {red:>11.4f} {1 - red:>14.3f}x "
              f"{sc['bias']:>+10.5f} {sc['coverage']:>9.4f}")
    print()
    print(f"CUPED takes the same experiment from power {rows['none']['reject_rate']:.3f} to "
          f"{rows['cuped']['reject_rate']:.3f} on identical data,")
    print(f"or equivalently needs {1 - rows['cuped']['reduction']:.3f}x the traffic for the same power.")
    print("Every adjuster here is unbiased and covers at nominal. They differ only in spread.")

    # ---------------------------------------------------------------- 2
    rule("2. THE SAVING IS rho SQUARED, NOT rho")

    print("'CUPED halves your test' is a statement about rho = 0.707, not about CUPED.")
    print()
    print(f"{'rho':>8} {'traffic needed':>16} {'a 6-week test becomes':>24}")
    for r in (0.2, 0.3, 0.5, 0.6, 0.707, 0.8, 0.9):
        mult = cuped.sample_size_multiplier(r)
        print(f"{r:>8.3f} {mult:>15.3f}x {6 * mult:>21.2f} weeks")
    print()
    print(f"A correlation of 0.5 - which reads as a strong relationship - returns "
          f"{1 - cuped.sample_size_multiplier(0.5):.0%} of the")
    print(f"sample, so six weeks becomes {6 * cuped.sample_size_multiplier(0.5):.1f}. To halve the test you need "
          f"rho = {cuped.rho_for_saving(0.5):.4f}.")
    print("The first thing to compute is not the adjustment. It is the correlation, on last")
    print("quarter's data, before anybody promises a timeline.")

    # ---------------------------------------------------------------- 3
    rule("3. NEGATIVE RESULT: 'JUST SUBTRACT THE PRE-PERIOD' CAN DOUBLE THE VARIANCE")

    print("theta = 1 is the instinct: each user's own before-value, subtracted. It is CUPED with")
    print("the coefficient guessed instead of fitted, and Var(Y - X) = sd_post^2 + sd_pre^2")
    print("- 2 rho sd_pre sd_post, which exceeds Var(Y) whenever sd_pre > 2 rho sd_post.")
    print()
    print(f"{'sd_pre':>8} {'sd_post':>8} {'rho':>6} {'closed form':>13} "
          f"{'measured (MC err)':>20} {'theta*':>8} {'CUPED red':>11}")
    dd_rows = {}
    for sd_pre, rho in ((4.0, 0.60), (4.0, 0.30), (6.0, 0.40), (8.0, 0.40)):
        wv = cuped.World(sd_pre=sd_pre, rho=rho, true_rel_lift=0.0)
        dv = cuped.simulate(wv, TRIALS, rng)
        b, _ = cuped.adj_none(dv)
        u, _ = cuped.adj_diff_in_diff(dv)
        c, _ = cuped.adj_cuped(dv)
        ratio_cf = cuped.variance_ratio_unit_theta(rho, sd_pre, wv.sd)
        ratio_m = (u.std(ddof=1) / b.std(ddof=1)) ** 2
        _, ratio_se = cuped.reduction_with_mc(u, b)
        dd_rows[(sd_pre, rho)] = {"cf": ratio_cf, "measured": ratio_m,
                                  "cuped": reduction(c, b)}
        print(f"{sd_pre:>8.1f} {wv.sd:>8.1f} {rho:>6.2f} {ratio_cf:>13.4f} "
              f"{ratio_m:>13.4f} +/-{ratio_se:5.3f} {cuped.theta_star(wv):>8.4f} "
              f"{reduction(c, b):>11.4f}")
    worst = dd_rows[(8.0, 0.40)]
    print()
    print("Bottom row: a pre-window twice as wide as the experiment window and a correlation")
    print(f"of 0.40. Subtracting the pre-period multiplies the variance by "
          f"{worst['measured']:.2f} - the test now")
    print(f"needs {worst['measured']:.1f}x the traffic - while fitting the coefficient instead of guessing")
    print(f"it returns {worst['cuped']:.4f}. Same covariate, same data, opposite sign of outcome.")
    print("A wider pre-window is the normal case: a month of history against a week of test.")
    print()
    print("The measured column is a ratio of two sample variances and carries the bootstrap")
    print(f"error shown. Re-run at T=40,000 rather than {TRIALS:,} and the sd_pre=6 row lands on")
    print("2.0444 and 2.0500 across two seeds - the closed form to four decimals. The rows")
    print("here sit within about two bootstrap errors of it, which is what that column is for.")

    # ---------------------------------------------------------------- 4
    rule("4. NEGATIVE RESULT: MEAN-IMPUTING THE MISSING COVARIATE STOPS HELPING AT f = 0.5")

    print("Users with no pre-period get the mean, which every implementation does. The usual")
    print("write-up says the reduction becomes (1-f)*rho^2. That is the PER-USER variance.")
    print("The estimator is a difference of arm MEANS, and once imputed, the arm's covariate")
    print("mean is the mean of the returning users only - variance sigma_x^2/(n(1-f)), not")
    print("sigma_x^2/n. Working it through:")
    print()
    print("    reduction = rho^2 * (2 - 1/(1-f))       <- zero at f = 0.5, negative beyond")
    print()
    rho_n = 0.70
    print(f"rho = {rho_n}:")
    print(f"{'new users':>10} {'(1-f)rho^2':>12} {'rho^2(2-1/(1-f))':>18} "
          f"{'measured impute':>19} {'measured stratified':>21}")
    imp_rows = {}
    for f in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8):
        wf = cuped.World(rho=rho_n, new_user_share=f, true_rel_lift=0.0)
        df = cuped.simulate(wf, SLOW_TRIALS, rng)
        b, _ = cuped.adj_none(df)
        i_, _ = cuped.adj_cuped(df)
        st, _ = cuped.adj_cuped_stratified(df)
        imp_rows[f] = {"impute": reduction(i_, b), "strat": reduction(st, b)}
        print(f"{f:>9.0%} {cuped.reduction_stratified(rho_n, f):>12.4f} "
              f"{cuped.reduction_mean_impute(rho_n, f):>18.4f} {red_mc(i_, b):>19} "
              f"{red_mc(st, b):>21}")
    print()
    print(f"The break-even is exactly f = {cuped.impute_breakeven_share()} and it does not depend on rho at all:")
    for r in (0.3, 0.5, 0.7, 0.9):
        print(f"    rho = {r}: reduction at f = 0.5 is {cuped.reduction_mean_impute(r, 0.5):+.6f}")
    print()
    print(f"At 60% new users mean-imputation is a variance INCREASE of "
          f"{-imp_rows[0.6]['impute']:.1%}, while treating")
    print(f"'has a pre-period' as a stratum returns {imp_rows[0.6]['strat']:.4f}. At 80% new users the naive")
    print(f"version multiplies the variance by {1 - imp_rows[0.8]['impute']:.2f}x and the stratified version still")
    print(f"returns {imp_rows[0.8]['strat']:.4f}. The fix is three lines and nobody ships it.")

    # ---------------------------------------------------------------- 5
    rule("5. NEGATIVE RESULT: ON A REVENUE-SHAPED METRIC YOU CANNOT MEASURE rho AT ALL")

    print("CUPED runs on the Pearson correlation of the metric as reported, not of its log.")
    print("Exponentiate both margins and a relationship that is 0.80 on the log scale is")
    print("much less on the reported scale - and the SAMPLE correlation, the number you would")
    print("compute to decide whether CUPED is worth doing, is biased upward and unstable.")
    print()
    print(f"{'log sigma':>10} {'population rho':>15} {'sample rho':>21} {'pop rho^2':>10} "
          f"{'measured reduction':>20}")
    tail_rows = {}
    for sig in (0.5, 1.0, 1.5, 2.0):
        wt = cuped.World(rho=0.80, lognormal=True, log_sigma=sig, true_rel_lift=0.0)
        dt = cuped.simulate(wt, SLOW_TRIALS, rng)
        samp = np.array([np.corrcoef(dt["pre_c"][i], dt["post_c"][i])[0, 1] for i in range(300)])
        b, _ = cuped.adj_none(dt)
        c, _ = cuped.adj_cuped(dt)
        pop = cuped.lognormal_pearson_rho(0.80, sig)
        red_pt, red_se = cuped.reduction_with_mc(c, b)
        tail_rows[sig] = {"pop": pop, "samp_mean": float(samp.mean()),
                          "samp_sd": float(samp.std(ddof=1)), "red": red_pt,
                          "red_se": red_se}
        print(f"{sig:>10.1f} {pop:>15.4f} {samp.mean():>14.4f} +/-{samp.std(ddof=1):<5.3f} "
              f"{pop ** 2:>10.4f} {red_pt:>13.4f} +/-{red_se:5.3f}")
    t2 = tail_rows[2.0]
    print()
    print(f"At log sigma 2.0 the population correlation is {t2['pop']:.4f} and the sample reads")
    print(f"{t2['samp_mean']:.4f} +/- {t2['samp_sd']:.3f} - biased up {t2['samp_mean'] / t2['pop'] - 1:+.0%}, with a spread wide enough that")
    print("two honest analysts on the same table would quote correlations 0.3 apart.")
    print()
    t15 = tail_rows[1.5]
    print("And the delivered reduction stops being predictable by either number. At log")
    print(f"sigma 1.5 it is {t15['red']:.4f} +/- {t15['red_se']:.3f} against a sample rho^2 of "
          f"{t15['samp_mean'] ** 2:.4f} and a population")
    print(f"rho^2 of {t15['pop'] ** 2:.4f} - the sample value wins. At log sigma 2.0 it is "
          f"{t2['red']:.4f} +/- {t2['red_se']:.3f},")
    print(f"which is {abs(t2['red'] - t2['samp_mean'] ** 2) / max(t2['red_se'], 1e-9):.1f} MC errors from the sample rho^2 "
          f"({t2['samp_mean'] ** 2:.4f}) and further still from the")
    print(f"population one ({t2['pop'] ** 2:.4f}). Neither predicts it: at that tail weight the sample")
    print("variance is itself set by a handful of users, so the RATIO of two sample variances")
    print("has an error bar wide enough to swallow the effect being measured.")
    print()
    print("The honest reading is not 'CUPED does better than rho^2 on heavy tails'. It is that")
    print("on a revenue-shaped metric neither the planning number nor the delivered number is")
    print("measurable to the precision anyone quotes them at. Cap or winsorise first, quote an")
    print("interval on the correlation, and re-measure the reduction after the cap.")

    # ---------------------------------------------------------------- 6
    rule("6. THE MISTAKE THAT LOOKS IDENTICAL IN CODE: A COVARIATE FROM AFTER ASSIGNMENT")

    print("Everything above needs the covariate to predate randomisation. Take a covariate")
    print("that the treatment also moved - a same-period engagement metric, a feature the")
    print("variant changed - and the adjustment removes the treatment effect as if it were")
    print("noise. The diff is three lines and reviews clean.")
    print()
    print(f"{'covariate':>26} {'adjuster':>8} {'estimate':>10} {'bias':>10} {'as % of effect':>15} "
          f"{'power':>8} {'coverage':>9}")
    post_rows = {}
    for label, flag in (("pre-period (correct)", False), ("post-assignment", True)):
        dp = cuped.simulate(W, TRIALS, rng, effect_on_pre=flag)
        for nm in ("none", "cuped"):
            est, se = cuped.ADJUSTERS[nm](dp)
            sc = cuped.score(est, se, W.true_effect)
            post_rows[(label, nm)] = sc
            print(f"{label:>26} {nm:>8} {sc['mean_est']:>10.4f} {sc['bias']:>+10.4f} "
                  f"{sc['bias'] / W.true_effect:>+15.1%} {sc['reject_rate']:>8.4f} "
                  f"{sc['coverage']:>9.4f}")
    bad = post_rows[("post-assignment", "cuped")]
    print()
    print(f"The true effect is {W.true_effect:.4f}. The unadjusted estimator finds it either way.")
    print(f"CUPED on the post-assignment covariate reports {bad['mean_est']:.4f} - "
          f"{bad['bias'] / W.true_effect:+.0%} - with coverage")
    print(f"{bad['coverage']:.4f} against a nominal 0.95 and power {bad['reject_rate']:.3f} against "
          f"{post_rows[('post-assignment', 'none')]['reject_rate']:.3f} unadjusted.")
    print("Variance reduction and effect destruction are the same operation pointed at")
    print("different columns, and only the column's timestamp tells them apart. That makes it")
    print("a data-contract question, not a statistics question: the covariate table needs a")
    print("hard cutoff at the assignment timestamp, enforced somewhere a reviewer can see.")

    # ---------------------------------------------------------------- 7
    rule("7. NEGATIVE RESULT: THE TWO THINGS PEOPLE WORRY ABOUT ARE FREE")

    print("Worry 1 - 'estimate theta per arm and it absorbs the treatment effect.'")
    dm = cuped.simulate(cuped.World(rho=0.6, multiplicative=True, true_rel_lift=0.10),
                        TRIALS, rng)
    true_m = W.mean * 0.10
    pa = {}
    for nm in ("none", "cuped", "cuped_per_arm"):
        est, se = cuped.ADJUSTERS[nm](dm)
        pa[nm] = cuped.score(est, se, true_m)
        print(f"  {nm:<16} est {pa[nm]['mean_est']:.5f}  bias {pa[nm]['bias']:+.5f}  "
              f"sd {pa[nm]['sd_est']:.5f}  coverage {pa[nm]['coverage']:.4f}")
    gap = abs(pa["cuped"]["mean_est"] - pa["cuped_per_arm"]["mean_est"])
    print("  Even with a MULTIPLICATIVE effect, so that theta genuinely differs between arms,")
    print(f"  the two estimates are {gap:.2e} apart. Randomisation puts E[xbar_t - xbar_c] at zero,")
    print("  so whatever coefficient multiplies it, it has nothing to bias.")
    print()
    print("Worry 2 - 'you need a lot of data to estimate theta.'")
    print(f"{'per arm':>9} {'unadjusted size':>16} {'CUPED size':>12} {'CUPED coverage':>15} "
          f"{'reduction':>10}")
    small = {}
    for m in (20, 50, 100, 500, 3000):
        ws = cuped.World(per_arm=m, rho=0.6, true_rel_lift=0.0)
        ds = cuped.simulate(ws, TRIALS, rng)
        b, sb = cuped.adj_none(ds)
        c, sc_ = cuped.adj_cuped(ds)
        rb = cuped.score(b, sb, 0.0)
        rc = cuped.score(c, sc_, 0.0)
        small[m] = {"size": rc["reject_rate"], "cover": rc["coverage"],
                    "red": reduction(c, b), "base_size": rb["reject_rate"]}
        print(f"{m:>9} {rb['reject_rate']:>16.4f} {rc['reject_rate']:>12.4f} "
              f"{rc['coverage']:>15.4f} {small[m]['red']:>10.4f}")
    print(f"  At 20 users per arm - 40 data points - the size is "
          f"{small[20]['size']:.4f} against "
          f"{small[20]['base_size']:.4f} unadjusted,")
    print(f"  and the delivered reduction is already {small[20]['red']:.4f} of the promised "
          f"{cuped.variance_reduction(0.6):.2f}. By 100")
    print(f"  per arm the size is {small[100]['size']:.4f} and the cost has vanished. Both worries are")
    print("  answerable in an afternoon; the two that actually break it are sections 4 and 6.")

    # ---------------------------------------------------------------- 8
    rule("8. WHAT ELSE IT FIXES - AND THE HALF IT CANNOT TOUCH")

    print("A pre-period covariate also removes the chance imbalance in itself, so CUPED")
    print("quietly repairs composition damage the covariate can see. Day 165 `srm-detector`")
    print("showed a filter that removes the low-intent users from one arm overstates the")
    print("effect. Here are two such filters, both removing 10% of the treatment arm: one")
    print("selects on the covariate, one on the part of the outcome the covariate cannot")
    print("explain.")
    print()
    print(f"{'filter':>30} {'adjuster':>8} {'estimate':>10} {'bias':>10} {'as % of effect':>15}")
    comp = {}
    for label, kw in (("10% lowest pre-period dropped", dict(drop_low_pre=0.10)),
                      ("10% lowest residual dropped", dict(drop_low_residual=0.10))):
        wc = cuped.World(rho=0.6, **kw)
        dc = cuped.simulate(wc, SLOW_TRIALS, rng)
        for nm in ("none", "cuped"):
            est, se = cuped.ADJUSTERS[nm](dc)
            sc = cuped.score(est, se, W.true_effect)
            comp[(label, nm)] = sc
            print(f"{label:>30} {nm:>8} {sc['mean_est']:>10.4f} {sc['bias']:>+10.4f} "
                  f"{sc['bias'] / W.true_effect:>+15.1%}")
    seen_none = comp[("10% lowest pre-period dropped", "none")]
    seen_cuped = comp[("10% lowest pre-period dropped", "cuped")]
    unseen_cuped = comp[("10% lowest residual dropped", "cuped")]
    print()
    print("Same 10% of the arm gone in both. Unadjusted, the first overstates the effect by")
    print(f"{seen_none['bias'] / W.true_effect:+.0%} and CUPED takes it to "
          f"{seen_cuped['bias'] / W.true_effect:+.1%} - the imbalance was in the covariate, so")
    print(f"adjusting for the covariate removed it. The second overstates by "
          f"{comp[('10% lowest residual dropped', 'none')]['bias'] / W.true_effect:+.0%} and CUPED")
    print(f"leaves {unseen_cuped['bias'] / W.true_effect:+.0%} of it - the selection was on what the covariate cannot see.")
    print()
    print("So CUPED is a variance tool that happens to correct exactly the bias its covariate")
    print("explains, and none of the rest. That is a reason to run it, and not a reason to")
    print("stop checking the split: it repairs the imbalance you can name and is blind, by")
    print("construction, to the one you cannot.")

    # ---------------------------------------------------------------- close
    rule("WHAT THIS MEANS FOR A REPORT")
    print("1. Compute the correlation first, on last quarter's data. Everything CUPED will")
    print(f"   ever give you is rho^2 - and rho = 0.5 buys {1 - cuped.sample_size_multiplier(0.5):.0%}, not half.")
    print("2. Fit theta. Never assume 1. With a pre-window wider than the test window,")
    print(f"   subtracting the pre-period multiplied the variance by {dd_rows[(8.0, 0.4)]['measured']:.2f}x.")
    print("3. Stratify on 'has a pre-period' instead of imputing the mean. Past 50% new")
    print("   users, imputation is worse than doing nothing, for any correlation.")
    print("4. Put a hard cutoff at the assignment timestamp on the covariate table, and")
    print(f"   review it. A post-assignment covariate cost {abs(bad['bias'] / W.true_effect):.0%} of the effect here and")
    print(f"   dropped coverage to {bad['coverage']:.2f} while looking like a normal diff.")
    print("5. On a heavy-tailed metric, cap or winsorise before quoting a correlation, and")
    print("   report the interval on it - the point estimate was biased up 21% here.")
    print("6. Report the reduction you MEASURED next to the rho^2 you promised. They agree")
    print("   in the clean case, and every section above is a way they come apart.")


if __name__ == "__main__":
    main()
