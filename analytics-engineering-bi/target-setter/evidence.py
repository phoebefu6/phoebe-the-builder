"""Every claim the README makes, printed from the code that produces it.

Run ``python evidence.py``. Each section prints the numbers that
``test_targets.py`` asserts, so the README, the notebook and the tests
cannot drift apart from the module.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np
import targets as T
from scipy import stats

ORIGIN_REF = 120  # a target set at the start of year eleven
LINE = "=" * 78


def rule(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def pct(x: float, dp: int = 1) -> str:
    return f"{x * 100:.{dp}f}%"


# --------------------------------------------------------------------------


def section_1_the_room(series: np.ndarray) -> Dict[str, float]:
    rule("1. TWELVE DEFENSIBLE TARGETS FOR THE SAME QUARTER")
    tg = T.targets_at(series, ORIGIN_REF)
    truth_mean, _ = T.truth_quarter(ORIGIN_REF)
    last_q = float(series[ORIGIN_REF - T.HORIZON : ORIGIN_REF].sum())
    actual = float(series[ORIGIN_REF : ORIGIN_REF + T.HORIZON].sum())

    print(f"History: {T.N_MONTHS} months, {pct(T.G)}/month trend, "
          f"sigma={T.SIGMA}, seed={T.SEED}")
    print(f"Target covers months {ORIGIN_REF}-{ORIGIN_REF + T.HORIZON - 1}. "
          f"Last quarter actual = {last_q:,.0f}. "
          f"E[next quarter] = {truth_mean:,.0f}.\n")
    print(f"{'method':24} {'target':>10} {'vs last q':>10} "
          f"{'vs truth':>9}  provenance")
    order = sorted(tg, key=lambda k: tg[k])
    for name in order:
        v = tg[name]
        print(f"{name:24} {v:>10,.0f} {pct(v / last_q - 1):>10} "
              f"{v / truth_mean:>9.3f}  {T.PROVENANCE[name]}")

    lo, hi = tg[order[0]], tg[order[-1]]
    spread = hi / lo
    growth_implied_lo = lo / last_q - 1
    growth_implied_hi = hi / last_q - 1
    print(f"\nLowest {order[0]} = {lo:,.0f};  highest {order[-1]} = {hi:,.0f}")
    print(f"Ratio highest/lowest              : {spread:.3f}x")
    print(f"Spread as a share of last quarter : {pct((hi - lo) / last_q)}")
    print(f"Growth each target asks for       : {pct(growth_implied_lo)} "
          f"to {pct(growth_implied_hi)}")
    print(f"Actual, when the quarter finished : {actual:,.0f} "
          f"({pct(actual / last_q - 1)} on last quarter)")
    print("\nThe methods disagree by "
          f"{pct((hi - lo) / last_q)} of the base. The quarter itself moved "
          f"{pct(actual / last_q - 1)}.")
    print("The choice of method is a larger number than the thing being "
          "targeted.")
    return {"spread": spread, "lo": lo, "hi": hi, "last_q": last_q,
            "actual": actual, "truth_mean": truth_mean}


def section_2_hit_rate_is_the_method(mp) -> Dict[str, float]:
    rule("2. THE HIT RATE MEASURES THE METHOD, NOT THE TEAM")
    names = list(T.METHODS)
    hit = np.array([mp[n]["hit_rate"].mean() for n in names])
    amb = np.array([mp[n]["ambition"].mean() for n in names])
    rho, p = stats.spearmanr(amb, hit)

    print(f"{T.N_PATHS} independent draws of the same eleven years, "
          f"{len(T.origins(T.make_history()))} origins each.\n")
    print(f"{'method':24} {'ambition':>9} {'hit rate':>9} {'sd':>7} "
          f"{'min':>7} {'max':>7}")
    for i in np.argsort(amb):
        n = names[i]
        h = mp[n]["hit_rate"]
        print(f"{n:24} {amb[i]:>9.3f} {hit[i]:>9.3f} {h.std():>7.3f} "
              f"{h.min():>7.3f} {h.max():>7.3f}")

    print(f"\nSpearman(ambition, hit rate) = {rho:.4f}   p = {p:.2e}")
    print(f"Hit rate runs from {hit.min():.3f} ({names[int(hit.argmin())]}) "
          f"to {hit.max():.3f} ({names[int(hit.argmax())]}) with no change "
          "to the work.")

    # Ambition does not fully determine the hit rate: find the pairs where a
    # more ambitious target is hit MORE often.
    inversions: List[Tuple[str, str, float, float]] = []
    for i, j in itertools.combinations(range(len(names)), 2):
        if (amb[i] - amb[j]) * (hit[i] - hit[j]) > 0:
            lo_i, hi_i = (i, j) if amb[i] < amb[j] else (j, i)
            inversions.append(
                (names[lo_i], names[hi_i], hit[lo_i], hit[hi_i])
            )
    print(f"\nPairs where the MORE ambitious target is hit more often: "
          f"{len(inversions)} of {len(names) * (len(names) - 1) // 2}")
    for a, b, ha, hb in inversions:
        print(f"   {b} (harder) hits {hb:.3f} vs {a} {ha:.3f}")
    print("A target is a random variable too. How often it is hit depends on "
          "its correlation\nwith the actual, not only on how high it is set.")
    return {"rho": float(rho), "hit_min": float(hit.min()),
            "hit_max": float(hit.max()), "n_inversions": len(inversions)}


def section_3_unbiased_is_missed(mp, oracle) -> Dict[str, float]:
    rule("3. AN UNBIASED TARGET IS MISSED MORE OFTEN THAN IT IS HIT")
    single_month = 1 - stats.norm.cdf(T.SIGMA / 2.0)
    skew = float(np.exp(T.SIGMA**2 / 2.0))
    print("The metric is lognormal, so its mean sits above its median by "
          f"exp(sigma^2/2) = {skew:.4f}.")
    print("A target set at the expectation is therefore above the middle of "
          "the distribution\nbefore anybody does any work.\n")
    print(f"P(one month >= its own mean)          : {single_month:.4f}")
    print(f"Oracle target at the TRUE mean        : "
          f"{oracle['mean_target']:.4f}  (sd {oracle['mean_target_sd']:.4f})")
    print(f"Oracle target at the TRUE median      : "
          f"{oracle['median_target']:.4f}  (sd {oracle['median_target_sd']:.4f})")
    print("The quarter sum is less skewed than a single month, which is why "
          "the oracle\nrecovers part of the gap. It never reaches 0.5 at the "
          "mean.\n")
    a = mp["trend_seasonal"]["hit_rate"].mean()
    b = mp["trend_seasonal_median"]["hit_rate"].mean()
    ta = T.targets_at(T.make_history(), ORIGIN_REF)
    gap = ta["trend_seasonal"] / ta["trend_seasonal_median"] - 1
    print(f"Same model, mean vs median target     : {pct(gap, 2)} apart "
          f"in the number")
    print(f"                                        {a:.3f} vs {b:.3f} "
          f"in the hit rate ({(b - a) * 100:.1f} points)")
    print("A 0.7% change in the target moves the hit rate by "
          f"{(b - a) * 100:.1f} points. Nobody in the\nmeeting will notice "
          "the 0.7%.")
    return {"single_month": float(single_month), "skew": skew,
            "mean_vs_median_gap": float(gap),
            "hit_mean": float(a), "hit_median": float(b)}


def section_4_not_reproducible(mp) -> Dict[str, float]:
    rule("4. A HIT RATE IS NOT A REPRODUCIBLE MEASUREMENT")
    n_or = len(T.origins(T.make_history()))
    worst = max(T.METHODS, key=lambda n: mp[n]["hit_rate"].std())
    h = mp[worst]["hit_rate"]
    print(f"Re-running the same eleven years from a fresh draw changes the "
          f"hit rate.\nAcross {T.N_PATHS} paths, each scoring {n_or} "
          f"overlapping quarters:\n")
    print(f"{'method':24} {'mean':>7} {'sd':>7} {'p05':>7} {'p95':>7} "
          f"{'range':>7}")
    for n in sorted(T.METHODS, key=lambda k: -mp[k]["hit_rate"].std()):
        x = mp[n]["hit_rate"]
        print(f"{n:24} {x.mean():>7.3f} {x.std():>7.3f} "
              f"{np.quantile(x, 0.05):>7.3f} {np.quantile(x, 0.95):>7.3f} "
              f"{x.max() - x.min():>7.3f}")
    print(f"\nThe least reproducible method is {worst}: "
          f"sd {h.std():.3f}, range {h.max() - h.min():.3f}.")
    print("The two least reproducible methods are the two best-specified "
          "forecasts in the\nlist -- the ones that match the process that "
          "generated the data. Forecasting well\nputs the target in the "
          "middle of the distribution, which is exactly where the\nhit/miss "
          "verdict is most sensitive to noise. The most reproducible hit "
          "rates\nbelong to the targets nobody would call forecasts.\n")
    n_needed = T.quarters_to_distinguish(0.50, 0.65)
    print(f"Quarters needed to tell a 0.50 hitter from a 0.65 hitter "
          f"(alpha 0.05, power 0.80): {n_needed}")
    print(f"That is {n_needed / 4:.1f} years. A company with eleven years of "
          f"history has 44 quarters.")
    obs, n_obs = 8, 12
    pv = stats.binomtest(obs, n_obs, 0.5, alternative="greater").pvalue
    print(f"\n'We hit {obs} of our last {n_obs} targets': "
          f"p = {pv:.3f} against a coin. Not evidence.")
    return {"worst": worst, "worst_sd": float(h.std()),
            "n_needed": n_needed, "binom_p": float(pv)}


def section_5_when_you_ask(series: np.ndarray) -> Dict[str, float]:
    rule("5. THE SAME METHOD ANSWERS DIFFERENTLY DEPENDING ON WHEN YOU ASK")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ors = T.origins(series)
    truth = {o: T.truth_quarter(o)[0] for o in ors}
    rows: Dict[str, Dict[int, float]] = {}
    for name in ("run_rate", "last_quarter", "trend_ols", "trend_seasonal"):
        fn = T.METHODS[name]
        by_month: Dict[int, List[float]] = {m: [] for m in range(12)}
        for o in ors:
            by_month[o % 12].append(fn(series[:o], o) / truth[o])
        rows[name] = {m: float(np.mean(v)) for m, v in by_month.items()}

    print("Target as a multiple of the truth it aims at, by the calendar "
          "month it is set in:\n")
    print(f"{'set in':7} " + " ".join(f"{n:>7}" for n in rows))
    for m in range(12):
        print(f"{month_names[m]:7} " + " ".join(f"{rows[n][m]:>7.3f}"
                                               for n in rows))
    out = {}
    for name, d in rows.items():
        lo_m = min(d, key=lambda k: d[k])
        hi_m = max(d, key=lambda k: d[k])
        swing = d[hi_m] / d[lo_m] - 1
        out[name] = swing
        print(f"\n{name:22} best month {month_names[hi_m]} {d[hi_m]:.3f}, "
              f"worst {month_names[lo_m]} {d[lo_m]:.3f}  ->  "
              f"swing {pct(swing)}")
    print("\nThe planning calendar is a parameter of the target. Moving the "
          "planning meeting\nfrom one month to another changes the number by "
          f"up to {pct(max(out.values()))} on the same history.")
    return out


def section_6_reconciliation(series: np.ndarray, mp) -> Dict[str, float]:
    rule("6. TOP-DOWN AND BOTTOM-UP DO NOT RECONCILE, AND THE MIDPOINT IS "
         "ACHIEVABLE BY NEITHER")
    ors = T.origins(series)
    cap = np.array([T.m_capacity(series[:o], o) for o in ors])
    top = np.array([T.m_top_down(series[:o], o) for o in ors])
    spl = 0.5 * (cap + top)
    ref = np.array([T.m_trend_seasonal(series[:o], o) for o in ors])
    gap = top / cap - 1

    print(f"Board multiple {T.BOARD_MULTIPLE:.2f}x on last year's quarter; "
          f"headcount plan adds {T.HEADCOUNT_STEP:.0f} head every "
          f"{T.HEADCOUNT_STEP_MONTHS} months.\n")
    print(f"Top-down above bottom-up at        : {pct((top > cap).mean())} "
          f"of {len(ors)} origins")
    print(f"Mean gap (top-down / capacity - 1) : {pct(gap.mean())}  "
          f"(min {pct(gap.min())}, max {pct(gap.max())})")
    print(f"Midpoint above capacity at         : {pct((spl > cap).mean())} "
          "of origins")
    print(f"Midpoint below top-down at         : {pct((spl < top).mean())} "
          "of origins")
    print("\nBy construction the compromise is unachievable under the "
          "resourcing argument\nand unacceptable under the board argument. "
          "It is not a third position; it is the\ntwo objections added "
          "together.\n")
    d_cap = float(np.abs(cap / ref - 1).mean())
    d_spl = float(np.abs(spl / ref - 1).mean())
    d_top = float(np.abs(top / ref - 1).mean())
    print(f"|distance| from the best forecast  capacity {pct(d_cap)}  "
          f"midpoint {pct(d_spl)}  top-down {pct(d_top)}")
    print(f"signed distance (target/forecast)  capacity "
          f"{pct(float((cap / ref - 1).mean()))}  midpoint "
          f"{pct(float((spl / ref - 1).mean()))}  top-down "
          f"{pct(float((top / ref - 1).mean()))}")
    print(f"hit rate across {T.N_PATHS} paths      capacity "
          f"{mp['capacity']['hit_rate'].mean():.3f}  midpoint "
          f"{mp['split_difference']['hit_rate'].mean():.3f}  top-down "
          f"{mp['top_down']['hit_rate'].mean():.3f}")
    print(f"\nThe midpoint is the CLOSEST of the three to the best available "
          f"forecast\n({pct(d_spl)} against capacity's {pct(d_cap)}) and it "
          f"is met "
          f"{mp['capacity']['hit_rate'].mean() / mp['split_difference']['hit_rate'].mean():.1f}x "
          "less often.")
    print("Distance to a forecast is two-sided; being met is one-sided. "
          "Capacity sits BELOW\nthe forecast and the midpoint sits ABOVE it, "
          "so the number that looks better on\nthe accuracy slide is the one "
          "that will be missed. Any accuracy statistic that\ntakes an "
          "absolute value has thrown away the only thing a target cares "
          "about.")
    return {"gap_mean": float(gap.mean()), "d_cap": d_cap, "d_spl": d_spl,
            "d_top": d_top, "split_above_cap": float((spl > cap).mean())}


def section_7_sandbagging(mp) -> Dict[str, float]:
    rule("7. A HIT-RATE INCENTIVE PAYS FOR THE CHOICE OF METHOD")
    lo, hi = "seasonal_naive", "trend_seasonal"
    amb_lo = mp[lo]["ambition"].mean()
    amb_hi = mp[hi]["ambition"].mean()
    hit_lo = mp[lo]["hit_rate"].mean()
    hit_hi = mp[hi]["hit_rate"].mean()
    gap = amb_hi / amb_lo - 1
    months = np.log(1 + gap) / np.log(1 + T.G)

    print("Both of these are said out loud in planning meetings:\n")
    print(f"  {lo:22} {T.PROVENANCE[lo]}")
    print(f"  {hi:22} {T.PROVENANCE[hi]}\n")
    print(f"Ambition   {amb_lo:.3f} vs {amb_hi:.3f}   -> the softer target "
          f"is {pct(gap)} lower")
    print(f"Hit rate   {hit_lo:.3f} vs {hit_hi:.3f}   -> "
          f"{(hit_lo - hit_hi) * 100:.1f} points, for identical work")
    print(f"\n{pct(gap)} of this metric is {months:.1f} months of real trend "
          f"growth at {pct(T.G)}/month.")
    print("Choosing which sentence to say in the meeting is worth more than "
          f"{months:.0f} months of\nthe growth the target exists to "
          "encourage.\n")

    bonus = 100_000.0
    print(f"{'method':24} {'hit':>7} {'E[bonus]':>12}   "
          f"(threshold bonus {bonus:,.0f})")
    for n in sorted(T.METHODS, key=lambda k: -mp[k]["hit_rate"].mean()):
        h = mp[n]["hit_rate"].mean()
        print(f"{n:24} {h:>7.3f} {h * bonus:>12,.0f}")
    return {"gap": float(gap), "months": float(months),
            "hit_lo": float(hit_lo), "hit_hi": float(hit_hi)}


def section_8_stretch_payout(mp) -> Dict[str, float]:
    rule("8. WHAT A STRETCH TARGET WOULD HAVE TO PAY")
    real, stretch = "trend_seasonal", "stretch_best_ever"
    p_real = mp[real]["hit_rate"].mean()
    p_str = mp[stretch]["hit_rate"].mean()
    p_top = mp["top_down"]["hit_rate"].mean()
    m_str = p_real / p_str
    m_top = p_real / p_top if p_top > 0 else float("inf")
    print("A threshold bonus pays B if the target is met, nothing if not.\n")
    print(f"{real:24} p = {p_real:.3f}")
    print(f"{stretch:24} p = {p_str:.3f}  -> needs "
          f"{m_str:.2f}x the payout to be worth the same attempt")
    print(f"{'top_down':24} p = {p_top:.3f}  -> needs "
          f"{m_top:.1f}x the payout")
    print(f"\nStretch targets are almost never paid at {m_str:.1f}x, and the "
          f"board target is never paid\nat {m_top:.0f}x. The expected value "
          "of trying is lower under the harder target,\nwhich is the "
          "opposite of what setting it was meant to do.")
    return {"p_real": float(p_real), "p_str": float(p_str),
            "m_str": float(m_str), "m_top": float(m_top)}


def section_9_inside_the_noise(series: np.ndarray) -> Dict[str, float]:
    rule("9. MOST OF THE ARGUMENT IS INSIDE THE PREDICTION INTERVAL")
    lo, hi = T.prediction_interval(series, ORIGIN_REF, level=0.80)
    width = hi - lo
    tg = T.targets_at(series, ORIGIN_REF)
    names = list(tg)
    pairs = list(itertools.combinations(names, 2))
    inside = [(a, b) for a, b in pairs if abs(tg[a] - tg[b]) < width]
    mid = (lo + hi) / 2
    print(f"80% interval for the quarter: {lo:,.0f} to {hi:,.0f}   "
          f"width {width:,.0f} ({pct(width / mid)} of the point forecast)\n")
    print(f"Pairs of methods:                       {len(pairs)}")
    print(f"Pairs closer together than the interval: {len(inside)} "
          f"({pct(len(inside) / len(pairs))})")
    n_in = sum(1 for v in tg.values() if lo <= v <= hi)
    print(f"Targets that fall inside the interval:   {n_in} of {len(tg)}")
    print(f"\n{len(inside)} of the {len(pairs)} disagreements in that meeting "
          "are smaller than the uncertainty\nin the forecast they are all "
          "arguing about.")
    return {"pi_lo": lo, "pi_hi": hi, "width": width,
            "n_pairs": len(pairs), "n_inside": len(inside), "n_in_pi": n_in}


def section_10_ensemble(series: np.ndarray, mp) -> Dict[str, float]:
    rule("10. NEGATIVE RESULT: AVERAGING THE TWELVE IMPROVES THE FORECAST "
         "AND NOT THE TARGET")
    ens_hit, ens_mape, ens_amb = [], [], []
    for i in range(T.N_PATHS):
        e = T.ensemble_result(T.make_history(seed=T.PATH_SEED0 + i))
        ens_hit.append(e.hit_rate)
        ens_mape.append(e.mape)
        ens_amb.append(e.ambition)
    ens_hit_m = float(np.mean(ens_hit))
    ens_mape_m = float(np.mean(ens_mape))
    best_mape = min(T.METHODS, key=lambda n: mp[n]["mape"].mean())
    print(f"Across {T.N_PATHS} paths:\n")
    print(f"{'':32} {'hit':>7} {'mape':>7} {'ambition':>9}")
    print(f"{'ensemble (mean of 12 targets)':32} {ens_hit_m:>7.3f} "
          f"{ens_mape_m:>7.3f} {float(np.mean(ens_amb)):>9.3f}")
    print(f"{'best single (' + best_mape + ')':32} "
          f"{mp[best_mape]['hit_rate'].mean():>7.3f} "
          f"{mp[best_mape]['mape'].mean():>7.3f} "
          f"{mp[best_mape]['ambition'].mean():>9.3f}")
    print(f"{'trend_seasonal':32} {mp['trend_seasonal']['hit_rate'].mean():>7.3f} "
          f"{mp['trend_seasonal']['mape'].mean():>7.3f} "
          f"{mp['trend_seasonal']['ambition'].mean():>9.3f}")
    beaten = sum(1 for n in T.METHODS if mp[n]["mape"].mean() < ens_mape_m)
    met_more = sum(
        1 for n in T.METHODS if mp[n]["hit_rate"].mean() > ens_hit_m
    )
    print(f"\nSingle methods MORE ACCURATE than the ensemble : {beaten} of "
          f"{len(T.METHODS)}")
    print(f"Single methods MET MORE OFTEN than the ensemble: {met_more} of "
          f"{len(T.METHODS)}")
    print("\nAveraging does what averaging always does: the ensemble is "
          f"{beaten + 1} of {len(T.METHODS) + 1} on accuracy,\nbetter than "
          "ten of the twelve inputs. It still fails as a target. Its "
          "ambition is\n"
          f"{float(np.mean(ens_amb)):.3f}, because four of the twelve inputs "
          "are not estimates of one quantity --\nthey are estimates of what "
          "somebody wants, and the mean inherits the wanting. So\nthe "
          f"ensemble is met less often than {met_more} of the twelve methods "
          "it averages.")

    # Rank by accuracy vs rank by hit rate.
    names = list(T.METHODS)
    mape = np.array([mp[n]["mape"].mean() for n in names])
    hit = np.array([mp[n]["hit_rate"].mean() for n in names])
    rho, _ = stats.spearmanr(-mape, hit)
    print(f"\nSpearman(accuracy rank, hit-rate rank) = {rho:.4f}")
    print("Accuracy and hit rate are close to unrelated. A target can be the "
          "best estimate\nof the future and one of the least often met.")
    return {"ens_hit": ens_hit_m, "ens_mape": ens_mape_m,
            "beaten": beaten, "rho_acc_hit": float(rho)}


def main() -> Dict[str, Dict]:
    series = T.make_history()
    mp = T.multipath()
    oracle = T.oracle_hit_rates()
    out = {
        "s1": section_1_the_room(series),
        "s2": section_2_hit_rate_is_the_method(mp),
        "s3": section_3_unbiased_is_missed(mp, oracle),
        "s4": section_4_not_reproducible(mp),
        "s5": section_5_when_you_ask(series),
        "s6": section_6_reconciliation(series, mp),
        "s7": section_7_sandbagging(mp),
        "s8": section_8_stretch_payout(mp),
        "s9": section_9_inside_the_noise(series),
        "s10": section_10_ensemble(series, mp),
    }
    rule("WHAT A TARGET IS")
    print("A target is a method plus a claim about the future. The method "
          "moves the number\nby more than the business does, and the "
          "statistic used to grade the result --\nthe hit rate -- is a "
          "property of the method that was chosen, not of the work\nthat "
          "was done.")
    print("\nSo write down the method, the claim, and the interval. A "
          "target that arrives as a\nsingle number has already thrown away "
          "everything needed to defend it.")
    return out


if __name__ == "__main__":
    main()
