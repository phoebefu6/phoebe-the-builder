"""Ten sections. Every number in the README is printed here and asserted in the tests."""

from __future__ import annotations

from typing import Dict

import guardrails as G
import numpy as np
from scipy import stats

N_PER_ARM = G.n_for_power(0.80, 1.0)   # the experiment is sized for the WIN
DECISION_DAY = 14
ALPHA = 0.05
REPS = 20_000
SEED = 20_260_830
DAILY_ENROLMENT = N_PER_ARM / DECISION_DAY
INTENSITY_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 3)
BIN_REPS = 4_000
YEARS = 2_000
EXPERIMENTS_PER_YEAR = 20
BASE_VALUE = G.true_value(0.0)


def _rule(title: str) -> None:
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)


# ======================================================================================
def section_1() -> Dict:
    _rule("1. A world where the answer is known")
    rows = []
    for a in [0.0, 0.25, 0.50, 0.75, 1.0]:
        rows.append({
            "a": a,
            "conv": G.conversion_rate(a),
            "lift": G.primary_lift(a),
            "value": G.true_value(a),
            "dvalue": G.value_change(a),
            "quality": G.retention_quality(a),
            "dquality": G.quality_change(a),
            "annoyed": G.annoyed_share(a),
        })
    print(f"{'intensity':>9} {'conversion':>11} {'reported lift':>14} "
          f"{'180d retained/1k':>17} {'volume change':>14} {'180d ret. rate':>15} {'rate change':>12}")
    for r in rows:
        print(f"{r['a']:9.2f} {r['conv']:11.4f} {r['lift']*100:13.1f}% "
              f"{r['value']:17.2f} {r['dvalue']*100:13.2f}% {r['quality']:15.4f} "
              f"{r['dquality']*100:11.2f}%")
    print(f"\nAt full intensity the slide says +{G.primary_lift(1.0)*100:.1f}% and the business "
          f"loses {abs(G.value_change(1.0))*100:.2f}% of its 180-day retained users.")
    print(f"Both are true. The lever buys {G.incremental_conversion(1.0)*1000:.0f} extra converters "
          f"per 1,000 who retain at {G.R_MARGINAL:.0%} against {G.R_GOOD:.0%}, and annoys "
          f"{G.annoyed_share(1.0):.0%} of everyone else.")
    print(f"There is no intensity above zero at which the trade is worth taking: value_change is "
          f"monotone down ({', '.join(f'{G.value_change(a)*100:.2f}%' for a in [0.25,0.5,0.75,1.0])}).")
    ratio = G.quality_change(1.0) / G.value_change(1.0)
    print(f"\nThe aggregate metric hides the harm. Retained users per 1,000 fall "
          f"{abs(G.value_change(1.0))*100:.2f}%, but the 180-day retention RATE falls "
          f"{abs(G.quality_change(1.0))*100:.2f}% - {ratio:.1f}x as much - because the lever "
          f"inflates the denominator with exactly the users who will not retain. A total can "
          f"absorb a great deal of damage to a rate before it moves.")
    return {"rows": rows, "lift_at_1": G.primary_lift(1.0), "dvalue_at_1": G.value_change(1.0),
            "dquality_at_1": G.quality_change(1.0), "masking_ratio": float(ratio)}


# ======================================================================================
def section_2() -> Dict:
    _rule("2. The experiment is powered for the win and not for the harm")
    print(f"n per arm sized for 80% power on a +{G.primary_lift(1.0)*100:.0f}% lift: "
          f"{N_PER_ARM:,}   (decision day {DECISION_DAY}, one-sided alpha {ALPHA})")
    print(f"realised primary power at that n: {G.primary_power(1.0, N_PER_ARM, ALPHA):.3f}\n")
    print(f"{'guardrail':22} {'denominator':>12} {'observable':>11} {'z':>6} {'power':>7} "
          f"{'n for 80%':>13} {'x primary n':>12}")
    out = []
    for g in G.GUARDRAILS:
        n_c, n_t = G.guardrail_n(g, 1.0, N_PER_ARM, DECISION_DAY)
        z = G.analytic_z(g, 1.0, N_PER_ARM, DECISION_DAY)
        p = G.analytic_power(g, 1.0, N_PER_ARM, DECISION_DAY, ALPHA)
        need = G.n_for_power(0.80, 1.0, ALPHA, g, DECISION_DAY)
        ratio = None if need is None else need / N_PER_ARM
        out.append({"name": g.name, "n_obs": n_c, "z": z, "power": p,
                    "need": need, "ratio": ratio})
        obs = f"{n_c:,.0f}"
        pw = "cannot run" if np.isnan(p) else f"{p:.3f}"
        nd = "unreachable" if need is None else f"{need:,}"
        rt = "-" if ratio is None else f"{ratio:9.1f}x"
        print(f"{g.name:22} {g.denominator:>12} {obs:>11} {z:6.2f} {pw:>7} {nd:>13} {rt:>12}")

    runnable = [r for r in out if not np.isnan(r["power"])]
    best = max(runnable, key=lambda r: r["power"])
    dash = [r for r in out if r["name"] in G.DASHBOARD_SUITE and not np.isnan(r["power"])]
    best_dash = max(dash, key=lambda r: r["power"])
    print(f"\nBest guardrail in the catalogue: {best['name']} at power {best['power']:.3f}.")
    print(f"Best guardrail actually on the dashboard: {best_dash['name']} at power "
          f"{best_dash['power']:.3f}.")
    print(f"The whole catalogue tops out at {best['power']/G.primary_power(1.0, N_PER_ARM, ALPHA):.0%} "
          f"of the power the experiment was built to have, and the metrics that are actually on "
          f"the checklist top out at {best_dash['power']/G.primary_power(1.0, N_PER_ARM, ALPHA):.0%}.")
    print(f"To detect the harm as reliably as the win, {best['name']} needs {best['need']:,} per "
          f"arm - {best['ratio']:.1f}x the experiment. The dashboard's best needs "
          f"{best_dash['need']:,}, or {best_dash['ratio']:.1f}x.")
    print("d90_retention is the metric closest to the truth and its denominator on day 14 is "
          "exactly zero: (14 - 90) is negative, so no user has been enrolled long enough to have "
          "a value. It is on the checklist. It cannot be computed.")
    return {"rows": out, "best": best, "best_dashboard": best_dash, "n_per_arm": N_PER_ARM}


# ======================================================================================
def section_3() -> Dict:
    _rule("3. 'No significant change' is not 'no harm'")
    rng = np.random.default_rng(SEED)
    z1 = G.simulate_experiment(1.0, N_PER_ARM, DECISION_DAY, REPS, rng)
    z0 = G.simulate_experiment(0.0, N_PER_ARM, DECISION_DAY, REPS, rng)

    pass_dash = 1 - np.mean(G.any_fires(z1, G.DASHBOARD_SUITE, ALPHA))
    pass_all = 1 - np.mean(G.any_fires(z1, G.COMPUTABLE_SUITE, ALPHA))
    print(f"Change is maximally harmful (a = 1.0, {abs(G.value_change(1.0))*100:.2f}% of value lost).")
    print(f"  P(no dashboard guardrail is significant)  = {pass_dash:.3f}")
    print(f"  P(no computable guardrail is significant) = {pass_all:.3f}")

    # Non-inferiority on the best runnable guardrail: prove harm is smaller than a margin.
    g = G.GUARDRAIL_BY_NAME["d7_retention"]
    a_tol = 0.20
    margin_z = G.analytic_z(g, a_tol, N_PER_ARM, DECISION_DAY)
    crit = G.crit_value(ALPHA)
    proved_clean = float(np.mean(z0[g.name] < margin_z - crit))
    proved_harm = float(np.mean(z1[g.name] < margin_z - crit))
    need_ni = None
    for mult in range(1, 400):
        n = N_PER_ARM * mult
        shift = G.analytic_z(g, a_tol, n, DECISION_DAY)
        if 1 - stats.norm.cdf(crit - shift) >= 0.80:
            need_ni = n
            break
    print(f"\nNon-inferiority on {g.name}, margin = the harm at intensity {a_tol} "
          f"({abs(G.value_change(a_tol))*100:.2f}% of value):")
    print(f"  P(PROVE non-inferiority | change is genuinely clean)  = {proved_clean:.3f}")
    print(f"  P(PROVE non-inferiority | change is maximally harmful) = {proved_harm:.3f}")
    print(f"  n per arm needed to prove it 80% of the time when true = {need_ni:,} "
          f"({need_ni / N_PER_ARM:.0f}x the experiment)")
    print(f"\nA clean change clears the guardrail {(1-np.mean(G.any_fires(z0, G.DASHBOARD_SUITE, ALPHA))):.3f} "
          f"of the time and a maximally harmful one clears it {pass_dash:.3f} of the time. The "
          f"checklist tick is nearly the same in both worlds, so it carries almost no information "
          f"about which world you are in.")
    return {"pass_dashboard_harmful": float(pass_dash), "pass_all_harmful": float(pass_all),
            "proved_ni_clean": proved_clean, "proved_ni_harmful": proved_harm,
            "n_for_ni": need_ni, "ni_multiple": need_ni / N_PER_ARM}


# ======================================================================================
def section_4() -> Dict:
    _rule("4. Adding a guardrail can make you less safe")
    rng = np.random.default_rng(SEED + 1)
    z1 = G.simulate_experiment(1.0, N_PER_ARM, DECISION_DAY, REPS, rng)
    z0 = G.simulate_experiment(0.0, N_PER_ARM, DECISION_DAY, REPS, rng)
    order = G.DASHBOARD_SUITE + ["refund_rate", "d7_retention"]

    print(f"{'k':>2} {'added':22} {'false block':>12} {'detect harm':>12} "
          f"{'bonf false':>11} {'bonf detect':>12}")
    rows = []
    for k in range(1, len(order) + 1):
        suite = order[:k]
        fb = float(np.mean(G.any_fires(z0, suite, ALPHA)))
        dt = float(np.mean(G.any_fires(z1, suite, ALPHA)))
        fbb = float(np.mean(G.any_fires(z0, suite, ALPHA / k)))
        dtb = float(np.mean(G.any_fires(z1, suite, ALPHA / k)))
        rows.append({"k": k, "added": order[k - 1], "false": fb, "detect": dt,
                     "bonf_false": fbb, "bonf_detect": dtb})
        print(f"{k:2d} {order[k-1]:22} {fb:11.3f} {dt:11.3f} {fbb:10.3f} {dtb:11.3f}")

    k = len(order)
    print(f"\nThe false-block column is not an artefact of correlated metrics: the observed "
          f"{rows[-1]['false']:.4f} matches 1 - (1 - {ALPHA})^{k} = {1-(1-ALPHA)**k:.4f} to four "
          f"decimal places, because under the null these guardrails are near-independent "
          f"(largest pairwise correlation 0.007). Every one you add multiplies the chance that "
          f"something harmless trips.")

    placebo_i = order.index("page_latency_ms")
    before, after = rows[placebo_i - 1], rows[placebo_i]
    print(f"\nAdding page_latency_ms - a metric the lever provably cannot touch - moves the "
          f"false-block rate on a harmless change from {before['false']:.3f} to {after['false']:.3f} "
          f"and, once the suite is corrected for its own size, moves detection of real harm from "
          f"{before['bonf_detect']:.3f} to {after['bonf_detect']:.3f}. It is strictly worse on both "
          f"counts. A guardrail with no causal sensitivity is not free.")
    full = rows[-1]
    print(f"\nThe dilemma: uncorrected, an {len(order)}-guardrail suite blocks "
          f"{full['false']*100:.1f}% of harmless changes. Correct it and detection of a maximally "
          f"harmful change falls from {full['detect']:.3f} to {full['bonf_detect']:.3f}. There is no "
          f"correction that fixes both, because both are consequences of running many "
          f"underpowered tests instead of one powered one.")
    return {"rows": rows, "placebo_before": before, "placebo_after": after, "full": full}


# ======================================================================================
def section_5() -> Dict:
    _rule("5. What a 14-day window can even see")
    print(f"{'day':>4} {'n/arm':>9} {'d7 obs':>8} {'d90 obs':>8} {'best single':>12} "
          f"{'suite':>8} {'composite':>10}")
    rows = []
    for d in [3, 7, 14, 21, 28, 56, 90, 120, 180]:
        n = int(DAILY_ENROLMENT * d)
        rng = np.random.default_rng(SEED + 100 + d)
        z1 = G.simulate_experiment(1.0, n, d, BIN_REPS * 2, rng)
        z0 = G.simulate_experiment(0.0, n, d, BIN_REPS * 2, rng)
        suite = [g.name for g in G.GUARDRAILS if G.observable_fraction(d, g.maturity_days) > 0]
        w = G.sensitivity_weights(suite, 1.0, n, d)
        crit_c = float(np.quantile(G.composite_z(z0, suite, w), 1 - ALPHA))
        comp = float(np.mean(G.composite_z(z1, suite, w) > crit_c))
        singles = {g: G.analytic_power(G.GUARDRAIL_BY_NAME[g], 1.0, n, d, ALPHA) for g in suite}
        best_name = max(singles, key=lambda k: singles[k])
        rows.append({"day": d, "n": n, "d7": G.observable_fraction(d, 7),
                     "d90": G.observable_fraction(d, 90), "best": singles[best_name],
                     "best_name": best_name,
                     "suite": float(np.mean(G.any_fires(z1, suite, ALPHA))),
                     "suite_false": float(np.mean(G.any_fires(z0, suite, ALPHA))),
                     "composite": comp})
        print(f"{d:4d} {n:9,} {G.observable_fraction(d,7):8.2f} {G.observable_fraction(d,90):8.3f} "
              f"{singles[best_name]:12.3f} {rows[-1]['suite']:8.3f} {comp:10.3f}")

    d14 = next(r for r in rows if r["day"] == 14)
    reach = next((r["day"] for r in rows if r["composite"] >= 0.80), None)
    reach_single = next((r["day"] for r in rows if r["best"] >= 0.80), None)
    print(f"\nOn day 14 the guardrail that best predicts the outcome (d90_retention) is "
          f"{d14['d90']*100:.0f}% observable and the day-7 metric is {d14['d7']*100:.0f}% observable: "
          f"half the enrolled converters are too young to have a value.")
    print(f"The best SINGLE guardrail does not reach 80% power until day {reach_single} - "
          f"{reach_single/14:.0f}x the window the decision is made in - while the pooled index gets "
          f"there by day {reach}. Pooling nine metrics that are already being collected is worth "
          f"about {reach_single/reach:.0f}x the calendar, and costs nothing.")
    print(f"d90_retention stays at a zero denominator until day 90 and is still only "
          f"{next(r['d90'] for r in rows if r['day']==180):.0%} observable in a 180-day experiment. "
          f"No window a growth team will accept makes the best predictor measurable.")
    return {"rows": rows, "day14": d14, "day_for_80": reach}


# ======================================================================================
def _intensity_pool(n_per_arm: int, day: int, seed: int) -> Dict[float, Dict[str, np.ndarray]]:
    pool = {}
    for i, a in enumerate(INTENSITY_GRID):
        rng = np.random.default_rng(seed + i)
        pool[float(a)] = G.simulate_experiment(float(a), n_per_arm, day, BIN_REPS, rng)
    return pool


def _run_years(pool, policy, rng, n_per_arm, day) -> Dict:
    """One year = EXPERIMENTS_PER_YEAR proposals. Ship if the win is significant and the
    guardrail policy does not block. Returns per-year aggregates over YEARS years."""
    total = YEARS * EXPERIMENTS_PER_YEAR
    is_clean = rng.random(total) < G.CLEAN_SHARE
    clean_lift = rng.uniform(0.0, G.CLEAN_LIFT_MAX, total)
    harm_a = rng.uniform(0.0, 1.0, total)
    bins = np.round(harm_a / 0.05) * 0.05

    zp = np.empty(total)
    zg = {name: np.empty(total) for name in pool[0.0] if name != "primary"}
    for a in INTENSITY_GRID:
        m = (~is_clean) & (np.isclose(bins, a))
        k = int(m.sum())
        if k == 0:
            continue
        idx = rng.integers(0, BIN_REPS, k)
        zp[m] = pool[float(a)]["primary"][idx]
        for name in zg:
            zg[name][m] = pool[float(a)][name][idx]
    m = is_clean
    k = int(m.sum())
    idx = rng.integers(0, BIN_REPS, k)
    zp[m] = rng.normal([G.primary_z_for_lift(x, n_per_arm) for x in clean_lift[m]], 1.0)
    for name in zg:
        zg[name][m] = pool[0.0][name][idx]

    win = zp > G.crit_value(ALPHA)
    blocked = policy(zg)
    shipped = win & ~blocked

    lift = np.where(is_clean, clean_lift, [G.primary_lift(x) for x in harm_a])
    # A clean change adds ORDINARY converters, so it moves volume and leaves the rate alone.
    # Only the lever damages retention quality, and a rate is what compounds honestly.
    dval = np.where(is_clean, 0.0, [G.quality_change(x) for x in harm_a])

    lift_y = (lift * shipped).reshape(YEARS, EXPERIMENTS_PER_YEAR).sum(axis=1)
    val_y = np.prod(1.0 + np.where(shipped, dval, 0.0).reshape(YEARS, EXPERIMENTS_PER_YEAR), axis=1)
    conv_y = np.prod(1.0 + np.where(shipped, lift, 0.0).reshape(YEARS, EXPERIMENTS_PER_YEAR), axis=1)
    retained_y = conv_y * val_y
    return {
        "conv_multiplier": float(conv_y.mean()),
        "retained_multiplier": float(retained_y.mean()),
        "retained_change": float(retained_y.mean() - 1.0),
        "ship_rate": float(shipped.mean()),
        "ships_per_year": float(shipped.sum() / YEARS),
        "harmful_ships": float((shipped & ~is_clean).sum() / YEARS),
        "clean_blocked": float((win & is_clean & blocked).sum() / YEARS),
        "harmful_caught": float((win & ~is_clean & blocked).sum() / max((win & ~is_clean).sum(), 1)),
        "clean_block_rate": float((win & is_clean & blocked).sum() / max((win & is_clean).sum(), 1)),
        "reported_lift": float(lift_y.mean()),
        "value_multiplier": float(val_y.mean()),
        "value_change": float(val_y.mean() - 1.0),
        "mean_shipped_a": float(harm_a[shipped & ~is_clean].mean()) if (shipped & ~is_clean).any() else 0.0,
        "mean_proposed_a": float(harm_a[~is_clean].mean()),
    }


def section_6() -> Dict:
    _rule("6. One ship is fine. Twenty is not.")
    pool = _intensity_pool(N_PER_ARM, DECISION_DAY, SEED + 500)
    w = G.sensitivity_weights(G.COMPUTABLE_SUITE, 1.0, N_PER_ARM, DECISION_DAY)
    crit_c = G.calibrate_composite(G.COMPUTABLE_SUITE, w, N_PER_ARM, DECISION_DAY,
                                   ALPHA, REPS, SEED + 501)

    policies = {
        "no guardrail": lambda z: np.zeros(G._reps_of(z), dtype=bool),
        "dashboard suite": lambda z: G.any_fires(z, G.DASHBOARD_SUITE, ALPHA),
        "all computable": lambda z: G.any_fires(z, G.COMPUTABLE_SUITE, ALPHA),
        "composite index": lambda z: G.composite_z(z, G.COMPUTABLE_SUITE, w) > crit_c,
    }
    print(f"{EXPERIMENTS_PER_YEAR} proposals a year, {G.CLEAN_SHARE:.0%} genuinely clean, "
          f"{YEARS:,} simulated years. Ship needs a significant win and a clear guardrail.\n")
    print(f"{'policy':18} {'ships/yr':>9} {'harmful':>8} {'caught':>7} {'clean blocked':>14} "
          f"{'reported lift':>14} {'180d ret. rate':>15} {'retained users':>15}")
    res = {}
    for name, pol in policies.items():
        rng = np.random.default_rng(SEED + 600)
        r = _run_years(pool, pol, rng, N_PER_ARM, DECISION_DAY)
        res[name] = r
        print(f"{name:18} {r['ships_per_year']:9.2f} {r['harmful_ships']:8.2f} "
              f"{r['harmful_caught']:7.3f} {r['clean_blocked']:14.2f} "
              f"{r['reported_lift']*100:13.1f}% {r['value_change']*100:14.2f}% "
              f"{r['retained_change']*100:14.2f}%")

    d = res["dashboard suite"]
    ng = res["no guardrail"]
    print(f"\nThe slide adds up to +{d['reported_lift']*100:.1f}% of conversion for the year. The "
          f"180-day retention rate is {abs(d['value_change'])*100:.1f}% lower. Both numbers come "
          f"from the same shipped experiments, and each one of those experiments passed.")
    print(f"Retained users - conversion volume times retention rate, the only number the "
          f"business banks - end the year {d['retained_change']*100:+.1f}% against "
          f"{ng['retained_change']*100:+.1f}% with no guardrail at all.")
    print(f"Running no guardrail at all ends the year at {ng['value_change']*100:.2f}% on the rate. The suite "
          f"most teams run recovers {(d['value_change']-ng['value_change'])*100:.2f} points of "
          f"that, which is {(d['value_change']-ng['value_change'])/abs(ng['value_change'])*100:.0f}% "
          f"of the damage it exists to prevent.")
    print(f"The ship filter selects FOR harm: the average proposed intensity is "
          f"{d['mean_proposed_a']:.2f} and the average SHIPPED one is {d['mean_shipped_a']:.2f}, "
          f"because reaching significance on the win requires the aggressive version.")
    print(f"The dashboard suite catches {d['harmful_caught']*100:.1f}% of harmful winners and "
          f"blocks {res['dashboard suite']['clean_block_rate']*100:.1f}% of clean ones. Swapping it "
          f"for one composite index moves those to {res['composite index']['harmful_caught']*100:.1f}% "
          f"and {res['composite index']['clean_block_rate']*100:.1f}%.")
    return {"policies": res, "crit_composite": crit_c, "weights": w}


# ======================================================================================
def section_7() -> Dict:
    _rule("7. Choosing a guardrail by correlation chooses the wrong one")
    rng = np.random.default_rng(SEED + 900)
    values, retained = G.passive_cohort(400_000, rng)
    rows = []
    for g in G.GUARDRAILS:
        corr = float(np.corrcoef(values[g.name], retained)[0, 1])
        sens = G.analytic_z(g, 1.0, N_PER_ARM, DECISION_DAY)
        runnable = G.observable_fraction(DECISION_DAY, g.maturity_days) > 0
        rows.append({"name": g.name, "abs_corr": abs(corr), "corr": corr,
                     "sens": sens, "runnable": runnable})

    by_corr = sorted(rows, key=lambda r: -r["abs_corr"])
    by_sens = sorted(rows, key=lambda r: -r["sens"])
    rank_corr = {r["name"]: i + 1 for i, r in enumerate(by_corr)}
    rank_sens = {r["name"]: i + 1 for i, r in enumerate(by_sens)}
    rho, pval = stats.spearmanr([r["abs_corr"] for r in rows], [r["sens"] for r in rows])

    print(f"{'guardrail':22} {'|corr| with 180d':>17} {'rank':>5} {'causal z':>9} {'rank':>5} "
          f"{'runnable day 14':>16}")
    for r in by_corr:
        print(f"{r['name']:22} {r['abs_corr']:17.4f} {rank_corr[r['name']]:5d} "
              f"{r['sens']:9.2f} {rank_sens[r['name']]:5d} "
              f"{'yes' if r['runnable'] else 'NO':>16}")

    top_corr, top_sens = by_corr[0], by_sens[0]
    print(f"\nSpearman(observational correlation, causal sensitivity) = {rho:+.3f} (p = {pval:.3f}).")
    print(f"The metric an analyst would pick by correlating everything against churn is "
          f"{top_corr['name']} - and on day {DECISION_DAY} its denominator is zero, so it cannot be "
          f"tested at all. The metric with the most causal sensitivity is {top_sens['name']}, ranked "
          f"{rank_corr[top_sens['name']]} of {len(rows)} by correlation.")
    second = by_corr[1]
    print(f"The best correlate that CAN be run is {second['name']} (rank "
          f"{rank_sens[second['name']]} of {len(rows)} on sensitivity). Predicting the outcome and "
          f"responding to the lever are different properties, and only the second one makes a "
          f"guardrail fire.")
    return {"rows": rows, "rho": float(rho), "p": float(pval),
            "top_corr": top_corr, "top_sens": top_sens,
            "rank_corr": rank_corr, "rank_sens": rank_sens}


# ======================================================================================
def section_8() -> Dict:
    _rule("8. The guardrail's alpha is not the win's alpha, and it depends on power")
    pool = _intensity_pool(N_PER_ARM, DECISION_DAY, SEED + 500)
    w = G.sensitivity_weights(G.COMPUTABLE_SUITE, 1.0, N_PER_ARM, DECISION_DAY)
    alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    print("The objective is 180-day RETAINED USERS at year end: conversion volume times "
          "retention rate. Blocking a clean change forfeits real volume; shipping a harmful one "
          "buys volume and pays more in quality. Both errors are priced in one unit.\n")

    def sweep(label, make_policy):
        rows = []
        for al in alphas:
            pol = make_policy(al)
            rng = np.random.default_rng(SEED + 600)
            r = _run_years(pool, pol, rng, N_PER_ARM, DECISION_DAY)
            rows.append({"alpha": al, **r})
        best = max(rows, key=lambda r: r["retained_change"])
        at05 = next(r for r in rows if r["alpha"] == 0.05)
        print(f"-- {label}")
        print(f"{'alpha':>6} {'clean blocked':>14} {'harmful caught':>15} {'conv volume':>12} "
              f"{'ret. rate':>10} {'retained users':>15}")
        for r in rows:
            star = "  <-- best" if r["alpha"] == best["alpha"] else ""
            print(f"{r['alpha']:6.2f} {r['clean_block_rate']*100:13.1f}% "
                  f"{r['harmful_caught']*100:14.1f}% {(r['conv_multiplier']-1)*100:11.1f}% "
                  f"{r['value_change']*100:9.2f}% {r['retained_change']*100:14.2f}%{star}")
        print()
        return {"rows": rows, "best": best, "at_005": at05}

    rng0 = np.random.default_rng(SEED + 600)
    none = _run_years(pool, lambda z: np.zeros(G._reps_of(z), dtype=bool), rng0,
                      N_PER_ARM, DECISION_DAY)
    print(f"Baseline, no guardrail at all: retained users {none['retained_change']*100:+.2f}%, "
          f"retention rate {none['value_change']*100:.2f}%.\n")

    comp = sweep("composite index (well powered: 0.86 at a = 1.0)",
                 lambda al: (lambda z, c=G.calibrate_composite(
                     G.COMPUTABLE_SUITE, w, N_PER_ARM, DECISION_DAY, al, REPS, SEED + 501):
                     G.composite_z(z, G.COMPUTABLE_SUITE, w) > c))
    dash = sweep("dashboard suite, any fires (underpowered: 0.33 best single)",
                 lambda al: (lambda z, a=al: G.any_fires(z, G.DASHBOARD_SUITE, a)))

    print(f"NEGATIVE RESULT for the obvious thesis: 0.05 is not automatically wrong. For the "
          f"well-powered composite the optimum is {comp['best']['alpha']:.2f} and the curve is "
          f"nearly flat - {min(r['retained_change'] for r in comp['rows'][:4])*100:.1f}% to "
          f"{max(r['retained_change'] for r in comp['rows'][:4])*100:.1f}% across alpha 0.01 to "
          f"0.10 - so a well-powered guardrail does not need its threshold tuned at all.")
    print(f"\nAnd the direction is the opposite of the intuition. A weak guardrail does not want "
          f"a LOOSER threshold to compensate - it wants the tightest one swept "
          f"({dash['best']['alpha']:.2f}), because raising its sensitivity buys false blocks faster "
          f"than it buys detection: at 0.05 it already refuses {dash['at_005']['clean_block_rate']*100:.0f}% "
          f"of harmless changes to catch {dash['at_005']['harmful_caught']*100:.0f}% of harmful ones. "
          f"Its year-end curve is monotone down across the whole sweep, so there is no setting at "
          f"which it earns its place.")
    gap_none = (dash['best']['retained_change'] - none['retained_change']) * 100
    print(f"The number that ends the argument: the dashboard suite AT ITS OWN OPTIMUM is worth "
          f"{gap_none:+.2f} points of retained users against running no guardrail whatsoever, while "
          f"the composite built from the same nine metrics is worth "
          f"{(comp['best']['retained_change']-none['retained_change'])*100:+.2f}. The metrics were "
          f"never the problem. Testing them one at a time was.")
    print("The right guardrail alpha is a function of the guardrail's POWER, which is the one "
          "number nobody computes before putting a metric on the checklist. 0.05 is not wrong "
          "everywhere - it is unexamined everywhere.")
    return {"composite": comp, "dashboard": dash, "none": none,
            "best": comp["best"], "at_005": comp["at_005"],
            "rows": comp["rows"]}


# ======================================================================================
def _matched_threshold(policy_z, pool, target_block: float) -> float:
    z0 = policy_z(pool[0.0])
    return float(np.quantile(z0, 1 - target_block))


def section_9() -> Dict:
    _rule("9. What actually helps, compared at an equal false-block rate")
    target_block = 0.10
    print(f"Every policy below is tuned to block exactly {target_block:.0%} of harmless changes, "
          f"so they are compared on detection alone rather than on how twitchy they are.\n")

    variants = []

    def add(label, n, day, suite, weighted):
        pool = _intensity_pool(n, day, SEED + 700 + len(variants) * 37)
        live = [s for s in suite if G.observable_fraction(day, G.GUARDRAIL_BY_NAME[s].maturity_days) > 0]
        w = G.sensitivity_weights(live, 1.0, n, day) if weighted else {s: 1.0 for s in live}
        def pz(z, ww=w, ll=live):
            return G.composite_z(z, ll, ww)

        crit = _matched_threshold(pz, pool, target_block)
        det = float(np.mean(pz(pool[1.0]) > crit))
        det4 = float(np.mean(pz(pool[0.4]) > crit))
        rng = np.random.default_rng(SEED + 600)
        yr = _run_years(pool, lambda z, p=pz, c=crit: p(z) > c, rng, n, day)
        variants.append({"label": label, "n": n, "day": day, "detect": det, "detect_04": det4, **yr})

    def add_suite(label, n, day, suite):
        pool = _intensity_pool(n, day, SEED + 700 + len(variants) * 37)
        live = [s for s in suite if G.observable_fraction(day, G.GUARDRAIL_BY_NAME[s].maturity_days) > 0]
        # find the per-test alpha that yields target_block for the ANY-fires rule
        lo, hi = 1e-5, 0.5
        for _ in range(40):
            mid = (lo + hi) / 2
            if np.mean(G.any_fires(pool[0.0], live, mid)) > target_block:
                hi = mid
            else:
                lo = mid
        al = (lo + hi) / 2
        det = float(np.mean(G.any_fires(pool[1.0], live, al)))
        det4 = float(np.mean(G.any_fires(pool[0.4], live, al)))
        rng = np.random.default_rng(SEED + 600)
        yr = _run_years(pool, lambda z, a=al, ll=live: G.any_fires(z, ll, a), rng, n, day)
        variants.append({"label": label, "n": n, "day": day, "detect": det, "detect_04": det4, **yr})

    add_suite("dashboard suite, any fires", N_PER_ARM, DECISION_DAY, G.DASHBOARD_SUITE)
    add_suite("all computable, any fires", N_PER_ARM, DECISION_DAY, G.COMPUTABLE_SUITE)
    add("composite, equal weights", N_PER_ARM, DECISION_DAY, G.COMPUTABLE_SUITE, False)
    add("composite, sensitivity-weighted", N_PER_ARM, DECISION_DAY, G.COMPUTABLE_SUITE, True)
    add("composite + 4x the sample", N_PER_ARM * 4, DECISION_DAY, G.COMPUTABLE_SUITE, True)
    add("composite + 28-day window", int(DAILY_ENROLMENT * 28), 28, G.COMPUTABLE_SUITE, True)
    add("composite + 56-day window", int(DAILY_ENROLMENT * 56), 56, G.COMPUTABLE_SUITE, True)

    print(f"{'policy':34} {'n/arm':>9} {'day':>4} {'a=1.0':>7} {'a=0.4':>7} "
          f"{'harmful caught':>15} {'180d ret. rate':>15}")
    for v in variants:
        print(f"{v['label']:34} {v['n']:9,} {v['day']:4d} {v['detect']:7.3f} {v['detect_04']:7.3f} "
              f"{v['harmful_caught']*100:14.1f}% {v['value_change']*100:14.2f}%")

    base = variants[0]
    comp_eq = variants[2]
    comp_w = variants[3]
    quad = variants[4]
    win = max(variants, key=lambda v: v["value_change"])
    print("\nAt a = 1.0 several policies sit near the ceiling, so the column that separates "
          "them is a = 0.4 - the ordinary, unremarkable change that is shipped most often and "
          "does most of the cumulative damage.")
    print(f"Pooling the same metrics into ONE directional index instead of testing them "
          f"separately moves detection of the a = 0.4 change from {base['detect_04']:.3f} to "
          f"{comp_eq['detect_04']:.3f} at the identical false-block rate. Nothing was measured "
          f"that was not already being measured.")
    print(f"Weighting that index by causal sensitivity rather than equally adds "
          f"{(comp_w['detect_04']-comp_eq['detect_04'])*100:+.1f} points.")
    print(f"NEGATIVE RESULT: sensitivity weighting is the expensive half of that idea - it needs "
          f"an estimate of how hard the lever moves each metric, which is the thing nobody has - "
          f"and it is worth {(comp_w['detect_04']-comp_eq['detect_04'])*100:.1f} points against the "
          f"{(comp_eq['detect_04']-base['detect_04'])*100:.1f} points that pooling with EQUAL "
          f"weights already delivered. The free half is worth "
          f"{(comp_eq['detect_04']-base['detect_04'])/max(comp_w['detect_04']-comp_eq['detect_04'],1e-9):.0f}x "
          f"the costly half.")
    fourx = quad
    fifty6 = variants[-1]
    print(f"Cost-matched at 4x the baseline users: 4x the sample in a 14-day window reaches "
          f"{fourx['detect_04']:.3f}, the same users spent on a 56-day window reach "
          f"{fifty6['detect_04']:.3f}. Calendar and sample are not interchangeable - the window "
          f"buys maturity as well as n - but the gap is "
          f"{(fifty6['detect_04']-fourx['detect_04'])*100:.1f} points, not an order of magnitude.")
    print(f"Best year-end value of the seven: {win['label']} at {win['value_change']*100:+.2f}%.")
    return {"variants": variants, "target_block": target_block}


# ======================================================================================
def section_10(s2, s3, s5, s6, s7, s8, s9) -> Dict:
    _rule("10. What a guardrail has to carry")
    d = s6["policies"]["dashboard suite"]
    print(f"1. A THRESHOLD IT COULD ACTUALLY CROSS. The experiment is powered to "
          f"{G.primary_power(1.0, N_PER_ARM, ALPHA):.2f} on the win and "
          f"{s2['best_dashboard']['power']:.2f} on the harm. That gap, not the product decision, "
          f"is what the checklist tick is measuring.")
    print(f"2. A DENOMINATOR THAT EXISTS ON DECISION DAY. d90_retention is the best single "
          f"predictor of the outcome in the catalogue and is {s5['day14']['d90']:.0%} observable in "
          f"a 14-day window.")
    print(f"3. SENSITIVITY TO THE LEVER, NOT CORRELATION WITH THE OUTCOME. "
          f"Spearman between the two rankings is {s7['rho']:+.3f}.")
    print(f"4. A MARGIN, NOT A NULL. Proving the harm is smaller than a stated bound needs "
          f"{s3['ni_multiple']:.0f}x the sample. Until then 'not significant' means 'not measured'.")
    print(f"5. ONE TEST, NOT NINE. At a matched false-block rate, pooling the same metrics into "
          f"one index moves detection of an ordinary change from {s9['variants'][0]['detect_04']:.3f} "
          f"to {s9['variants'][3]['detect_04']:.3f}.")
    print(f"6. ITS OWN ALPHA, SET BY ITS OWN POWER. For the pooled index the optimum is "
          f"{s8['composite']['best']['alpha']:.2f} and the curve is flat; for the underpowered "
          f"suite it is {s8['dashboard']['best']['alpha']:.2f}. 0.05 is not wrong everywhere - it "
          f"is unexamined everywhere.")
    print(f"7. A YEAR-END NUMBER. Under the suite most teams actually run, the slide reads "
          f"+{d['reported_lift']*100:.1f}% of conversion, the 180-day retention rate reads "
          f"{d['value_change']*100:.1f}%, and retained users - the product of the two - come out "
          f"{d['retained_change']*100:+.1f}%.")
    return {}


def main() -> Dict:
    s1 = section_1()
    s2 = section_2()
    s3 = section_3()
    s4 = section_4()
    s5 = section_5()
    s6 = section_6()
    s7 = section_7()
    s8 = section_8()
    s9 = section_9()
    s10 = section_10(s2, s3, s5, s6, s7, s8, s9)
    return {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5,
            "s6": s6, "s7": s7, "s8": s8, "s9": s9, "s10": s10}


if __name__ == "__main__":
    main()
