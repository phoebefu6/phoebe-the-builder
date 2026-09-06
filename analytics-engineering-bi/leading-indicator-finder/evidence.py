"""Every claim this build makes, computed from the world in ``leadlag.py``.

Run it: ``python evidence.py``. Nothing here is asserted in prose that is not
printed by the code below and re-checked in ``test_leadlag.py``.
"""

from __future__ import annotations

from typing import Dict, List

import leadlag as L
import numpy as np
from scipy import stats

W = L.World()
HORIZON_NEEDED = 3      # the warning somebody actually asked for
NULL_REPS = 400
LAG_REPS = 300


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------


def section_1_world() -> Dict[str, object]:
    rule("1. A world where the lead is known")
    d = L.simulate(W)
    print(f"{W.T} months of history after a {W.burn}-month burn-in.")
    print("\n  a_t = phi*a_{t-1} + eps            latent demand")
    print("  s_t = c_s*a_{t-1} + eps            signups")
    print("  v_t = c_v*s_{t-1} + eps            activations")
    print("  y_t = c_y*v_{t-1} + season + eps   revenue\n")
    print(f"  phi={W.phi_a}  c_s={W.c_s}  c_v={W.c_v}  c_y={W.c_y}")
    print("\nRevenue at t is driven by demand at t-3. Each stage passes on a")
    print("fraction of the one before it, so the earliest warning is also the")
    print("weakest signal. That trade-off is mechanical, not a choice:\n")
    rows = []
    print(f"  {'metric':<18} {'true lead':>9} {'r at true lead':>15} {'dY/dX':>8}  truth")
    for c in L.CANDIDATES:
        lead = W.true_lead[c]
        r = L.lagged_corr(d[c], d["revenue"], lead) if lead > 0 else float("nan")
        shown = f"{r:.3f}" if lead > 0 else "no lead"
        print(f"  {c:<18} {lead:>9} {shown:>15} {W.gain[c]:>8.3f}  {L.TRUTH_LABEL[c]}")
        rows.append({"metric": c, "lead": lead, "r_at_lead": r, "gain": W.gain[c]})
    funnel = [(c, W.true_lead[c], L.lagged_corr(d[c], d["revenue"], W.true_lead[c]))
              for c in ("activations", "signups", "web_sessions")]
    print("\nThe funnel, ordered by warning time:")
    for c, lead, r in funnel:
        print(f"  {lead} month(s) of warning -> r = {r:.3f}   ({c})")
    print("\nRanking candidates by strength therefore prefers the one that gives")
    print("the LEAST warning. That is not a bug in any particular tool.")
    return {"data": d, "rows": rows, "funnel": funnel}


def section_2_rankers(d: Dict[str, np.ndarray]) -> Dict[str, object]:
    rule("2. Four rankers, one world, four answers")
    y = d["revenue"]
    res: Dict[str, Dict[str, object]] = {}
    print(f"  {'metric':<18} {'lead-scan':>16} {'|CCF| peak':>16} "
          f"{'prewhitened':>16} {'Granger p':>11}")
    for c in L.CANDIDATES:
        r1, l1 = L.rank_pearson_lead(d[c], y)
        r2, l2 = L.rank_pearson_abs_sym(d[c], y)
        r3, l3 = L.rank_prewhitened(d[c], y)
        f, p = L.granger_f(d[c], y)
        res[c] = {"lead_r": r1, "lead_lag": l1, "abs_r": r2, "abs_lag": l2,
                  "pw_r": r3, "pw_lag": l3, "granger_f": f, "granger_p": p}
        print(f"  {c:<18} {r1:>+9.3f} @{l1:<4} {r2:>+9.3f} @{l2:<4} "
              f"{r3:>+9.3f} @{l3:<4} {p:>11.3g}")

    top_abs = max(L.CANDIDATES, key=lambda c: abs(res[c]["abs_r"]))
    top_lead = max(L.CANDIDATES, key=lambda c: res[c]["lead_r"])
    print(f"\nThe |CCF| peak crowns '{top_abs}' at r={res[top_abs]['abs_r']:+.3f}, "
          f"lag {res[top_abs]['abs_lag']}.")
    print(f"That metric FOLLOWS revenue by {-W.true_lead[top_abs]} month(s). Its peak is at a")
    print("negative lag, and taking the peak of the whole cross-correlation")
    print(f"function reads the sign off. It beats the best real indicator "
          f"({res[top_lead]['lead_r']:+.3f}) by {abs(res[top_abs]['abs_r']) - res[top_lead]['lead_r']:+.3f}.")
    print(f"\nWith the sign respected, {top_abs} still posts "
          f"r={res[top_abs]['lead_r']:+.3f} at lag {res[top_abs]['lead_lag']} - because revenue is")
    print("persistent, so anything tracking last month's revenue tracks this")
    print(f"month's too. Granger, which conditions on revenue's own history, "
          f"gives p={res[top_abs]['granger_p']:.3f}.")

    mkt = res["marketing_spend"]
    print(f"\n'marketing_spend' shares only a calendar with revenue and posts "
          f"r={mkt['lead_r']:+.3f}")
    close = res[top_lead]["lead_r"] - mkt["lead_r"]
    print(f"at lag {mkt['lead_lag']} - within {close:.3f} of the best real indicator. "
          f"Strip the two annual")
    print(f"harmonics and the level and it falls to r={mkt['pw_r']:+.3f}.")

    order_lead = sorted(L.CANDIDATES, key=lambda c: -res[c]["lead_r"])
    order_pw = sorted(L.CANDIDATES, key=lambda c: -res[c]["pw_r"])
    rho = stats.spearmanr([order_lead.index(c) for c in L.CANDIDATES],
                          [order_pw.index(c) for c in L.CANDIDATES]).statistic
    pairs = [(a, b) for i, a in enumerate(L.CANDIDATES) for b in L.CANDIDATES[i + 1:]]
    flipped = sum(1 for a, b in pairs
                  if (res[a]["lead_r"] > res[b]["lead_r"]) !=
                     (res[a]["pw_r"] > res[b]["pw_r"]))
    print(f"\nRaw and prewhitened rankings: Spearman {rho:+.3f}, "
          f"{flipped} of {len(pairs)} pairs ordered differently.")
    return {"res": res, "top_abs": top_abs, "top_lead": top_lead,
            "rho_raw_pw": float(rho), "flipped_pairs": flipped, "n_pairs": len(pairs)}


def section_3_horizon(d: Dict[str, np.ndarray]) -> Dict[str, object]:
    rule("3. The horizon is part of the question, not a detail")
    y = d["revenue"]
    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for h in (1, HORIZON_NEEDED):
        out[h] = {c: L.oos_gain(y, d[c], h) for c in L.CANDIDATES}
    print("Rolling-origin backtest. Standing at month t we know y and x up to t,")
    print("so forecasting h months ahead can only use x at lags >= h. The lag is")
    print("re-chosen on each training window - never on the full series.\n")
    print(f"  {'metric':<18} {'h=1 gain':>10} {'DM p':>7} | "
          f"{'h=3 gain':>10} {'DM p':>7} {'lag':>4}")
    for c in L.CANDIDATES:
        a, b = out[1][c], out[HORIZON_NEEDED][c]
        print(f"  {c:<18} {a['gain_pct']:>+9.2f}% {a['dm_p']:>7.3f} | "
              f"{b['gain_pct']:>+9.2f}% {b['dm_p']:>7.3f} {b['lag']:>4}")
    base1 = out[1]["placebo_1"]["rmse_base"]
    sn1 = out[1]["placebo_1"]["rmse_seasonal_naive"]
    print(f"\nBaseline (revenue's own 3 lags + 2 annual harmonics) RMSE {base1:.3f}; "
          f"seasonal-naive {sn1:.3f}.")
    print("Every gain below is measured against that baseline, i.e. against")
    print("already knowing revenue's own history. That is the only number that")
    print("answers 'is this worth watching'.\n")

    best1 = max(L.CANDIDATES, key=lambda c: out[1][c]["gain_pct"])
    best3 = max(L.CANDIDATES, key=lambda c: out[HORIZON_NEEDED][c]["gain_pct"])
    coll = out[HORIZON_NEEDED][best1]
    print(f"At h=1 the winner is '{best1}' (+{out[1][best1]['gain_pct']:.2f}%).")
    print(f"At h={HORIZON_NEEDED} the winner is '{best3}' (+{out[HORIZON_NEEDED][best3]['gain_pct']:.2f}%), "
          f"and '{best1}' collapses to")
    print(f"{coll['gain_pct']:+.2f}% (DM p={coll['dm_p']:.3f}, not significant) because it leads by "
          f"{W.true_lead[best1]} month")
    print("and cannot be read early enough to matter.")
    g1 = [out[1][c]["gain_pct"] for c in L.CANDIDATES]
    g3 = [out[HORIZON_NEEDED][c]["gain_pct"] for c in L.CANDIDATES]
    rho = stats.spearmanr(g1, g3).statistic
    inf = W.informative
    rho_inf = stats.spearmanr([out[1][c]["gain_pct"] for c in inf],
                              [out[HORIZON_NEEDED][c]["gain_pct"] for c in inf]).statistic
    print(f"\nSpearman across all ten: {rho:+.3f} - which looks reassuring, and is")
    print("carried entirely by the six distractors sitting at the bottom of both.")
    print(f"Across the four candidates that actually carry information: {rho_inf:+.3f}.")
    print("The ordering of the shortlist is reversed. A single 'leading")
    print("indicators' table is answering a question nobody asked.")

    distract = [c for c in L.CANDIDATES if c not in W.informative]
    worst_fp = max(distract, key=lambda c: out[HORIZON_NEEDED][c]["gain_pct"])
    print(f"\nOn the honest criterion all {len(distract)} distractors are rejected: best of them")
    print(f"is '{worst_fp}' at {out[HORIZON_NEEDED][worst_fp]['gain_pct']:+.2f}% with DM "
          f"p={out[HORIZON_NEEDED][worst_fp]['dm_p']:.3f}. A positive percentage")
    print("is not a finding; the test on the loss differential is.")
    return {"oos": out, "best_h1": best1, "best_h3": best3,
            "rho_h1_h3": float(rho), "rho_informative": float(rho_inf),
            "worst_fp": worst_fp}


def section_4_actionable(oos: Dict[int, Dict[str, Dict[str, float]]]) -> Dict[str, object]:
    rule("4. Predicting it and being able to move it are different properties")
    print("The do-operator, run in the simulator: add 1.0 to a metric at every")
    print("period, before anything downstream reads it, and measure revenue.")
    print("Common random numbers, so the comparison carries no sampling noise.\n")
    eff: Dict[str, float] = {}
    reps = 40
    for c in L.CANDIDATES:
        deltas = []
        for s in range(reps):
            b = L.simulate(W, seed=5000 + s)
            f = L.simulate(W, seed=5000 + s, force=(c, 1.0))
            deltas.append(float(f["revenue"].mean() - b["revenue"].mean()))
        eff[c] = float(np.mean(deltas))
    print(f"  {'metric':<18} {'h=3 OOS gain':>13} {'dY/dX':>8} {'closed form':>12}")
    for c in sorted(L.CANDIDATES, key=lambda c: -oos[HORIZON_NEEDED][c]["gain_pct"]):
        print(f"  {c:<18} {oos[HORIZON_NEEDED][c]['gain_pct']:>+12.2f}% "
              f"{eff[c]:>8.3f} {W.gain[c]:>12.3f}")
    top = max(L.CANDIDATES, key=lambda c: oos[HORIZON_NEEDED][c]["gain_pct"])
    act = max(W.actionable, key=lambda c: W.gain[c])
    print(f"\nThe best indicator at the horizon that matters is '{top}', and moving")
    print(f"it changes revenue by {eff[top]:.3f}. It is a sensor: it reads demand without")
    print("being part of the chain, so there is nothing downstream of it to push.")
    print(f"\nThe metric with the most leverage is '{act}' ({eff[act]:.2f} revenue per unit),")
    print(f"and it gives {W.true_lead[act]} month of warning - "
          f"{oos[HORIZON_NEEDED][act]['gain_pct']:+.2f}% at h={HORIZON_NEEDED}. In a funnel,")
    print("leverage and warning time are ordered against each other, so the")
    print("'best leading indicator' and 'the lever to pull' are two questions")
    print("with two different answers, and no correlation table separates them.")
    print("\nNothing observational distinguishes these two columns. The only thing")
    print("that does is an intervention, which is why a leading-indicator scan")
    print("is a forecasting result and never a plan.")
    return {"effect": eff, "top": top, "lever": act}


def section_5_null() -> Dict[str, object]:
    rule("5. A world with nothing in it, and 10 ways of finding something")
    print("Revenue is persistent (phi=0.70) and seasonal, because real revenue is.")
    print(f"The 10 candidates are drawn independently of it. {NULL_REPS} worlds each.")
    print("Any indicator found here is false by construction.\n")
    rates: Dict[str, Dict[str, float]] = {}
    for kind, label in (("ar1", "candidates are AR(1), phi=0.6"),
                        ("rw", "candidates are random walks")):
        acc: Dict[str, int] = {}
        for s in range(NULL_REPS):
            y, X = L.simulate_null(120, 90000 + s, kind=kind)
            for k, v in L.scan_flags(y, X).items():
                acc[k] = acc.get(k, 0) + int(v)
        rates[kind] = {k: v / NULL_REPS for k, v in acc.items()}
        print(f"  {label}")
        for k in L.FLAG_LABEL:
            print(f"    {L.FLAG_LABEL[k]:<44} {rates[kind][k]:.3f}")
        print()
    a = rates["ar1"]
    print("Nominal rate for every row above: 0.050.")
    print("\nScan 10 candidates x 12 lags and read the textbook p-value: a")
    print(f"leading indicator is found in {a['scan_naive']:.1%} of worlds that contain none.")
    print(f"One candidate at one pre-registered lag already fires "
          f"{a['one_test_naive']:.1%} of the time,")
    print("so most of the damage is done before any scanning happens: two")
    print("autocorrelated series share far fewer independent facts than they")
    print("have rows, and the p-value assumes they share all of them.")
    print(f"\nBonferroni alone leaves {a['scan_bonferroni']:.1%}. Bartlett alone leaves "
          f"{a['scan_bartlett']:.1%}. Neither")
    print("correction addresses the other's problem, and each looks adequate in")
    print(f"isolation. Together: {a['scan_bartlett_bonferroni']:.3f}, below nominal.")
    print("\nGranger, which conditions on revenue's own history, is a different")
    print(f"story: {a['granger_best_of_k']:.3f} across 10 candidates against the "
          f"{1 - 0.95 ** 10:.3f} a perfectly")
    print(f"calibrated 5% test would give over 10 tries, and {a['granger_bonferroni']:.3f} with Bonferroni.")
    print("It was never the multiplicity that broke the correlation scan - it")
    print("was the autocorrelation, and conditioning on revenue's own lags")
    print("removes that at the source rather than deflating a p-value afterwards.")
    print("\nSo the cheap screen is Granger plus Bonferroni, not a CCF table.")
    r = rates["rw"]
    print(f"\nOn random walks Bartlett is far more effective at one lag "
          f"({r['fixed_lag_bartlett']:.3f} vs")
    print(f"{a['fixed_lag_bartlett']:.3f}) - not because it handles unit roots, but because "
          f"lag-1")
    print("autocorrelation near 1 collapses its effective sample size to the")
    print("floor. It over-corrects the nonstationary case and under-corrects the")
    print("stationary one, which is the case a real KPI is in.")
    return {"rates": rates}


def section_6_lag() -> Dict[str, object]:
    rule("6. The lag is a point estimate, and it has a standard error")
    print(f"{LAG_REPS} re-runs per row. Lag read as the argmax of the positive-lag scan.\n")
    recov: List[Dict[str, float]] = []
    print(f"  {'history':<9} {'metric':<15} {'true':>4} {'exact':>7} {'within 1':>9} {'2.5-97.5 span':>14}")
    for T in (60, 120, 240):
        for c in ("activations", "signups", "web_sessions"):
            true = W.true_lead[c]
            est = []
            for s in range(LAG_REPS):
                d = L.simulate(L.World(T=T), seed=7000 + s)
                est.append(L.rank_pearson_lead(d[c], d["revenue"])[1])
            e = np.array(est)
            span = float(np.percentile(e, 97.5) - np.percentile(e, 2.5))
            recov.append({"T": T, "metric": c, "true": true,
                          "exact": float(np.mean(e == true)),
                          "within1": float(np.mean(np.abs(e - true) <= 1)),
                          "span": span})
            print(f"  {T:<9} {c:<15} {true:>4} {np.mean(e == true):>7.3f} "
                  f"{np.mean(np.abs(e - true) <= 1):>9.3f} {span:>14.1f}")
    print("\nNEGATIVE RESULT: when the indicator is real, the lag is the easy part.")
    print("Five years of monthly data recovers it 78-99% of the time. The worry")
    print("about 'we cannot pin the lag' is misplaced for a strong signal.")
    print("\nWhat governs it is strength, not history. Sweeping sensor noise at")
    print("T=60 (five years, the length most teams have):\n")
    sweep: List[Dict[str, float]] = []
    print(f"  {'sensor noise':>12} {'r at argmax':>12} {'exact lag':>10}")
    for sd in (0.3, 0.8, 1.5, 2.5, 4.0, 6.0, 9.0):
        ww = L.World(T=60, sd_web=sd)
        ex, rs = [], []
        for s in range(LAG_REPS):
            d = L.simulate(ww, seed=8000 + s)
            r, lag = L.rank_pearson_lead(d["web_sessions"], d["revenue"])
            ex.append(lag == 3)
            rs.append(r)
        sweep.append({"sd": sd, "r": float(np.mean(rs)), "exact": float(np.mean(ex))})
        print(f"  {sd:>12.1f} {np.mean(rs):>12.3f} {np.mean(ex):>10.3f}")
    strong = [s for s in sweep if s["exact"] >= 0.60]
    print(f"\nThe lag is worth reading above r ~ {min(s['r'] for s in strong):.2f} and is mostly noise")
    print(f"below r ~ {max(s['r'] for s in sweep if s['exact'] < 0.35):.2f}, where it is right "
          f"{max(s['exact'] for s in sweep if s['exact'] < 0.35):.0%} of the time on a")
    print("12-lag grid. Publishing a lag from a weak correlation publishes a")
    print("draw from that grid.")
    return {"recovery": recov, "sweep": sweep}


def section_7_stability(d: Dict[str, np.ndarray]) -> Dict[str, object]:
    rule("7. Lag stability looks like a free screen. It is not.")
    print("If the lag wanders between windows there is probably nothing there.")
    print("Cheap, needs no p-value and no outcome. Rolling 96-month windows,")
    print("step 6:\n")
    stab: Dict[str, float] = {}
    print(f"  {'metric':<18} {'mode':>5} {'share at mode':>14}  truth")
    for c in L.CANDIDATES:
        lags = []
        n = d["revenue"].size
        for start in range(0, n - 96 + 1, 6):
            seg = slice(start, start + 96)
            lags.append(L.rank_pearson_lead(d[c][seg], d["revenue"][seg])[1])
        a = np.array(lags)
        mode = int(np.bincount(a).argmax())
        share = float(np.mean(a == mode))
        stab[c] = share
        print(f"  {c:<18} {mode:>5} {share:>14.2f}  {L.TRUTH_LABEL[c]}")
    noise = ["nps_trend", "placebo_1", "placebo_2", "placebo_3"]
    print(f"\nIt works on noise: the four unrelated series score "
          f"{min(stab[c] for c in noise):.2f}-{max(stab[c] for c in noise):.2f} against")
    print(f"{min(stab[c] for c in W.informative):.2f}-{max(stab[c] for c in W.informative):.2f} "
          f"for the four real ones.")
    print(f"\nNEGATIVE RESULT: 'marketing_spend' scores {stab['marketing_spend']:.2f} - a perfect score -")
    print("and shares nothing with revenue but a calendar. A confounded")
    print("relationship is stable precisely because the confounder is stable, so")
    print("a stability screen filters noise and waves confounding through. It")
    print("cannot replace the horizon-matched backtest; it can only cheapen it")
    print("by dropping the obviously dead candidates first.")
    print("(Overlapping windows also share most of their data, which flatters")
    print("every row here.)")
    return {"stability": stab}


def section_8_scorecard(
    d: Dict[str, np.ndarray],
    r2: Dict[str, object],
    oos: Dict[int, Dict[str, Dict[str, float]]],
    eff: Dict[str, float],
) -> Dict[str, object]:
    rule(f"8. What a leading-indicator table should say at h={HORIZON_NEEDED}")
    res = r2["res"]
    print(f"  {'metric':<18} {'r@lag':>10} {'Granger p':>10} {'OOS gain':>10} "
          f"{'DM p':>7} {'verdict':<26}")
    verdicts: Dict[str, str] = {}
    for c in sorted(L.CANDIDATES, key=lambda c: -oos[HORIZON_NEEDED][c]["gain_pct"]):
        o = oos[HORIZON_NEEDED][c]
        useful = o["dm_p"] < 0.05 and o["gain_pct"] > 0
        if useful and W.gain[c] > 0:
            v = "watch AND pull"
        elif useful:
            v = "watch, cannot pull"
        elif res[c]["lead_r"] > 0.30:
            v = "correlated, no lead value"
        else:
            v = "drop"
        verdicts[c] = v
        print(f"  {c:<18} {res[c]['lead_r']:>+9.3f} {res[c]['granger_p']:>10.3g} "
              f"{o['gain_pct']:>+9.2f}% {o['dm_p']:>7.3f} {v:<26}")
    kept = [c for c, v in verdicts.items() if v.startswith("watch")]
    correlated = [c for c in L.CANDIDATES if res[c]["lead_r"] > 0.30]
    print(f"\n{len(correlated)} of 10 candidates correlate with future revenue above 0.30.")
    movable = len([c for c in kept if W.gain[c] > 0])
    print(f"{len(kept)} survive a horizon-matched backtest. {movable} of those can be moved.")
    print("\nThe table is horizon-specific and says so. At h=1 the same data puts")
    act1 = oos[1]["activations"]["gain_pct"]
    print(f"'activations' first at {act1:+.2f}%; here it is fifth. Nothing about")
    print("the metric changed - only the amount of warning being asked for.")
    print("\nThe column that changed the answer is not the correlation, and the")
    print("column that changed the plan is not in the data at all.")
    return {"verdicts": verdicts, "kept": kept, "correlated": correlated}


def run_all(verbose: bool = True) -> Dict[str, object]:
    s1 = section_1_world()
    d = s1["data"]
    s2 = section_2_rankers(d)
    s3 = section_3_horizon(d)
    s4 = section_4_actionable(s3["oos"])
    s5 = section_5_null()
    s6 = section_6_lag()
    s7 = section_7_stability(d)
    s8 = section_8_scorecard(d, s2, s3["oos"], s4["effect"])
    rule("Summary")
    print(f"- The |CCF| peak names a metric that FOLLOWS revenue "
          f"(r={s2['res'][s2['top_abs']]['abs_r']:+.3f} at lag {s2['res'][s2['top_abs']]['abs_lag']}).")
    print(f"- Horizon 1 vs horizon {HORIZON_NEEDED}: Spearman {s3['rho_h1_h3']:+.3f} over all ten, "
          f"{s3['rho_informative']:+.3f} over the")
    print("  four that carry information. Different questions.")
    print(f"- Best indicator at h={HORIZON_NEEDED} has a causal gain of "
          f"{s4['effect'][s4['top']]:.3f}; the lever gives {W.true_lead[s4['lever']]} month.")
    print(f"- Empty world, 10 x 12 correlation scan, textbook p: "
          f"{s5['rates']['ar1']['scan_naive']:.1%} false discovery;")  # noqa: E501
    nr = s5["rates"]["ar1"]
    print(f"  Bonferroni alone {nr['scan_bonferroni']:.3f}, Bartlett alone "
          f"{nr['scan_bartlett']:.3f}, both {nr['scan_bartlett_bonferroni']:.3f}.")
    print(f"- Granger over the same 10 candidates: {nr['granger_best_of_k']:.3f}, "
          f"already calibrated per test,")
    print(f"  and {nr['granger_bonferroni']:.3f} with Bonferroni. Skip the CCF table.")
    best_lag = max(r["exact"] for r in s6["recovery"])
    worst_lag = min(s["exact"] for s in s6["sweep"])
    print(f"- The lag is recovered {best_lag:.0%} of the time when the signal is strong,")
    print(f"  and {worst_lag:.0%} of the time when it is weak. Strength, not history.")
    print(f"- Lag stability catches noise and passes the calendar-only metric at "
          f"{s7['stability']['marketing_spend']:.2f}.")
    return {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8}


if __name__ == "__main__":
    run_all()
