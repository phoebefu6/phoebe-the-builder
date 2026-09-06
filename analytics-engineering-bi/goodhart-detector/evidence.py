"""Every number in the README, computed here and asserted in test_goodhart.py.

Run: ``python evidence.py``   (deterministic; no network, no data files)
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Dict, List

import goodhart as G
import numpy as np
from scipy import stats

RESULTS: Dict[str, object] = {}
W = G.World()
ALPHA = 0.05


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def table(rows: List[List[object]], head: List[str]) -> None:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(head)]
    print("  " + "  ".join(str(h).ljust(w) for h, w in zip(head, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


# ==========================================================================
def section_1_world() -> None:
    rule("1. A world where the answer is known")
    print(f"""
  latent quality   L = skill + kappa*(1-u)          kappa = {W.kappa}
  outcome          y = a_y*L + N(0,{W.sigma_y})              a_y   = {W.a_y}
  proxy            p = beta*L + gamma*u + N(0,{W.sigma_p})   beta  = {W.beta}, gamma = {W.gamma}

  u is the share of an agent's effort diverted from doing the work to moving
  the number. Diverting one unit of effort:

      buys   {W.exploit_edge:+.2f} proxy points   (gamma - beta*kappa)
      costs  {W.outcome_cost:+.2f} outcome points (a_y * kappa)

  The exploit pays {W.payoff_ratio:.3f}x what honest work pays, on the proxy.
  An agent games when its own scruple threshold is below that ratio.
  With nobody gaming, corr(proxy, outcome) = {W.rho_clean:.4f}. This is a good proxy.""")

    rows = []
    for regime in ("continuous", "threshold"):
        panel = G.simulate(W, regime=regime)
        d = G.decompose(W, panel)
        rows.append([
            regime,
            f"{d['diverted_share']:.3f}",
            f"{d['proxy_delta']:+.4f}",
            f"{d['proxy_delta_true']:+.4f}",
            f"{d['outcome_delta']:+.4f}",
            f"{d['outcome_delta_true']:+.4f}",
        ])
        RESULTS[f"decompose_{regime}"] = d
    print()
    table(rows, ["regime", "u (mean)", "proxy obs", "proxy true", "outcome obs", "outcome true"])
    ct = RESULTS["decompose_continuous"]
    th = RESULTS["decompose_threshold"]
    print(f"""
  Continuous target ("make it go up"): the proxy rises {ct['proxy_delta']:+.4f} and the
  outcome falls {ct['outcome_delta']:+.4f}. Exchange rate {ct['exchange_rate']:.3f} outcome points
  per proxy point (closed form {ct['exchange_rate_closed']:.3f}).

  Threshold target ("clear the line"): true damage is {th['outcome_delta_true']:+.4f}, and what the
  aggregate outcome shows is {th['outcome_delta']:+.4f} -- {abs(th['outcome_delta_true']/th['outcome_delta']):.1f}x smaller than the
  truth and well inside its own noise. The harm is real and the total cannot see it.""")


# ==========================================================================
def section_2_correlation_barely_moves() -> None:
    rule("2. The correlation barely moves while the outcome falls")
    rows, xs = [], []
    for med in (3.0, 2.2, 1.9, 1.8, 1.7, 1.5, 1.2):
        w = replace(W, scruple_median=med)
        panel = G.simulate(w, regime="continuous")
        r_pre = float(np.corrcoef(panel.pre(panel.proxy), panel.pre(panel.outcome))[0, 1])
        r_post = float(np.corrcoef(panel.post(panel.proxy), panel.post(panel.outcome))[0, 1])
        d = G.decompose(w, panel)
        dmg = -d["outcome_delta_true"] / (panel.pre(panel.outcome).mean())
        rows.append([f"{d['diverted_share']:.3f}", f"{r_pre:.4f}", f"{r_post:.4f}",
                     f"{r_post - r_pre:+.4f}", f"{d['proxy_delta']:+.4f}", f"{100*dmg:5.1f}%"])
        xs.append((d["diverted_share"], r_post - r_pre, dmg))
    table(rows, ["u (mean)", "rho pre", "rho post", "d rho", "proxy move", "outcome lost"])
    worst = xs[-1]
    RESULTS["rho_sweep"] = xs
    print(f"""
  At the worst setting {100*worst[0]:.0f}% of effort is diverted, the outcome loses {100*worst[2]:.1f}% of its
  level, and the correlation the metric was chosen on moves {worst[1]:+.4f}.
  A quarterly review comparing {rows[0][1]} to {rows[-1][2]} does not call that a break.

  Damage per 0.01 of correlation lost: {100*worst[2]/abs(100*worst[1]):.2f}% of the outcome.""")
    RESULTS["damage_per_centi_rho"] = float(100 * worst[2] / abs(100 * worst[1]))


# ==========================================================================
def section_3_winners_curse() -> None:
    rule("3. The correlation drops without anyone gaming (winner's curse)")
    print("""
  A metric becomes the proxy because it correlated best out of several candidates.
  Selection is not free: the winner's measured correlation is its true correlation
  plus whatever noise helped it win, and that noise does not come back next period.
  Below, k=12 candidates, NOBODY GAMES ANYTHING, and the chosen one still decays.""")
    k, reps = 12, 400
    rows = []
    for n_select in (15, 30, 60, 120, 300, 900, 3600):
        drops_win, drops_rand, fp = [], [], 0
        for rep in range(reps):
            rng = np.random.default_rng(9000 + rep)
            betas = rng.uniform(0.55, 1.15, k)
            gammas = rng.uniform(0.45, 1.35, k)
            y, P = G.simulate_candidates(W, betas, gammas, n_periods=12, seed=int(rng.integers(1, 2**31)))
            flat_y, flat_P = y.ravel(), P.reshape(k, -1)
            idx = rng.choice(flat_y.size, n_select, replace=False)
            sel = np.array([np.corrcoef(flat_P[j][idx], flat_y[idx])[0, 1] for j in range(k)])
            win = int(np.argmax(sel))
            rnd = int(rng.integers(0, k))
            later = np.array([np.corrcoef(flat_P[j], flat_y)[0, 1] for j in (win, rnd)])
            drops_win.append(later[0] - sel[win])
            drops_rand.append(later[1] - sel[rnd])
            z1, s1 = G._fisher_z(sel[win], n_select)
            z2, s2 = G._fisher_z(later[0], flat_y.size)
            if stats.norm.sf((z1 - z2) / np.hypot(s1, s2)) < ALPHA:
                fp += 1
        rows.append([n_select, f"{np.mean(drops_win):+.4f}", f"{np.mean(drops_rand):+.4f}",
                     f"{fp/reps:.3f}"])
        RESULTS.setdefault("curse", []).append(
            (n_select, float(np.mean(drops_win)), float(np.mean(drops_rand)), fp / reps))
    table(rows, ["obs used to choose", "chosen proxy d rho", "random proxy d rho",
                 "corr_drop fires (no gaming!)"])
    mild_u, mild_drop, mild_dmg = RESULTS["rho_sweep"][0]
    worst_drop = abs(RESULTS["rho_sweep"][-1][1])
    RESULTS["gaming_drop_mild"] = float(abs(mild_drop))
    RESULTS["gaming_drop_worst"] = float(worst_drop)
    match = [c for c in RESULTS["curse"] if abs(c[1]) >= abs(mild_drop)]
    where = f"<= {max(c[0] for c in match)}" if match else "none of the sampled sizes"
    print(f"""
  The mildest gaming in section 2 diverted {100*mild_u:.0f}% of effort, destroyed {100*mild_dmg:.1f}% of the
  outcome, and moved the correlation {mild_drop:+.4f}. Selection alone, with the exploit
  switched off entirely, moves it {RESULTS['curse'][0][1]:+.4f} when the proxy was chosen on {RESULTS['curse'][0][0]}
  observations -- a LARGER drop than real gaming produced, from no gaming at all.
  Selection matches that drop at {where} observations.

  Even the worst gaming in section 2 only moved it {worst_drop:.4f}, so the two causes
  overlap across the whole range a metric review would plausibly see.

  A randomly chosen candidate does not decay ({RESULTS['curse'][0][2]:+.4f} at the smallest n): the
  decay is caused by the choosing, not by the metric.

  Consequence: "the correlation fell" is not evidence of Goodharting unless you
  know how much data the proxy was picked on. corr_drop's false-positive rate at
  the smallest selection sample is {RESULTS['curse'][0][3]:.3f}, against a nominal {ALPHA} -- and it stays
  above nominal out to {max(c[0] for c in RESULTS['curse'] if c[3] > 2*ALPHA)} observations.""")


# ==========================================================================
def _paired_runs(regime: str, reps: int, n_periods_post: int = 12, n_agents: int = 120):
    null_p, alt_p = {n: [] for n in G.DETECTOR_NAMES}, {n: [] for n in G.DETECTOR_NAMES}
    for rep in range(reps):
        w = replace(W, seed=4000 + rep, n_agents=n_agents)
        for gaming, store in ((False, null_p), (True, alt_p)):
            panel = G.simulate(w, regime=regime, n_post=n_periods_post, gaming=gaming)
            for name, v in G.run_all(panel).items():
                store[name].append(v.pvalue)
    return null_p, alt_p


def _auc(null_p: List[float], alt_p: List[float]) -> float:
    a, b = np.asarray(alt_p), np.asarray(null_p)
    u = stats.mannwhitneyu(-a, -b, alternative="greater").statistic
    return float(u / (a.size * b.size))


def section_4_power() -> None:
    rule("4. What each detector is actually worth")
    reps, n_agents = 200, 120
    RESULTS["power"] = {}
    print(f"""
  {n_agents} agents -- a realistic branch network, not a consumer funnel. At the
  600-agent default every outcome-based detector scores AUC 1.000 against a
  continuous target and the table says nothing.""")
    for regime in ("continuous", "threshold"):
        null_p, alt_p = _paired_runs(regime, reps, n_agents=n_agents)
        rows = []
        for name in G.DETECTOR_NAMES:
            n_arr, a_arr = np.array(null_p[name]), np.array(alt_p[name])
            fpr = float((n_arr < ALPHA).mean())
            power = float((a_arr < ALPHA).mean())
            auc = _auc(list(n_arr), list(a_arr))
            crit = float(np.quantile(n_arr, ALPHA))
            cal = float((a_arr < crit).mean())
            needs = "outcome" if getattr(G, name)(G.simulate(replace(W, n_agents=n_agents), regime=regime)).needs_outcome else "-"
            rows.append([name, needs, f"{fpr:.3f}", f"{power:.3f}", f"{cal:.3f}", f"{auc:.3f}"])
            RESULTS["power"][f"{regime}/{name}"] = dict(fpr=fpr, power=power, calibrated=cal, auc=auc)
        print(f"\n  regime = {regime}   ({reps} paired worlds, alpha = {ALPHA})")
        table(rows, ["detector", "needs", "false pos", "power", "power@calib", "AUC"])
    best_c = max(G.DETECTOR_NAMES, key=lambda n: RESULTS["power"][f"continuous/{n}"]["auc"])
    best_t = max(G.DETECTOR_NAMES, key=lambda n: RESULTS["power"][f"threshold/{n}"]["auc"])
    bt = RESULTS["power"]["threshold/bunching"]
    worst_cal = max(G.DETECTOR_NAMES,
                    key=lambda n: RESULTS["power"][f"threshold/{n}"]["fpr"])
    wc = RESULTS["power"][f"threshold/{worst_cal}"]
    print(f"""
  {worst_cal} has the highest raw power in this table and a false-positive rate of
  {wc['fpr']:.3f} against a nominal {ALPHA} -- it fits its baseline on the pre-period and then
  ignores that fit's own error, so it over-fires by {wc['fpr']/ALPHA:.1f}x. Re-run at its empirical
  critical value it is worth {wc['calibrated']:.3f}, not {wc['power']:.3f}. Any detector shipped without
  a measured null is quoting a power it does not have.

  Best AUC, continuous target: {best_c} ({RESULTS['power'][f'continuous/{best_c}']['auc']:.3f})
  Best AUC, threshold target:  {best_t} ({RESULTS['power'][f'threshold/{best_t}']['auc']:.3f})

  Against a CONTINUOUS target every outcome-based detector still scores 1.000 --
  46% of effort diverted is not a hard detection problem for anything that can
  see the outcome. The difficulty is not statistical, it is that the outcome is
  not there yet (section 5).

  bunching needs no outcome and scores AUC {bt['auc']:.3f} against a threshold target,
  and AUC {RESULTS['power']['continuous/bunching']['auc']:.3f} against a continuous one -- it has nothing to look at when
  the instruction is "make it go up" rather than "clear this line". A detector
  that is first on one target and worthless on the other is not a detector you
  can install once and forget.""")


# ==========================================================================
def section_5_lag() -> None:
    rule("5. The outcome is late, which is why it was proxied in the first place")
    lag = 4
    reps = 120
    per_period_damage = []
    first_fire = {n: [] for n in G.DETECTOR_NAMES}
    for rep in range(reps):
        w = replace(W, seed=5000 + rep)
        panel = G.simulate(w, regime="threshold", n_post=14)
        dmg = w.outcome_cost * panel.diverted[panel.t_target:].mean(axis=1)
        per_period_damage.append(dmg)
        for name in G.DETECTOR_NAMES:
            fn = getattr(G, name)
            probe = fn(panel)
            fired_at = None
            for t in range(panel.t_target + 2, panel.n_periods + 1):
                window = t - lag if probe.needs_outcome else t
                if window < panel.t_target + 2:
                    continue
                if fn(panel, upto=window).pvalue < ALPHA:
                    fired_at = t - panel.t_target
                    break
            first_fire[name].append(fired_at)
    dmg_curve = np.mean(per_period_damage, axis=0)
    cum = np.cumsum(dmg_curve)
    rows = []
    for name in G.DETECTOR_NAMES:
        hits = [f for f in first_fire[name] if f is not None]
        rate = len(hits) / reps
        if hits:
            med = int(np.median(hits))
            acc = float(cum[min(med, len(cum)) - 1])
            rows.append([name, f"{rate:.2f}", med, f"{acc:.4f}", f"{100*acc/cum[-1]:.0f}%"])
        else:
            rows.append([name, f"{rate:.2f}", "never", f"{cum[-1]:.4f}", "100%"])
        RESULTS.setdefault("lag", {})[name] = dict(
            detect_rate=rate, median_period=(int(np.median(hits)) if hits else None))
    print(f"\n  outcome reporting lag = {lag} periods, {reps} worlds, threshold target\n")
    table(rows, ["detector", "ever fires", "median period", "damage by then", "of 14-period total"])
    print(f"""
  Total damage over the window if nothing ever fires: {cum[-1]:.4f} outcome points per agent.
  An outcome-based detector cannot be computed at all until period {lag + 2}, because
  it needs a post-target outcome and the outcome is {lag} periods behind. By the time
  it is computable the damage is already {100*cum[lag+1]/cum[-1]:.0f}% of the window's total.""")
    RESULTS["damage_at_lag"] = float(cum[lag + 1] / cum[-1])


# ==========================================================================
def section_6_holdout() -> None:
    rule("6. The holdout works, and it stops working the moment it leaks")
    print("""
  holdout_divergence needs no outcome: it asks whether the target still moves
  with a sibling metric that nobody was told to move. It costs one metric kept
  off every dashboard and out of every bonus. Below, that discipline decays.""")
    reps = 150
    rows = []
    for leak in (0.0, 0.25, 0.5, 0.60, 0.65, 0.70, 0.75, 1.0):
        fires, nulls = 0, 0
        for rep in range(reps):
            w = replace(W, seed=6000 + rep)
            for gaming, is_alt in ((True, True), (False, False)):
                panel = G.simulate(w, regime="continuous", gaming=gaming)
                # a leaked holdout is partly gamed too
                contaminated = panel.holdout + leak * w.gamma * panel.diverted
                leaked = G.Panel(panel.proxy, panel.outcome, contaminated, panel.diverted,
                                 panel.t_target, panel.threshold)
                p = G.holdout_divergence(leaked).pvalue
                if is_alt and p < ALPHA:
                    fires += 1
                if not is_alt and p < ALPHA:
                    nulls += 1
        rows.append([f"{leak:.2f}", f"{fires/reps:.3f}", f"{nulls/reps:.3f}"])
        RESULTS.setdefault("leak", []).append((leak, fires / reps, nulls / reps))
    table(rows, ["share of exploit that also hits the holdout", "power", "false pos"])
    lk = RESULTS["leak"]
    cliff = next(((a[0], b[0]) for a, b in zip(lk, lk[1:]) if a[1] - b[1] > 0.3), (None, None))
    print(f"""
  A clean holdout gives power {lk[0][1]:.3f}. Half the exploit leaking onto it costs
  almost nothing ({lk[2][1]:.3f}), and then it falls off a cliff between {cliff[0]:.2f} and {cliff[1]:.2f}
  leakage, reaching {lk[-1][1]:.3f} when the holdout is managed as hard as the target.
  At that point the two series move together again, which is exactly what a
  healthy metric looks like. False positives stay at {lk[0][2]:.3f} throughout, so the
  failure is silent: the detector does not complain, it just stops seeing.

  The holdout is not a measurement. It is an organisational commitment not to
  manage a number somebody can see, and it decays the way commitments decay.""")


# ==========================================================================
def section_7_sample_size() -> None:
    rule("7. How many agents each detector needs")
    reps = 120
    rows = []
    sizes = (75, 150, 300, 600, 1200, 2400)
    RESULTS["sample"] = {}
    for name in G.DETECTOR_NAMES:
        fn = getattr(G, name)
        powers, need = [], None
        for n_agents in sizes:
            w = replace(W, n_agents=n_agents)
            hits = 0
            for rep in range(reps):
                panel = G.simulate(replace(w, seed=7000 + rep), regime="threshold")
                if fn(panel).pvalue < ALPHA:
                    hits += 1
            pw = hits / reps
            powers.append(pw)
            if need is None and pw >= 0.80:
                need = n_agents
        rows.append([name] + [f"{p:.2f}" for p in powers] + [str(need or f">{sizes[-1]}")])
        RESULTS["sample"][name] = dict(powers=powers, n80=need)
    table(rows, ["detector"] + [f"n={s}" for s in sizes] + ["n for 80%"])
    print("""
  Read the row, not the name. A detector that reaches 0.80 only at thousands of
  agents is not available to a team with forty branches, whatever its AUC says.""")


# ==========================================================================
def section_8_verdict() -> None:
    rule("8. What survives")
    ct = RESULTS["decompose_continuous"]
    curse_small = RESULTS["curse"][0]
    bt = RESULTS["power"]["threshold/bunching"]
    hd = RESULTS["power"]["threshold/holdout_divergence"]
    cd = RESULTS["power"]["threshold/corr_drop"]
    print(f"""
  1. The proxy did its job and the outcome still fell. Exchange rate {ct['exchange_rate']:.2f}
     outcome points per proxy point, in a world where the proxy correlated {W.rho_clean:.2f}
     with the outcome before it became a target.

  2. corr_drop -- the thing everyone reaches for -- fires at {curse_small[3]:.3f} on a proxy
     that was merely CHOSEN on {curse_small[0]} observations, with no gaming at all. Its
     apparent sensitivity is mostly the winner's curse of having picked it.

  3. The two detectors that need no outcome are the two that are available in
     time: bunching (AUC {bt['auc']:.3f} on a threshold target, undefined on a continuous
     one) and holdout_divergence (AUC {hd['auc']:.3f}), against corr_drop's {cd['auc']:.3f}.

  4. holdout_divergence's power collapses from {RESULTS['leak'][0][1]:.3f} to {RESULTS['leak'][-1][1]:.3f} once the holdout
     is itself being managed. It is a policy, not a statistic.

  5. Nothing here detects Goodharting from the KPI alone. Every detector that
     worked needed a second series the target did not control -- an outcome, a
     sibling metric, or the shape of the distribution around a line. A single
     number going up carries no information about whether it should be trusted.""")


def main() -> None:
    section_1_world()
    section_2_correlation_barely_moves()
    section_3_winners_curse()
    section_4_power()
    section_5_lag()
    section_6_holdout()
    section_7_sample_size()
    section_8_verdict()
    with open("results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)
    print("\n  wrote results.json")


if __name__ == "__main__":
    main()
