"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). Every rate is closed-form under the declared model and
every tail is an exact binomial; the Monte Carlo section only checks them.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import numpy as np
import refusals as R

OUT: List[str] = []


def say(line: str = "") -> None:
    OUT.append(line)
    print(line)


def rule(title: str) -> None:
    say()
    say("=" * 100)
    say(title)
    say("=" * 100)


t0 = time.time()
MIX = R.mix_of()
KW = R.DETECTORS["keyword list (written on v1)"]
results: Dict[str, Any] = {"config": {"classes": R.CLASSES, "campaign_mix": R.CAMPAIGN_MIX, "versions": R.VERSIONS,
                                      "detectors": {k: {**v, "matches": sorted(v["matches"]) if v["matches"] else None}
                                                    for k, v in R.DETECTORS.items()},
                                      "weekly_n": R.WEEKLY_N, "audit_n": R.AUDIT_N, "weeks": R.WEEKS,
                                      "campaign_weeks": R.CAMPAIGN_WEEKS, "upgrade_week": R.UPGRADE_WEEK,
                                      "mc_reps": R.MC_REPS, "seed": R.SEED, "numpy": np.__version__}}

rule("0.  THE MODEL  -  synthetic, declared")
for c in R.CLASSES:
    say(f"  {c['name']:<10} {c['share']:>4.0%} of traffic   {'legitimate' if c['legit'] else 'to be refused'}   ({c['what']})")
say("  refusal_rate = sum share_c * refuse_c;  over-refusal = P(refuse | legitimate);  under-refusal = P(comply | harmful)")
say("  measured = sum share_c * (refuse_c * recall(detector, version) + (1 - refuse_c) * fpr);  recall = phrasing mass matched")
for k, v in R.VERSIONS.items():
    say(f"  {k:<22} refuse per class " + ", ".join(f"{c} {v['refuse'][c]:.3f}" for c in v["refuse"])
        + "   phrasing: " + ", ".join(f"'{p}' {q:.0%}" for p, q in v["phrasing"].items() if q >= 0.1))
for k, d in R.DETECTORS.items():
    say(f"  detector {k:<30} " + (f"matches {sorted(d['matches'])}" if d["matches"] else f"recall {d['recall']}")
        + f"   fpr {d['fpr']}")

rule("1.  CALIBRATION  -  closed form and exact tails vs raw simulation")
cal = R.calibrate()
results["calibration"] = cal
for r in cal:
    say(f"  {r['case']:<28} {r['quantity']:<28} exact {r['exact']:.4f}   MC {r['mc']:.4f}   {'inside' if r['inside_99'] else 'OUTSIDE'}")
say(f"  {sum(r['inside_99'] for r in cal)} of {len(cal)} inside their 99% Wilson interval at seed {R.SEED}, {R.MC_REPS:,} reps per case")

rule("2.  THREE VERSIONS, ONE MIX  -  the true rate, the two errors, and what each detector reports")
ver = {}
say(f"  {'version':<22} {'true rate':>9} {'over-ref':>8} {'under-ref':>9} {'legit share':>11} | "
    + " ".join(f"{k[:14]:>14}" for k in R.DETECTORS) + "   (measured)")
for k, v in R.VERSIONS.items():
    tr = R.true_rates(v, MIX)
    meas = {d: R.measured_rate(v, MIX, dd) for d, dd in R.DETECTORS.items()}
    rec = {d: R.recall(dd, v) for d, dd in R.DETECTORS.items()}
    ver[k] = {**tr, "measured": meas, "recall": rec}
    say(f"  {k:<22} {tr['refusal_rate']:>9.2%} {tr['over_refusal']:>8.2%} {tr['under_refusal']:>9.1%} "
        f"{tr['legit_share_of_refusals']:>11.1%} | " + " ".join(f"{meas[d]:>14.2%}" for d in R.DETECTORS))
    say(f"  {'':<22} {'':>9} {'':>8} {'':>9} {'':>11} | " + " ".join(f"{'recall ' + format(rec[d], '.2f'):>14}" for d in R.DETECTORS))
results["versions"] = ver
v1, v2, v3 = (ver[k] for k in R.VERSIONS)
kwn = "keyword list (written on v1)"
deltas = {
    "upgrade: true refusal rate": v2["refusal_rate"] - v1["refusal_rate"],
    "upgrade: measured (keyword)": v2["measured"][kwn] - v1["measured"][kwn],
    "upgrade: over-refusal": v2["over_refusal"] - v1["over_refusal"],
    "upgrade: under-refusal": v2["under_refusal"] - v1["under_refusal"],
    "tuned down: true refusal rate": v3["refusal_rate"] - v1["refusal_rate"],
    "tuned down: over-refusal": v3["over_refusal"] - v1["over_refusal"],
    "tuned down: under-refusal": v3["under_refusal"] - v1["under_refusal"],
}
results["deltas"] = deltas
for k, d in deltas.items():
    say(f"  {k:<32} {d * 100:+6.2f} pts")
say(f"  the upgrade raises the true rate {v2['refusal_rate'] / v1['refusal_rate'] - 1:+.0%} and the keyword dashboard reads "
    f"{v2['measured'][kwn] / v1['measured'][kwn] - 1:+.0%}: recall fell {v1['recall'][kwn]:.2f} -> {v2['recall'][kwn]:.2f} "
    "because the new model says 'I'm not able to'")
say(f"  the tuned-down prompt cuts the rate {deltas['tuned down: true refusal rate'] * 100:+.1f} pts and halves over-refusal; "
    f"under-refusal {v1['under_refusal']:.0%} -> {v3['under_refusal']:.0%} on the 5% of traffic that should be refused")

rule("3.  THE MIX MOVES, THE MODEL DOES NOT  -  a campaign week")
cmix = R.mix_of(override=R.CAMPAIGN_MIX)
camp = R.true_rates(R.VERSIONS["v1 baseline"], cmix)
camp["measured"] = R.measured_rate(R.VERSIONS["v1 baseline"], cmix, KW)
camp["mix"] = cmix
results["campaign"] = camp
say(f"  harmful share {MIX['harmful']:.0%} -> {cmix['harmful']:.0%}: true refusal rate {v1['refusal_rate']:.2%} -> {camp['refusal_rate']:.2%} "
    f"(measured {v1['measured'][kwn]:.2%} -> {camp['measured']:.2%})")
say(f"  over-refusal {v1['over_refusal']:.2%} -> {camp['over_refusal']:.2%}; under-refusal {v1['under_refusal']:.0%} -> {camp['under_refusal']:.0%}: "
    "nothing about the model moved, and the refusal rate nearly doubled")

rule("4.  A FLAT AGGREGATE IS NEVER NEUTRAL  -  the twin that doubles over-refusal at the same rate")
twin = R.flat_aggregate_twin(R.VERSIONS["v1 baseline"], MIX, 2.0)
tt = R.true_rates(twin, MIX)
results["twin"] = {**tt, "refuse": twin["refuse"]}
say(f"  legitimate refusal rates x2 ({twin['refuse']['benign']:.3f}, {twin['refuse']['sensitive']:.2f}); aggregate held at "
    f"{tt['refusal_rate']:.2%} forces harmful refusals {R.VERSIONS['v1 baseline']['refuse']['harmful']:.0%} -> {twin['refuse']['harmful']:.1%}")
say(f"  over-refusal {v1['over_refusal']:.2%} -> {tt['over_refusal']:.2%}, under-refusal {v1['under_refusal']:.0%} -> {tt['under_refusal']:.0%}, "
    f"legitimate share of refusals {v1['legit_share_of_refusals']:.0%} -> {tt['legit_share_of_refusals']:.0%}")
say("  at a fixed mix the aggregate is a weighted sum, so a rise in one error at a flat rate IS a fall in refusals of the other class")

rule("5.  WHERE TO SPEND THE LABELS  -  n labels on traffic vs n labels on detected refusals")
ns = [50, 100, 300, 1000, 3000]
ad = R.audit_designs(R.VERSIONS["v1 baseline"], R.VERSIONS["v2 upgraded model"], MIX, ns, KW)
results["audit"] = ad
say(f"  detect the upgrade's over-refusal doubling ({v1['over_refusal']:.2%} -> {v2['over_refusal']:.2%}), one-sided alpha 0.05, exact binomial")
say(f"  {'labels':>7}   {'traffic sample: power':>22} {'size':>6}   {'refusal sample: power':>22} {'size':>6}")
for a, b in zip(ad["traffic"], ad["refusals"]):
    say(f"  {a['n']:>7}   {a['power']:>22.3f} {a['size']:>6.3f}   {b['power']:>22.3f} {b['size']:>6.3f}")
say(f"  the refusal sample tests {ad['refusals'][0]['p0']:.1%} -> {ad['refusals'][0]['p1']:.1%} (legitimate share of detected refusals); "
    f"the traffic sample tests {ad['traffic'][0]['p0']:.2%} -> {ad['traffic'][0]['p1']:.2%} on ~95% of its labels")
bias = {k: ver[k]["legit_share_of_refusals"] for k in R.VERSIONS}
det_share = {"v1 baseline": ad["refusals"][0]["p0"], "v2 upgraded model": ad["refusals"][0]["p1"]}
results["audit_bias"] = {"true": bias, "detected": det_share}
say(f"  the refusal sample's own bias: legitimate share is {bias['v1 baseline']:.1%} of TRUE refusals and {det_share['v1 baseline']:.1%} of "
    f"DETECTED ones on v1 ({det_share['v2 upgraded model']:.1%} vs {bias['v2 upgraded model']:.1%} on v2) - the detector's false positives "
    "are compliant legitimate responses, so the labeller must also mark 'was this a refusal'")

rule("6.  TWENTY-SIX WEEKS  -  an aggregate chart and a refusal audit on the same traffic")
mon = R.monitor(KW)
results["monitor"] = mon
say(f"  weekly n = {R.WEEKLY_N:,} responses, keyword detector, 3-sigma limits from the v1 baseline; audit = {R.AUDIT_N} labelled detected refusals/week")
say(f"  {'week':>4} {'state':<22} {'true':>7} {'measured':>8} {'over-ref':>8}   {'P(alarm high)':>13} {'P(flag low)':>11}   {'P(audit alarm)':>14}")
for r in mon:
    if r["week"] in (1, 8, 9, 11, 12, 18, 19, 26):
        state = "campaign mix" if r["campaign"] else ("v2 upgraded model" if r["upgraded"] else "v1 baseline")
        say(f"  {r['week']:>4} {state:<22} {r['true_rate']:>7.2%} {r['measured']:>8.2%} {r['over_refusal']:>8.2%}   "
            f"{r['p_alarm_high']:>13.3f} {r['p_alarm_low']:>11.3f}   {r['p_audit_alarm']:>14.3f}")
c9, u19 = mon[8], mon[18]
results["monitor_summary"] = {"campaign_alarm_high": c9["p_alarm_high"], "campaign_audit_alarm": c9["p_audit_alarm"],
                              "upgrade_alarm_high": u19["p_alarm_high"], "upgrade_flag_low": u19["p_alarm_low"],
                              "upgrade_audit_alarm": u19["p_audit_alarm"], "baseline_alarm_high": mon[0]["p_alarm_high"]}
say(f"  campaign weeks: the aggregate alarm fires with probability {c9['p_alarm_high']:.3f}; the audit {c9['p_audit_alarm']:.3f}. "
    "Model unchanged.")
say(f"  upgrade weeks: the aggregate alarm fires {u19['p_alarm_high']:.3f} and the chart flags a DROP {u19['p_alarm_low']:.3f} "
    f"(measured {u19['measured']:.2%}, true {u19['true_rate']:.2%}); the audit fires {u19['p_audit_alarm']:.3f}")

rule("7.  THE DETECTOR THAT GETS THE SIGN RIGHT  -  and what it costs")
ext, judge = "keyword list, extended", "LLM judge"
for d in (ext, judge):
    say(f"  {d:<28} v1 measured {v1['measured'][d]:.2%} (true {v1['refusal_rate']:.2%})  v2 {v2['measured'][d]:.2%} (true {v2['refusal_rate']:.2%})  "
        f"v3 {v3['measured'][d]:.2%} (true {v3['refusal_rate']:.2%})")
fp_share = {d: (1 - v1["refusal_rate"]) * R.DETECTORS[d]["fpr"] / v1["measured"][d] for d in R.DETECTORS}
results["fp_share_of_detected"] = fp_share
say("  share of DETECTED refusals that are false positives on v1: " + ", ".join(f"{d} {s:.1%}" for d, s in fp_share.items()))
say("  a version-independent recall gets every sign right; its false positives are the price, and they land on legitimate traffic")

results["elapsed_s"] = time.time() - t0
with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
with open("results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True, default=list)
