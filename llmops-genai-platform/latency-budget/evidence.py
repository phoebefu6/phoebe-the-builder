"""Run the study once, write results.json (read by the chart, the app and the notebook test) and evidence.txt."""

from __future__ import annotations

import json

import latency as L


def main() -> None:
    s = L.study()
    fixes = L.fix_table()
    cal = L.calibrate()
    res = {"study": s, "fixes": fixes, "calibration": cal, "slo_ms": L.SLO_MS, "shards": L.SHARDS, "base": L.BASE}
    json.dump(res, open("results.json", "w"), indent=1)

    out = []
    w = out.append
    w(f"latency-budget evidence  (seed {L.SEED}, exact on a 1 ms grid, Monte Carlo n = {L.MC_REPS:,})")
    w("")
    w("1. Per-hop percentiles vs end-to-end (ms)")
    w(f"{'hop':<10} {'p50':>6} {'p95':>6} {'p99':>6} {'mean':>8} {'p99/p50':>8}")
    for h in L.HOPS:
        r = s["hops"][h]
        w(f"{h:<10} {r['p50']:>6} {r['p95']:>6} {r['p99']:>6} {r['mean']:>8.1f} {r['p99'] / r['p50']:>8.1f}")
    e = s["e2e"]
    w(f"{'END-TO-END':<10} {e['p50']:>6} {e['p95']:>6} {e['p99']:>6} {e['mean']:>8.1f}")
    w(f"sum of per-hop p99 = {s['sum_of_hop_p99']} ms vs true p99 {e['p99']} ms  (overstates by {s['sum_of_hop_p99'] - e['p99']} ms)")
    w(f"SLO p99 <= {L.SLO_MS} ms: per-hop sum says {s['sum_of_hop_p99'] - L.SLO_MS} ms over; truth {e['p99'] - L.SLO_MS} ms over, "
      f"P(T > SLO) = {s['p_over_slo']:.2%}")
    w("")
    w(f"2. Fan-out: one shard p99 = {s['shard']['p99']} ms (GC pause {L.BASE['shard_gc_p']:.1%} sits above it)")
    w(f"   search step waits for {L.SHARDS} shards: P(some shard pauses) = {s['p_any_shard_gc']:.1%}, search p95 = {s['hops']['search']['p95']} ms")
    w("")
    w(f"3. Who owns the slow requests (T > p99 = {e['p99']} ms, P = {s['p_tail']:.3%})")
    w(f"{'hop':<10} {'mean share':>11} {'tail excess ms':>15} {'tail share':>11}")
    for h in L.HOPS:
        w(f"{h:<10} {s['mean_share'][h]:>11.1%} {s['tail_excess_ms'][h]:>15.1f} {s['tail_share'][h]:>11.1%}")
    w("")
    w("4. Four fixes and their sum, ranked two ways")
    w(f"{'fix':<30} {'d mean':>8} {'d p50':>6} {'d p99':>6} {'p99':>6} {'P(T>SLO)':>9}")
    for r in fixes:
        w(f"{r['fix']:<30} {r['d_mean']:>+8.1f} {r['d_p50']:>+6} {r['d_p99']:>+6} {r['p99']:>6} {r['p_over_slo']:>9.2%}")
    by_mean = [r["fix"] for r in sorted(fixes, key=lambda r: r["d_mean"])]
    by_p99 = [r["fix"] for r in sorted(fixes, key=lambda r: r["d_p99"])]
    w(f"ranked by mean: {' > '.join(by_mean)}")
    w(f"ranked by p99 : {' > '.join(by_p99)}")
    w("")
    inside = sum(r["inside"] for r in cal)
    w(f"5. Calibration: {inside}/{len(cal)} exact tail probabilities inside the 99% Wilson interval of the simulation")
    for r in cal:
        w(f"   {r['chain']:<30} t={r['t']:>5}  exact {r['exact']:.5f}  mc {r['mc']:.5f}  {'ok' if r['inside'] else 'OUTSIDE'}")
    open("evidence.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
