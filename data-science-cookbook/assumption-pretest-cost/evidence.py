"""Run the whole study and print the evidence. Writes results.json for the chart and the README.

Every rate below is measured under a null that is EXACTLY true (both groups have the same mean),
so every rejection counted is a false positive by construction. No verdict rests on a gap the
harness cannot resolve: a rate is called broken only when its whole 99% Wilson interval, computed
at that cell's replicate count, clears [0.045, 0.055].
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import pretest as P

REPS = P.REPS


def bar(rate: float, width: int = 22, full: float = 0.35) -> str:
    filled = max(0, min(width, int(round(rate / full * width))))
    return "#" * filled + "." * (width - filled)


def main() -> None:
    t0 = time.time()
    print("=" * 100)
    print("ASSUMPTION-PRETEST COST: what 'we checked the variances first' actually does")
    print("=" * 100)
    print()
    print("Both groups always have the SAME mean, so the null is exactly true everywhere and")
    print("every rejection printed is a false positive. A correct 0.05 procedure reads 0.05.")
    print()

    # ------------------------------------------------------------------ 0. calibration
    print("-" * 100)
    print("0. THE HARNESS MEASURING ITSELF")
    print("-" * 100)
    lo, hi = P.noise_floor(runs=10)
    a, b = P.wilson(0.05, REPS)
    print(f"  Student's t on a balanced, equal-variance, normal design, {REPS:,} reps, 10 seeds.")
    print("  This cell is EXACT - its true error rate is 0.0500 with no approximation anywhere.")
    print(f"  Observed spread: {lo:.4f} to {hi:.4f}   (half-width {(hi - lo) / 2:.4f})")
    print(f"  99% Wilson interval on a true 0.05 at {REPS:,} reps: [{a:.4f}, {b:.4f}], width {b - a:.4f}")
    print(f"  The band is {P.BAND_HI - P.BAND_LO:.3f} wide, so this replicate count can resolve it.")
    print("  Nothing below is called broken on a point estimate.")
    print()

    # ------------------------------------------------------------------ 1. do the pretests work
    print("-" * 100)
    print("1. FOUR PROCEDURES ANSWER 'ARE THE VARIANCES EQUAL', AND THEY DO NOT AGREE")
    print("-" * 100)
    print("   Rejection rate of each pretest. The sd=1:1 rows are false positives (should be")
    print("   0.05); the rest are power. All at alpha = 0.05.")
    print()
    probe = [P.Design(20, 20, 1.0), P.Design(50, 10, 1.0), P.Design(50, 10, 1.5),
             P.Design(50, 10, 2.0), P.Design(50, 10, 3.0)]
    print(f"  {'design':<22} " + " ".join(f"{p:>16}" for p in P.PRETESTS))
    pretest_table = {}
    for d in probe:
        row = []
        for pt in P.PRETESTS:
            c = P.run_cell(d, pt, REPS, seed=11)
            row.append(c.pretest_reject_rate)
            pretest_table[(d.label, pt)] = c.pretest_reject_rate
        print(f"  {d.label:<22} " + " ".join(f"{v:>16.3f}" for v in row))
    print()
    print("   Read the two sd=1:1 rows first: that is each pretest's own false-positive rate,")
    print("   and a pretest that cannot hit 0.05 on equal variances has no business gating")
    print("   anything. Then read down the power columns.")
    print()

    print("   The same four on NON-NORMAL data with EQUAL variances - still all false positives:")
    print()
    print(f"  {'population':<22} " + " ".join(f"{p:>16}" for p in P.PRETESTS))
    robustness = {}
    for dist in P.DISTRIBUTIONS:
        row = []
        for pt in P.PRETESTS:
            c = P.run_cell(P.Design(20, 20, 1.0, dist), pt, REPS, seed=23)
            row.append(c.pretest_reject_rate)
            robustness[(dist, pt)] = c.pretest_reject_rate
        print(f"  {dist:<22} " + " ".join(f"{v:>16.3f}" for v in row))
    print()
    print("   Bartlett and the F-test are not variance tests on real data - they are normality")
    print("   detectors wearing a variance test's name. Their rejections on the heavy-tailed and")
    print("   skewed rows are reporting a shape difference that is not there, on groups whose")
    print("   variances are IDENTICAL. Brown-Forsythe is the one that holds.")
    print()

    # ------------------------------------------------------------------ 2. headline grid
    print("-" * 100)
    print("2. THE THREE PROCEDURES, ON THE SAME SAMPLES")
    print("-" * 100)
    print("   Student's always. Welch's always. And the two-stage move: Brown-Forsythe first,")
    print("   Welch if it rejects, Student's if it does not. Null exactly true throughout.")
    print()
    cells = P.run_grid("brown-forsythe", reps=REPS)
    print(f"  {'design':<22} {'pretest':>8} {'student':>9} {'welch':>9} {'GATED':>9}  {'gated vs nominal'}")
    for c in cells:
        mark = "*" if c.gated_is_broken else " "
        print(
            f"  {c.design.label:<22} {c.pretest_reject_rate:>8.3f} {c.student_error:>9.4f} "
            f"{c.welch_error:>9.4f} {c.gated_error:>8.4f}{mark}  {bar(c.gated_error)}"
        )
    print("   (* = the gated procedure's whole 99% interval is outside [0.045, 0.055])")
    print()

    beats = [c for c in cells if c.gated_beats_welch]
    beats_real = [c for c in cells if c.gated_beats_welch_materially]
    welch_broken = [c for c in cells if c.welch_is_broken]
    gated_broken = [c for c in cells if c.gated_is_broken]
    hw = (P.wilson(P.ALPHA, REPS)[1] - P.wilson(P.ALPHA, REPS)[0]) / 2
    print(f"   Cells where the two-stage procedure beat unconditional Welch AT ALL: {len(beats)} of {len(cells)}")
    for c in beats:
        gap = abs(c.welch_error - P.ALPHA) - abs(c.gated_error - P.ALPHA)
        print(f"      {c.design.label}: gated {c.gated_error:.4f} vs welch {c.welch_error:.4f} "
              f"- closer by {gap:.4f}, which is {gap / hw:.2f} of one interval half-width")
    print(f"   Cells where it beat Welch BY MORE THAN THE MEASUREMENT:         {len(beats_real)} of {len(cells)}")
    print(f"   Cells where unconditional Welch is outside the band:          {len(welch_broken)} of {len(cells)}")
    print(f"   Cells where the two-stage procedure is outside the band:      {len(gated_broken)} of {len(cells)}")
    print()
    print("   Those two lines are the whole build. `gated_beats_welch` is a real comparison that")
    print("   CAN come out either way - the test suite asserts it returns True on a synthetic")
    print("   cell built for it - and across this grid the only cell it flags wins by a margin")
    print("   far smaller than the noise. Counting that as a win would be reading the")
    print("   measurement error as a result.")
    print()

    # ------------------------------------------------------------------ 3. the mechanism
    print("-" * 100)
    print("3. THE MECHANISM: THE GATE HANDS STUDENT'S ITS WORST CASES")
    print("-" * 100)
    print("   The pretest reads the sample variances. Student's pooled standard error reads the")
    print("   same sample variances. So 'the pretest passed' does not select a random subset of")
    print("   samples - it selects the ones whose variance ratio happened to look small.")
    print()
    print(f"  {'design':<22} {'share passed':>13} {'Student overall':>16} {'Student | PASSED':>18} {'ratio':>7}")
    for c in cells:
        if c.share_passed < 0.01 or c.design.sd_ratio == 1.0:
            continue
        r = c.student_error_given_pass / c.student_error if c.student_error > 0 else float("inf")
        print(
            f"  {c.design.label:<22} {c.share_passed:>13.3f} {c.student_error:>16.4f} "
            f"{c.student_error_given_pass:>18.4f} {r:>6.2f}x"
        )
    print()
    print("   Every ratio above 1.00 is the procedure working backwards: on the samples the")
    print("   pretest cleared, Student's is MORE wrong than it is on average, not less. The gate")
    print("   is not filtering out the dangerous cases - it is concentrating them, because a")
    print("   sample where the small group's variance came out low both passes the pretest and")
    print("   understates the pooled standard error.")
    print()

    # ------------------------------------------------------------------ 4. the danger zone
    print("-" * 100)
    print("4. THE DANGER ZONE IS A MODERATE GAP, NOT A LARGE ONE")
    print("-" * 100)
    print("   n=50/10, walking the variance ratio. Watch the gated column rise and then fall.")
    print()
    ratios = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
    print(f"  {'sd ratio':>9} {'pretest fires':>14} {'student':>9} {'welch':>9} {'GATED':>9}   {'gated / nominal'}")
    ladder = []
    for r in ratios:
        c = P.run_cell(P.Design(50, 10, r), "brown-forsythe", REPS, seed=31)
        ladder.append(c)
        print(
            f"  {r:>9.2f} {c.pretest_reject_rate:>14.3f} {c.student_error:>9.4f} "
            f"{c.welch_error:>9.4f} {c.gated_error:>9.4f}   {c.gated_error / P.ALPHA:>5.2f}x"
        )
    peak = max(ladder, key=lambda c: c.gated_error)
    print()
    print(f"   Worst at sd ratio {peak.design.sd_ratio:g}: the gated procedure runs at "
          f"{peak.gated_error:.4f}, {peak.gated_error / P.ALPHA:.1f}x nominal,")
    print(f"   while Welch on the identical samples holds {peak.welch_error:.4f}.")
    print()
    print("   The shape is the point. At a huge gap the pretest always fires, the procedure")
    print("   collapses into Welch, and it is fine. At no gap there is nothing to get wrong. The")
    print("   damage is in the middle - which is where the pretest returns p > 0.05 and the")
    print("   analyst writes down 'variances were checked and found equal'.")
    print()

    # ------------------------------------------------------------------ 5. liberal alpha
    print("-" * 100)
    print("5. DOES A LIBERAL PRETEST ALPHA FIX IT?")
    print("-" * 100)
    print("   The standard advice for a low-power pretest is to run it at a generous alpha so it")
    print("   errs toward rejecting. Measured on the worst design found above:")
    print()
    worst_design = P.Design(50, 10, peak.design.sd_ratio)
    print(f"  {'pretest alpha':>14} {'fires':>8} {'GATED error':>13}   {'vs Welch'}")
    alpha_sweep = []
    for pa in (0.01, 0.05, 0.10, 0.20, 0.50, 1.0):
        c = P.run_cell(worst_design, "brown-forsythe", REPS, pretest_alpha=pa, seed=41)
        alpha_sweep.append((pa, c))
        delta = c.gated_error - c.welch_error
        print(
            f"  {pa:>14.2f} {c.pretest_reject_rate:>8.3f} {c.gated_error:>13.4f}   "
            f"{delta:+.4f}"
        )
    print()
    print("   alpha = 1.00 is the pretest always rejecting, i.e. unconditional Welch - it is in")
    print("   the table as the known-answer anchor, and it lands on Welch's own rate. Raising")
    print("   the pretest alpha does help, monotonically. What it converges to is the procedure")
    print("   that skips the pretest entirely, which is the cheapest possible way to get there.")
    print()

    # ------------------------------------------------------------------ 6. power
    print("-" * 100)
    print("6. THE OTHER HALF: WHAT EACH PROCEDURE DETECTS WHEN THERE IS SOMETHING TO DETECT")
    print("-" * 100)
    print("   A procedure that never rejects is not safe, it is deaf. Same designs, now with a")
    print("   real difference in means. Higher is better here.")
    print()
    print(f"  {'design':<22} {'shift':>7} {'student':>9} {'welch':>9} {'gated':>9}   comparable?")
    power_rows = []
    null_by_label = {c.design.label: c for c in cells}
    for d, shift in ((P.Design(50, 10, 3.0), 1.5), (P.Design(10, 50, 3.0), 1.5),
                     (P.Design(20, 20, 1.0), 0.8), (P.Design(50, 10, 1.5), 1.0)):
        pw = P.power_cost(d, shift, "brown-forsythe", REPS)
        null = null_by_label.get(d.label)
        # A power number from a test that does not hold its false-positive rate is not power,
        # it is the same inflation showing up again. Say so in the row rather than in a footnote.
        direction = null.student_type_one_direction if null is not None else "ok"
        note = {
            "ok": "yes - both hold 0.05 under the null",
            "inflated": f"NO - Student's runs at {null.student_error:.4f} under the null, so its"
                        " lead here is that same inflation",
            "conservative": f"YES, and it is the finding - Student's runs at {null.student_error:.4f}"
                            " under the null, so this deficit is real",
        }[direction]
        power_rows.append((d.label, shift, pw, direction))
        print(f"  {d.label:<22} {shift:>7.2f} {pw['student']:>9.4f} {pw['welch']:>9.4f} "
              f"{pw['gated']:>9.4f}   {note}")
    print()
    print("   Read the last column before the numbers. Where Student's is INFLATED under the")
    print("   null, its apparent lead here is that same inflation - it rejects more often")
    print("   because it rejects more often, not because it detects more. Those rows are not a")
    print("   power comparison at all.")
    print()
    print("   The n=10/50 row is the opposite case and the one nobody audits. There Student's")
    print(f"   holds {power_rows[1][2]['student']:.4f} detection against Welch's "
          f"{power_rows[1][2]['welch']:.4f} - a real deficit,")
    print("   honestly paid for, because its false-positive rate under the null is BELOW")
    print("   nominal. A Type I audit would have called that design safe and said nothing")
    print("   about the detections it was quietly giving up.")
    print()

    # ------------------------------------------------------------------ 7. scorecard
    print("-" * 100)
    print("7. THE PRETEST SCORED AS A DECISION ABOUT WHETHER STUDENT'S IS SAFE")
    print("-" * 100)
    score = P.score_gate(cells)
    print("  Positive  = the pretest rejects on the majority of samples from this design")
    print(f"  Condition = Student's real error rate is outside [{P.BAND_LO}, {P.BAND_HI}]")
    print()
    print(f"  designs                       {score.cells}")
    print(f"  designs where Student IS bad  {score.unsafe_cells}")
    print(f"  sensitivity                   {score.sensitivity:.3f}")
    print(f"  specificity                   {score.specificity:.3f}")
    print(f"  false alarms                  {score.false_alarm_cells}")
    print(f"  misses                        {score.missed_cells}")
    print()
    w = P.worst_cell(cells)
    print(f"  Worst two-stage cell: {w.design.label} at {w.gated_error:.4f} "
          f"({w.gated_error / P.ALPHA:.1f}x nominal), where Welch holds {w.welch_error:.4f}.")
    print()

    payload = {
        "meta": {
            "alpha": P.ALPHA,
            "band": [P.BAND_LO, P.BAND_HI],
            "reps": REPS,
            "pretests": list(P.PRETESTS),
            "distributions": list(P.DISTRIBUTIONS),
            "noise_floor": [lo, hi],
            "runtime_seconds": round(time.time() - t0, 1),
        },
        "pretest_power": {f"{k[0]}|{k[1]}": v for k, v in pretest_table.items()},
        "pretest_robustness": {f"{k[0]}|{k[1]}": v for k, v in robustness.items()},
        "cells": [_cell_json(c) for c in cells],
        "ladder": [_cell_json(c) for c in ladder],
        "alpha_sweep": [{"pretest_alpha": pa, **_cell_json(c)} for pa, c in alpha_sweep],
        "power": [{"design": lab, "shift": sh, "student_null_direction": dr, **pw}
                  for lab, sh, pw, dr in power_rows],
        "derived": {
            "gate_score": asdict(score),
            "gated_beats_welch_count": len(beats),
            "gated_beats_welch_materially_count": len(beats_real),
            "welch_broken_count": len(welch_broken),
            "gated_broken_count": len(gated_broken),
            "worst": _cell_json(w),
        },
    }
    with open("results.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  wrote results.json  ({time.time() - t0:.0f}s total)")


def _cell_json(c: P.CellResult) -> dict:
    d = asdict(c)
    d["design"] = {**asdict(c.design), "label": c.design.label}
    d["gated_is_broken"] = c.gated_is_broken
    d["welch_is_broken"] = c.welch_is_broken
    d["student_is_broken"] = c.student_is_broken
    d["gated_beats_welch"] = c.gated_beats_welch
    d["gated_interval"] = list(c.gated_interval)
    return d


if __name__ == "__main__":
    main()
