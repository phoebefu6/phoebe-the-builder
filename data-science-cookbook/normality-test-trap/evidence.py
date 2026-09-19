"""Run the whole study and print the evidence. Writes results.json for the chart and the README.

Everything printed here is a measured rejection rate under a null that is EXACTLY true, so
every rejection counted is a false positive by construction. No cell reports a claim that the
harness cannot resolve: the noise floor is measured first and printed at the top, and no verdict
below is allowed to rest on a gap narrower than it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Dict, List

import normality as N

BAND = (0.045, 0.055)


def bar(rate: float, width: int = 20) -> str:
    filled = int(round(rate * width))
    return "#" * filled + "." * (width - filled)


def main() -> None:
    t0 = time.time()
    print("=" * 96)
    print("NORMALITY-TEST TRAP: what a Shapiro-Wilk gate actually does to a t-test")
    print("=" * 96)
    print()
    print("Every population below is standardised to mean 0, sd 1. The null tested is 'the")
    print("population mean is 0'. It is true in every single cell. So every rejection printed")
    print("is a false positive, and a correct 0.05 test should show 0.05 everywhere.")
    print()

    # ------------------------------------------------------------------ 0. noise floor
    print("-" * 96)
    print("0. THE HARNESS MEASURING ITSELF")
    print("-" * 96)
    lo, hi = N.noise_floor(n=50, runs=10)
    print(f"  One-sample t on standard normal data, n=50, {N.reps_for(50):,} reps, 10 different seeds.")
    print("  This cell is EXACT - its true error rate is 0.0500 with no approximation anywhere.")
    print(f"  Observed spread: {lo:.4f} to {hi:.4f}   (half-width {(hi - lo) / 2:.4f})")
    print()
    print("  Replicates are tiered across the grid, so 'is this cell broken' is decided from a")
    print("  99% Wilson interval computed at THAT cell's replicate count - never from the point")
    print("  estimate. A cell is called broken only when its whole interval clears the band.")
    print()
    print(f"  {'n':>6} {'reps':>8}   99% interval on a TRUE 0.05     width   can it resolve the band?")
    for n in N.GRID_N:
        r = N.reps_for(n)
        a, b = N.wilson(0.05, r)
        ok = "yes" if (b - a) < 2 * (BAND[1] - BAND[0]) else "NO - absence of evidence only"
        print(f"  {n:>6} {r:>8}   [{a:.4f}, {b:.4f}]            {b - a:.4f}   {ok}")
    print()
    print("  The first version of this harness ran 1,500 replicates at n=5000 and promptly")
    print("  flagged its own exact calibration cell as broken. The interval is what stops that.")
    print()

    cells = N.run_grid()
    by: Dict[str, List[N.CellResult]] = {}
    for c in cells:
        by.setdefault(c.dist, []).append(c)
    for v in by.values():
        v.sort(key=lambda c: c.n)
    ns = list(N.GRID_N)

    # ------------------------------------------------------------------ 1. power curve
    print("-" * 96)
    print("1. SHAPIRO-WILK IS A POWER CURVE, NOT A VERDICT")
    print("-" * 96)
    print("   Share of samples where Shapiro REJECTS normality. The population never changes")
    print("   down a column; only n changes.")
    print()
    print(f"  {'population':<18} " + " ".join(f"{n:>7}" for n in ns))
    for d in N.DISTRIBUTIONS:
        row = " ".join(f"{c.shapiro_reject_rate:>7.3f}" for c in by[d])
        print(f"  {d:<18} {row}")
    print()
    print("   Read the 'normal' row as the calibration: a correct test rejects true normality")
    print("   at 0.05 regardless of n, and it does. Every other row is a power curve climbing")
    print("   to 1.000 - the same population, called normal at one n and not at another.")
    print()

    # ------------------------------------------------------------------ 2. what t does
    print("-" * 96)
    print("2. WHAT THE T-TEST ACTUALLY DOES ON THAT SAME DATA")
    print("-" * 96)
    print("   Two-sided Type I error of the one-sample t-test. Nominal 0.05. Same samples as")
    print("   the table above - not a separate simulation.")
    print()
    print(f"  {'population':<18} " + " ".join(f"{n:>7}" for n in ns))
    for d in N.DISTRIBUTIONS:
        cs = []
        for c in by[d]:
            mark = "*" if c.t_is_broken else " "
            cs.append(f"{c.t_error:>6.3f}{mark}")
        print(f"  {d:<18} " + " ".join(cs))
    print("   (* = 99% interval lies entirely outside [0.045, 0.055])")
    print()
    print("   The two tables run in OPPOSITE directions. Shapiro's certainty grows with n;")
    print("   the t-test's error shrinks with n, because the CLT is repairing exactly the")
    print("   thing Shapiro is getting better at detecting.")
    print()

    # ------------------------------------------------------------------ 3. crossover
    print("-" * 96)
    print("3. THE CROSSOVER: WHERE THE GATE FIRES ON A TEST THAT WORKS")
    print("-" * 96)
    print(f"  {'population':<18} {'gate fires from n=':<20} {'t not shown broken from':<24} {'crossover n'}")
    crossovers = {}
    for d in N.DISTRIBUTIONS:
        fires = next((c.n for c in by[d] if c.shapiro_reject_rate >= 0.50), None)
        fine = next((c.n for c in by[d] if not c.t_is_broken), None)
        x = N.crossover_n(cells, d)
        crossovers[d] = x
        print(
            f"  {d:<18} {str(fires) if fires else 'never':<20} "
            f"{str(fine) if fine else 'never':<24} {str(x) if x else '-'}"
        )
    print()
    print("   The crossover column is the trap in one number: at and above that n, Shapiro")
    print("   rejects the majority of samples drawn from a population on which the t-test")
    print("   shows no detectable error inflation. Following the gate there costs you")
    print("   something and buys you nothing.")
    print()

    # ------------------------------------------------------------------ 4. blind spot
    print("-" * 96)
    print("4. THE BLIND SPOT: WHERE THE T-TEST IS BROKEN AND SHAPIRO IS QUIET")
    print("-" * 96)
    any_blind = False
    for d in N.DISTRIBUTIONS:
        bs = N.blind_spot_n(cells, d)
        if bs:
            any_blind = True
            detail = ", ".join(
                f"n={c.n} (shapiro {c.shapiro_reject_rate:.3f}, t {c.t_error:.4f})"
                for c in by[d]
                if c.n in bs
            )
            print(f"  {d:<18} {detail}")
    if not any_blind:
        print("  none in this grid")
    print()

    # ------------------------------------------------------------------ 5. tail asymmetry
    print("-" * 96)
    print("5. THE DEFECT THE TWO-SIDED NUMBER HIDES")
    print("-" * 96)
    print("   Same t-tests, split by which tail rejected. Under a symmetric true null each")
    print("   side should be 0.025.")
    print()
    print(f"  {'population':<18} {'n':>6} {'lower':>8} {'upper':>8} {'two-sided':>11} {'asymmetry':>11}")
    for d in N.DISTRIBUTIONS:
        for c in by[d]:
            if c.n in (20, 50, 500, 5000):
                print(
                    f"  {d:<18} {c.n:>6} {c.t_error_lower_tail:>8.4f} {c.t_error_upper_tail:>8.4f} "
                    f"{c.t_error:>11.4f} {c.tail_asymmetry:>10.1f}x"
                )
    print()
    print("   A right-skewed population makes the t-test reject DOWNWARD far more often than")
    print("   upward, because a sample that happens to miss the long right tail has both a low")
    print("   mean and a low variance - and the low variance shrinks the standard error that")
    print("   the low mean is divided by. The two errors do not cancel, they compound. A cell")
    print("   can sit at 0.050 two-sided and still be a one-directional test.")
    print()

    # ------------------------------------------------------------------ 6. conditional
    print("-" * 96)
    print("6. THE PROCEDURE ITSELF: 'RUN SHAPIRO, THEN PICK'")
    print("-" * 96)
    print("   Three procedures on identical data. Unconditional t. Unconditional Wilcoxon")
    print("   signed-rank. And the conditional move: Wilcoxon if Shapiro rejected, else t.")
    print()
    print(f"  {'population':<18} {'n':>6} {'t alone':>9} {'wilcoxon':>9} {'gated':>9}   {'gated error rate'}")
    for d in N.DISTRIBUTIONS:
        for c in by[d]:
            if c.n in (20, 50, 200, 1000):
                print(
                    f"  {d:<18} {c.n:>6} {c.t_error:>9.4f} {c.wilcoxon_error:>9.4f} "
                    f"{c.conditional_error:>9.4f}   {bar(c.conditional_error)}"
                )
    print()

    # ------------------------------------------------------------------ 7. scorecard
    print("-" * 96)
    print("7. SHAPIRO-WILK SCORED AS WHAT PEOPLE USE IT FOR: A GATE ON THE T-TEST")
    print("-" * 96)
    score = N.score_as_diagnostic(cells, threshold=0.50)
    print("  Positive  = the gate fires (Shapiro rejects the majority of samples at this n)")
    print(f"  Condition = the t-test's real error rate is outside [{BAND[0]}, {BAND[1]}]")
    print()
    print(f"  grid cells                 {score.cells}")
    print(f"  cells where t IS broken    {score.broken_cells}")
    print(f"  sensitivity (catches it)   {score.sensitivity:.3f}")
    print(f"  specificity (stays quiet)  {score.specificity:.3f}")
    print(f"  false alarms               {score.false_alarm_cells} cells")
    print(f"  misses                     {score.missed_cells} cells")
    unresolvable = [c for c in cells if not c.resolvable]
    print(f"  cells too noisy to judge   {len(unresolvable)}")
    print()
    print("   Specificity is the number that matters. The gate's job, as people use it, is to")
    print("   stop you using a t-test that would mislead you. A gate that fires on cell after")
    print("   cell where the t-test is already correct is not being cautious - it is routing")
    print("   you to a substitute test, and section 6 is what the substitute costs.")
    print()

    payload = {
        "meta": {
            "alpha": N.ALPHA,
            "grid_n": ns,
            "distributions": list(N.DISTRIBUTIONS),
            "reps_for_n": N.REPS_FOR_N,
            "band": list(BAND),
            "noise_floor": [lo, hi],
            "runtime_seconds": round(time.time() - t0, 1),
        },
        "shape": {k: list(v) for k, v in N.SHAPE.items()},
        "cells": [
            dict(
                asdict(c),
                t_error_interval=list(c.t_error_interval),
                t_is_broken=c.t_is_broken,
                resolvable=c.resolvable,
                tail_asymmetry=c.tail_asymmetry,
            )
            for c in cells
        ],
        "derived": {
            "crossover_n": crossovers,
            "blind_spots": {d: N.blind_spot_n(cells, d) for d in N.DISTRIBUTIONS},
            "diagnostic": asdict(score),
        },
    }
    with open("results.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  wrote results.json  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
