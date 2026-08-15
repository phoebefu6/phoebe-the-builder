"""Every number in the README, computed. Each experiment isolates one mechanism.

Run: python3 evidence.py
"""

from __future__ import annotations

import random
import statistics
from typing import List

import retry as R

RULE = "-" * 78

FLEET = 500
OUTAGE = 20.0
CAPACITY = 50.0
BASE = 0.1
CAP = 20.0
ATTEMPTS = 10


def head(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


# ---------------------------------------------------------------------------


def exp1_backoff_moves_total_work_not_peak() -> None:
    head(1, "Exponential backoff reduces total work. It does not touch the peak.")
    print("500 clients fail together. Where does the first retry land?\n")
    print(f"{'attempt':>8}{'delay (s)':>12}{'arrives at':>13}{'clients arriving':>19}")
    print(RULE)
    s = R.Schedule("no_jitter", BASE, CAP, ATTEMPTS)
    rng = random.Random(0)
    t = 0.0
    for i, d in enumerate(s.delays(rng)):
        t += d
        print(f"{i:>8}{d:>12.2f}{t:>13.2f}{FLEET:>19}")
    print(RULE)
    print("the gaps grow. the height of each spike is 500 at every single one.")
    print("total work falls with each doubling; instantaneous concurrency never does.")
    print(f"\nspike height 500 against {CAPACITY:.0f} rps of capacity = "
          f"{FLEET / CAPACITY:.0f}x over.")


def exp2_the_outage_run() -> None:
    head(2, "The headline run: 20s outage, 500 clients, 50 rps of capacity.")
    print("shed load burns an attempt exactly like failed load.\n")
    print(f"{'policy':<22}{'reqs':>7}{'wasted':>8}{'peak rps':>10}{'waves':>7}"
          f"{'recovered':>11}{'gave up':>9}{'clear at':>10}{'verdict':>11}")
    print(RULE)
    res = R.compare(FLEET, OUTAGE, CAPACITY, BASE, CAP, ATTEMPTS)
    for name in R.POLICY_ORDER:
        v = res[name]
        verdict, _ = R.audit(v.schedule, FLEET, OUTAGE, CAPACITY, sim=v)
        ct = v.completion_time()
        print(f"{name:<22}{v.total_requests():>7}{v.wasted_requests():>8}"
              f"{v.recovery_peak_rps():>10.0f}{v.over_capacity_waves():>7}"
              f"{v.succeeded:>11}{v.gave_up:>9}"
              f"{(f'{ct:.1f}s' if ct else '-'):>10}{verdict.value:>11}")
    print(RULE)
    nj, ej, fj = res["no_jitter"], res["equal_jitter"], res["full_jitter"]
    print(f"no_jitter leaves {nj.gave_up} of {FLEET} clients permanently failed after an")
    print(f"outage the dependency itself recovered from in {OUTAGE:.0f} seconds.\n")
    print("and the inversion nobody expects:")
    print(f"  full_jitter  peak {fj.recovery_peak_rps():>4.0f} rps  ->  "
          f"{fj.gave_up} clients lost")
    print(f"  equal_jitter peak {ej.recovery_peak_rps():>4.0f} rps  ->  "
          f"{ej.gave_up} clients lost")
    print("the policy with the LOWER peak has the WORSE outcome. peak and recovery")
    print("are different objectives and 'add full jitter' optimises the measured one.")


def exp3_jitter_halves_coverage() -> None:
    head(3, "Why: jitter is drawn downward, so it halves your coverage.")
    print("same attempt count, same cap. different reach.\n")
    print(f"{'policy':<22}{'worst case':>12}{'expected':>11}{'median':>10}"
          f"{'covers 20s outage?':>21}")
    print(RULE)
    for name in R.POLICY_ORDER:
        s = R.Schedule(name, BASE, CAP, ATTEMPTS)
        med, mean = R.sampled_totals(s, n=5000, seed=11)
        ok = "yes" if med >= OUTAGE else "NO"
        print(f"{name:<22}{s.worst_case_total():>12.1f}{s.expected_total():>11.1f}"
              f"{med:>10.1f}{ok:>21}")
    print(RULE)
    print("full_jitter draws uniform(0, window): E = window/2. the ladder is the same,")
    print("the reach is half. a budget of 10 attempts sized against the un-jittered")
    print("schedule stops trying at ~33s instead of 65.5s - and that is the mean.")


def exp4_mean_is_not_the_typical_client() -> None:
    head(4, "For decorrelated jitter the mean describes a client that does not exist.")
    print("each delay is drawn from a range set by the PREVIOUS delay, so the walk")
    print("compounds. draw low early and you stay low for all ten attempts.\n")
    print(f"{'policy':<22}{'mean':>9}{'median':>9}{'p10':>9}{'p90':>9}{'med/mean':>10}")
    print(RULE)
    for name in ("equal_jitter", "full_jitter", "decorrelated_jitter"):
        s = R.Schedule(name, BASE, CAP, ATTEMPTS)
        rng = random.Random(11)
        tot = sorted(sum(s.delays(rng)) for _ in range(5000))
        mean = statistics.fmean(tot)
        med = tot[len(tot) // 2]
        print(f"{name:<22}{mean:>9.1f}{med:>9.1f}{tot[len(tot)//10]:>9.1f}"
              f"{tot[9*len(tot)//10]:>9.1f}{med/mean:>9.0%}")
    print(RULE)
    print("expectation is linear so the mean is exact - and useless as a budget.")
    print("half the fleet gives up sooner than the number in the design doc.")

    print("\nand the closed form for the walk is not even the right mean.")
    print("E[d_n] = min(cap, (base + 3*E[d_n-1])/2) iterated looks obviously")
    print("correct. min() is concave, so Jensen runs the wrong way:")
    print("E[min(cap, X)] <= min(cap, E[X]). the error compounds over the walk.\n")
    print(f"{'cap (s)':>9}{'naive recurrence':>19}{'sampled mean':>15}{'overstated by':>16}")
    print(RULE)
    for cap in (5.0, 20.0, 60.0, 1e9):
        s = R.Schedule("decorrelated_jitter", BASE, cap, ATTEMPTS)
        naive = s.naive_recurrence_total()
        true = s.expected_total()
        label = "no cap" if cap > 1e6 else f"{cap:.0f}"
        print(f"{label:>9}{naive:>19.1f}{true:>15.1f}{naive/true - 1:>15.0%}")
    print(RULE)
    print("with no cap the recurrence is linear and exact. the cap is what breaks")
    print("it, and the cap is the part everyone adds. retry.py samples instead.")


def exp5_the_cap_is_a_load_floor() -> None:
    head(5, "The cap stops the thinning. It is a load floor, not a safety valve.")
    print("once base*2^n >= cap the jitter window stops widening, so the arrival")
    print("process stops spreading out. it settles at a fixed aggregate rate.\n")
    print(f"{'cap (s)':>9}{'caps at attempt':>17}{'floor rps (full jitter)':>26}"
          f"{'vs 50 rps capacity':>20}")
    print(RULE)
    for cap in (5.0, 10.0, 20.0, 30.0, 60.0, 120.0):
        s = R.Schedule("full_jitter", BASE, cap, ATTEMPTS)
        floor = s.steady_state_rate(FLEET)
        at = s.cap_reached_at()
        flag = "OVER" if floor > CAPACITY else "ok"
        print(f"{cap:>9.0f}{(str(at) if at is not None else 'never'):>17}"
              f"{floor:>26.0f}{flag:>20}")
    print(RULE)
    print(f"floor = fleet / (cap * mean_factor). with {FLEET} clients and full jitter")
    print(f"that is {FLEET}/(cap*0.5) = {2*FLEET:.0f}/cap requests per second, forever.")
    print(f"a 20s cap on a 500-client fleet is {2*FLEET/20:.0f} rps of steady load -")
    print(f"exactly the {CAPACITY:.0f} rps the dependency serves, with nothing left over")
    print("for the users who did not fail. the fix is not more jitter. it is a")
    print("client-side retry budget, or fewer clients retrying at all.")

    print("\nmeasured against the model, at t in [outage, outage+60s]:")
    print(f"{'cap (s)':>9}{'predicted floor':>18}{'measured rps':>15}")
    print(RULE)
    for cap in (10.0, 20.0, 60.0):
        s = R.Schedule("full_jitter", BASE, cap, 40)
        sim = R.simulate(s, fleet=FLEET, outage_s=OUTAGE, capacity_rps=0.0,
                         seed=3, horizon_s=200.0)
        win = [t for t in sim.arrivals if OUTAGE <= t < OUTAGE + 60]
        print(f"{cap:>9.0f}{s.steady_state_rate(FLEET):>18.0f}{len(win)/60:>15.0f}")
    print(RULE)
    print("(capacity 0 so nothing is admitted - this isolates the arrival process.)")


def exp6_amplification() -> None:
    head(6, "Nested retries multiply. The bottom service pays the product.")
    print(f"{'stack':<44}{'requests at the bottom':>26}")
    print(RULE)
    stacks = [
        ("browser 1", [1]),
        ("browser 3", [3]),
        ("browser 3 -> api gateway 3", [3, 3]),
        ("browser 3 -> gateway 3 -> service 3", [3, 3, 3]),
        ("+ db driver 2", [3, 3, 3, 2]),
        ("sidecar 2 -> gateway 3 -> svc 3 -> db 2", [2, 3, 3, 2]),
    ]
    for label, layers in stacks:
        print(f"{label:<44}{R.amplification(layers):>26}")
    print(RULE)
    print("54 requests reach the database for one click. every layer was")
    print("individually reasonable. nobody owns the product.")


def exp7_clock_alignment() -> None:
    head(7, "A scheduler tick wider than the jitter window re-synchronises the fleet.")
    s = R.Schedule("full_jitter", BASE, CAP, ATTEMPTS)
    tick = 1.0
    rng = random.Random(5)
    d = s.delays(rng)
    q = R.quantise(d, tick)
    print(f"jitter window per attempt vs a {tick:g}s timer-wheel tick:\n")
    print(f"{'attempt':>8}{'window':>10}{'drawn':>10}{'after tick':>13}{'collapsed?':>12}")
    print(RULE)
    for i, (a, b) in enumerate(zip(d, q)):
        w = s.window(i)
        print(f"{i:>8}{w:>10.2f}{a:>10.3f}{b:>13.2f}"
              f"{('YES' if w < tick else '-'):>12}")
    print(RULE)
    narrow = [i for i in range(ATTEMPTS) if s.window(i) < tick]
    print(f"attempts {narrow[0]}-{narrow[-1]} have a window narrower than the tick, so")
    print("every client in the fleet rounds into the same slot. the jitter was")
    print("computed, then quantised away. this is what a cron-driven retry does.")


def exp8_findings_and_verdicts() -> None:
    head(8, "Full audit output.")
    res = R.compare(FLEET, OUTAGE, CAPACITY, BASE, CAP, ATTEMPTS)
    counts = {"critical": 0, "warning": 0, "info": 0}
    for name in R.POLICY_ORDER:
        v = res[name]
        verdict, fs = R.audit(v.schedule, FLEET, OUTAGE, CAPACITY,
                              deadline_s=30.0, nested_layers=[3, 3],
                              tick_s=1.0, sim=v)
        print(f"\n--- {name}  ->  {verdict.value.upper()}")
        for x in fs:
            counts[x.severity.value] += 1
            print(f"  [{x.severity.value:<8}] {x.code}: {x.message}")
            print(f"             {x.detail}")
    print(f"\n{RULE}")
    total = sum(counts.values())
    print(f"{total} findings across 5 policies: {counts['critical']} critical, "
          f"{counts['warning']} warning, {counts['info']} info.")


def exp9_what_actually_fixes_it() -> None:
    head(9, "What actually fixes it: fewer retries, not better-shaped ones.")
    print("same outage. equal_jitter. vary the fleet that is allowed to retry -")
    print("which is what a client-side retry budget (gRPC retryThrottling) does.\n")
    print(f"{'retrying clients':>18}{'peak rps':>10}{'recovered':>11}{'gave up':>9}"
          f"{'clear at':>10}{'verdict':>11}")
    print(RULE)
    for n in (500, 250, 100, 50):
        s = R.Schedule("equal_jitter", BASE, CAP, ATTEMPTS)
        sim = R.simulate(s, fleet=n, outage_s=OUTAGE, capacity_rps=CAPACITY, seed=7)
        verdict, _ = R.audit(s, n, OUTAGE, CAPACITY, sim=sim)
        ct = sim.completion_time()
        print(f"{n:>18}{sim.recovery_peak_rps():>10.0f}{sim.succeeded:>11}"
              f"{sim.gave_up:>9}{(f'{ct:.1f}s' if ct else '-'):>10}"
              f"{verdict.value:>11}")
    print(RULE)
    print("the arrival process is fleet/cap. jitter only decides its shape.")
    print("to lower the floor you must reduce the numerator.")


def main() -> None:
    print("RETRY SCHEDULE - evidence")
    print(f"fleet={FLEET}  outage={OUTAGE:g}s  capacity={CAPACITY:g} rps  "
          f"base={BASE:g}s  cap={CAP:g}s  attempts={ATTEMPTS}")
    exp1_backoff_moves_total_work_not_peak()
    exp2_the_outage_run()
    exp3_jitter_halves_coverage()
    exp4_mean_is_not_the_typical_client()
    exp5_the_cap_is_a_load_floor()
    exp6_amplification()
    exp7_clock_alignment()
    exp8_findings_and_verdicts()
    exp9_what_actually_fixes_it()


if __name__ == "__main__":
    main()
