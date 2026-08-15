"""Retry schedules, and the arrival process they create across a fleet.

A backoff function returns a delay. The delay is not the interesting thing.

The interesting thing is what happens when N clients fail at the *same* instant,
because that is the only time retries matter. Each of them then runs the same
deterministic function over the same attempt counter and schedules its next
request for the same moment. Exponential backoff spaces those moments further
and further apart, which reduces the *total* work the dependency absorbs and
does nothing at all to the *peak*: the peak is still N, arriving together, and
peak is what takes the service down a second time.

Core ideas
----------
1. Backoff is about total work. Jitter is about peak. They are different
   problems and exponential backoff only solves the first one.
2. A cap is not free. Once every client's window has reached `cap`, the jitter
   window stops widening, so the arrival process stops thinning. Capped full
   jitter has a load *floor* of `2N/cap` requests per second that no amount of
   randomness reduces.
3. Jitter halves your coverage. At a fixed attempt count, full jitter's expected
   elapsed time is half the un-jittered schedule's, so a budget tuned to outlast
   a 20-second outage stops trying after 10.
4. The verdict is three-valued:
   `dispersed` - peak arrival rate stays inside the dependency's capacity.
   `bursty`    - peak exceeds capacity, but the schedule self-clears.
   `herding`   - peak exceeds capacity and the shed load re-synchronises into
                 another spike, so the retries are now the outage.

Standard library only: `random`, `heapq`, `math`, `dataclasses`, `enum`.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
#
# Five schedules that are actually deployed. `full_jitter`, `equal_jitter` and
# `decorrelated_jitter` are written from the algorithms published in the AWS
# Architecture Blog post "Exponential Backoff And Jitter" (Brooker, 2015);
# `no_jitter` is the textbook capped exponential every tutorial prints; and
# `fixed_interval` is what a `while True: sleep(5)` loop does, which is still
# the most common retry in production code.
#
# Each takes (attempt, base, cap, prev, rng) and returns the delay in seconds
# before attempt number `attempt` (0-indexed: attempt 0 is the first retry).


def no_jitter(attempt: int, base: float, cap: float, prev: float,
              rng: random.Random) -> float:
    """`min(cap, base * 2**attempt)` - capped exponential, no randomness.

    Deterministic, which is the whole problem: every client in the fleet runs
    the same function over the same attempt counter and gets the same number.
    Two clients that failed together retry together, forever.
    """
    return min(cap, base * (2.0 ** attempt))


def full_jitter(attempt: int, base: float, cap: float, prev: float,
                rng: random.Random) -> float:
    """`uniform(0, min(cap, base * 2**attempt))`.

    The window grows exponentially; the delay is uniform inside it. Lowest
    total work of the published options, because the expected delay is half
    the un-jittered one - which is also its cost: at a fixed attempt count it
    covers half the wall-clock time.
    """
    return rng.uniform(0.0, min(cap, base * (2.0 ** attempt)))


def equal_jitter(attempt: int, base: float, cap: float, prev: float,
                 rng: random.Random) -> float:
    """`t/2 + uniform(0, t/2)` where `t = min(cap, base * 2**attempt)`.

    Half the delay is guaranteed, half is random. Expected delay is 0.75t, so
    it keeps more of the coverage than full jitter while still spreading
    arrivals - but only over a window of width t/2, so the peak it produces is
    twice full jitter's for the same t.
    """
    t = min(cap, base * (2.0 ** attempt))
    return t / 2.0 + rng.uniform(0.0, t / 2.0)


def decorrelated_jitter(attempt: int, base: float, cap: float, prev: float,
                        rng: random.Random) -> float:
    """`min(cap, uniform(base, prev * 3))` - a random walk, not a ladder.

    The delay is derived from the *previous delay* rather than from the attempt
    counter, so two clients diverge after their first retry instead of tracking
    each other. E[next | prev] = (base + 3*prev)/2, so it grows at roughly 1.5x
    rather than 2x, with a long right tail.

    It is not monotone. A draw near the bottom of the range makes the next
    delay *shorter* than the last one. That is the documented behaviour and not
    a bug to be patched out - the non-monotonicity is what decorrelates it.
    """
    lo = base
    hi = max(base, prev * 3.0)
    return min(cap, rng.uniform(lo, hi))


def fixed_interval(attempt: int, base: float, cap: float, prev: float,
                   rng: random.Random) -> float:
    """`cap` every time - the `while True: sleep(5)` loop.

    Included because it is the most deployed retry in the world and because it
    makes the ceiling behaviour of every other policy legible: capped
    exponential *becomes* this, once the cap is reached.
    """
    return cap


POLICIES: Dict[str, Callable[..., float]] = {
    "no_jitter": no_jitter,
    "equal_jitter": equal_jitter,
    "full_jitter": full_jitter,
    "decorrelated_jitter": decorrelated_jitter,
    "fixed_interval": fixed_interval,
}

POLICY_ORDER = ["no_jitter", "fixed_interval", "equal_jitter",
                "full_jitter", "decorrelated_jitter"]


def bucket_index(t: float, width: float, since: float = 0.0) -> int:
    """Index of the half-open bucket `[since + i*w, since + (i+1)*w)` holding `t`.

    `int((t - since) / width)` is the obvious version and it is wrong. With
    width 0.1, `1.5 / 0.1` evaluates to 14.999999999999998, so an arrival
    exactly on a bucket edge is filed one bucket early. On a deterministic
    policy every client lands on the same edge, so the whole fleet is
    misplaced together and the histogram still looks plausible.

    Comparing against reconstructed edges does not fix it either, and that is
    the trap worth naming: `i * width` does not tile the line. With width 0.1,
    bucket 199 ends at 20.000000000000004 while bucket 200 begins at 20.0, so
    the two overlap and `t = 20.0` satisfies *both* half-open tests. A
    "reference implementation" written that way is wrong in the same place the
    fast one is, and agrees with it often enough to look verified.

    What works is snapping the quotient: compute it once, and treat a value
    within a relative hair of the next integer as being on that edge. The
    tests check this against integer arithmetic on scaled times, which is the
    only version with no float grid at all.
    """
    q = (t - since) / width
    i = int(math.floor(q))
    if q - i > 1.0 - 1e-9:
        i += 1
    return max(0, i)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Schedule:
    """One retry configuration."""

    policy: str
    base: float = 0.1
    cap: float = 20.0
    max_attempts: int = 10

    def delays(self, rng: random.Random) -> List[float]:
        """One client's full delay sequence."""
        fn = POLICIES[self.policy]
        out: List[float] = []
        prev = self.base
        for i in range(self.max_attempts):
            d = fn(i, self.base, self.cap, prev, rng)
            out.append(d)
            prev = d
        return out

    def arrivals(self, rng: random.Random, t0: float = 0.0) -> List[float]:
        """Absolute times at which one client's retries hit the dependency."""
        t = t0
        out: List[float] = []
        for d in self.delays(rng):
            t += d
            out.append(t)
        return out

    # -- deterministic analysis, no simulation -----------------------------

    def window(self, attempt: int) -> float:
        """The un-jittered ceiling for this attempt: `min(cap, base*2**n)`."""
        if self.policy == "fixed_interval":
            return self.cap
        return min(self.cap, self.base * (2.0 ** attempt))

    def naive_recurrence_total(self) -> float:
        """The closed form for `decorrelated_jitter` that looks right and is not.

        E[d_n] = min(cap, (base + 3*E[d_{n-1}]) / 2), iterated. Every step is
        individually defensible and the result is wrong, because `min` is
        concave and Jensen's inequality runs the wrong way:
        E[min(cap, X)] <= min(cap, E[X]). Substituting the mean into the
        truncation over-estimates the mean once the cap starts binding, and the
        error compounds over the walk.

        Kept, and named, because the honest version below is only interesting
        next to the version most people would ship. See `evidence.py` §4.
        """
        total, prev = 0.0, self.base
        for _ in range(self.max_attempts):
            e = min(self.cap, (self.base + 3.0 * prev) / 2.0)
            total += e
            prev = e
        return total

    def expected_total(self, samples: int = 4000, seed: int = 0) -> float:
        """Expected elapsed time from first failure to giving up.

        Exact and closed-form for the four *ladder* policies: each delay's
        window depends only on the attempt index, so linearity of expectation
        gives the sum directly no matter how the draws are correlated.

        `decorrelated_jitter` is not a ladder - each window is set by the
        previous *draw* - and its truncation at the cap makes the obvious
        recurrence biased (see `naive_recurrence_total`). It is estimated by
        sampling instead, with a fixed seed so the number is reproducible.
        """
        if self.policy == "decorrelated_jitter":
            rng = random.Random(seed)
            return sum(sum(self.delays(rng)) for _ in range(samples)) / samples
        factor = {"no_jitter": 1.0, "fixed_interval": 1.0,
                  "full_jitter": 0.5, "equal_jitter": 0.75}[self.policy]
        return factor * sum(self.window(i) for i in range(self.max_attempts))

    def worst_case_total(self) -> float:
        """The longest the schedule can possibly run - every draw at its max."""
        if self.policy == "decorrelated_jitter":
            total, prev = 0.0, self.base
            for _ in range(self.max_attempts):
                d = min(self.cap, prev * 3.0)
                total += d
                prev = d
            return total
        return sum(self.window(i) for i in range(self.max_attempts))

    def steady_state_rate(self, fleet: int) -> float:
        """Requests per second the dependency absorbs once every client caps.

        Once `base * 2**attempt >= cap` the window stops widening, so the mean
        inter-arrival time per client stops growing. It settles at
        `cap * factor`, and `fleet` clients each firing every `cap*factor`
        seconds is an aggregate rate of `fleet / (cap*factor)`.

        This is a *floor*. It does not decay, and jitter does not reduce it -
        jitter only decides whether the load arrives smoothly or in a spike.
        """
        factor = {"no_jitter": 1.0, "fixed_interval": 1.0, "full_jitter": 0.5,
                  "equal_jitter": 0.75, "decorrelated_jitter": 0.5}[self.policy]
        return fleet / (self.cap * factor)

    def cap_reached_at(self) -> Optional[int]:
        """First attempt index whose window is clamped by the cap."""
        if self.policy == "fixed_interval":
            return 0
        for i in range(self.max_attempts):
            if self.base * (2.0 ** i) >= self.cap:
                return i
        return None


# ---------------------------------------------------------------------------
# Fleet simulation
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    schedule: Schedule
    fleet: int
    outage_s: float
    capacity_rps: float
    bucket_s: float
    arrivals: List[float] = field(default_factory=list)
    admitted: List[float] = field(default_factory=list)
    rejected: List[float] = field(default_factory=list)
    succeeded: int = 0
    gave_up: int = 0
    horizon: float = 0.0

    # -- derived -----------------------------------------------------------

    def histogram(self, width: Optional[float] = None,
                  upto: Optional[float] = None,
                  since: float = 0.0) -> Tuple[List[float], List[int]]:
        """Arrival counts per bucket from `since`. Returns (left edges, counts).

        Buckets are aligned to `since`, not to zero, so a recovery-window
        histogram is not offset by wherever the outage happened to end.
        """
        w = width or self.bucket_s
        end = upto if upto is not None else (max(self.arrivals) if self.arrivals else since + w)
        nb = max(1, int(math.ceil((end - since) / w)))
        counts = [0] * nb
        for t in self.arrivals:
            if t < since:
                continue
            i = bucket_index(t, w, since)
            if 0 <= i < nb:
                counts[i] += 1
        return [since + i * w for i in range(nb)], counts

    def peak_rps(self, window_s: float = 1.0, since: float = 0.0) -> float:
        """Highest arrival rate over any window of `window_s` after `since`.

        Default `since=0` includes the requests thrown at the dependency while
        it is still down. Pass `since=outage_s` for the number that decides
        whether the *recovery* survives, which is the one the verdict uses.
        """
        if not self.arrivals:
            return 0.0
        _, counts = self.histogram(width=window_s, since=since)
        return max(counts) / window_s if counts else 0.0

    def recovery_peak_rps(self, window_s: float = 1.0) -> float:
        """Peak rate against the service once it is actually back up."""
        return self.peak_rps(window_s, since=self.outage_s)

    def total_requests(self) -> int:
        return len(self.arrivals)

    def wasted_requests(self) -> int:
        """Arrivals that hit the dependency while it was still down."""
        return sum(1 for t in self.arrivals if t < self.outage_s)

    def completion_time(self) -> Optional[float]:
        return max(self.admitted) if self.admitted else None

    def over_capacity_waves(self, window_s: float = 1.0,
                            since: Optional[float] = None) -> int:
        """Number of separated stretches where arrivals exceeded capacity."""
        s = self.outage_s if since is None else since
        _, counts = self.histogram(width=window_s, since=s)
        thresh = self.capacity_rps * window_s
        waves, inside = 0, False
        for c in counts:
            if c > thresh and not inside:
                waves += 1
                inside = True
            elif c <= thresh:
                inside = False
        return waves


def simulate(schedule: Schedule, fleet: int = 500, outage_s: float = 20.0,
             capacity_rps: float = 50.0, bucket_s: float = 0.1,
             seed: int = 7, horizon_s: float = 300.0) -> SimResult:
    """N clients all fail at t=0. Play the retries forward.

    Model, stated plainly so the numbers can be argued with:

    * The dependency is down for `outage_s`. Every request that arrives before
      then fails and burns one attempt.
    * After recovery it serves `capacity_rps`, enforced per bucket of width
      `bucket_s` (so `capacity_rps * bucket_s` requests per bucket). Excess
      arrivals in a bucket are shed - which is the *good* failure mode; a
      dependency without shedding falls over instead, and then the retries have
      caused a second outage rather than merely prolonged the first.
    * A shed request burns an attempt, exactly like a failed one. This is the
      step most retry discussions skip, and it is where herding comes from:
      load shed by an overloaded service is re-scheduled by the *same* policy
      that produced the spike.
    * A client that exhausts `max_attempts` gives up permanently.

    Ties inside a bucket are broken by client id, so the result is a pure
    function of `seed`.
    """
    rng = random.Random(seed)
    # Per-client RNG streams, so a client's draws do not depend on the
    # interleaving of other clients' events.
    streams = [random.Random(rng.randrange(1 << 30)) for _ in range(fleet)]

    prev_delay = [schedule.base] * fleet
    attempt = [0] * fleet
    heap: List[Tuple[float, int]] = []

    fn = POLICIES[schedule.policy]

    def push(cid: int) -> None:
        n = attempt[cid]
        if n >= schedule.max_attempts:
            return
        d = fn(n, schedule.base, schedule.cap, prev_delay[cid], streams[cid])
        prev_delay[cid] = d
        attempt[cid] = n + 1
        heapq.heappush(heap, (round(_now[cid] + d, 9), cid))

    _now = [0.0] * fleet
    for cid in range(fleet):
        push(cid)

    res = SimResult(schedule=schedule, fleet=fleet, outage_s=outage_s,
                    capacity_rps=capacity_rps, bucket_s=bucket_s)
    # A positive capacity always admits at least one request per bucket - a
    # service that serves 1 rps is not a service that serves nothing. Zero
    # capacity means zero, which is how the arrival process is isolated from
    # the service's response to it.
    per_bucket_cap = (0 if capacity_rps <= 0
                      else max(1, int(round(capacity_rps * bucket_s))))
    done = [False] * fleet

    while heap:
        t, _ = heap[0]
        if t > horizon_s:
            break
        bucket = bucket_index(t, bucket_s)
        batch: List[Tuple[float, int]] = []
        while heap and bucket_index(heap[0][0], bucket_s) == bucket:
            batch.append(heapq.heappop(heap))
        batch.sort(key=lambda e: e[1])  # deterministic tie-break

        served = 0
        for at, cid in batch:
            res.arrivals.append(at)
            _now[cid] = at
            if at < outage_s:
                res.rejected.append(at)
                push(cid)
            elif served < per_bucket_cap:
                served += 1
                res.admitted.append(at)
                done[cid] = True
                res.succeeded += 1
            else:
                res.rejected.append(at)
                push(cid)

    res.gave_up = sum(1 for c in range(fleet) if not done[c])
    res.horizon = max(res.arrivals) if res.arrivals else 0.0
    return res


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class Severity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    detail: str = ""


class Verdict(Enum):
    DISPERSED = "dispersed"
    BURSTY = "bursty"
    HERDING = "herding"


def amplification(layers: Sequence[int]) -> int:
    """Requests the bottom service sees per one user request.

    Each layer that retries `a` times multiplies the layer below it. Three
    layers at three attempts each is 27, not 9, and the bottom service is the
    one that was already unhealthy.
    """
    out = 1
    for a in layers:
        out *= max(1, a)
    return out


def sampled_totals(schedule: Schedule, n: int = 2000,
                   seed: int = 7) -> Tuple[float, float]:
    """(median, mean) elapsed time over `n` independent clients.

    `Schedule.expected_total()` is exact for the mean because expectation is
    linear. It says nothing about the median, and for `decorrelated_jitter` -
    where each delay is drawn from a range defined by the *previous* delay -
    the two are far apart: a client that draws low early stays low for the rest
    of the walk, while a rare client that draws high inflates the mean for
    everybody. The mean describes no client in particular.
    """
    rng = random.Random(seed)
    totals = sorted(sum(schedule.delays(rng)) for _ in range(n))
    med = totals[len(totals) // 2]
    return med, sum(totals) / len(totals)


def quantise(delays: Sequence[float], tick_s: float) -> List[float]:
    """Round delays up to a scheduler tick - what a 1s cron or timer wheel does."""
    return [math.ceil(d / tick_s) * tick_s for d in delays]


def audit(schedule: Schedule, fleet: int = 500, outage_s: float = 20.0,
          capacity_rps: float = 50.0, deadline_s: Optional[float] = None,
          nested_layers: Optional[Sequence[int]] = None,
          tick_s: Optional[float] = None, seed: int = 7,
          sim: Optional[SimResult] = None) -> Tuple[Verdict, List[Finding]]:
    """Everything the schedule will do that its signature does not say."""
    res = sim or simulate(schedule, fleet=fleet, outage_s=outage_s,
                          capacity_rps=capacity_rps, seed=seed)
    f: List[Finding] = []

    # 1 - the synchronous wave
    _, counts = res.histogram(width=res.bucket_s, since=res.outage_s)
    peak_bucket = max(counts) if counts else 0
    if schedule.policy in ("no_jitter", "fixed_interval"):
        f.append(Finding(
            "SYNCHRONOUS_WAVE", Severity.CRITICAL,
            f"policy is deterministic: all {fleet} clients retry at identical instants",
            f"peak of {peak_bucket} arrivals in one {res.bucket_s*1000:.0f}ms bucket "
            f"({peak_bucket/res.bucket_s:.0f} rps against {capacity_rps:.0f} rps of capacity)",
        ))

    # 2 - peak over capacity, measured on the recovering service
    peak = res.recovery_peak_rps(1.0)
    if peak > capacity_rps:
        f.append(Finding(
            "PEAK_OVER_CAPACITY", Severity.CRITICAL,
            f"peak arrival rate {peak:.0f} rps on the recovering service exceeds "
            f"capacity {capacity_rps:.0f} rps",
            f"{res.over_capacity_waves()} separate stretch(es) above capacity after "
            f"recovery; {len(res.rejected)} requests shed in total",
        ))

    # 2b - work spent on a service that is known to be down
    wasted = res.wasted_requests()
    if wasted:
        share = wasted / max(1, res.total_requests())
        f.append(Finding(
            "OUTAGE_HAMMERING", Severity.WARNING if share < 0.9 else Severity.CRITICAL,
            f"{wasted} of {res.total_requests()} requests ({share:.0%}) arrive while "
            f"the dependency is still down",
            f"peak {res.peak_rps(1.0):.0f} rps during the outage itself - connection "
            f"attempts against a dead socket are cheap for the client and are still "
            f"load on whatever is in front of it (LB, proxy, service mesh)",
        ))

    # 3 - the cap plateau
    floor = schedule.steady_state_rate(fleet)
    capped_at = schedule.cap_reached_at()
    if capped_at is not None and capped_at < schedule.max_attempts - 1:
        sev = Severity.CRITICAL if floor > capacity_rps else Severity.WARNING
        shape = ("delivered as one spike per cap interval, not as a rate"
                 if schedule.policy in ("no_jitter", "fixed_interval")
                 else "spread across the cap window, but never decaying")
        f.append(Finding(
            "CAP_PLATEAU", sev,
            f"window stops widening at attempt {capped_at}; load floor {floor:.0f} rps",
            f"cap={schedule.cap:g}s with {fleet} clients is an aggregate floor of "
            f"{floor:.0f} rps ({'above' if floor > capacity_rps else 'within'} the "
            f"{capacity_rps:.0f} rps the dependency serves), {shape}; the floor is "
            f"fleet/cap and no amount of jitter changes the numerator",
        ))

    # 4 - the budget dies before the outage does
    exp = schedule.expected_total()
    if exp < outage_s:
        f.append(Finding(
            "BUDGET_SHORTER_THAN_OUTAGE", Severity.CRITICAL,
            f"expected elapsed {exp:.1f}s is shorter than the {outage_s:g}s outage",
            f"{res.gave_up} of {fleet} clients exhaust {schedule.max_attempts} attempts "
            f"before the dependency is back",
        ))
    elif res.gave_up:
        f.append(Finding(
            "CLIENTS_GAVE_UP", Severity.CRITICAL,
            f"{res.gave_up} of {fleet} clients exhausted their attempts",
            "shed load burns attempts exactly like failed load, so the spike "
            "consumes the budget it created",
        ))

    # 5 - jitter halves coverage
    if schedule.policy in ("full_jitter", "equal_jitter", "decorrelated_jitter"):
        worst = schedule.worst_case_total()
        if worst > outage_s > exp:
            f.append(Finding(
                "JITTER_SHORTENS_COVERAGE", Severity.WARNING,
                f"same attempt count covers {exp:.1f}s expected vs {worst:.1f}s worst case",
                "jitter is drawn downward from the ceiling, so a budget sized "
                "against the un-jittered schedule loses roughly half its reach",
            ))

    # 6 - deadline
    if deadline_s is not None:
        if schedule.expected_total() > deadline_s:
            n_fit = 0
            acc = 0.0
            factor = exp / max(schedule.worst_case_total(), 1e-9)
            for i in range(schedule.max_attempts):
                acc += schedule.window(i) * factor
                if acc > deadline_s:
                    break
                n_fit += 1
            f.append(Finding(
                "DEADLINE_OVERRUN", Severity.CRITICAL,
                f"schedule runs {exp:.1f}s against a {deadline_s:g}s caller deadline",
                f"only ~{n_fit} of {schedule.max_attempts} attempts happen before the "
                f"caller has already given up; the rest are work nobody is waiting for",
            ))
        elif schedule.worst_case_total() < deadline_s * 0.5:
            f.append(Finding(
                "BUDGET_UNDERUSE", Severity.INFO,
                f"worst case {schedule.worst_case_total():.1f}s uses under half "
                f"the {deadline_s:g}s deadline",
                "there is room for more attempts or a longer cap",
            ))

    # 7 - amplification
    if nested_layers:
        amp = amplification(nested_layers)
        if amp > 1:
            f.append(Finding(
                "RETRY_AMPLIFICATION", Severity.CRITICAL if amp >= 8 else Severity.WARNING,
                f"nested retries multiply to {amp}x at the bottom service",
                f"layers {list(nested_layers)} - each layer retries the layer below, "
                f"so one user request becomes {amp} requests against the dependency "
                f"that was already unhealthy",
            ))

    # 8 - clock alignment
    if tick_s:
        rng = random.Random(seed)
        d = schedule.delays(rng)
        q = quantise(d, tick_s)
        collapsed = sum(1 for a, b in zip(d, q) if schedule.window(0) and b - a > 0)
        narrow = [i for i in range(schedule.max_attempts)
                  if schedule.window(i) < tick_s]
        if narrow:
            f.append(Finding(
                "CLOCK_ALIGNMENT", Severity.WARNING,
                f"attempts {narrow[0]}-{narrow[-1]} have a jitter window narrower "
                f"than the {tick_s:g}s scheduler tick",
                f"rounding to the tick puts every client back in the same slot; "
                f"{collapsed}/{len(d)} of this client's delays were rounded up",
            ))

    # 9 - decorrelated jitter is not monotone (documented, not a defect)
    if schedule.policy == "decorrelated_jitter":
        rng = random.Random(seed)
        d = schedule.delays(rng)
        drops = sum(1 for i in range(1, len(d)) if d[i] < d[i - 1])
        f.append(Finding(
            "NON_MONOTONE_BY_DESIGN", Severity.INFO,
            f"{drops} of {len(d)-1} steps in this client's schedule are shorter "
            f"than the previous step",
            "decorrelated jitter draws from uniform(base, 3*prev), which can go "
            "down; that is what decorrelates two clients and should not be "
            "'fixed' by adding a max() against the previous delay",
        ))

    # 10 - the mean schedule is not the typical schedule
    if schedule.policy in ("full_jitter", "equal_jitter", "decorrelated_jitter"):
        med, mean = sampled_totals(schedule, n=2000, seed=seed)
        if med < 0.85 * mean:
            f.append(Finding(
                "SKEWED_BUDGET", Severity.WARNING,
                f"median client covers {med:.1f}s but the mean is {mean:.1f}s "
                f"({med/mean:.0%} of it)",
                "the delay distribution is right-skewed, so sizing the attempt "
                "budget off the expected total describes a client that does not "
                "exist; half the fleet gives up sooner than that",
            ))

    # -- verdict ------------------------------------------------------------
    if peak <= capacity_rps:
        v = Verdict.DISPERSED
    elif res.gave_up == 0 and res.over_capacity_waves() <= 1:
        v = Verdict.BURSTY
    else:
        v = Verdict.HERDING

    order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    f.sort(key=lambda x: order[x.severity])
    return v, f


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def compare(fleet: int = 500, outage_s: float = 20.0, capacity_rps: float = 50.0,
            base: float = 0.1, cap: float = 20.0, max_attempts: int = 10,
            seed: int = 7) -> Dict[str, SimResult]:
    """Run every policy through the same outage."""
    out: Dict[str, SimResult] = {}
    for name in POLICY_ORDER:
        s = Schedule(policy=name, base=base, cap=cap, max_attempts=max_attempts)
        out[name] = simulate(s, fleet=fleet, outage_s=outage_s,
                             capacity_rps=capacity_rps, seed=seed)
    return out
