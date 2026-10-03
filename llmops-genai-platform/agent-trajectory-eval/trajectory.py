"""Outcome-only and reference-match scoring of agent trajectories, measured exactly.

One declared task: a refund agent. A customer asks for a refund; it is eligible with probability P_ELIG.
A valid run looks up the order, checks the policy, then refunds (once, to the looked-up order) or declines.
Five things are declared invariants of a valid path (`violations`). The agent is a stochastic policy over a
handful of discrete choices, so every trajectory it can produce - and its probability - is enumerated
(`enumerate_runs`), and every scorer's error rate is an exact sum over that list. `simulate` builds the same
trajectories step by step from raw draws and is the check on the enumeration.

Scorers compare what real harnesses compare: the final answer only, or the tool-name sequence against one
reference path (strict / unordered / superset - the usual match modes), or both.
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

P_ELIG = 0.7
ORDER = "A17"
WRONG = "A71"
PLANS = ("lookup>policy", "policy>lookup", "lookup only", "no reads")
PLAN_STEPS = {"lookup>policy": ("lookup_order", "check_policy"), "policy>lookup": ("check_policy", "lookup_order"),
              "lookup only": ("lookup_order",), "no reads": ()}

# v2 is v1 after a "use fewer tool calls" prompt change: cheaper, and it skips reads more often.
AGENTS: Dict[str, Dict[str, object]] = {
    "v1 careful": {"plan": (0.60, 0.30, 0.07, 0.03), "kb": 0.35, "rash": 0.03, "dup": 0.03, "badarg": 0.02,
                   "misread": 0.02},
    "v2 fewer calls": {"plan": (0.45, 0.15, 0.25, 0.15), "kb": 0.10, "rash": 0.10, "dup": 0.03, "badarg": 0.02,
                       "misread": 0.02},
}
# "outcome+strict" is left out: here a strict match implies the right answer (the action IS the decision).
SCORERS = ("outcome", "strict", "unordered", "superset", "outcome+superset")
INVARIANTS = ("unsupported decision", "acted before reading", "duplicate action", "wrong argument",
              "wrong action")
MC_REPS = 200_000
SEED = 189

Step = Tuple[str, Optional[str]]


def reference(eligible: bool) -> List[str]:
    return ["lookup_order", "check_policy"] + (["issue_refund"] if eligible else [])


def build(eligible: bool, plan: str, kb: bool, rash: bool, guess_yes: bool, misread: bool, dup: bool,
          badarg: bool) -> Tuple[List[Step], bool]:
    """One trajectory and whether its final answer is right.

    A rash agent decides before reading (a guess, and any refund goes first). A careful agent that read the
    policy decides correctly unless it misreads; one that never read the policy guesses. A refund issued
    without a lookup can only carry an invented order id.
    """
    reads = PLAN_STEPS[plan]
    informed = "check_policy" in reads and not rash
    decide_yes = (eligible != misread) if informed else guess_yes
    looked = "lookup_order" in reads and not rash
    arg = WRONG if (badarg or not looked) else ORDER
    refunds: List[Step] = [("issue_refund", arg)] * ((2 if dup else 1) if decide_yes else 0)
    steps: List[Step] = [("search_kb", None)] if kb else []
    read_steps: List[Step] = [(s, ORDER if s == "lookup_order" else None) for s in reads]
    steps += (refunds + read_steps) if rash else (read_steps + refunds)
    return steps, decide_yes == eligible


def violations(steps: Sequence[Step], eligible: bool) -> List[str]:
    """The declared spec of a valid path. Ground truth by construction - the point is what cheaper scorers miss."""
    names = [s for s, _ in steps]
    out = []
    if not {"lookup_order", "check_policy"} <= set(names):
        out.append("unsupported decision")
    seen: set = set()
    looked: Optional[str] = None
    for s, a in steps:
        if s == "issue_refund":
            if not {"lookup_order", "check_policy"} <= seen and "acted before reading" not in out:
                out.append("acted before reading")
            if a != looked and "wrong argument" not in out:
                out.append("wrong argument")
        if s == "lookup_order":
            looked = a
        seen.add(s)
    if names.count("issue_refund") > 1:
        out.append("duplicate action")
    if ("issue_refund" in names) != eligible:
        out.append("wrong action")
    return out


def score(steps: Sequence[Step], eligible: bool, correct: bool) -> Dict[str, bool]:
    names = [s for s, _ in steps]
    ref = reference(eligible)
    have, want = Counter(names), Counter(ref)
    strict = names == ref
    sup = all(have[k] >= v for k, v in want.items())
    return {"outcome": correct, "strict": strict, "unordered": have == want, "superset": sup,
            "outcome+superset": correct and sup}


def enumerate_runs(agent: Dict[str, object]) -> List[Dict[str, object]]:
    """Every (task, choice) combination with its probability. Choices that cannot matter are still enumerated."""
    runs = []
    for elig, pi, kb, rash, gy, mis, dup, bad in itertools.product(
            (True, False), range(4), (True, False), (True, False), (True, False), (True, False), (True, False),
            (True, False)):
        p = P_ELIG if elig else 1 - P_ELIG
        p *= agent["plan"][pi]
        for flag, q in ((kb, agent["kb"]), (rash, agent["rash"]), (gy, P_ELIG), (mis, agent["misread"]),
                        (dup, agent["dup"]), (bad, agent["badarg"])):
            p *= q if flag else 1 - q
        steps, correct = build(elig, PLANS[pi], kb, rash, gy, mis, dup, bad)
        v = violations(steps, elig)
        runs.append({"p": p, "eligible": elig, "steps": steps, "correct": correct, "violations": v,
                     "good": correct and not v, "scores": score(steps, elig, correct)})
    return runs


def summarise(runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    pg = sum(r["p"] for r in runs if r["good"])
    out: Dict[str, object] = {"p_good": pg, "p_correct": sum(r["p"] for r in runs if r["correct"]),
                              "calls": sum(r["p"] * len(r["steps"]) for r in runs), "scorers": {}}
    for s in SCORERS:
        pp = sum(r["p"] for r in runs if r["scores"][s])
        pass_bad = sum(r["p"] for r in runs if r["scores"][s] and not r["good"])
        fail_good = sum(r["p"] for r in runs if not r["scores"][s] and r["good"])
        out["scorers"][s] = {"reported": pp, "false_pass": pass_bad / (1 - pg), "false_fail": fail_good / pg,
                             "bad_among_passes": pass_bad / pp if pp else 0.0}
    out["hidden_by_outcome"] = {k: sum(r["p"] for r in runs if r["correct"] and k in r["violations"])
                                for k in INVARIANTS}
    return out


def evaluate() -> Dict[str, Dict[str, object]]:
    return {name: summarise(enumerate_runs(a)) for name, a in AGENTS.items()}


def simulate(agent: Dict[str, object], reps: int = MC_REPS, seed: int = SEED) -> Dict[str, float]:
    """Raw draws, step by step, through the same build/violations/score code. Checks the enumeration."""
    rng = np.random.default_rng(seed)
    elig = rng.random(reps) < P_ELIG
    plan = rng.choice(4, size=reps, p=agent["plan"])
    flags = {k: rng.random(reps) < q for k, q in (("kb", agent["kb"]), ("rash", agent["rash"]), ("gy", P_ELIG),
                                                   ("mis", agent["misread"]), ("dup", agent["dup"]),
                                                   ("bad", agent["badarg"]))}
    tally: Counter = Counter()
    for i in range(reps):
        steps, correct = build(bool(elig[i]), PLANS[plan[i]], *(bool(flags[k][i]) for k in
                                                                 ("kb", "rash", "gy", "mis", "dup", "bad")))
        good = correct and not violations(steps, bool(elig[i]))
        tally["good"] += good
        for s, ok in score(steps, bool(elig[i]), correct).items():
            tally[s] += ok
    return {k: v / reps for k, v in tally.items()}


def wilson(p: float, n: int, z: float = 2.576) -> Tuple[float, float]:
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def calibrate(reps: int = MC_REPS) -> List[Dict[str, object]]:
    rows = []
    for name, a in AGENTS.items():
        ex = summarise(enumerate_runs(a))
        mc = simulate(a, reps)
        for k in ("good",) + SCORERS:
            e = ex["p_good"] if k == "good" else ex["scorers"][k]["reported"]
            lo, hi = wilson(mc[k], reps)
            rows.append({"agent": name, "quantity": k, "exact": e, "mc": mc[k], "inside_99": bool(lo <= e <= hi)})
    return rows


def parse_steps(text: str) -> List[Step]:
    """`search_kb;lookup_order:A17;check_policy;issue_refund:A17` -> [(tool, arg), ...]."""
    out: List[Step] = []
    for tok in [t.strip() for t in text.split(";") if t.strip()]:
        name, _, arg = tok.partition(":")
        out.append((name.strip(), arg.strip() or None))
    return out


def audit(rows: Sequence[Tuple[bool, bool, str]]) -> Dict[str, object]:
    """Score your own runs: (eligible, answer_correct, steps-text) each. Returns per-scorer pass rates against
    the invariant spec, and which invariant each outcome-passing run broke."""
    scored = []
    for elig, correct, text in rows:
        steps = parse_steps(text)
        v = violations(steps, elig)
        scored.append({"good": correct and not v, "violations": v, "scores": score(steps, elig, correct),
                       "correct": correct})
    n = len(scored)
    res: Dict[str, object] = {"n": n, "good": sum(r["good"] for r in scored) / n, "scorers": {}}
    for s in SCORERS:
        res["scorers"][s] = {"reported": sum(r["scores"][s] for r in scored) / n,
                             "passed_bad": sum(r["scores"][s] and not r["good"] for r in scored),
                             "failed_good": sum((not r["scores"][s]) and r["good"] for r in scored)}
    res["hidden"] = Counter(v for r in scored if r["correct"] for v in r["violations"])
    return res
