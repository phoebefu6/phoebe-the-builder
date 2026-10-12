"""Generate demo.ipynb.

Plumbing copied from `latency-budget` (Day 194), content written fresh. The engine is EXTRACTED from
escalation.py by AST at build time, so the notebook holds the library's own source text. The write lives behind
__main__: test_notebook.py imports this module, and an import-time write would replace the executed notebook
with a blank one (the Day 177 defect).

Every cost is exact on the confidence grid, so the notebook reproduces evidence.txt to the digit. Only the Monte
Carlo calibration runs NB_DAYS instead of the study's 40 days - stated in the notebook, and a test asserts the note.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "llmops-genai-platform/human-escalation-policy"
NB_DAYS = 8


def extract(path: str) -> str:
    """Every top-level def and assignment, in source order. Imports are left to HEADER."""
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
            out.append("".join(lines[start:node.end_lineno]))
    return "\n\n".join(out).rstrip()


ENGINE = extract("escalation.py")

EXPORTS = [
    "SEED", "MC_DAYS", "BINS", "CENTERS", "SEGMENTS", "SEGS", "BASE", "segment_tables", "human_error", "evaluate",
    "chow_threshold", "snap", "EDGES", "best_global", "best_per_segment", "rate_target", "policies", "policy_table",
    "load_curve", "INFLATE_V2", "upgrade", "value_curve", "simulate", "wilson", "calibrate", "parse_params",
]

HEADER = """from __future__ import annotations

import math
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
"""

cells: List[Dict[str, Any]] = []
_N = 0


def _lines(src: str) -> List[str]:
    """nbformat wants each source line to KEEP its trailing newline."""
    return src.strip("\n").splitlines(keepends=True)


def _nid() -> str:
    global _N
    _N += 1
    return f"c{_N:02d}"


def md(src: str) -> None:
    cells.append({"cell_type": "markdown", "id": _nid(), "metadata": {}, "source": _lines(src)})


def code(src: str) -> None:
    cells.append({"cell_type": "code", "id": _nid(), "execution_count": None,
                  "metadata": {}, "outputs": [], "source": _lines(src)})


# ---------------------------------------------------------------------------

md(f"""
# When should a human take over?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

An AI support agent handles 10,000 tickets a day and reports a confidence for each one. Below some cutoff, the
ticket goes to a human. This notebook prices that cutoff by the cost of the error a review prevents, and then adds
the two things textbook escalation rules leave out: the human team has a daily capacity, and humans make more
mistakes as their queue fills. Everything is a constructed, declared world, and every cost is exact.

1. The engine
2. Calibration - exact vs simulation
3. The segments: same agent, three prices of being wrong
4. What one review is worth
5. Seven policies, one day
6. Escalating more is not always safer
7. Same skill, louder confidence
8. The chart
9. Try your own
""")

md("""
## 1. The engine

Extracted from `escalation.py` at build time. Reported confidence lives on 40 bins. Each segment declares an
underlying score x ~ Beta(a, b), a true accuracy x ** gamma (gamma > 1 is overconfident), and a reporting
inflation x ** (1 / k). A policy is one cutoff per segment. Escalations past capacity time out and the agent acts.
""")

code(HEADER + "\n" + ENGINE)

md(f"""
## 2. Calibration - exact vs simulation

`simulate` plays whole days ticket by ticket: draw a segment and a confidence, draw whether the agent is right,
pick which escalations a full queue drops, draw whether the human is right. Each exact per-segment error rate
should sit inside the 99% Wilson interval of the simulated share.

**Resolution of the instrument.** The study in `evidence.txt` simulates {40} days; this notebook simulates
{NB_DAYS} so it runs in seconds, which makes its intervals about 2x wider. The exact numbers below do not depend
on this.
""")

code(f"""rows = []
for name, thr in policies().items():
    ex = evaluate(thr)
    mc = simulate(thr, days={NB_DAYS})
    for s in SEGS:
        exact = ex["errors"][s] / (BASE["tickets"] * SEGMENTS[s]["share"])
        lo, hi = wilson(mc[s]["errors"], mc[s]["tickets"])
        rows.append(lo <= exact <= hi)
print(f"{{sum(rows)}}/{{len(rows)}} exact error rates inside the simulation's 99% Wilson interval")""")

md("""
## 3. The segments: same agent, three prices of being wrong

A wrong refund decision costs $120, a wrong account change $40, a wrong FAQ answer $3. The agent is least accurate
exactly where errors cost most, and it is most overconfident there too: on refunds it reports 88.6% and is right
80.0% of the time.
""")

code("""tab = segment_tables()
print(f"{'segment':<9} {'share':>6} {'cost of a wrong action':>23} {'true accuracy':>14} {'mean reported conf':>19}")
for s in SEGS:
    v = SEGMENTS[s]
    acc = float((tab[s]["pmf"] * tab[s]["p_true"]).sum())
    conf = float((tab[s]["pmf"] * CENTERS).sum())
    print(f"{s:<9} {v['share']:>6.0%} {'$' + format(v['cost'], '.0f'):>23} {acc:>14.1%} {conf:>19.1%}")""")

md("""
## 4. What one review is worth

With a calm queue, sending a ticket to a human saves (1 - p) * cost and spends the review plus the human's own
error rate times the cost. That is Chow's reject rule with a fallible reviewer. A FAQ review never pays for itself:
the best FAQ ticket a human could see is still a $3 problem.
""")

code("""vc = value_curve()
for s in SEGS:
    vals = [(c, v) for c, v in zip(CENTERS, vc[s]) if v == v]
    pos = [c for c, v in vals if v > 0]
    hi = f"pays below conf {max(pos) + 0.5 / BINS:.3f}" if pos else "never pays"
    print(f"   {s:<8} {hi:<26} best bin ${max(v for _, v in vals):.1f}")""")

md("""
## 5. Seven policies, one day

The common rule (escalate anything under 90% confidence) sends 3,655 tickets to a 700-a-day team and costs MORE
than never escalating. Pricing each segment with Chow's rule is right on paper and still loses to one plain cutoff,
because it asks for two to three times the team's capacity. The policy that wins prices the errors AND respects
capacity: every review goes to refunds, and the queue stops at 79% full, before humans start to slip.
""")

code("""table = policy_table()
never = table[0]
print(f"{'policy':<28} {'escalated':>9} {'reviewed':>9} {'human err':>10} {'error rate':>11} {'cost/day':>9} {'vs never':>9}")
for r in table:
    print(f"{r['policy']:<28} {r['escalated']:>9.0f} {r['reviewed_share']:>9.0%} {r['human_err']:>10.1%} "
          f"{r['error_rate']:>11.2%} {'$' + format(r['cost'], ',.0f'):>9} {r['cost'] / never['cost'] - 1:>+9.1%}")""")

md("""
## 6. Escalating more is not always safer

Scale the calibrated priced cutoffs from 0.6x to 1.2x. Cost falls while the queue has room, bottoms out at 80% of
capacity, then climbs as overflow and tired reviewers eat the gain.
""")

code("""for r in load_curve():
    print(f"scale {r['scale']:.2f}  escalated {r['escalated']:>5.0f}  demand {r['utilisation_demand']:>5.2f}x  "
          f"human err {r['human_err']:.1%}  ${r['cost']:,.0f}/day")""")

md("""
## 7. Same skill, louder confidence

A new model version is exactly as accurate and reports higher numbers (0.81 skill now reads 0.90). Keep the same
cutoffs and the escalation rate falls from 5.5% to 1.2%. A deflection dashboard calls that a win; the day costs 27%
more, almost all of it refunds the humans no longer see. Refitting the cutoffs on labels from the new version brings
the cost back. A threshold belongs to a model version, not to the product.
""")

code("""u = upgrade()
print(f"{'':<22} {'escalation rate':>16} {'error rate':>11} {'refund error $':>15} {'cost/day':>10}")
for k, label in (("before", "v1, joint thresholds"), ("after", "v2, same thresholds"), ("refit", "v2, refitted")):
    r = u[k]
    print(f"{label:<22} {r['escalation_rate']:>16.2%} {r['error_rate']:>11.2%} {'$' + format(r['refund_err_cost'], ',.0f'):>15} "
          f"{'$' + format(r['cost'], ',.0f'):>10}")""")

md("""
## 8. The chart

The full audit figure is `escalation_audit.png` (built by `make_chart.py` from `results.json`). Here, the cost of a
day as the queue fills, for the calibrated priced rule scaled up and down.
""")

code("""lc = load_curve()
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot([r["utilisation_demand"] for r in lc], [r["cost"] / 1000 for r in lc], color="#c8562b", lw=2.2, marker="o")
ax.axvline(1.0, color="#2e8b57", ls="--", lw=1.2)
ax.text(1.03, 49, "team capacity", color="#2e8b57")
ax.axhline(policy_table()[0]["cost"] / 1000, color="#8a94a3", ls=":", lw=1)
ax.text(3.2, 50.8, "never escalate", color="#6b7684")
ax.set_xlabel("escalations demanded / team capacity")
ax.set_ylabel("cost per day, $ thousands")
ax.set_title("More escalation helps until the queue fills, then it costs more", loc="left")
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- The common rule, escalate below 90% confidence, costs 0.7% MORE than never escalating: it floods a 700/day team
  with 3,655 tickets, the queue drops 81% of them, and the reviewers it does reach make 12% errors.
- Pricing each segment by the error it prevents is right on paper. With capacity binding, it loses to one cutoff
  ($48,277 vs $43,633), because calibration reveals more cases worth a review than the team can take.
- Pricing AND respecting capacity wins: $35,153/day, 19.4% below the best single cutoff, all 551 reviews on refunds,
  queue 79% full.
- With unlimited capacity the textbook priced rule is within about 1% of the optimum. The trap is the ceiling.
- Same skill, louder confidence: escalations 5.5% -> 1.2%, cost +27%. Refit cutoffs on every model version.
""")

md("## 9. Try your own")

code("""# Your team, your prices, e.g. a bigger team with slower, careful reviewers:
# mine = parse_params("capacity=1500, review_cost=12, human_err=0.02")
# for r in policy_table(mine): print(r["policy"], round(r["escalated"]), round(r["cost"]))""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version takes your
own team: `pip install -r requirements.txt && streamlit run app.py`.

Sibling builds: [`threshold-explorer`](../../ml-engineering-toolkit/threshold-explorer) picks one binary cutoff with
two error prices; [`calibration-checker`](../../ml-engineering-toolkit/calibration-checker) asks whether a 0.9
means 90%. This build adds a human team with a ceiling and its own error rate.
[`refusal-rate-monitor`](../refusal-rate-monitor) is an earlier audit of a rate that moved for the wrong reason.
""")


def write() -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open("demo.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote demo.ipynb with {len(cells)} cells ({len(ENGINE.splitlines())} engine lines embedded)")


if __name__ == "__main__":
    write()
