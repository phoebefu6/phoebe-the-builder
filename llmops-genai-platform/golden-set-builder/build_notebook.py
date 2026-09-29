"""Generate demo.ipynb.

Plumbing copied from `bayes-vs-p` (Day 185), content written fresh. The engine is EXTRACTED from
golden.py by AST at build time, so the notebook holds the library's own source text. The write lives
behind __main__: test_notebook.py imports this module, and an import-time write would replace the
executed notebook with a blank one (the Day 177 defect).

The estimation numbers are exact (binomial moments), so the notebook reproduces them to the digit.
Only the Monte Carlo parts run at NB_REPS instead of the study's 20,000 - stated in the notebook (the
Day 179 resolution note), and a test asserts the note exists.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "llmops-genai-platform/golden-set-builder"
NB_REPS = 4000


def extract(path: str) -> str:
    """Every top-level def and assignment, in source order. Imports are left to HEADER.

    Tuple targets are walked, because an extractor reading only `targets[0].id` drops them
    silently - the Day 178 defect.
    """
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
            out.append("".join(lines[start:node.end_lineno]))
    return "\n\n".join(out).rstrip()


ENGINE = extract("golden.py")

EXPORTS = [
    "ALPHA", "M", "MONTHS", "INTENTS", "EXISTING", "W0", "W12", "Q", "DESIGNS", "FLAKY", "REG_INTENT", "REG_FLIP",
    "MC_REPS", "mix", "truth", "allocate", "moments", "moments_random", "headline", "coverage_gap", "refresh",
    "mcnemar_p", "regression_power", "audit", "wilson", "calibrate",
]

HEADER = """from __future__ import annotations

from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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
# Is your golden set still a sample of your traffic?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A team labels 240 production conversations once, calls it the golden set, and reports the model's pass
rate on it for a year. This notebook measures how that number drifts away from production, how much of
the drift a free fix removes, which part only new labels remove, and whether the set can see a
regression confined to one small intent. The traffic is synthetic, with its structure declared in the
engine, so every claim can be checked against a known truth.

1. The traffic, and calibration of the exact moments
2. Four ways to allocate 240 cases, scored at month 0 and month 12
3. Staleness, split into mix drift and coverage gap
4. Negative result: a bigger frozen set does not help
5. The refresh: 40 labels, not 240
6. Catching a regression that halves one intent
7. Try your own
""")

md("""
## The engine

Extracted from `golden.py` at build time, character for character. Where a design fixes the per-intent
counts n_k, the golden-set pass rate is a weighted sum of independent binomials, so its bias and SD are
exact. Simple random sampling (random counts) with reweighting, and regression power, are Monte Carlo.
""")

code(HEADER + "\n" + ENGINE + "\n\nprint('engine loaded')")

md(f"""
**Resolution of the instrument.** The estimation numbers are exact and match `evidence.txt` to the digit.
The regression-power cells run at {NB_REPS:,} replicates here instead of the study's 20,000, so they can
differ from `evidence.txt` in the second decimal - lower resolution, not disagreement. `evidence.txt` is
the reference.
""")

md("## 1. The traffic and the calibration")

code("""
for i, name in enumerate(INTENTS):
    print(f"{name:<17} month 0 {W0[i]:.3f}   month 12 {W12[i]:.3f}   pass rate {Q[i]:.2f}")
print(f"production pass rate: month 0 {truth(0):.4f}, month 12 {truth(12):.4f}")
cal = calibrate(4000)
for r in cal["moments"]:
    print(f"{r['design']:<12} m{r['month']:<2} {'rew' if r['reweight'] else 'raw'}  bias z {r['bias_z']:+.2f}  sd ratio {r['sd_ratio']:.3f}")
print("McNemar vs scipy binomtest mismatches:", cal["mcnemar_vs_binomtest_mismatches"])
""")

md("## 2. Four allocations of 240 cases")

code("""
for d in DESIGNS:
    cells = [headline(d, t, rw) for t, rw in ((0, False), (0, True), (12, False), (12, True))]
    print(f"{d:<13}" + "  ".join(f"{100 * c['bias']:+5.1f}+/-{100 * c['sd']:.1f}" for c in cells))
print("columns: month 0 raw, month 0 reweighted, month 12 raw, month 12 reweighted (pts)")
""")

md("""
Proportional (and random) allocation is right at month 0 and 5.4 points high by month 12. Equal
allocation is 5.1 points LOW at month 0 if you report its raw pass rate, because it over-samples the
hard tail intents; reweighted it is unbiased, at an RMSE of 2.34 instead of 1.90. At month 12 its raw
number happens to land near the truth (+0.4): two errors cancelling, not a property worth relying on.
Reweighted, every design ends at exactly the same +4.1 - the part no design fixes.
""")

md("## 3. Staleness, decomposed")

code("""
n = allocate("proportional")
months = list(range(0, 13, 2))
raw = [100 * moments(n, t, False)["bias"] for t in months]
rew = [100 * moments(n, t, True)["bias"] for t in months]
gap = [coverage_gap(t)["uncovered_share"] for t in months]
for t, a, b, g in zip(months, raw, rew, gap):
    print(f"month {t:>2}: raw {a:+.1f}  reweighted {b:+.1f}  uncovered traffic {g:.1%}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].plot(months, raw, "o-", color="#c8562b", label="raw")
axes[0].plot(months, rew, "o-", color="#2f6fdb", label="reweighted")
axes[0].axhline(0, color="#1f2733", lw=0.8)
axes[0].set_xlabel("months since frozen"); axes[0].set_ylabel("bias (pts)"); axes[0].legend(frameon=False)
axes[0].set_title("Staleness", loc="left")
designs = ["proportional", "sqrt", "equal"]
pw = [regression_power(allocate(d), reps=4000)["aggregate"] for d in designs]
axes[1].bar(designs, pw, color="#2f6fdb")
axes[1].set_ylim(0, 1); axes[1].set_ylabel("P(regression flagged)")
axes[1].set_title("Catching a halved 'bulk export'", loc="left")
plt.tight_layout(); plt.savefig("notebook_chart.png", dpi=120); plt.show()
""")

md("""
Reweighting per-intent pass rates by this month's traffic counts - which need no labels - removes the
1.3 points caused by the mix shifting between intents that already existed. The other 4.1 points come
from the 13% of traffic in two intents launched after the set was built. They have no cases, so no
weighting scheme can see them.
""")

md("## 4. Negative result - a bigger frozen set")

code("""
for m in (60, 240, 960, 2400):
    a, b = headline("proportional", 12, False, m), headline("proportional", 12, True, m)
    print(f"m = {m:>5}: raw {100 * a['bias']:+.1f} (SD {100 * a['sd']:.2f})   reweighted {100 * b['bias']:+.1f} (SD {100 * b['sd']:.2f})")
""")

md("""
Ten times the labels shrinks the SD and leaves the bias where it was: a more precise wrong number. The
instinct "our eval set is too small" is right about noise and wrong about staleness.
""")

md("## 5. The refresh")

code("""
for add in (0, 5, 10, 20, 40):
    nn = refresh(allocate("proportional"), add)
    a, b = moments(nn, 12, False), moments(nn, 12, True)
    print(f"+{add:>2} per new intent: raw bias {100 * a['bias']:+.1f}, reweighted bias {100 * b['bias']:+.1f}, RMSE {100 * b['rmse']:.2f}")
""")

md("""
Five labels per new intent remove the bias entirely (once reweighted); 20 each - 40 labels, a sixth of a
rebuild - bring the RMSE back to 2.16 against 1.90 at month 0. Without the reweighting the raw number
overshoots as soon as the new intents are over-represented, so the two fixes go together.
""")

md("## 6. Catching a regression confined to one intent")

code(f"""
for d in ("proportional", "sqrt", "equal"):
    nn = allocate(d)
    p, f = regression_power(nn, reps={NB_REPS}), regression_power(nn, flip=0, reps={NB_REPS})
    print(f"{{d:<13}} {{p['cases_in_regressed']:>2}} cases there   aggregate {{p['aggregate']:.3f}}   per-intent {{p['per_intent']:.3f}}"
          f"   false alarms {{f['aggregate']:.3f}} / {{f['per_intent']:.3f}}")
""")

md("""
v2 breaks half of the passing cases in "bulk export" (0.72 -> 0.36), a 1.6-point drop in production.
The proportional set holds 8 such cases and flags it about 12% of the time. The equal set holds 24 and
flags it 57%. Per-intent alerting with a Bonferroni correction does NOT help here - the correction costs
more than the slicing buys - so slice to diagnose, not to detect.

The first version of this simulation flipped every outcome with probability 0.03 between runs. That
lowers a re-run's pass rate (there are nine passes to flip for every fail), and the aggregate test
flagged the SAME model 58% of the time. Flakiness is now modelled per case, and a test pins the false-
alarm rate under 0.06.

## Summary

| claim | what the numbers say |
|---|---|
| "our golden set tracks production" | frozen for 12 months: 89.9% on the set, 84.4% in production |
| "reweight it and it is fine" | removes 1.3 of 5.4 pts; 4.1 is traffic with no cases |
| "we need a bigger set" | 10x the labels, same +4.1 bias |
| "the headline set also catches regressions" | proportional: 12% chance on a halved 3% intent; equal + reweighting: 57%, unbiased headline |
""")

md("## 7. Try your own")

code("""
# rows = [
#     {"intent": "billing", "cases": 90, "passes": 85, "traffic": 26000},
#     {"intent": "refund", "cases": 19, "passes": 16, "traffic": 7600},
#     {"intent": "voice agent", "cases": 0, "passes": 0, "traffic": 8000},
# ]
# a = audit(rows)
# print(f"raw {a['raw']:.3f}  reweighted {a['reweighted']:.3f}  uncovered {a['uncovered_share']:.1%}  gaps {a['gaps']}")
print("uncomment to explore")
""")

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you paste your own golden set against this period's traffic:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`rag-eval`](../rag-eval) and [`../../ai-agent-workshop/agent-eval-dashboard`](../../ai-agent-workshop/agent-eval-dashboard)
run a fixed suite; this one asks whether the suite still represents traffic.
[`../../data-science-cookbook/proportion-test`](../../data-science-cookbook/proportion-test) shows why the test
you pick on a 2x2 changes the answer.
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
    print(f"wrote demo.ipynb with {len(cells)} cells "
          f"({len(ENGINE.splitlines())} engine lines embedded)")


if __name__ == "__main__":
    write()
