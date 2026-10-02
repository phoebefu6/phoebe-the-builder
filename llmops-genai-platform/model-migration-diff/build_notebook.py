"""Generate demo.ipynb.

Plumbing copied from `prompt-regression-ci` (Day 187), content written fresh. The engine is EXTRACTED from
migrate.py by AST at build time, so the notebook holds the library's own source text. The write lives
behind __main__: test_notebook.py imports this module, and an import-time write would replace the
executed notebook with a blank one (the Day 177 defect).

The gate numbers are exact (2D convolution over cases), so the notebook reproduces them to the digit.
Only the Monte Carlo parts run at NB_REPS instead of the study's 20,000 - stated in the notebook (the
Day 179 resolution note), and a test asserts the note exists.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "llmops-genai-platform/model-migration-diff"
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


ENGINE = extract("migrate.py")

EXPORTS = [
    "ALPHA", "MARGIN", "INTENTS", "N", "FLAKY_SHARE", "FAIL_SHARE", "STABLE_P", "N_SWAP", "CONCENTRATE_IN", "SEED",
    "SCENARIOS", "GATES", "MC_REPS", "eval_set", "scenario", "joint_pmf", "mcnemar_p", "reject", "churn_pmf",
    "churn_threshold", "rerun_pmf", "sign_reject", "gate_probs", "evaluate", "sweep", "decide", "simulate", "wilson",
    "calibrate", "audit",
]

HEADER = """from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

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
# The score didn't move. What did?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A team swaps model A for model B, runs the same 400-case eval set once under each, and the score reads
88.3% both times. This notebook shows, exactly, that two very different migrations produce that same
number - and which comparison can tell them apart.

1. Calibration - exact probabilities vs raw simulated runs
2. Four migrations, five gates
3. The sweep - McNemar gets quieter as churn grows
4. Negative result - slicing by intent does not detect a spread regression
5. Try your own
""")

md("""
## The engine

Extracted from `migrate.py` at build time, character for character. Per case, A passes with probability
pa and B with pb. Each case adds (1, 0) - broke - or (0, 1) - fixed - or nothing, so the joint
distribution of (broke, fixed) is a 2D convolution over cases. Every gate is a function of that pair,
so its firing rate is an exact sum over the table.
""")

code(HEADER + "\n" + ENGINE + "\n\nprint('engine loaded')")

md(f"""
**Resolution of the instrument.** Every gate number here is exact and matches `evidence.txt` to the digit.
Only the calibration's raw simulation runs at {NB_REPS:,} replicates instead of the study's 20,000.
""")

md("## 1. Calibration")

code(f"""
cal = calibrate({NB_REPS})
print(f"2D convolution vs scipy binom: max gap {{cal['conv_vs_binom_gap']:.1e}}")
print(f"gate rates inside their 99% interval: {{sum(r['inside_99'] for r in cal['rates'])}} of {{len(cal['rates'])}}")
""")

md("## 2. Four migrations, five gates")

code("""
ev = evaluate()
print(f"noise churn, A vs A run again: {ev['noise_churn']:.2f} cases")
for r in ev["scenarios"]:
    g = r["gates"]
    print(f"{r['scenario']:<24} score {r['score_a']:.1%} -> {r['score_b']:.1%}  changed {r['cases_changed']:>2}  "
          + "  ".join(f"{k} {g[k]:.3f}" for k in GATES))
print("refunds under the concentrated migration:", [round(x, 3) for x in ev["scenarios"][1]["intent_scores"]["refunds"]])
""")

md("""
Two migrations post the same 88.3%. One breaks 20 of the 30 refunds cases (88% -> 22%) and fixes 20
cases elsewhere; the other spreads the same 40 changes thinly. The score-delta gate fires 4% of the time
on both - exactly its rate on no change at all. McNemar, the textbook paired test, is quieter on them
(0.0001) than on no change (0.029), because it compares broke with fixed and they balance.

Two gates see them. Per-intent McNemar finds the concentrated one and is blind to the spread one. The
churn test - run A a second time, and ask whether B disagrees with A more often than A disagrees with
itself - catches both, at 3.2% false alarms and 1.5x the calls.

TOST equivalence is not fooled: the paired variance *is* churn / N, so churn widens the interval and it
refuses to certify. But on 400 cases it certifies an unchanged model only 22% of the time, so "not
certified" carries little news.
""")

md("## 3. The sweep")

code("""
sw = sweep()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for g, col in (("mcnemar", "#2f6fdb"), ("tost", "#8a6d3b"), ("churn vs rerun", "#2e8b57"), ("per-intent", "#e3b23c")):
    axes[0].plot([r["churn_cases"] for r in sw], [r[g] for r in sw], "o-", color=col, ms=3, label=g)
axes[0].axhline(ALPHA, color="grey", ls=":"); axes[0].legend(frameon=False)
axes[0].set_xlabel("cases changed (net zero)"); axes[0].set_ylabel("P(gate fires)"); axes[0].set_title("Net-zero sweep", loc="left")
names = [n for n, _ in INTENTS]
conc = ev["scenarios"][1]["intent_scores"]
axes[1].bar(np.arange(len(names)) - 0.2, [conc[n][0] for n in names], 0.4, color="#9fb3c8", label="model A")
axes[1].bar(np.arange(len(names)) + 0.2, [conc[n][1] for n in names], 0.4, color="#c8562b", label="model B")
axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, rotation=30)
axes[1].set_ylabel("expected pass rate"); axes[1].legend(frameon=False); axes[1].set_title("Same total, one intent gutted", loc="left")
plt.tight_layout(); plt.savefig("notebook_chart.png", dpi=120); plt.show()
for r in sw[::2]:
    print(f"changed {r['churn_cases']:>2}   mcnemar {r['mcnemar']:.4f}   churn vs rerun {r['churn vs rerun']:.3f}")
""")

md("## 4. Negative result - slicing by intent")

code("""
pr = ev["scenarios"][3]["gates"]
print(f"plain regression, 20 breaks spread: whole-set McNemar {pr['mcnemar']:.3f}   per-intent Bonferroni {pr['per-intent']:.3f}")
""")

md("""
Each intent sees 2 or 3 breaks; an exact test at 0.05 / 8 needs about 8. The same regression the
whole-set test catches 95% of the time is caught 0.2% of the time by slicing. Slice to *diagnose* a
change you have already detected, not to detect one.

## Summary

| claim | what the numbers say |
|---|---|
| "the score didn't move, so nothing changed" | 40 of 400 cases changed; refunds went 88% -> 22% |
| "McNemar is the right paired test" | it is a test of net change; on a net-zero swap it is quieter than on no change |
| "slice the eval by intent" | finds a concentrated break, misses spread ones (0.002 vs 0.949) |
| "run the old model twice" | the exact sign test catches both swaps at 3.2% false alarms, 1.5x calls |
""")

md("## 5. Try your own")

code("""
# rows  = [("refunds", 1, 0), ("refunds", 1, 0), ("billing", 0, 1), ("billing", 1, 1)]   # intent, A, B
# rerun = [1, 1, 0, 1]                                                                  # A run again
# res = audit(rows, rerun)
# print(res["broke"], res["fixed"], res["mcnemar_p"], res["sign_p"])
print("uncomment to explore")
""")

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you paste your own per-case results:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`prompt-regression-ci`](../prompt-regression-ci) asks how often a prompt gate goes red on
noise; [`golden-set-builder`](../golden-set-builder) asks whether the eval set still represents traffic.
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
