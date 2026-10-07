"""Generate demo.ipynb.

Plumbing copied from `context-packing` (Day 192), content written fresh. The engine is EXTRACTED from refusals.py by
AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank
one (the Day 177 defect).

Every rate is closed-form and every tail exact, so the notebook reproduces evidence.txt to the digit. Only the
Monte Carlo calibration runs at NB_REPS instead of the study's 200,000 - stated in the notebook (the Day 179
resolution note), and a test asserts the note exists.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "llmops-genai-platform/refusal-rate-monitor"
NB_REPS = 20_000


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


ENGINE = extract("refusals.py")

EXPORTS = [
    "SEED", "MC_REPS", "CLASSES", "CAMPAIGN_MIX", "PHRASES", "VERSIONS", "DETECTORS", "mix_of", "true_rates", "recall",
    "measured_rate", "flat_aggregate_twin", "log_pmf", "upper_tail", "critical_k", "audit_power", "audit_designs",
    "WEEKS", "WEEKLY_N", "AUDIT_N", "week_state", "monitor", "wilson", "simulate", "calibrate", "parse_version", "parse_mix",
]

HEADER = """from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

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
# It started refusing valid requests

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

The dashboard reports a refusal rate: the share of responses a detector flags as refusals. Behind that one number
sit four declared objects - a traffic mix, two error rates (refusing legitimate requests, complying with harmful
ones), the model's refusal phrasing, and a detector written against one version's phrasing. This notebook
declares all four, derives the dashboard's number and the two errors in closed form, checks them against a raw
simulation, and shows three of the four moving the dashboard while the model stays put - and the model moving
while the dashboard reads the wrong sign.

1. The model
2. Calibration - closed form vs raw simulation
3. Three versions, three detectors
4. The mix moves, the model does not
5. A flat aggregate is never neutral
6. Where to spend the labels
7. Twenty-six weeks on a chart
8. Try your own
""")

md("""
## The engine

Extracted from `refusals.py` at build time, character for character. `true_rates` gives the aggregate and the two
errors; `recall` and `measured_rate` put the detector in front of them; `flat_aggregate_twin` solves for the
version that moves one error at a flat rate; `audit_power` and `monitor` are exact binomial tails.
""")

code(HEADER + "\n" + ENGINE + "\n\nMIX = mix_of()\nKW = DETECTORS['keyword list (written on v1)']\nprint('engine loaded')")

md(f"""
**Resolution of the instrument.** Every rate and tail here is exact and matches `evidence.txt` to the digit. Only
the calibration's raw simulation runs at {NB_REPS:,} replicates per case instead of the study's 200,000.
""")

md("## 1. The model")

code("""
for c in CLASSES:
    print(f"{c['name']:<10} {c['share']:>4.0%} of traffic   {'legitimate' if c['legit'] else 'to be refused'}   ({c['what']})")
print()
for k, v in VERSIONS.items():
    print(f"{k:<22} refuses " + ", ".join(f"{c} {v['refuse'][c]:.3f}" for c in v["refuse"])
          + "   says: " + ", ".join(f"'{p}' {q:.0%}" for p, q in v["phrasing"].items() if q >= 0.1))
print()
for k, d in DETECTORS.items():
    print(f"{k:<30} recall on v1 {recall(d, VERSIONS['v1 baseline']):.2f}, on v2 {recall(d, VERSIONS['v2 upgraded model']):.2f}, fpr {d['fpr']}")
""")

md("## 2. Calibration")

code(f"""
cal = calibrate(reps={NB_REPS})
for r in cal:
    print(f"{{r['case']:<28}} {{r['quantity']:<28}} exact {{r['exact']:.4f}}   MC {{r['mc']:.4f}}   {{'inside' if r['inside_99'] else 'OUTSIDE'}}")
print(f"{{sum(r['inside_99'] for r in cal)}} of {{len(cal)}} inside their 99% Wilson interval")
""")

md("## 3. Three versions, three detectors")

code("""
kwn = "keyword list (written on v1)"
ver = {}
print(f"{'version':<22} {'true rate':>9} {'over-ref':>8} {'under-ref':>9} | " + " ".join(f"{k[:14]:>14}" for k in DETECTORS))
for k, v in VERSIONS.items():
    tr = true_rates(v, MIX)
    tr["measured"] = {d: measured_rate(v, MIX, dd) for d, dd in DETECTORS.items()}
    ver[k] = tr
    print(f"{k:<22} {tr['refusal_rate']:>9.2%} {tr['over_refusal']:>8.2%} {tr['under_refusal']:>9.1%} | "
          + " ".join(f"{tr['measured'][d]:>14.2%}" for d in DETECTORS))
v1, v2, v3 = (ver[k] for k in VERSIONS)
deltas = {
    "upgrade: true refusal rate": v2["refusal_rate"] - v1["refusal_rate"],
    "upgrade: measured (keyword)": v2["measured"][kwn] - v1["measured"][kwn],
    "upgrade: over-refusal": v2["over_refusal"] - v1["over_refusal"],
    "upgrade: under-refusal": v2["under_refusal"] - v1["under_refusal"],
    "tuned down: true refusal rate": v3["refusal_rate"] - v1["refusal_rate"],
    "tuned down: over-refusal": v3["over_refusal"] - v1["over_refusal"],
    "tuned down: under-refusal": v3["under_refusal"] - v1["under_refusal"],
}
print()
for k, d in deltas.items():
    print(f"{k:<32} {d * 100:+6.2f} pts")
""")

code("""
fig, ax = plt.subplots(figsize=(8, 3.6))
names = list(ver)
ys = np.arange(len(names))[::-1]
ax.barh(ys + 0.19, [ver[n]["refusal_rate"] * 100 for n in names], 0.36, color="#1f2733", label="true refusal rate")
ax.barh(ys - 0.19, [ver[n]["measured"][kwn] * 100 for n in names], 0.36, color="#8a94a3", label="keyword dashboard reads")
ax.set_yticks(ys); ax.set_yticklabels(names); ax.set_xlabel("refusal rate (%)"); ax.legend(loc="lower right")
ax.set_title("The upgrade raises refusals 39%; the dashboard reads a 63% fall", loc="left")
plt.tight_layout(); plt.savefig("notebook_chart.png", dpi=120); plt.show()
""")

md("""
The upgraded model refuses more of everything - twice as many legitimate requests, a third as many harmful ones
- and says "I'm not able to" where the old one said "I cannot". The keyword detector's recall falls from 0.85 to
0.20, and the dashboard reports the upgrade as a two-thirds cut in refusals. The prompt tuned to refuse less
halves over-refusal and triples under-refusal; on the dashboard it is a 2-point improvement.

## 4. The mix moves, the model does not
""")

code("""
cm = mix_of(override=CAMPAIGN_MIX)
a, b = true_rates(VERSIONS["v1 baseline"], MIX), true_rates(VERSIONS["v1 baseline"], cm)
print(f"harmful share {MIX['harmful']:.0%} -> {cm['harmful']:.0%}: true refusal rate {a['refusal_rate']:.2%} -> {b['refusal_rate']:.2%}")
print(f"over-refusal {a['over_refusal']:.2%} -> {b['over_refusal']:.2%}; under-refusal {a['under_refusal']:.0%} -> {b['under_refusal']:.0%}")
""")

md("## 5. A flat aggregate is never neutral")

code("""
tw = flat_aggregate_twin(VERSIONS["v1 baseline"], MIX, 2.0)
tt = true_rates(tw, MIX)
print(f"legitimate refusal rates x2; aggregate held at {tt['refusal_rate']:.2%} forces harmful refusals "
      f"{VERSIONS['v1 baseline']['refuse']['harmful']:.0%} -> {tw['refuse']['harmful']:.1%}")
print(f"over-refusal {a['over_refusal']:.2%} -> {tt['over_refusal']:.2%}, under-refusal {a['under_refusal']:.0%} -> {tt['under_refusal']:.0%}")
""")

md("""
At a fixed mix the aggregate is a weighted sum. If refusals of legitimate requests rise and the total does not,
refusals of harmful requests fell by exactly the same count. A flat refusal rate after a change is evidence that
both errors moved, or that nothing did.

## 6. Where to spend the labels
""")

code("""
ad = audit_designs(VERSIONS["v1 baseline"], VERSIONS["v2 upgraded model"], MIX, [50, 100, 300, 1000, 3000], KW)
print(f"{'labels':>7}   {'traffic sample: power':>22} {'size':>6}   {'refusal sample: power':>22} {'size':>6}")
for t, r in zip(ad["traffic"], ad["refusals"]):
    print(f"{t['n']:>7}   {t['power']:>22.3f} {t['size']:>6.3f}   {r['power']:>22.3f} {r['size']:>6.3f}")
print(f"refusal sample tests {ad['refusals'][0]['p0']:.1%} -> {ad['refusals'][0]['p1']:.1%}; "
      f"true legitimate share of refusals is {v1['legit_share_of_refusals']:.1%} -> {v2['legit_share_of_refusals']:.1%} - "
      "the detector's false positives inflate it, so label 'was this a refusal' too")
""")

md("""
Labelling random traffic estimates a 2.4% event on 95% of the labels; labelling detected refusals estimates a
38% event on all of them. One hundred labels a week on refusals has the power one thousand on traffic has.

## 7. Twenty-six weeks on a chart
""")

code("""
mon = monitor(KW)
print(f"{'week':>4} {'state':<18} {'true':>7} {'measured':>8} {'over-ref':>8}   {'P(alarm high)':>13} {'P(flag low)':>11}   {'P(audit alarm)':>14}")
for r in mon:
    if r["week"] in (1, 9, 12, 19, 26):
        state = "campaign mix" if r["campaign"] else ("v2 upgraded" if r["upgraded"] else "v1 baseline")
        print(f"{r['week']:>4} {state:<18} {r['true_rate']:>7.2%} {r['measured']:>8.2%} {r['over_refusal']:>8.2%}   "
              f"{r['p_alarm_high']:>13.3f} {r['p_alarm_low']:>11.3f}   {r['p_audit_alarm']:>14.3f}")
""")

md("""
Weeks 9-11 are a jailbreak campaign: the harmful share goes 5% to 12%, the model is untouched, and the
aggregate chart alarms with certainty while over-refusal moves 0.01 points. Week 19 is the upgrade: the true
rate rises 2.6 points, over-refusal doubles, and the aggregate chart flags a drop. The audit of 300 detected
refusals a week - what share came from legitimate requests - is silent on the campaign and certain on the
upgrade.

## Summary

| claim | what the numbers say |
|---|---|
| "refusals fell two thirds after the upgrade" | the phrasing changed; true refusals +39%, over-refusal doubled |
| "refusals doubled this week" | the harmful share did; both error rates unchanged to 0.01 pt |
| "the rate is flat, nothing changed" | at a fixed mix, 2x over-refusal at a flat rate is under-refusal 10% -> 56% |
| "the tuned prompt cut refusals 2 points" | and tripled compliance with harmful requests |
| "we label 300 responses a week" | 300 on traffic: power 0.72; 100 on refusals: 0.98 |
""")

md("## 8. Try your own")

code("""
# A, B = parse_version("0.006, 0.12, 0.90"), parse_version("0.004, 0.08, 0.95")
# mix = parse_mix("70, 20, 10")
# for name, v in (("A", A), ("B", B)):
#     tr = true_rates(v, mix)
#     print(name, f"true {tr['refusal_rate']:.2%}  over {tr['over_refusal']:.2%}  under {tr['under_refusal']:.1%}  "
#           f"keyword reads {measured_rate(v, mix, KW):.2%}")
print("uncomment to explore")
""")

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you describe two versions, a traffic mix and a detector:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`llm-guardrails`](../llm-guardrails) is the filter that refuses; this build is the monitor on
its rate. [`model-migration-diff`](../model-migration-diff) showed a net score hiding a gross swap on an
upgrade; this is the same shape on a safety metric, with the detector added. [`context-packing`](../context-packing)
is the previous day's audit of a metric that ranked the fixes backwards.
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
