"""Streamlit front end: score your own agent runs by outcome, by reference match, and by the invariant spec.

Paste one run per line: `eligible,answer_correct,steps` with steps as
`search_kb;lookup_order:A17;check_policy;issue_refund:A17` (eligible / answer_correct are 0 or 1). The app
shows what each scorer would report, how many broken runs it passed and valid runs it failed, and which
invariant each right-answer run broke. The exact study from results.json is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import numpy as np
import streamlit as st
import trajectory as T

st.set_page_config(page_title="Agent trajectory eval", layout="wide")
R = json.load(open("results.json"))


def _default() -> str:
    a = T.AGENTS["v2 fewer calls"]
    rng = np.random.default_rng(7)
    lines = []
    for _ in range(40):
        elig = bool(rng.random() < T.P_ELIG)
        plan = T.PLANS[rng.choice(4, p=a["plan"])]
        flags = [bool(rng.random() < q) for q in (a["kb"], a["rash"], T.P_ELIG, a["misread"], a["dup"], a["badarg"])]
        steps, correct = T.build(elig, plan, *flags)
        text = ";".join(s + (f":{x}" if x else "") for s, x in steps)
        lines.append(f"{int(elig)},{int(correct)},{text}")
    return "\n".join(lines)


def _parse(text: str) -> list:
    rows = []
    for ln in [x.strip() for x in text.strip().splitlines() if x.strip()]:
        parts = ln.split(",", 2)
        if len(parts) < 3 or parts[0].strip() not in ("0", "1") or parts[1].strip() not in ("0", "1"):
            raise ValueError(f"each line must be eligible,answer_correct,steps with 0 or 1 flags: {ln[:60]!r}")
        rows.append((parts[0].strip() == "1", parts[1].strip() == "1", parts[2]))
    if not rows:
        raise ValueError("paste at least one run")
    return rows


st.title("The answer was right. Was the path?")
tab_use, tab_study = st.tabs(["Score your runs", "The study"])

with tab_use:
    text = st.text_area("One run per line: eligible,answer_correct,steps", _default(), height=220, key="runs")
    try:
        res = T.audit(_parse(text))
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    out = res["scorers"]["outcome"]
    hidden = sum(res["hidden"].values())
    if out["passed_bad"]:
        top = res["hidden"].most_common(1)[0][0]
        st.error(f"OUTCOME SCORE OVERSTATES: it reports {out['reported']:.0%}, the spec passes {res['good']:.0%}. "
                 f"{out['passed_bad']} right-answer runs broke the path; most common: {top}.")
    elif res["scorers"]["strict"]["failed_good"]:
        st.warning(f"STRICT MATCH UNDERSTATES: {res['scorers']['strict']['failed_good']} valid runs failed only for "
                   "taking another valid path.")
    else:
        st.success(f"PATHS CLEAN: every right-answer run also kept the invariants ({res['n']} runs).")
    st.table([{"scorer": k, "reports": f"{v['reported']:.1%}", "passed broken": v["passed_bad"],
               "failed valid": v["failed_good"]} for k, v in res["scorers"].items()]
             + [{"scorer": "invariant spec", "reports": f"{res['good']:.1%}", "passed broken": 0, "failed valid": 0}])
    if hidden:
        st.caption("Right answer, broken path: " + ", ".join(f"{k} {v}" for k, v in res["hidden"].items()))

with tab_study:
    st.image("trajectory_audit.png")
    st.write({k: f"{v * 100:+.1f} pts" for k, v in R["deltas"].items()})
    st.caption("Exact enumeration over every trajectory the declared agent can produce; see evidence.txt.")
