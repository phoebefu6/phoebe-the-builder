"""Streamlit front end: describe your human team and see which escalation policy actually pays.

Override any world parameter as `key=value` (capacity, review cost, human error rates, ticket volume) and read the
seven policies side by side, with a banner for the two traps: a rule that floods the queue past capacity, and a
policy that costs more than never escalating. The exact study from results.json is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import escalation as E
import streamlit as st

st.set_page_config(page_title="Human escalation policy", layout="wide")
RES = json.load(open("results.json"))

st.title("When should a human take over?")
tab_use, tab_study = st.tabs(["Your team", "The study"])

with tab_use:
    text = st.text_input("Parameter overrides (key=value, comma separated)", "", key="overrides",
                         help="Keys: " + ", ".join(E.BASE))
    try:
        prm = E.parse_params(text)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    rows = E.policy_table(prm)
    by = {r["policy"]: r for r in rows}
    never, joint = by["never escalate"], by["priced + capacity (joint)"]
    flooded = [r["policy"] for r in rows if r["escalated"] > prm["capacity"] * 1.05]
    worse = [r["policy"] for r in rows if r["cost"] > never["cost"] and r["policy"] != "never escalate"]
    st.success(f"Best policy: escalate {joint['escalated']:.0f} tickets/day ("
               + ", ".join(f"{s} below {joint['thr'][s]:.3f}" for s in E.SEGS if joint["esc_by_seg"][s] >= 1)
               + f"), \\${joint['cost']:,.0f}/day vs \\${never['cost']:,.0f} never escalating.")
    if worse:
        st.error("Costs more than never escalating: " + "; ".join(worse) + ".")
    if flooded:
        st.warning(f"Floods a {prm['capacity']:.0f}/day team (overflow goes back to the agent, humans err more): "
                   + "; ".join(flooded) + ".")
    st.table([{"policy": r["policy"], "escalated/day": round(r["escalated"]), "reviewed": f"{r['reviewed_share']:.0%}",
               "human error": f"{r['human_err']:.1%}", "error rate": f"{r['error_rate']:.2%}",
               "cost/day": f"${r['cost']:,.0f}"} for r in rows])

with tab_study:
    st.image("escalation_audit.png")
    st.code(open("evidence.txt").read())
    st.caption(f"Calibration: {sum(r['inside'] for r in RES['calibration'])}/{len(RES['calibration'])} exact error "
               f"rates inside the 99% Wilson interval of {E.MC_DAYS} simulated days.")
