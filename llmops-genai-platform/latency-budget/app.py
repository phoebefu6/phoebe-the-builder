"""Streamlit front end: describe your chain, see where the latency and the tail actually are.

Override any hop parameter as `key=value`, pick the SLO, and read three things the per-hop dashboards get wrong:
whether the chain is over its SLO, which hop owns the slow requests, and which fix moves the p99. The exact study
from results.json is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import latency as L
import streamlit as st

st.set_page_config(page_title="Latency budget", layout="wide")
RES = json.load(open("results.json"))

st.title("It is slow and nobody knows where")
tab_use, tab_study = st.tabs(["Your chain", "The study"])

with tab_use:
    text = st.text_input("Parameter overrides (key=value, comma separated)", "", key="overrides",
                         help="Keys: " + ", ".join(L.BASE))
    slo = st.number_input("SLO: p99 must be at or under (ms)", 100, 7000, L.SLO_MS, step=50, key="slo")
    try:
        prm = L.parse_params(text)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    s = L.study(prm)
    e = s["e2e"]
    over_sum = s["sum_of_hop_p99"] - slo
    over_true = e["p99"] - slo
    if over_sum > 0 and over_true <= 0:
        st.success(f"Within SLO: true p99 {e['p99']} ms. Adding per-hop p99s ({s['sum_of_hop_p99']} ms) would say "
                   f"{over_sum} ms over - percentiles do not add.")
    elif over_true > 0:
        st.error(f"Over SLO by {over_true} ms (true p99 {e['p99']} ms). The per-hop sum says {over_sum:+} ms.")
    else:
        st.success(f"Within SLO on both readings: true p99 {e['p99']} ms, per-hop sum {s['sum_of_hop_p99']} ms.")
    top_mean = max(L.HOPS, key=lambda h: s["mean_share"][h])
    top_tail = max(L.HOPS, key=lambda h: s["tail_share"][h])
    if top_mean != top_tail:
        st.warning(f"The average is mostly **{top_mean}** ({s['mean_share'][top_mean]:.0%}); the slow requests are "
                   f"mostly **{top_tail}** ({s['tail_share'][top_tail]:.0%} of tail excess).")
    st.table([{"hop": h, "p50": s["hops"][h]["p50"], "p95": s["hops"][h]["p95"], "p99": s["hops"][h]["p99"],
               "mean share": f"{s['mean_share'][h]:.1%}", "tail share": f"{s['tail_share'][h]:.1%}"} for h in L.HOPS])
    st.subheader("Fixes")
    st.table([{"fix": r["fix"], "mean saved (ms)": round(-r["d_mean"], 1), "p99 saved (ms)": -r["d_p99"],
               "p99 after": r["p99"]} for r in L.fix_table(prm)])

with tab_study:
    st.image("latency_audit.png")
    st.code(open("evidence.txt").read())
    st.caption(f"Calibration: {sum(r['inside'] for r in RES['calibration'])}/{len(RES['calibration'])} exact tail "
               f"probabilities inside the 99% Wilson interval of {L.MC_REPS:,} simulated requests.")
