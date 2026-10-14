"""Streamlit front end: describe your world and see which features stand in for the protected attribute.

Override any world parameter as `key=value` (how strong each proxy is, how biased the historical labels were) and
read the three proxy screens and seven models. Interactive runs use one seed on a smaller sample so they finish in
seconds; the full three-seed study is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import proxy as P
import streamlit as st

APP_N = 20_000
st.set_page_config(page_title="Proxy variable finder", layout="wide")
RES = json.load(open("results.json"))

st.title("We removed the attribute and it still discriminates")
tab_use, tab_study = st.tabs(["Your world", "The study"])

with tab_use:
    text = st.text_input("World overrides (key=value, comma separated)", "", key="overrides",
                         help="Keys: " + ", ".join(P.WORLD))
    try:
        w = P.parse_world(text)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    d = P.generate(APP_N, P.SEEDS[0], w)
    sc = P.screen(d)
    hidden = [p for p, v in sc["pairs"].items() if v > 0.6 and not any(sc["single"][f] > 0.6 for f in p.split("+"))]
    if hidden:
        st.error("Joint proxy a one-feature screen misses: " + ", ".join(f"{p} (AUC {sc['pairs'][p]:.2f})" for p in hidden))
    else:
        st.success("No pair predicts the group unless one of its features already does on its own.")
    rows = {k: P.evaluate(d, v) for k, v in P.MODELS.items()}
    unaware = rows["unaware (group removed)"]
    if unaware["ratio"] < 0.8:
        st.warning(f"Removing the group leaves a selection ratio of {unaware['ratio']:.2f} (B/A): the proxies carry it.")
    st.caption(f"One seed, n = {APP_N:,}. Single-feature AUC for predicting the group: "
               + ", ".join(f"{f} {v:.2f}" for f, v in sc["single"].items()) + f"; all five {sc['joint']:.2f}.")
    st.table([{"model": k, "B/A ratio": f"{r['ratio']:.3f}", "AUC vs history": f"{r['auc_hist']:.3f}",
               "AUC vs truth": f"{r['auc_true']:.3f}", "group from score": f"{r['group_from_score']:.3f}"}
              for k, r in rows.items()])

with tab_study:
    st.image("proxy_audit.png")
    st.code(open("evidence.txt").read())
