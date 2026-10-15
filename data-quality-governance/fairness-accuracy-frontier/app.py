"""Streamlit front end: describe your world and see what fairness costs, measured against both rulers.

Override any world parameter as `key=value` (the real skill gap, how biased the historical labels were) and read the
frontier and the price of four-fifths and of parity. Interactive runs use one seed on a smaller sample so they finish
in seconds; the full three-seed study is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import frontier as F
import streamlit as st

APP_N = 20_000
st.set_page_config(page_title="Fairness-accuracy frontier", layout="wide")

st.title("What does fairness cost?")
tab_use, tab_study = st.tabs(["Your world", "The study"])

with tab_use:
    text = st.text_input("World overrides (key=value, comma separated)", "", key="overrides",
                         help="Keys: " + ", ".join(F.WORLD))
    try:
        w = F.parse_world(text)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    r = F.one_seed(F.SEEDS[0], APP_N, w, with_audit=False)
    pr = r["prices"]
    ruler = pr["parity_hit_hist"] - pr["parity_hit_true"]
    if ruler > 1.0:
        st.warning(f"Parity costs {pr['parity_hit_hist']:.1f} pts against past decisions and "
                   f"{pr['parity_hit_true']:+.1f} against true merit: {ruler:.1f} pts of the price is the biased ruler.")
    else:
        st.success(f"The two rulers agree (difference {ruler:.1f} pts): parity costs {pr['parity_hit_true']:.1f} pts "
                   "of hit rate, and that is a real price.")
    st.caption(f"One seed, n = {APP_N:,}. Unconstrained ratio {r['start']:.2f}; hit vs truth peaks at ratio "
               f"{r['peak_true']['ratio']:.2f}.")
    st.line_chart({"hit vs history": [v * 100 for v in r["frontier"]["hit_hist"]],
                   "hit vs truth": [v * 100 for v in r["frontier"]["hit_true"]]})
    st.table([{"model": k, "B/A ratio": f"{p['ratio']:.3f}", "hit vs history": f"{p['hit_hist']:.1%}",
               "hit vs truth": f"{p['hit_true']:.1%}"} for k, p in r["points"].items()])

with tab_study:
    st.image("frontier.png")
    st.code(open("evidence.txt").read())
