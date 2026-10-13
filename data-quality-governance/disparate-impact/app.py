"""Streamlit front end: describe your two groups and see which fairness definitions your threshold can meet.

Override any group parameter as `group.key=value` (share, base rate, score means, sd) and read the five policies
side by side, with banners for the four-fifths verdict and for the impossibility: when base rates differ, no
threshold pair meets equal precision, equal true positive rate and equal false positive rate at once.
The exact study from results.json is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import fairness as F
import streamlit as st

st.set_page_config(page_title="Disparate impact", layout="wide")
RES = json.load(open("results.json"))

st.title("The model is fine overall")
tab_use, tab_study = st.tabs(["Your groups", "The study"])

with tab_use:
    text = st.text_input("Overrides (group.key=value, comma separated)", "", key="overrides",
                         help="Groups A, B. Keys: " + ", ".join(F.WORLD["A"]))
    try:
        wd = F.parse_world(text)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    pols = F.policies(wd)
    one = pols["one threshold (max accuracy)"]
    ir = one["overall"]["impact_ratio"]
    msg = (f"One threshold: accuracy {one['overall']['acc']:.1%}, qualified selected A {one['per']['A']['tpr']:.0%} "
           f"vs B {one['per']['B']['tpr']:.0%}, four-fifths ratio {ir:.3f}.")
    if ir < F.FOUR_FIFTHS:
        st.error(msg + " Fails the four-fifths rule.")
    else:
        st.success(msg + " Passes the four-fifths rule.")
    if abs(wd["A"]["base"] - wd["B"]["base"]) > 1e-9:
        st.info("Base rates differ, so equal precision, equal true positive rate and equal false positive rate "
                "cannot all hold at once. Pick the definition on purpose.")
    st.table([{"policy": n, "accuracy": f"{p['overall']['acc']:.1%}", "4/5 ratio": f"{p['overall']['impact_ratio']:.3f}",
               "TPR gap": f"{p['gaps']['tpr'] * 100:+.1f} pts", "FPR gap": f"{p['gaps']['fpr'] * 100:+.1f} pts",
               "PPV gap": f"{p['gaps']['ppv'] * 100:+.1f} pts"} for n, p in pols.items()])

with tab_study:
    st.image("fairness_audit.png")
    st.code(open("evidence.txt").read())
    st.caption(f"Calibration: {sum(r['inside'] for r in RES['calibration'])}/{len(RES['calibration'])} exact rates "
               f"inside the 99% Wilson interval of {F.MC_REPS:,} simulated applicants per policy.")
