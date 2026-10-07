"""Streamlit front end: two versions, a traffic mix, a detector - the true rate, the two errors, and what the
dashboard would read.

Describe version A and version B as `benign, sensitive, harmful` refusal rates, the traffic mix as three shares,
pick the detector, and say whether B uses the new model's phrasing. The banner says whether the measured change
has the right sign, whether a flat aggregate is hiding a swap, and whether the rate fell while under-refusal
rose. The exact study from results.json is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import refusals as R
import streamlit as st

st.set_page_config(page_title="Refusal rate monitor", layout="wide")
RES = json.load(open("results.json"))

st.title("It started refusing valid requests")
tab_use, tab_study = st.tabs(["Your versions and traffic", "The study"])

with tab_use:
    c1, c2, c3 = st.columns(3)
    a_text = c1.text_input("Version A refusal rates: benign, sensitive, harmful", "0.006, 0.12, 0.90", key="version_a")
    b_text = c2.text_input("Version B refusal rates: benign, sensitive, harmful", "0.012, 0.24, 0.97", key="version_b")
    mix_text = c3.text_input("Traffic mix: benign, sensitive, harmful", "80, 15, 5", key="mix")
    det_name = c1.selectbox("Refusal detector", list(R.DETECTORS), key="detector")
    new_phrasing = c2.checkbox("Version B uses the new model's phrasing", True, key="new_phrasing")
    try:
        A, B = R.parse_version(a_text), R.parse_version(b_text)
        if new_phrasing:
            B = {**B, "phrasing": R.VERSIONS["v2 upgraded model"]["phrasing"]}
        mix = R.parse_mix(mix_text)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    det = R.DETECTORS[det_name]
    rows = []
    out = {}
    for name, v in (("A", A), ("B", B)):
        tr = R.true_rates(v, mix)
        tr["measured"] = R.measured_rate(v, mix, det)
        tr["recall"] = R.recall(det, v)
        out[name] = tr
        rows.append({"version": name, "true refusal rate": f"{tr['refusal_rate']:.2%}", "measured (dashboard)": f"{tr['measured']:.2%}",
                     "detector recall": f"{tr['recall']:.2f}", "over-refusal P(refuse | legit)": f"{tr['over_refusal']:.2%}",
                     "under-refusal P(comply | harmful)": f"{tr['under_refusal']:.1%}",
                     "legit share of refusals": f"{tr['legit_share_of_refusals']:.1%}"})
    a, b = out["A"], out["B"]
    d_true, d_meas = b["refusal_rate"] - a["refusal_rate"], b["measured"] - a["measured"]
    over_up, under_up = b["over_refusal"] > a["over_refusal"] * 1.05, b["under_refusal"] > a["under_refusal"] * 1.05
    if abs(d_true) > 0.002 and (d_true > 0) != (d_meas > 0):
        st.error(f"MEASURED SIGN IS WRONG: the true refusal rate moves {d_true * 100:+.2f} pts, the dashboard reads {d_meas * 100:+.2f} pts. "
                 f"Detector recall {a['recall']:.2f} -> {b['recall']:.2f}: the phrasing changed, not the behaviour the dashboard describes.")
    elif abs(d_true) <= 0.002 and (over_up or under_up):
        st.warning(f"FLAT AGGREGATE HIDES A SWAP: refusal rate {a['refusal_rate']:.2%} -> {b['refusal_rate']:.2%}, over-refusal "
                   f"{a['over_refusal']:.2%} -> {b['over_refusal']:.2%}, under-refusal {a['under_refusal']:.1%} -> {b['under_refusal']:.1%}. "
                   "At a fixed mix a flat aggregate with movement is movement in both errors.")
    elif d_true < 0 and under_up:
        st.warning(f"RATE FELL, UNDER-REFUSAL ROSE: {d_true * 100:+.2f} pts on the dashboard; P(comply | harmful) "
                   f"{a['under_refusal']:.1%} -> {b['under_refusal']:.1%} on {mix['harmful']:.0%} of traffic.")
    elif d_true > 0 and over_up and not under_up:
        st.warning(f"RATE ROSE, OVER-REFUSAL ROSE: legitimate requests refused {a['over_refusal']:.2%} -> {b['over_refusal']:.2%}; "
                   f"under-refusal {a['under_refusal']:.1%} -> {b['under_refusal']:.1%}.")
    else:
        st.success("CONSISTENT: the measured change has the right sign and neither error got worse. Check the mix before "
                   "crediting the model - a change in the harmful share moves this number with the model untouched.")
    st.table(rows)
    st.caption("measured = sum share_c * (refuse_c * recall + (1 - refuse_c) * fpr); recall is the phrasing mass the detector matches, "
               "a property of the (detector, version) pair. Exact under the declared model; see evidence.txt.")

with tab_study:
    st.image("refusal_audit.png")
    st.write({k: f"{v * 100:+.2f} pts" for k, v in RES["deltas"].items()})
    st.caption("Closed form under the declared model, checked by raw simulation; see evidence.txt.")
