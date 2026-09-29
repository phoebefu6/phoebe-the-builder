"""Streamlit front end: audit a golden set against today's traffic.

Paste one row per intent - cases in the set, how many pass, and this period's traffic count (unlabelled
counts are enough). The app shows the raw pass rate the dashboard reports, the pass rate reweighted to
today's mix, which traffic has no cases at all, and which intents are too thin to catch a regression.
The study from results.json is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import csv
import io
import json

import golden as G
import streamlit as st

st.set_page_config(page_title="Golden set audit", layout="wide")
R = json.load(open("results.json"))

n0 = G.allocate("proportional")
DEFAULT = "intent,cases,passes,traffic\n" + "\n".join(
    f"{name},{n0[i]},{int(round(n0[i] * G.Q[i]))},{int(round(G.W12[i] * 100000))}" for i, name in enumerate(G.INTENTS))

st.title("Is your golden set still a sample of your traffic?")
st.markdown(
    "A golden set is drawn once and then trusted for months. Its pass rate is a claim about production "
    "only while the set still looks like production. Paste your set against this period's traffic."
)

tab1, tab2 = st.tabs(["Your golden set", "The study"])

with tab1:
    raw = st.text_area("CSV: intent, cases, passes, traffic", DEFAULT, height=300, key="table")
    try:
        rows = list(csv.DictReader(io.StringIO(raw.strip())))
        if not rows or not {"intent", "cases", "passes", "traffic"} <= set(rows[0]):
            raise ValueError("header must be intent,cases,passes,traffic")
        res = G.audit(rows)
    except (ValueError, TypeError) as err:
        st.warning(f"Cannot audit this input: {err}")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("raw pass rate (what the dashboard shows)", f"{res['raw']:.1%}")
    m2.metric("reweighted to today's traffic", f"{res['reweighted']:.1%}", f"{100 * (res['reweighted'] - res['raw']):+.1f} pts")
    m3.metric("traffic with no cases", f"{res['uncovered_share']:.1%}")

    if res["gaps"]:
        st.error(f"COVERAGE GAP. {res['uncovered_share']:.0%} of traffic ({', '.join(res['gaps'])}) has no cases, so no "
                 "reweighting can see it. In the study a 13% gap left the headline 4.1 pts too high after the free fix. "
                 "Label 20 cases per missing intent and reweight.")
    elif abs(res["reweighted"] - res["raw"]) > 0.02:
        st.warning(f"STALE MIX. The set's mix no longer matches traffic: raw {res['raw']:.1%} vs reweighted "
                   f"{res['reweighted']:.1%}. Report the reweighted number; no new labels needed.")
    elif res["thin"]:
        st.info(f"COVERED, BUT THIN: {', '.join(res['thin'])} have fewer than 10 cases. The headline is fine; a "
                "regression confined to one of them is close to invisible (the study caught a halved 8-case intent "
                f"{R['regression'][0]['aggregate']:.0%} of the time).")
    else:
        st.success("REPRESENTATIVE. Every intent above 2% of traffic has at least 10 cases, and the set's mix matches "
                   "traffic within 2 pts.")

    st.dataframe(res["rows"], width="stretch")

with tab2:
    st.subheader("Error of the headline vs production, m = 240 (bias, pts)")
    st.dataframe([{"design": h["design"], "month 0 raw": round(100 * h["m0_raw"]["bias"], 2),
                   "month 12 raw": round(100 * h["m12_raw"]["bias"], 2),
                   "month 12 reweighted": round(100 * h["m12_rew"]["bias"], 2)} for h in R["headline"]], width="stretch")
    st.subheader("Catching a regression that halves one intent")
    st.dataframe(R["regression"], width="stretch")
    st.subheader("The refresh: new-intent labels only, then reweight")
    st.dataframe([{"labels added": r["labels_added"], "reweighted bias (pts)": round(100 * r["rew"]["bias"], 2),
                   "RMSE (pts)": round(100 * r["rew"]["rmse"], 2)} for r in R["refresh"]], width="stretch")
