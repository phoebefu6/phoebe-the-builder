"""Streamlit front end: paste your groups, see which pairs each post-hoc procedure names.

The stored study comes from results.json. What the app computes live is the user's own data - the
F test and each procedure's list of significant pairs - plus the family-wise error each procedure
carries at the user's k, read from the study.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import posthoc as P
import streamlit as st

st.set_page_config(page_title="ANOVA says something differs. Which pair?", layout="wide")
R = json.load(open("results.json"))
EX = R["exemplar"]
DEFAULT = "\n".join(", ".join(f"{v:.2f}" for v in g) for g in EX["groups"])

st.title("ANOVA says something differs. Which pair?")
st.markdown(
    "Five procedures hunt for the pair behind a significant F, and they are reported as though "
    "they were interchangeable. They are not: on the same data they name different pairs, and one "
    "of them (protected LSD) stops protecting you the moment one group is clearly different."
)

tab1, tab2 = st.tabs(["Your groups", "The study"])

with tab1:
    raw = st.text_area("One group per line, values separated by commas (opens on the study's exemplar)",
                       DEFAULT, height=180, key="groups")
    try:
        groups = [[float(v) for v in line.replace(";", ",").split(",") if v.strip()]
                  for line in raw.splitlines() if line.strip()]
        res = P.analyse(groups)
    except ValueError as err:
        st.warning(f"Cannot analyse this input: {err}")
        st.stop()

    k = len(groups)
    st.metric("ANOVA F p-value", f"{res['F_p']:.4f}",
              "significant" if res["F_p"] < P.ALPHA else "not significant", delta_color="off")
    near = min(R["null"], key=lambda r: abs(r["k"] - k))
    rows = []
    for proc in P.PROCS:
        names = ", ".join(f"{chr(65 + i)}-{chr(65 + j)}" for i, j in res["significant"][proc])
        rows.append({"procedure": P.LABEL[proc], "pairs named": names or "none",
                     f"FWER, all means equal (k={near['k']})": round(near[proc + "_fwer"], 4),
                     "calibration": near[proc + "_fwer_verdict"]})
    st.dataframe(rows, use_container_width=True)

    named = {p: bool(res["significant"][p]) for p in P.PROCS}
    if res["F_p"] < P.ALPHA and not named["tukey"]:
        st.error("The F test is significant and Tukey names NO pair. That is not a contradiction: "
                 "F tests every contrast at once, Tukey only the pairwise ones. Report the F result "
                 "and say no single pair can be singled out.")
    elif len({named[p] for p in ("bonferroni", "holm", "tukey")}) > 1:
        st.warning("The family-wise procedures disagree on whether any pair differs. Tukey is "
                   "exact for equal n and normal data; Holm beat it only at k = 3 in the study.")
    else:
        st.success("The family-wise procedures agree on whether any pair differs.")

with tab2:
    st.subheader("Family-wise false-alarm rate, all means equal")
    st.dataframe([{"k": r["k"], **{P.LABEL[p]: f"{r[p + '_fwer']:.4f} {r[p + '_fwer_verdict']}"
                                    for p in P.PROCS}} for r in R["null"]], use_container_width=True)
    st.subheader("Same, with one group 3 SD away (F always significant)")
    st.dataframe([{"k": r["k"], **{P.LABEL[p]: f"{r[p + '_fwer']:.4f} {r[p + '_fwer_verdict']}"
                                    for p in P.PROCS}} for r in R["partial"]], use_container_width=True)
    st.subheader("Per-pair power and Tukey vs Holm")
    st.dataframe([{"design": f"k={r['k']} {r['shape']}",
                   **{P.LABEL[p]: round(r[p + "_per_pair"], 4) for p in P.PROCS},
                   "Tukey-Holm": f"{r['tukey_minus_holm']:+.4f}", "winner": r["tukey_vs_holm"]}
                  for r in R["power"]], use_container_width=True)
