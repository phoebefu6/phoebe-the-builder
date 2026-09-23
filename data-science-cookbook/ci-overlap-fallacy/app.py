"""Streamlit front end: draw two error bars, read them the way everybody does, then run the test.

Stored numbers come from results.json - the app never recomputes a verdict the study made. What it
computes live is the single chart the user builds, which is exactly the object the fallacy is
about: one picture, one test, and whether they agree.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import overlap as o
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Do the error bars overlap?", layout="wide")

R = json.load(open("results.json"))
C = R["config"]
TEST, RULE, INK = "#2f6fdb", "#c8562b", "#1f2733"

st.title("\"The error bars overlap, so it is not significant\"")
st.markdown(
    f"For two independent means, 95% bars that do not touch imply `p < 0.05` - but `p < 0.05` "
    f"does **not** imply bars that do not touch. At equal standard errors the bars agree with "
    f"the test only when drawn at **{C['matching_level_equal_se'] * 100:.1f}%**. On paired data "
    f"the rule fails the other way, and the chart cannot show which case you are in."
)

tab1, tab2, tab3 = st.tabs(["Draw two bars", "The study", "Proportions"])

with tab1:
    left, right = st.columns([1, 2])
    with left:
        diff = st.slider("observed difference of means", 0.0, 2.0, 0.55, 0.01, key="diff")
        n1 = st.slider("n in group A", 5, 400, 30, key="n1")
        n2 = st.slider("n in group B", 5, 400, 30, key="n2")
        sd1 = st.slider("SD of group A", 0.2, 5.0, 1.0, 0.1, key="sd1")
        sd2 = st.slider("SD of group B", 0.2, 5.0, 1.0, 0.1, key="sd2")
        rho = st.slider("correlation (0 = independent groups)", 0.0, 0.99, 0.0, 0.01, key="rho")
        level = st.select_slider("bar level", options=[0.834, 0.90, 0.95, 0.99], value=0.95,
                                 key="level")

    if rho > 0 and n1 != n2:
        st.warning("Correlated measurements are paired, so both groups use n in group A.")
        n2 = n1
    s1, s2 = sd1 / math.sqrt(n1), sd2 / math.sqrt(n2)
    zl = o.z_for_level(level)
    se_d = o.paired_se(s1, s2, rho)
    p = float(2 * stats.norm.sf(diff / se_d)) if se_d > 0 else 0.0
    gap = diff > zl * (s1 + s2)
    sig = p < o.ALPHA
    ml = o.matching_level(s1, s2)

    with right:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        for y, c, s, col, lab in ((1.0, 0.0, s1, TEST, "A"), (0.0, diff, s2, RULE, "B")):
            ax.plot([c - zl * s, c + zl * s], [y, y], color=col, lw=4)
            ax.plot([c], [y], "o", color=col, ms=10)
            ax.text(c + zl * s + 0.02, y, f" group {lab}", va="center", color=col,
                    fontweight="bold")
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([])
        ax.set_xlabel("value")
        ax.set_title(f"{level * 100:.1f}% intervals, drawn per group", loc="left", color=INK)
        ax.grid(True, axis="x", color="#dfe4ea")
        st.pyplot(fig)
        plt.close(fig)

        a, b, c = st.columns(3)
        a.metric("the picture says", "different" if gap else "overlap")
        b.metric("the test says (p)", f"{p:.4f}")
        c.metric("level where they agree", f"{ml * 100:.1f}%" if rho == 0 else "no such level")

        if sig and not gap:
            st.error("The bars overlap and the difference IS significant. "
                     "A reader applying the overlap rule would discard a real result.")
        elif gap and not sig:
            st.error("The bars are apart and the test does NOT reject - possible only with "
                     "bars drawn below the matching level.")
        else:
            st.success("Picture and test agree on this chart.")
        if rho > 0:
            st.info(f"With correlation {rho:.2f} the bars demand the means be "
                    f"{o.paired_slack(s1, s2, rho, level):.2f}x further apart than the test "
                    f"does. Nothing in the picture shows the correlation.")

with tab2:
    st.subheader("Significant results whose 95% bars overlap")
    rows = [{"n1/n2": f"{r['n1']}/{r['n2']}", "sd1/sd2": f"{r['sd1']:g}/{r['sd2']:g}",
             "power": round(r["power_approx"], 3), "P(sig)": round(r["sig"], 4),
             "P(overlap | sig)": round(r["overlap_given_sig"], 4),
             "99% CI": f"[{r['ogs_lo']:.4f}, {r['ogs_hi']:.4f}]"} for r in R["means"]]
    st.dataframe(rows, use_container_width=True)
    st.subheader("Paired data: the reversal")
    st.dataframe([{"rho": r["rho"], "slack": round(r["slack"], 3), "P(sig)": round(r["sig"], 4),
                   "P(overlap | sig)": round(r["overlap_given_sig"], 4)} for r in R["paired"]],
                 use_container_width=True)

with tab3:
    st.subheader("Exact coverage of a nominal 95% interval for a proportion")
    only_bad = st.checkbox("only rows where Wald is below 0.90", key="only_bad")
    cov = [r for r in R["coverage"] if not only_bad or r["wald"] < 0.90]
    st.dataframe([{k: (round(v, 4) if isinstance(v, float) and k != "p" else v)
                   for k, v in r.items()} for r in cov], use_container_width=True)
    st.caption("Summed over every outcome with its binomial weight - no Monte Carlo error.")
