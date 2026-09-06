"""Streamlit front end: check one split, and see what the check cannot see.

Deliberately reports four things the usual SRM widget leaves out - the n at
which this same ratio would flip verdict, the smallest mismatch this test size
can catch, the bias that mismatch already carries, and the alpha actually
spent if the check has been run every day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import srm
import streamlit as st

st.set_page_config(page_title="SRM detector", page_icon="⚖️", layout="wide")

st.title("Sample ratio mismatch")
st.caption(
    "A split is a hypothesis. Passing its test is not evidence the arms are comparable - "
    "see the two negative results at the bottom."
)

with st.sidebar:
    st.header("The split you observed")
    n_ctrl = st.number_input("control users", min_value=1, value=100_000, step=1_000)
    n_trt = st.number_input("treatment users", min_value=1, value=98_500, step=1_000)
    share = st.slider("intended share of control", 0.05, 0.95, 0.50, 0.01)
    alpha_label = st.radio(
        "threshold",
        ["0.0005 - what platforms publish", "0.05 - the reflex"],
        help="Section 8 of evidence.py gives two independent reasons for the strict one.",
    )
    alpha = srm.ALPHA_PLATFORM if alpha_label.startswith("0.0005") else srm.ALPHA_REFLEX
    looks = st.number_input("how many times has this check been run?", 1, 200, 1)

n_total = int(n_ctrl + n_trt)
obs_share = n_ctrl / n_total
p = srm.p_chi2(int(n_ctrl), int(n_trt), share)
p_exact = srm.p_binom_exact(int(n_ctrl), int(n_trt), share) if n_total <= 2_000_000 else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("observed split", f"{obs_share:.3%} / {1 - obs_share:.3%}")
c2.metric("chi-square p", f"{p:.3e}")
c3.metric("total users", f"{n_total:,}")
c4.metric("verdict", "MISMATCH" if p < alpha else "consistent")

if p < alpha:
    st.error(
        f"**p = {p:.3e} < {alpha}.** The counts are not consistent with a {share:.0%} split. "
        "This is a trigger to find out **who** is missing - it is not a severity score. "
        "Two mechanisms with the identical count loss produce a 0% and a +26% bias in the "
        "reported effect, and the p-value cannot tell them apart."
    )
else:
    st.success(
        f"**p = {p:.3e}.** The counts are consistent with the intended split. That is all it "
        "means: equal counts are also what a *balanced* selective loss produces, and that one "
        "overstates the effect by a quarter while flagging at exactly the null rate."
    )

st.subheader("Every detector on these two numbers")
rows = []
for name, fn in list(srm.DETECTORS.items()) + list(srm.EXACT.items()):
    if name in srm.EXACT and n_total > 2_000_000:
        rows.append({"detector": name, "value": float("nan"), "fires": "(skipped, O(n))"})
        continue
    v = fn(int(n_ctrl), int(n_trt), share)
    is_eyeball = name.startswith("eyeball")
    rows.append({
        "detector": name + (" (rule of thumb)" if is_eyeball else ""),
        "value": v,
        "fires": "FLAG" if v < (0.5 if is_eyeball else alpha) else "pass",
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(
    "The five statistical tests disagree on 0.15% of trials (6 of 4,000) and only at the "
    "boundary. The two rules of thumb are the ones that differ: 'within 1%' names two rules "
    "whose tolerances are 4x apart - one is inert, the other false-alarms on ~2.4% of healthy "
    "experiments."
)

st.subheader("The same ratio, at other sample sizes")
dev = abs(obs_share - share)
if dev < 1e-9:
    st.info("The split is exactly as intended, so there is no ratio to re-scale.")
else:
    ns = np.unique(np.round(np.logspace(3, 7.5, 10)).astype(int))
    tab = []
    for n in ns:
        a = int(round(n * (share + dev)))
        pp = srm.p_chi2(a, int(n) - a, share)
        tab.append({"total users": f"{int(n):,}", "chi-square p": f"{pp:.2e}",
                    "verdict at this threshold": "MISMATCH" if pp < alpha else "consistent"})
    st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")
    st.caption(
        "Nothing about the split changed down this column. A percentage is not a finding - "
        "49.3/50.7 is p = 0.66 at n = 1,000 and p = 2e-44 at n = 1,000,000."
    )

st.subheader("What this test size cannot see")
per_arm = max(int(n_total * share), 1)
mdd = srm.mdd_share(n_total, alpha, share=share)
min_loss = srm.loss_for_share_deviation(mdd)
world = srm.World()
rate = min(min_loss / world.low_share, 1.0)
bias = (srm.analytic_est_lift(world, "selective_loss", rate) - world.true_rel_lift) / world.true_rel_lift
mde = srm.mde_rel_lift(per_arm, world.base_rate, srm.ALPHA_REFLEX)

d1, d2, d3 = st.columns(3)
d1.metric("smallest mismatch reliably caught", f"{min_loss:.2%} of one arm",
          help="80% power at the selected threshold.")
d2.metric("bias a selective loss that size carries", f"+{bias:.0%}",
          help="On the reference world: 30% low-intent users converting at 2%.")
d3.metric("this experiment's own min detectable lift", f"{mde:.2%}",
          help="80% power, alpha 0.05, 10% base rate.")
st.warning(
    f"Detection scales with n. **Bias does not.** At this size the smallest mismatch the check "
    f"can reliably catch already overstates a selective-loss effect by **{bias:.0%}** - so "
    "everything below that line is invisible, and invisible is not the same as harmless. "
    "The check only becomes genuinely protective around a million per arm."
)

if looks > 1:
    st.subheader("Alpha actually spent")
    rng = np.random.default_rng(11)
    realized = srm.sequential_srm_fpr(min(n_total, 400_000), int(looks), alpha, 1500, rng)
    st.metric(f"realized false-positive rate over {looks} looks", f"{realized:.4f}",
              delta=f"{realized / alpha:.1f}x nominal", delta_color="inverse")
    st.caption(
        "Optional stopping applies to the health check exactly as Day 164 priced it for the "
        "effect test. A daily SRM check for three weeks is a 0.257 test at 0.05. At 0.0005 it "
        "lands near 0.0035 - inflated, but two orders of magnitude fewer false alarms."
    )

with st.expander("Segment sweep - where the check should actually be pointed"):
    st.write(
        "Paste one row per segment as `name,control,treatment`. A 6% loss confined to a "
        "15%-of-traffic segment is seen 0.064 of the time by the aggregate check at 0.0005 and "
        "0.941 of the time by a Bonferroni-corrected per-segment sweep - and the corrected "
        "sweep false-alarms *less* than the same three tests uncorrected."
    )
    raw = st.text_area("segments", "chrome,62000,61900\nandroid,23000,22950\nsafari,15000,14100",
                       height=110)
    seg_rows = []
    for line in raw.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            nm, a, b = parts[0], int(parts[1]), int(parts[2])
        except ValueError:
            continue
        seg_rows.append({"segment": nm, "control": a, "treatment": b,
                         "p": srm.p_chi2(a, b, share)})
    if seg_rows:
        k = len(seg_rows)
        for r in seg_rows:
            r["fires (Bonferroni)"] = "FLAG" if r["p"] < alpha / k else "pass"
            r["p"] = f"{r['p']:.3e}"
        tot_a = sum(r["control"] for r in seg_rows)
        tot_b = sum(r["treatment"] for r in seg_rows)
        st.dataframe(pd.DataFrame(seg_rows), hide_index=True, width="stretch")
        agg_p = srm.p_chi2(tot_a, tot_b, share)
        st.write(
            f"Aggregate over all {k} segments: **p = {agg_p:.3e}** "
            f"({'MISMATCH' if agg_p < alpha else 'consistent'} at {alpha}), "
            f"corrected per-segment threshold **{alpha / k:.2e}**."
        )

st.divider()
st.caption(
    "Day 165 of the FDE portfolio. Full measured argument: `python evidence.py`; assertions in "
    "`test_srm.py`; figure from `make_chart.py`."
)
