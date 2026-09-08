"""Pick a world, run the scan an analyst would run, and see what it reports about it.

Every panel is a live refit - nothing is read from results.json - so this is a way of poking
at the two estimators rather than a slideshow of the evidence run.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import hte  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

INK, BAD, COOL, GOOD, GRID = "#16222e", "#b3402f", "#2b6ca3", "#1f7a5c", "#dfe5ea"

WORLDS = [
    "Homogeneous - the launch helped everyone equally",
    "One genuinely harmed segment",
    "Heterogeneity along a covariate the segments do not see",
    "Segment measured AFTER assignment ('engaged users')",
]


def main() -> None:
    st.set_page_config(page_title="It worked overall and hurt one segment", layout="wide")
    st.title("It worked overall and hurt one segment")
    st.caption(
        "A segment scan asks K questions of 1/K of the data each. This app runs one, on a world "
        "you choose, and shows what the scan says next to what is actually true."
    )

    with st.sidebar:
        world = st.selectbox("What is true in this world", WORLDS)
        n = st.select_slider("users in the test", [2000, 8000, 20000, 60000, 200000], value=8000)
        k = st.slider("segments in the scan", 2, 50, 20)
        ate = st.slider("true effect for everyone", -0.10, 0.30, 0.05, 0.01)
        delta = st.slider("harm in the affected segment", 0.0, 1.0, 0.30, 0.05)
        beta = st.slider("effect slope on x0 (covariate world)", 0.0, 0.6, 0.20, 0.05)
        seed = st.number_input("seed", 0, 9999, 7, step=1)

    rng = np.random.default_rng(int(seed))
    truth_note = ""
    if world == WORLDS[0]:
        d = hte.trial_partition(rng, n=n, k=k, ate=ate)
        groups, seg_truth = hte.partition_matrix(d["seg"], k), np.full(k, ate)
        truth_note = f"Every segment's true effect is {ate:+.2f}. Every flag below is a false alarm."
    elif world == WORLDS[1]:
        d = hte.trial_partition(rng, n=n, k=k, ate=ate, delta=delta, harmed=0)
        groups, seg_truth = hte.partition_matrix(d["seg"], k), np.full(k, ate)
        seg_truth[0] = ate - delta
        truth_note = f"Segment 0's true effect is {ate - delta:+.2f}; every other segment is {ate:+.2f}."
    elif world == WORLDS[2]:
        d = hte.trial_continuous(rng, n=n, ate=ate, beta=beta, k=k)
        groups = hte.partition_matrix(d["seg"], k)
        seg_truth = np.array([d["tau"][d["seg"] == j].mean() for j in range(k)])
        truth_note = (
            f"tau(x) = {ate:+.2f} {beta:+.2f}*x0, so the effect changes sign across users - but the "
            "segments are independent of x0, so every segment's TRUE average is about the same."
        )
    else:
        d = hte.trial_post_gate(rng, n=n, ate=ate, gate_effect=0.30)
        groups = np.stack([d["engaged"], ~d["engaged"]], axis=1)
        seg_truth = np.full(2, ate)
        truth_note = f"The true effect is {ate:+.2f} for everybody. There is no heterogeneity at all."

    res = hte.scan(d["y"], d["w"], groups)
    overall, overall_se = hte.diff_means(d["y"], d["w"])
    st.info(truth_note)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("overall effect", f"{overall:+.4f}", f"+/- {1.96 * overall_se:.4f}")
    c2.metric("segments flagged as harmed (raw)", int(np.sum((res["est"] < 0) & (res["p"] < 0.05))))
    c3.metric("after Bonferroni", int(np.sum((res["est"] < 0) & (hte.bonferroni(res["p"]) < 0.05))))
    gap = hte.consistency_gap(d["y"], d["w"], groups)
    c4.metric("parts-vs-whole gap (overall SEs)", f"{gap:+.3f}", "algebra, not a test")
    if abs(gap) > 1.0:
        c4.error("The segments do not add up to the whole - this split is not pre-assignment.")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("The scan")
        fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=140)
        order = np.argsort(res["est"])
        xs = np.arange(order.size)
        flagged = (res["est"] < 0) & (res["p"] < 0.05)
        ax.errorbar(
            xs, res["est"][order], yerr=1.96 * res["se"][order], fmt="o", ms=4, lw=1,
            color=INK, ecolor=GRID, capsize=2, zorder=3,
        )
        f = flagged[order]
        if f.any():
            ax.errorbar(
                xs[f], res["est"][order][f], yerr=1.96 * res["se"][order][f], fmt="o", ms=6, lw=1.4,
                color=BAD, ecolor=BAD, capsize=2, zorder=4, label="called harmed (raw p<0.05)",
            )
        ax.plot(xs, seg_truth[order], "_", ms=12, color=GOOD, mew=2, label="true segment effect", zorder=5)
        ax.axhline(0.0, color=INK, lw=0.8)
        ax.axhline(overall, color=COOL, lw=1.2, ls="--", label="overall effect")
        ax.set_xlabel("segments, sorted by reported effect")
        ax.set_ylabel("effect")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        st.pyplot(fig, clear_figure=True)

        worst = int(np.argmin(res["est"]))
        pred = ate + hte.segment_se(n, k) * hte.expected_extreme_z(k, "min")
        st.markdown(
            f"**Worst segment:** #{worst}, reported **{res['est'][worst]:+.4f}** "
            f"(p={res['p'][worst]:.3f}), true value **{seg_truth[worst]:+.4f}**.  \n"
            f"Under a homogeneous world the minimum of {k} segments is expected at "
            f"**{pred:+.4f}** before any data is collected - that is the selection, not the segment."
        )

    with right:
        st.subheader("If you fit the subgroup instead")
        if world == WORLDS[2] or st.checkbox("fit a CATE tree on this world", value=world == WORLDS[2]):
            x = d.get("x")
            if x is None:
                x = rng.normal(size=(n, 5))
            h = hte.honest_fit(rng, x, d["w"], d["y"], depth=2, min_leaf=max(50, n // 40))
            ins, hon = h["in_sample"], h["honest"]
            j = int(np.argmin(ins["est"]))
            st.metric("worst leaf, read off the fitting data", f"{ins['est'][j]:+.4f}")
            st.metric("same leaf, read off held-out data", f"{hon['est'][j]:+.4f}")
            st.caption(
                "The split was chosen to make this difference large. Scoring it on the rows that "
                "chose it is reading the objective function back out - so the honest number is the "
                "only quotable one."
            )
            fig2, ax2 = plt.subplots(figsize=(4.4, 3.0), dpi=140)
            idx = np.arange(hte.n_leaves(h["tree"]))
            ax2.bar(idx - 0.2, ins["est"], 0.4, color=BAD, label="in-sample")
            ax2.bar(idx + 0.2, hon["est"], 0.4, color=GOOD, label="honest")
            ax2.axhline(0, color=INK, lw=0.8)
            ax2.set_xlabel("leaf")
            ax2.set_ylabel("effect")
            ax2.legend(fontsize=8, frameon=False)
            for s in ("top", "right"):
                ax2.spines[s].set_visible(False)
            st.pyplot(fig2, clear_figure=True)

        st.subheader("The arithmetic, before any data")
        st.markdown(
            f"- a segment holds `n/K` = **{n // k:,}** users, so its SE is "
            f"**{hte.segment_se(n, k):.4f}** = sqrt(K) x the experiment's **{hte.segment_se(n, 1):.4f}**\n"
            f"- P(the scan calls something harmed) with a true effect of {ate:+.2f} everywhere: "
            f"**{hte.harm_alarm_closed_form(k, n, ate):.3f}**\n"
            f"- the same number at a true effect of 0: "
            f"**{hte.harm_alarm_closed_form(k, n, 0.0):.3f}**\n"
            f"- MDE of the experiment **{hte.mde(n / 2):.4f}** vs of one segment "
            f"**{hte.mde(n / (2 * k)):.4f}**"
        )


if __name__ == "__main__":
    main()
