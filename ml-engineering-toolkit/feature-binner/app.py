"""Streamlit front end for the feature binner."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from binning import (
    audit,
    build_dataset,
    fit,
    iv_band,
    noise_screen,
)

st.set_page_config(page_title="Feature Binner", layout="wide")
st.title("Feature Binner")
st.caption(
    "Monotone optimal binning with WOE/IV - and a permutation null, because raw IV is not "
    "comparable across bin counts."
)


@st.cache_data
def load(n: int, seed: int) -> dict:
    return build_dataset(n=n, seed=seed)


with st.sidebar:
    st.header("Data")
    n_rows = st.select_slider("Rows", [800, 2000, 6000, 12000, 24000], value=12000)
    data = load(n_rows, 11)
    y = data["y"]
    tr, ho = data["train_idx"], data["holdout_idx"]
    st.caption(f"base default rate **{data['base_rate']:.2%}** · train {len(tr)} · holdout {len(ho)}")

    st.header("Binner settings")
    feature = st.selectbox("Feature", list(data["features"]), index=0)
    max_bins = st.slider("max_bins", 2, 20, 6)
    min_bin_share = st.slider("min bin share", 0.002, 0.20, 0.05, step=0.002)
    min_bin_events = st.slider("min events per bin", 0, 60, 20)
    smoothing = st.select_slider("WOE smoothing", [0.01, 0.1, 0.5, 1.0, 2.0], value=0.5)
    monotone = st.checkbox("Enforce monotone WOE", value=True)
    n_perm = st.slider("Permutations for the null", 0, 120, 40, step=20)

kwargs = dict(
    max_bins=max_bins,
    min_bin_share=min_bin_share,
    min_bin_events=min_bin_events,
    smoothing=smoothing,
    monotone=monotone,
    specials=data["specials"].get(feature, ()),
)

x = data["features"][feature]
result = audit(x[tr], y[tr], x[ho], y[ho], feature=feature, n_permutations=n_perm, **kwargs)
scheme = result.scheme

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("IV (train)", f"{result.iv_train:.4f}", iv_band(result.iv_train))
c2.metric(
    "IV above null",
    f"{result.excess_iv:.4f}" if n_perm else "n/a",
    help="IV minus the median IV this same procedure gets from shuffled labels",
)
c3.metric("p-value", f"{result.p_value:.3f}" if n_perm else "n/a")
c4.metric(
    "IV (holdout)",
    f"{result.iv_holdout:.4f}",
    f"{-result.shrinkage:+.0%} vs train",
    help="Negative means the in-sample IV did not survive; positive means it held up or grew",
)
c5.metric("PSI", f"{result.psi:.3f}")

verdict = result.verdict
if verdict.startswith("DROP"):
    st.error(f"**{verdict}**")
elif verdict.startswith(("SUSPECT", "REVIEW")):
    st.warning(f"**{verdict}**")
else:
    st.success(f"**{verdict}**")

if result.sparse_warning:
    st.warning(f"Sparse bins: {result.sparse_warning}")
for note in scheme.notes:
    st.caption(f"note: {note}")

tabs = st.tabs(
    ["Bin table", "WOE curve", "Train vs holdout", "Monotonicity cost", "All features", "Noise screen"]
)

with tabs[0]:
    table = pd.DataFrame(scheme.table())
    st.dataframe(
        table.style.format(
            {
                "share": "{:.1%}",
                "event_rate": "{:.2%}",
                "woe": "{:+.3f}",
                "iv_part": "{:.4f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Cut points that transfer to new data: `{[round(c, 4) for c in scheme.cuts]}`  ·  "
        f"specials kept out of the numeric scale: `{scheme.specials}`"
    )

with tabs[1]:
    plot = pd.DataFrame(
        {
            "bin": [b.label for b in scheme.bins],
            "WOE": [scheme.woe(b) for b in scheme.bins],
            "kind": [b.kind for b in scheme.bins],
        }
    ).set_index("bin")
    st.bar_chart(plot["WOE"], height=340)
    st.caption(
        "Missing and special bins are excluded from the monotonicity check by design - they "
        "are not on the numeric scale, so requiring them to sit in order is meaningless."
    )
    rates = pd.DataFrame(
        {"bin": [b.label for b in scheme.bins], "event rate": [b.event_rate for b in scheme.bins]}
    ).set_index("bin")
    st.bar_chart(rates, height=280)

with tabs[2]:
    left, right = st.columns(2)
    left.markdown("**Train** (cut points chosen here)")
    left.dataframe(
        pd.DataFrame(scheme.table())[["bin", "n", "event_rate", "woe"]].style.format(
            {"event_rate": "{:.2%}", "woe": "{:+.3f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    right.markdown("**Holdout** (same cut points, recounted)")
    right.dataframe(
        pd.DataFrame(result.holdout.table())[["bin", "n", "event_rate", "woe"]].style.format(
            {"event_rate": "{:.2%}", "woe": "{:+.3f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.info(
        f"Monotone on train: **{result.monotone_train or 'no'}** · "
        f"on holdout: **{result.monotone_holdout or 'no - WOE wobbles'}**. "
        "A shape that only holds on the rows that chose the cuts is not a shape."
    )

with tabs[3]:
    free = fit(x[tr], y[tr], feature=feature, **{**kwargs, "monotone": False})
    forced = fit(x[tr], y[tr], feature=feature, **{**kwargs, "monotone": True})
    cost = free.iv - forced.iv
    a, b = st.columns(2)
    a.metric("Unconstrained IV", f"{free.iv:.4f}", f"{len(free.bins)} bins")
    b.metric(
        "Monotone IV",
        f"{forced.iv:.4f}",
        f"-{cost / free.iv:.0%}" if free.iv else "n/a",
        delta_color="inverse",
    )
    if free.iv and cost / free.iv > 0.15:
        st.warning(
            f"Monotonicity costs **{cost / free.iv:.0%}** of the IV here. That is not a wiggle "
            "being cleaned up - the relationship is genuinely non-monotone, and you are "
            "deleting signal to buy a shape the review committee likes. Consider splitting "
            "the feature instead."
        )
    else:
        st.success(
            "The constraint costs little, which means the non-monotone wiggle was noise. "
            "Take the shape."
        )
    st.caption(f"unconstrained monotonicity: {free.is_monotone() or 'none - WOE wobbles'}")

with tabs[4]:
    audits = [
        audit(
            data["features"][name][tr],
            y[tr],
            data["features"][name][ho],
            y[ho],
            feature=name,
            n_permutations=n_perm,
            **{**kwargs, "specials": data["specials"].get(name, ())},
        )
        for name in data["features"]
    ]
    frame = pd.DataFrame(
        [
            {
                "feature": a.feature,
                "IV train": a.iv_train,
                "IV null": a.null_median,
                "excess": a.excess_iv,
                "p": a.p_value,
                "IV holdout": a.iv_holdout,
                "PSI": a.psi,
                "bins": a.n_bins,
                "verdict": a.verdict,
            }
            for a in sorted(audits, key=lambda a: -a.excess_iv)
        ]
    )
    st.dataframe(
        frame.style.format(
            {
                "IV train": "{:.4f}",
                "IV null": "{:.4f}",
                "excess": "{:.4f}",
                "p": "{:.3f}",
                "IV holdout": "{:.4f}",
                "PSI": "{:.3f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "`noise` is an rng.normal() column with zero relationship to the target. If it is not "
        "at the bottom with a high p-value, the settings above are too loose."
    )

with tabs[5]:
    st.markdown(
        "**12 independent pure-noise columns**, screened by raw IV against the conventional "
        "0.10 bar and by the permutation p-value. Every column has zero signal by construction."
    )
    if st.button("Run the noise screen (takes a few seconds)"):
        rows = noise_screen(y[tr][: min(len(tr), 480)], n_columns=12, n_permutations=40)
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {
                    "mean_bins": "{:.1f}",
                    "mean_iv": "{:.4f}",
                    "mean_excess": "{:.4f}",
                    "kept_by_iv": "{:.0%}",
                    "kept_by_permutation": "{:.0%}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.error(
            "At the loosest settings every pure-noise column clears the 0.10 'medium predictor' "
            "bar. The permutation screen stays near its nominal 5%, because the null is measured "
            "through the same procedure. (At 12 columns those rates are noisy to about +/-10pp.)"
        )

st.divider()
st.caption(
    "Dataset is generated deterministically - a small application scorecard with an 18% missing "
    "employment field, a -999 'no bureau record' sentinel, a genuinely U-shaped age effect, and "
    "one pure-noise column. No real credit data."
)
