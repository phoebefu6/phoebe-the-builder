"""Streamlit front end for the distribution fitter.

Deliberately does not lead with the ranking table. The first thing on screen is the
verdict, because the ranking is the part of this workflow people already trust too much.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from fitting import (
    DEFAULT_FAMILIES,
    FitReport,
    diagnose,
    family,
    fit_distributions,
    qq_points,
    sample_book,
)

st.set_page_config(page_title="Distribution Fitter", layout="wide")

PASS = "#2a7f62"
FAIL = "#b4451f"
MUTED = "#9a9aa8"


@st.cache_data(show_spinner=False)
def _sample() -> Dict[str, np.ndarray]:
    return sample_book()


@st.cache_data(show_spinner=False)
def _fit(
    values: bytes, n: int, names: List[str], n_boot: int, reps: int, alpha: float, seed: int
) -> FitReport:
    """Cached fit. `values` is the raw float64 buffer so the cache key is the data itself."""
    x = np.frombuffer(values, dtype=np.float64)
    fams = [family(name) for name in names]
    return fit_distributions(
        x, families=fams, n_boot=n_boot, stability_reps=reps, alpha=alpha, seed=seed
    )


st.title("Distribution Fitter")
st.caption(
    "Ranks candidate distributions by AIC, then answers the question the ranking cannot: "
    "does any of them actually fit?"
)

with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Sample book", "Upload CSV"], index=0)

    column_values: Optional[np.ndarray] = None
    column_name = ""
    if source == "Sample book":
        book = _sample()
        column_name = st.selectbox("Column", list(book))
        column_values = book[column_name]
        st.caption(
            {
                "session_seconds": "Genuinely lognormal. The tool should say yes.",
                "basket_value": "Gamma, rounded to cents. Watch the ties.",
                "latency_ms": "A two-component mixture. No candidate is right.",
                "daily_return": "Student-t. Normal is not rejected until n is large.",
            }[column_name]
        )
    else:
        upload = st.file_uploader("CSV", type=["csv"])
        if upload is not None:
            frame = pd.read_csv(upload)
            numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
            if not numeric:
                st.error("No numeric columns in that file.")
            else:
                column_name = st.selectbox("Column", numeric)
                column_values = frame[column_name].dropna().to_numpy(dtype=float)

    st.header("Candidates")
    default_names = [f.name for f in DEFAULT_FAMILIES]
    chosen = st.multiselect("Families", default_names, default=default_names)

    st.header("Resolution")
    n_boot = st.slider(
        "Bootstrap replicates (KS)", 0, 500, 200, step=50,
        help="The smallest reportable p-value is 1/(B+1). At B=0 the absolute test is skipped.",
    )
    reps = st.slider("Bootstrap resamples (stability)", 0, 400, 100, step=50)
    alpha = st.select_slider("alpha", options=[0.01, 0.05, 0.10], value=0.05)
    seed = st.number_input("Seed", value=0, step=1)

if column_values is None or len(chosen) == 0:
    st.info("Pick a column and at least one candidate family to start.")
    st.stop()

if column_values.size < 20:
    st.error(f"Only {column_values.size} finite values. Nothing here is meaningful below ~20.")
    st.stop()

diag = diagnose(column_values)
report = _fit(
    np.ascontiguousarray(column_values, dtype=np.float64).tobytes(),
    column_values.size,
    chosen,
    int(n_boot),
    int(reps),
    float(alpha),
    int(seed),
)

# ---- verdict first -------------------------------------------------------------------
verdict = report.verdict()
if report.adequate and report.best is not None and report.best.adequate(alpha) is True:
    st.success(verdict)
elif report.adequate:
    st.warning(verdict)
else:
    st.error(verdict)

cols = st.columns(4)
cols[0].metric("n", f"{diag.n:,}")
cols[1].metric("tie fraction", f"{diag.tie_fraction:.3f}")
cols[2].metric("skew", f"{diag.skew:+.2f}")
cols[3].metric(
    "candidates not rejected", f"{len(report.adequate)}/{len(report.ranked)}"
)

if diag.heavily_tied:
    st.warning(
        f"This column is rounded to {diag.decimals} decimal place(s) and {diag.tie_fraction:.0%} "
        "of the values are repeats. Ties inflate every KS distance on their own, so treat the "
        "absolute tests as advisory here and read the ranking instead."
    )

# ---- ranking -------------------------------------------------------------------------
st.subheader("Ranking")
rows = []
for r in report.ranked:
    adequate = r.adequate(alpha)
    rows.append(
        {
            "family": r.name,
            "k": r.family.n_free,
            "logLik": round(r.loglik, 2),
            "AIC": round(r.aic, 2),
            "AICc": round(r.aicc, 2),
            "dAIC": round(r.delta_aic, 2),
            "weight": round(r.aic_weight, 3),
            "KS D": round(r.ks.d_observed, 4) if r.ks else None,
            "p naive": round(r.ks.p_naive, 4) if r.ks else None,
            "p bootstrap": round(r.ks.p_bootstrap, 4) if r.ks else None,
            "win share": round(r.win_share, 3),
            "verdict": {True: "not rejected", False: "REJECTED", None: "not tested"}[adequate],
        }
    )
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

if report.excluded:
    with st.expander(f"{len(report.excluded)} candidate(s) excluded before fitting"):
        for r in report.excluded:
            st.write(f"**{r.name}** - {r.excluded_reason}")
        st.caption(
            "Excluded rather than fitted to the subset that survives: an AIC computed on "
            "different rows is not comparable to the others."
        )

# ---- plots ---------------------------------------------------------------------------
st.subheader("Diagnostics")
top = report.ranked[: min(3, len(report.ranked))]
fig, axes = plt.subplots(1, len(top) + 1, figsize=(4.4 * (len(top) + 1), 3.8), dpi=140)
axes = np.atleast_1d(axes)

ax = axes[0]
lo, hi = float(np.min(column_values)), float(np.max(column_values))
ax.hist(column_values, bins=min(70, max(10, diag.n_unique // 8)), density=True,
        color="#e6e6ee", edgecolor="#c9c9d6", linewidth=0.3)
grid = np.linspace(lo, hi, 500)
for r in top:
    ax.plot(grid, r.family.dist.pdf(grid, *r.params), linewidth=1.5, label=r.name)
ax.legend(fontsize=7, frameon=False)
ax.set_title(f"{column_name}: data and top fits", fontsize=9)

for ax, r in zip(axes[1:], top):
    theo, emp = qq_points(r.family, column_values, r.params)
    passed = r.adequate(alpha)
    colour = PASS if passed is True else (FAIL if passed is False else MUTED)
    e_lo, e_hi = float(np.min(emp)), float(np.max(emp))
    pad = 0.05 * (e_hi - e_lo)
    ax.plot([e_lo - pad, e_hi + pad], [e_lo - pad, e_hi + pad], color=MUTED,
            linestyle="--", linewidth=1.0)
    ax.scatter(theo, emp, s=5, color=colour, alpha=0.55, linewidths=0)
    ax.set_xlim(e_lo - pad, e_hi + pad)
    ax.set_ylim(e_lo - pad, e_hi + pad)
    tag = {True: "not rejected", False: "REJECTED", None: ""}[passed]
    ax.set_title(f"QQ: {r.name} {tag}", fontsize=9)
    ax.set_xlabel("theoretical", fontsize=8)
    ax.set_ylabel("observed", fontsize=8)

for ax in axes:
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---- the null distribution ------------------------------------------------------------
best = report.best
if best is not None and best.ks is not None and best.ks.null_distribution.size:
    with st.expander("Why the naive KS p-value is different from the bootstrap one"):
        fig2, ax2 = plt.subplots(figsize=(7.5, 3.0), dpi=140)
        ax2.hist(best.ks.null_distribution, bins=40, color="#dfe4f1",
                 edgecolor="#c2c9df", linewidth=0.3)
        ax2.axvline(best.ks.d_observed, color=FAIL, linewidth=1.8)
        ax2.set_xlabel("KS distance", fontsize=8)
        ax2.set_ylabel("replicates", fontsize=8)
        ax2.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax2.spines[side].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        st.markdown(
            f"""
The histogram is the KS distance you get by simulating **{best.ks.n_replicates}** fresh
samples from the fitted `{best.name}`, **refitting the parameters on each one**, and
measuring the distance again. The red line is the observed distance
(**{best.ks.d_observed:.4f}**).

The textbook KS test compares that red line against a reference distribution that assumes
the parameters were known before the data arrived. They were not - they were chosen to make
this distance small. That is why the naive p-value here is **{best.ks.p_naive:.3f}** and the
bootstrap p-value is **{best.ks.p_bootstrap:.3f}**.
"""
        )

# ---- free location --------------------------------------------------------------------
if report.location_probes:
    with st.expander("What a free location parameter would buy"):
        probe_rows = []
        for p in report.location_probes:
            probe_rows.append(
                {
                    "family": p.family_name,
                    "logLik (loc=0)": round(p.loglik_pinned, 2),
                    "logLik (loc free)": "-inf" if not np.isfinite(p.loglik_free)
                    else round(p.loglik_free, 2),
                    "loc hat": round(p.loc_free, 4),
                    "min(x)": round(p.data_min, 4),
                    "AIC prefers": "INVALID FIT" if p.free_fit_invalid
                    else ("free loc" if p.free_wins_aic else "loc=0"),
                }
            )
        st.dataframe(pd.DataFrame(probe_rows), width="stretch", hide_index=True)
        st.caption(
            "Positive-support families are fit with loc pinned at 0. Freeing it adds a "
            "parameter whose benefit grows with how wrong the family is, and whose MLE is "
            "unbounded as loc approaches min(x) - so scipy sometimes returns a loc above "
            "min(x), which assigns zero density to observed points."
        )

st.divider()
st.caption(
    "Day 136 of the FDE portfolio - Data Science Cookbook. "
    "`python3 test_fitting.py` and `python3 test_evidence.py` reproduce every claim above."
)
