from __future__ import annotations

# Streamlit UI for the Baseline Model ladder. Pick a task, pick a candidate,
# and see whether it actually beats a dumb rule. Runs offline on sklearn's
# bundled datasets.
import streamlit as st
from baseline import (
    MIN_LIFT,
    RANDOM_SEED,
    run_classification,
    run_regression,
    sample_classification,
    sample_regression,
    to_frame,
    verdict,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

st.set_page_config(page_title="Baseline Model", page_icon="📏", layout="wide")

st.title("📏 Baseline Model")
st.caption(
    '"We got 0.87 AUC" - compared to what? A dumb baseline is the only thing that '
    "turns a metric into a claim. This fits a ladder of deliberately stupid models, "
    "then reports whether your candidate actually earns its complexity."
)

with st.sidebar:
    st.header("Task")
    task = st.radio("Type", ["Classification", "Regression"], index=0)
    st.header("Candidate model")
    if task == "Classification":
        cand_name = st.selectbox(
            "Model", ["gradient boosting", "random forest", "k-nearest neighbours", "none"]
        )
        cands = {
            "gradient boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
            "random forest": RandomForestClassifier(random_state=RANDOM_SEED),
            "k-nearest neighbours": KNeighborsClassifier(),
            "none": None,
        }
        metric = st.selectbox("Decide on", ["roc_auc", "f1", "accuracy"], index=0)
    else:
        cand_name = st.selectbox(
            "Model", ["gradient boosting", "random forest", "k-nearest neighbours", "none"]
        )
        cands = {
            "gradient boosting": GradientBoostingRegressor(random_state=RANDOM_SEED),
            "random forest": RandomForestRegressor(random_state=RANDOM_SEED),
            "k-nearest neighbours": KNeighborsRegressor(),
            "none": None,
        }
        metric = st.selectbox("Decide on", ["r2", "mae"], index=0)
    st.header("Bar")
    min_lift = st.slider("Minimum lift to justify complexity", 0.0, 0.2, MIN_LIFT, 0.01)

candidate = cands[cand_name]
label_name = cand_name if candidate is not None else "candidate"

with st.spinner("Fitting the ladder..."):
    if task == "Classification":
        X, y, dataset = sample_classification()
        results = run_classification(X, y, candidate, label_name)
        order = ["roc_auc", "f1", "accuracy", "prevalence"]
    else:
        X, y, dataset = sample_regression()
        results = run_regression(X, y, candidate, label_name)
        order = ["r2", "mae"]

v = verdict(results, metric, min_lift)
st.caption(f"Dataset: **{dataset}** · 75/25 split · seed {RANDOM_SEED}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best baseline", v["best_baseline"], f"{v['best_baseline_score']} {metric}")
c2.metric("Best *trivial*", v["best_trivial"], f"{v['best_trivial_score']} {metric}")
if v.get("lift") is None:
    c3.metric("Candidate", "none selected")
    c4.metric("Verdict", "-")
else:
    c3.metric("Candidate", v["candidate"], f"{v['candidate_score']} {metric}")
    c4.metric("Lift over baseline", f"{v['lift']:+.4f}", "clears the bar" if v["ok"] else "below bar")

if v.get("lift") is None:
    st.info("Pick a candidate model to get a verdict.")
elif v["ok"]:
    st.success(f"**Worth it** - {v['reason']}")
else:
    st.warning(f"**Not worth it** - {v['reason']}")

tab_table, tab_chart, tab_why = st.tabs(["📋 The ladder", "📊 Chart", "🧠 Why baselines"])

df = to_frame(results, order)

with tab_table:
    st.dataframe(df, width="stretch", hide_index=True)
    if task == "Classification":
        prev = results[0].metrics.get("prevalence")
        st.info(
            f"Test-set prevalence is {prev:.1%}, which is exactly the accuracy the "
            "majority-class row scores. That is why accuracy alone is a useless metric "
            "on imbalanced data - and why the trivial rows belong in every model review."
        )

with tab_chart:
    chart_df = df.set_index("model")[[metric]]
    st.bar_chart(chart_df)
    st.caption(
        f"Ranked on {metric}. The gap that matters is candidate minus the tallest "
        "non-candidate bar - not candidate minus zero."
    )

with tab_why:
    st.markdown(
        """
**The ladder, rung by rung**

| Rung | What it is | What it exposes |
|---|---|---|
| majority class / mean | ignores every feature | the metric's floor - accuracy here *is* the prevalence |
| stratified guess / median | keeps the target's shape, learns nothing | whether your metric rewards guessing |
| best single rule | one feature, one brute-forced threshold | the model most likely to embarrass a pipeline |
| depth-2 tree | at most 3 splits, memorisable | how much of the signal is in a couple of cuts |
| logistic / linear | honest simple model | the thing most projects should actually ship |

**How to read a verdict.** Lift is measured against the *strongest* baseline, not the weakest.
Beating "predict the mean" is not an achievement. If a depth-2 tree gets within a hair of your
gradient booster, the tree is the better product: it trains in milliseconds, it can be explained
in a sentence, and it has nowhere to hide a bug.
        """
    )

st.divider()
st.caption(
    "Day 127 of Phoebe's daily FDE build - ML Engineering Toolkit line. "
    "Pairs with Day 122 (threshold explorer), Day 76 (train/eval harness)."
)
