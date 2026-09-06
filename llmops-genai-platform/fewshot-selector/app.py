from __future__ import annotations

# Streamlit UI for the Few-Shot Example Selector. Type a query and watch the
# selector pull the k nearest labeled examples from the pool - the ones you'd
# actually put in the prompt - and see how dynamic selection beats a static /
# random example block. Fully offline lexical embedder.
import pandas as pd
import streamlit as st
from selector import POOL, TEST_QUERIES, evaluate, select

st.set_page_config(page_title="Few-Shot Example Selector", page_icon="🎯",
                   layout="wide")

st.title("🎯 Few-Shot Example Selector")
st.caption(
    "Static few-shot blocks use the same examples for every input - and "
    "underperform. This picks the k nearest labeled examples to each query, "
    "so the model sees examples that actually resemble the task. Offline "
    "lexical embedder; swap in real embeddings for production."
)

k = st.slider("k (examples to select)", 1, 5, 3)

r = evaluate(k=k)
c1, c2, c3 = st.columns(3)
c1.metric("Relevant@k (nearest)", f"{r['near_relevant_mean']:.2f}",
          f"random {r['rand_relevant_mean']:.2f}")
c2.metric("k-NN accuracy (nearest)", f"{r['near_accuracy']:.2f}",
          f"random {r['rand_accuracy']:.2f}")
c3.metric("Pool size", len(POOL))

st.subheader("Nearest vs random on the test set")
df = pd.DataFrame(r["rows"])[
    ["query", "true_label", "nearest_pred", "near_relevant@k", "rand_relevant@k"]
]
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Try your own query")
preset = st.selectbox("Load a test query", ["(custom)"] + [q.text for q in TEST_QUERIES])
default = "" if preset == "(custom)" else preset
query = st.text_input("Query", value=default,
                      placeholder="e.g. I never received my package")

if query.strip():
    picks = select(query, POOL, k)
    st.markdown(f"**Selected {len(picks)} examples** (put these in the prompt):")
    st.dataframe(
        pd.DataFrame([{"similarity": s, "label": ex.label, "example": ex.text}
                      for ex, s in picks]),
        use_container_width=True, hide_index=True,
    )

st.divider()
st.caption("Day 89 of Phoebe's FDE build sprint · LLMOps & GenAI Platform · "
           "dynamic > static few-shot; k-NN label match is a proxy for prompt quality.")
