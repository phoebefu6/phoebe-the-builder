from __future__ import annotations

# Streamlit UI for the Semantic Response Cache. Replay a paraphrase-heavy
# query stream, watch cache hits reuse past answers, and sweep the similarity
# threshold to see the hit-rate-vs-safety tradeoff. Fully offline.

import pandas as pd
import streamlit as st

from cache import SAMPLE_QUERIES, run_sample

st.set_page_config(page_title="Semantic Response Cache", page_icon="♻️", layout="wide")

st.title("♻️ Semantic Response Cache")
st.caption(
    "Users ask the same thing in different words. An exact-match cache misses "
    "paraphrases; a semantic cache reuses the stored answer when meaning is "
    "close enough - so you stop paying for repeated similar queries. Offline "
    "lexical embedder (swap in real embeddings for production)."
)

threshold = st.slider("Similarity threshold (higher = safer, fewer hits)", 0.3, 1.0, 0.7, 0.05)

cache, results = run_sample(threshold=threshold)
s = cache.summary()

col1, col2, col3 = st.columns(3)
col1.metric("Hit rate", f"{int(s['hit_rate'] * 100)}%", f"{s['hits']}/{s['lookups']} calls")
col2.metric("Cost saved", f"${s['cost_saved']}")
col3.metric("Unique answers cached", s["unique_cached"])

tab_stream, tab_sweep = st.tabs(["🔁 Query stream", "🎚️ Threshold sweep"])

with tab_stream:
    rows = []
    for q, r in zip(SAMPLE_QUERIES, results):
        rows.append(
            {
                "query": q,
                "result": "♻️ HIT" if r.hit else "🧠 MISS (generated)",
                "similarity": r.similarity,
                "matched_query": r.matched_query or "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info(
        "A HIT returns a cached answer for free. Lower the threshold for more "
        "hits - but too low and you risk returning a confidently-wrong answer "
        "to a subtly different question. Tune it on your own traffic."
    )

with tab_sweep:
    sweep = []
    for th in [round(x / 20, 2) for x in range(6, 21)]:
        c, _ = run_sample(threshold=th)
        sm = c.summary()
        sweep.append({"threshold": th, "hit_rate": sm["hit_rate"], "cost_saved": sm["cost_saved"]})
    df = pd.DataFrame(sweep)
    st.line_chart(df.set_index("threshold")["hit_rate"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("As the threshold rises, hit rate falls but each hit is a safer match.")

st.divider()
st.caption("Day 86 of Phoebe's FDE portfolio · LLMOps & GenAI Platform · `python cache.py` for the CLI.")
