from __future__ import annotations

# Streamlit UI for the Chunking Strategy Tester. Pick k, run every strategy
# over the sample corpus + gold eval set, and see which chunking config
# retrieves the answer best - the retrieval-quality-vs-chunk-size tradeoff,
# made visible. Fully offline.

import pandas as pd
import streamlit as st

from chunker import (
    SAMPLE_CASES,
    SAMPLE_DOCS,
    build_strategies,
    compare_strategies,
    evaluate_strategy,
)

st.set_page_config(page_title="Chunking Strategy Tester", page_icon="🧩", layout="wide")

st.title("🧩 Chunking Strategy Tester")
st.caption(
    "Bad chunks = bad answers. The retriever can only surface a chunk that "
    "actually contains the answer. Compare chunking strategies on retrieval "
    "quality over a gold eval set - fully offline, no API keys."
)

k = st.slider("Retrieve top-k chunks per query", 1, 5, 3)

tab_corpus, tab_eval, tab_run, tab_detail = st.tabs(
    ["📚 Corpus", "✅ Eval set", "🏁 Leaderboard", "🔍 Per-query"]
)

with tab_corpus:
    st.subheader("Sample knowledge base")
    st.dataframe(
        pd.DataFrame([{"doc_id": d, "text": t} for d, t in SAMPLE_DOCS.items()]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"{len(SAMPLE_DOCS)} documents. Swap in your own in chunker.py.")

with tab_eval:
    st.subheader("Gold eval set")
    st.write("Each question maps to the doc + exact span a good chunk must contain.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "qid": c.qid,
                    "question": c.question,
                    "doc_id": c.doc_id,
                    "answer_span": c.answer_span,
                }
                for c in SAMPLE_CASES
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_run:
    st.subheader(f"Strategy leaderboard (top-{k})")
    results = compare_strategies(k=k)
    df = pd.DataFrame(
        [
            {
                "strategy": r.name,
                "chunks": r.n_chunks,
                "avg_chunk_words": r.avg_chunk_words,
                f"hit_rate@{k}": r.hit_rate_at_k,
                "mrr": r.mrr,
            }
            for r in results
        ]
    )
    best = results[0]
    st.metric("Best strategy", best.name, f"hit@{k} = {best.hit_rate_at_k}")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("strategy")[f"hit_rate@{k}"])
    st.info(
        "Watch the tradeoff: very small chunks split answers across boundaries "
        "so no single chunk is retrievable (hit rate drops); whole-document "
        "chunks score high but dump maximum tokens into the prompt and dilute "
        "the signal. The sweet spot is usually sentence- or paragraph-aware."
    )

with tab_detail:
    st.subheader("Inspect one strategy")
    strat_names = list(build_strategies().keys())
    choice = st.selectbox("Strategy", strat_names, index=strat_names.index("sentence_2"))
    res = evaluate_strategy(
        choice, build_strategies()[choice], dict(SAMPLE_DOCS), list(SAMPLE_CASES), k=k
    )
    st.write(
        f"**{res.n_chunks}** chunks · avg **{res.avg_chunk_words}** words · "
        f"hit@{k} = **{res.hit_rate_at_k}** · MRR = **{res.mrr}**"
    )
    st.dataframe(pd.DataFrame(res.per_query), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Day 83 of Phoebe's FDE portfolio · LLMOps & GenAI Platform · "
    "run `python chunker.py` for the CLI leaderboard."
)
