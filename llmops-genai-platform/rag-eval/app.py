from __future__ import annotations

# Streamlit UI for the RAG Evaluation Harness. Four tabs: Corpus, Eval set,
# Run (score the retriever), A/B (compare two k values as a proxy for old vs
# new retriever).
import pandas as pd
import streamlit as st
from harness import (
    SAMPLE_CASES,
    SAMPLE_DOCS,
    LexicalRetriever,
    compare,
    evaluate,
)

st.set_page_config(page_title="RAG Evaluation Harness", page_icon="🔎", layout="wide")

st.title("🔎 RAG Evaluation Harness")
st.caption(
    "Changed your chunking or retriever? Prove it got better. "
    "Ranking-aware metrics (recall@k, MRR, nDCG) over a gold eval set - fully offline."
)

docs = dict(SAMPLE_DOCS)
retriever = LexicalRetriever(docs)

tab_corpus, tab_eval, tab_run, tab_ab = st.tabs(
    ["📚 Corpus", "✅ Eval set", "▶️ Run", "🆚 A/B by k"]
)

with tab_corpus:
    st.subheader("Sample knowledge base")
    st.dataframe(
        pd.DataFrame(
            [{"doc_id": d, "text": t} for d, t in docs.items()]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_eval:
    st.subheader("Gold eval set")
    st.write("Each question maps to the doc(s) a good retriever *should* surface.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "qid": c.qid,
                    "question": c.question,
                    "gold_docs": ", ".join(c.gold_docs),
                    "gold_answer": c.gold_answer,
                }
                for c in SAMPLE_CASES
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_run:
    st.subheader("Run evaluation")
    k = st.slider("Top-k retrieved", min_value=1, max_value=8, value=3)
    result = evaluate(SAMPLE_CASES, retriever.retrieve, docs, k=k)

    agg = result["aggregate"]
    cols = st.columns(len(agg))
    for col, (metric, val) in zip(cols, agg.items()):
        col.metric(metric, f"{val:.2f}")

    st.markdown("#### Per-question breakdown")
    per_case = pd.DataFrame(result["per_case"])
    per_case["retrieved"] = per_case["retrieved"].apply(lambda r: ", ".join(r) or "—")
    per_case["gold_docs"] = per_case["gold_docs"].apply(lambda r: ", ".join(r))
    show_cols = [c for c in per_case.columns if c not in ("question",)]
    st.dataframe(per_case[["question"] + [c for c in show_cols if c != "qid"]],
                 use_container_width=True, hide_index=True)

    misses = per_case[per_case[f"hit@{k}"] == 0]
    if len(misses):
        st.warning(
            f"⚠️ {len(misses)} question(s) retrieved **zero** gold docs in top-{k}. "
            "These are your silent RAG failures - fix chunking or retrieval for: "
            + ", ".join(misses["qid"])
        )
    else:
        st.success(f"✅ Every question hit a gold doc within top-{k}.")

with tab_ab:
    st.subheader("A/B: does a wider top-k help?")
    st.write(
        "Same retriever, two different `k` values. In a real workflow you'd swap "
        "the retriever itself (new embeddings, new chunk size) and diff the runs."
    )
    c1, c2 = st.columns(2)
    k_base = c1.number_input("Baseline k", 1, 8, 1)
    k_cand = c2.number_input("Candidate k", 1, 8, 3)

    base = evaluate(SAMPLE_CASES, retriever.retrieve, docs, k=int(k_base))
    cand = evaluate(SAMPLE_CASES, retriever.retrieve, docs, k=int(k_cand))
    diff = compare(base, cand)

    rows = []
    for metric, d in diff.items():
        arrow = "🟢" if d["delta"] > 0 else ("🔴" if d["delta"] < 0 else "⚪")
        rows.append(
            {
                "metric": metric,
                f"k={int(k_base)}": round(d["baseline"], 3),
                f"k={int(k_cand)}": round(d["candidate"], 3),
                "delta": round(d["delta"], 3),
                "": arrow,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Note: precision usually *drops* as k grows even when recall rises - "
        "that trade-off is exactly why you evaluate instead of guessing."
    )

st.divider()
st.caption("Day 82 · LLMOps & GenAI Platform · Phoebe the Builder")
