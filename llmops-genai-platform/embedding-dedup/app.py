from __future__ import annotations

# Streamlit UI for the near-duplicate finder. Load the sample corpus or paste your
# own docs, tune the three thresholds, and see the clusters, the rejected pairs with
# reasons, the keep/drop plan, and what dedup does to a retrieval result page.
import pandas as pd
import streamlit as st

from dedup import (
    SAMPLE_DOCS,
    cluster_pairs,
    dedup_plan,
    distinct_answers,
    evaluate,
    find_duplicates,
    redundant_slots,
    retrieve,
    truth_pairs,
)

st.set_page_config(page_title="Embedding Dedup", page_icon="🧬", layout="wide")

st.title("🧬 Near-Duplicate Finder for a RAG Corpus")
st.caption(
    "Duplicates do not just cost embedding spend - they eat the top-k slots, so the "
    "model sees one document three times instead of three documents once. Cosine "
    "similarity alone cannot do this job: it merges template siblings and misses "
    "absorbed paragraphs. This gates on three signals and shows its reasoning."
)

with st.sidebar:
    st.header("Corpus")
    source = st.radio("Input", ["Sample knowledge base", "Paste your own"], index=0)
    st.header("Gate thresholds")
    cos_t = st.slider("Cosine (near-duplicate)", 0.0, 1.0, 0.80, 0.01)
    con_t = st.slider("Containment (near-duplicate)", 0.0, 1.0, 0.65, 0.01)
    sub_t = st.slider("Containment (subset link, any cosine)", 0.0, 1.0, 0.90, 0.01)
    num_t = st.slider("Numeric agreement veto below", 0.0, 1.0, 0.50, 0.05,
                      help="Two docs off the same template disagree on their figures. "
                           "Set to 0 to disable the veto and watch precision drop.")
    st.header("Cost")
    price = st.number_input("Embedding $ / 1M tokens", 0.0, 10.0, 0.13, 0.01, format="%.3f")

if source == "Sample knowledge base":
    docs = SAMPLE_DOCS
else:
    raw = st.text_area(
        "One document per line (or paste a blank-line-separated block)",
        "\n\n".join(d["text"] for d in SAMPLE_DOCS[:4]),
        height=200,
    )
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()] or \
             [ln.strip() for ln in raw.splitlines() if ln.strip()]
    docs = [{"id": f"doc-{i}", "source": "pasted", "dup_group": None, "text": b}
            for i, b in enumerate(blocks)]

result = find_duplicates(docs, cos_threshold=cos_t, contain_threshold=con_t,
                         subset_threshold=sub_t, numeric_threshold=num_t)
plan = dedup_plan(docs, result, price_per_mtok=price)
dropped = {d["index"] for d in plan["dropped"]}

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Documents", result["n_docs"])
c2.metric("Duplicate clusters", len(result["clusters"]))
c3.metric("Drop", plan["drop_count"], f"-{plan['pct_index_saved']}% tokens")
c4.metric("Pairs scored", result["n_candidate_pairs"],
          f"of {result['n_all_pairs']} all-pairs", delta_color="off")
c5.metric("Re-embed saved", f"${plan['embedding_cost_saved']:.6f}")

if not result["clusters"]:
    st.info("No duplicate clusters at these thresholds.")

tabs = st.tabs(["🧩 Clusters", "🚫 Rejected pairs", "📉 Retrieval impact",
                "📋 Keep / drop plan", "🎯 Accuracy"])

kind_of = {(p["a"], p["b"]): p.get("kind", "near") for p in result["pairs"]}

with tabs[0]:
    for c in result["clusters"]:
        via = sorted({k for (a, b), k in kind_of.items()
                      if a in c["members"] and b in c["members"]})
        st.markdown(f"**Keep `{docs[c['keep']].get('id', c['keep'])}`** "
                    f"— {c['size']} members, linked via `{'+'.join(via) or 'exact'}`")
        st.caption(docs[c["keep"]]["text"][:220] + "…")
        for i in c["drop"]:
            st.markdown(f"&nbsp;&nbsp;↳ drop `{docs[i].get('id', i)}` "
                        f"({docs[i].get('source', '?')})", unsafe_allow_html=True)
        st.divider()
    if result["fragments"]:
        st.warning(
            "Held out as fragments (too short for any signal to mean anything): "
            + ", ".join(f"`{docs[i].get('id', i)}`" for i in result["fragments"])
            + ". Nav bars and footers are similar to everything; merging them would "
              "drag unrelated pages into one cluster."
        )

with tabs[1]:
    st.caption("Pairs similar enough to consider that the gate refused, and why. "
               "This list is the point of the tool - a silent dedup is unauditable.")
    if result["rejected"]:
        st.dataframe(pd.DataFrame([{
            "doc A": docs[r["a"]].get("id", r["a"]),
            "doc B": docs[r["b"]].get("id", r["b"]),
            "cosine": r["cosine"],
            "containment": r["containment"],
            "numeric agreement": r["numeric"],
            "why not merged": "; ".join(r["reasons"]),
        } for r in result["rejected"]]), use_container_width=True, hide_index=True)
    else:
        st.info("Nothing was rejected at these thresholds.")

with tabs[2]:
    query = st.text_input(
        "Query", "how many business days for a refund to reach the original payment method")
    k = st.slider("top-k", 1, 10, 3)
    idx = result["index"]
    if idx is not None and query.strip():
        before, after = retrieve(idx, query, k=k), retrieve(idx, query, k=k, exclude=dropped)
        col_b, col_a = st.columns(2)
        for col, title, hits in ((col_b, "Before dedup", before), (col_a, "After dedup", after)):
            n_ans, n_red = distinct_answers(result, hits), redundant_slots(result, hits)
            col.markdown(f"**{title}** — {n_ans} distinct answer{'' if n_ans == 1 else 's'}, "
                         f"{n_red} redundant slot{'' if n_red == 1 else 's'}")
            for i, s in hits:
                col.markdown(f"`{s:.3f}` **{docs[i].get('id', i)}** — {docs[i]['text'][:110]}…")
        st.caption("Redundant slots are the real cost: the generator pays for them, "
                   "re-reads the same claim, and never sees the answer that got pushed out.")

with tabs[3]:
    if plan["dropped"]:
        st.dataframe(pd.DataFrame(plan["dropped"])[
            ["id", "duplicate_of", "tokens"]], use_container_width=True, hide_index=True)
    st.markdown(
        f"Keep **{plan['keep_count']}** of {result['n_docs']} documents · "
        f"~**{plan['tokens_saved']:,}** of {plan['total_tokens']:,} tokens dropped "
        f"(**{plan['pct_index_saved']}%**) · re-embedding saved "
        f"**${plan['embedding_cost_saved']:.6f}** at ${price}/1M."
    )
    st.caption("Token counts are a ~4 chars/token estimate, fine for sizing and wrong "
               "for billing. The survivor is the longest member of each cluster, so a "
               "superset keeps detail a shorter copy dropped.")

with tabs[4]:
    truth = truth_pairs(docs)
    if not truth:
        st.info("Accuracy needs labelled duplicate groups - available on the sample corpus.")
    else:
        rows = []
        for name, kwargs in (
            ("cosine only", {"contain_threshold": 0.0, "subset_threshold": 1.01,
                             "numeric_threshold": 0.0}),
            ("+ containment", {"contain_threshold": con_t, "subset_threshold": sub_t,
                               "numeric_threshold": 0.0}),
            ("+ numeric veto", {"contain_threshold": con_t, "subset_threshold": sub_t,
                                "numeric_threshold": num_t}),
        ):
            m = evaluate(cluster_pairs(find_duplicates(docs, cos_threshold=cos_t, **kwargs)), truth)
            rows.append({"gate": name, "precision": m["precision"], "recall": m["recall"],
                         "F1": m["f1"], "false merges": m["fp"], "missed": m["fn"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Precision matters more than recall here. A missed duplicate wastes "
                   "storage; a false merge deletes information that no longer exists "
                   "anywhere in the index.")
