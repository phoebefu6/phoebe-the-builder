from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from topics import SAMPLE_DOCS, fit_topics, label_documents, topic_sizes

st.set_page_config(page_title="Topic Modeling Tool", layout="wide")
st.title("Topic Modeling Tool")
st.caption('"What are these documents about?" — discover themes automatically with NMF over TF-IDF.')

with st.sidebar:
    n_topics = st.slider("Number of topics", 2, 6, 3)
    n_words = st.slider("Words per topic", 5, 12, 8)
    st.caption("Paste one document per line, or use the sample corpus.")

raw = st.text_area("Documents (one per line)", value="\n".join(SAMPLE_DOCS), height=260)
docs = [d.strip() for d in raw.splitlines() if d.strip()]

if len(docs) < n_topics + 1:
    st.warning("Add more documents than topics.")
    st.stop()

model = fit_topics(docs, n_topics=n_topics, n_top_words=n_words)

st.subheader("Discovered topics")
cols = st.columns(n_topics)
sizes = topic_sizes(model)
for t in model.topics:
    with cols[t.id % n_topics]:
        st.markdown(f"**Topic {t.id}** ({sizes[t.id]} docs)")
        st.write(", ".join(t.top_words))

st.subheader("Topic sizes")
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar([f"T{k}: {model.topics[k].label}" for k in sizes], list(sizes.values()), color="#3b6fd6")
ax.set_ylabel("Documents")
plt.xticks(rotation=20, ha="right", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig)

st.subheader("Documents labeled by topic")
st.dataframe(pd.DataFrame(label_documents(model, docs)), use_container_width=True)
