from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from classifier import predict, sample_data, top_features, train_classifier

st.set_page_config(page_title="Text Classification Trainer", layout="wide")
st.title("Text Classification Trainer")
st.caption('"Auto-tag support tickets" — train a TF-IDF + logistic-regression classifier and route new text.')

texts, labels = sample_data()

with st.sidebar:
    test_size = st.slider("Test split", 0.2, 0.5, 0.3)
    st.caption(f"{len(texts)} labeled examples · {len(set(labels))} classes")

model = train_classifier(texts, labels, test_size=test_size)

c1, c2, c3 = st.columns(3)
c1.metric("Test accuracy", f"{model.accuracy:.0%}")
c2.metric("Classes", len(model.labels))
c3.metric("Examples", len(texts))

st.subheader("Tag new text")
txt = st.text_input("Ticket text", value="I was double charged on my invoice this month")
if txt.strip():
    r = predict(model, txt)
    st.success(f"**{r['label']}** ({r['confidence']:.0%} confident)")
    st.caption("Scores: " + " · ".join(f"{k}: {v:.0%}" for k, v in r["scores"].items()))

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Top words per class")
    for cls, words in top_features(model).items():
        st.markdown(f"**{cls}:** {', '.join(words)}")

with col_b:
    st.subheader("Confusion matrix (held-out)")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(model.confusion, cmap="Blues")
    ax.set_xticks(range(len(model.labels)))
    ax.set_xticklabels(model.labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(model.labels)))
    ax.set_yticklabels(model.labels, fontsize=8)
    for i in range(len(model.labels)):
        for j in range(len(model.labels)):
            ax.text(j, i, model.confusion[i, j], ha="center", va="center",
                    color="white" if model.confusion[i, j] > model.confusion.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("Held-out predictions")
st.dataframe(
    pd.DataFrame({"text": model.test_texts, "true": model.test_true, "predicted": model.test_pred}),
    use_container_width=True,
)
