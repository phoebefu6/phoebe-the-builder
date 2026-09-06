from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
from recommender import ItemItemCF, sample_interactions

st.set_page_config(page_title="Recommendation Engine", layout="wide")
st.title("Recommendation Engine")
st.caption('"No you-may-also-like" — item-item collaborative filtering with explanations.')

df = sample_interactions()
cf = ItemItemCF().fit(df)

with st.sidebar:
    st.subheader("Interaction matrix")
    st.caption("Rows = users, columns = items, values = ratings (0 = none).")
    n = st.slider("Recommendations", 1, 6, 3)

st.subheader("Ratings")
st.dataframe(df, use_container_width=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Recommend for a user")
    user = st.selectbox("User", df.index.tolist())
    recs = cf.recommend_for_user(user, n)
    if recs:
        for r in recs:
            why = cf.explain(user, r.item)
            st.markdown(f"**{r.item}** · score {r.score:.2f}")
            st.caption("Because you liked: " + ", ".join(why) if why else "")
    else:
        st.write("No recommendations.")

with col_b:
    st.markdown("### Similar items")
    item = st.selectbox("Item", df.columns.tolist())
    for r in cf.similar_items(item, n):
        st.markdown(f"**{r.item}** · similarity {r.score:.2f}")

st.subheader("Item-item similarity")
fig, ax = plt.subplots(figsize=(7.5, 6))
sim = cf.item_sim
im = ax.imshow(sim.values, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(sim.columns)))
ax.set_xticklabels(sim.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(sim.index)))
ax.set_yticklabels(sim.index, fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
st.pyplot(fig)
