"""Streamlit front end: your corpus classes, your reindex policies, the three numbers for each.

Describe the document classes (one per line: name, docs, days between edits, share of edits that change a
fact, query share) and up to three reindex policies (a number of days for every class, or name=days pairs).
For each policy you get the dashboard's number (index freshness), the fact-level freshness, the share of
served answers that carry a changed fact, and the reindex cost; the banner says whether the dashboard
ranks the policies in the same order the answers do. The exact study from results.json is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import staleness as T
import streamlit as st

st.set_page_config(page_title="RAG staleness", layout="wide")
R = json.load(open("results.json"))

st.title("It answered from last quarter's document")
tab_use, tab_study = st.tabs(["Your corpus and policies", "The study"])

with tab_use:
    left, right = st.columns([3, 2])
    classes_text = left.text_area(
        "Document classes: name, docs, days between edits, fact-bearing share, query share",
        "\n".join(f"{c['name']}, {c['docs']}, {c['change_every']:g}, {c['fact_share']}, {c['query_share']}"
                  for c in T.CLASSES), height=120, key="classes")
    pol_texts = [right.text_input(f"Policy {i + 1} (days, or name=days per class)", v, key=f"policy{i}")
                 for i, v in enumerate(("30", "7", "hot=1, warm=7, cold=90"))]
    try:
        classes = T.parse_classes(classes_text)
        policies = {}
        for i, t in enumerate(pol_texts):
            if t.strip():
                policies[f"policy {i + 1}: {t.strip()}"] = T.parse_policy(t, classes)
        if not policies:
            raise ValueError("give at least one policy")
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    ev = {k: T.evaluate(p, classes) for k, p in policies.items()}
    rows = [{"policy": k, "index freshness (hash)": f"{e['hash_freshness']:.1%}",
             "fact freshness": f"{e['fact_freshness']:.1%}", "answers stale": f"{e['answer_staleness']:.1%}",
             "reindex / day": f"{e['reindex_per_day']:.1f}"} for k, e in ev.items()]
    by_dash = sorted(ev, key=lambda k: -ev[k]["hash_freshness"])
    by_user = sorted(ev, key=lambda k: ev[k]["answer_staleness"])
    worst = max(ev.values(), key=lambda e: e["answer_staleness"])
    if len(ev) > 1 and by_dash != by_user:
        st.error(f"RANKING FLIPS: index freshness puts '{by_dash[0]}' first; stale answers put '{by_user[0]}' first "
                 f"({ev[by_user[0]]['answer_staleness']:.1%} vs {ev[by_dash[0]]['answer_staleness']:.1%} of answers).")
    elif worst["answer_staleness"] > 2 * (1 - worst["fact_freshness"]):
        st.warning(f"ANSWERS STALER THAN THE INDEX: {worst['answer_staleness']:.1%} of answers carry a changed fact vs "
                   f"{1 - worst['fact_freshness']:.1%} of documents - the queries land on the classes that change.")
    else:
        st.success("CONSISTENT: the dashboard's number and the answers' number agree on the order, and the "
                   "answers are not much staler than the index. The similarity gate is still blind to staleness.")
    st.table(rows)
    st.caption("stale(mu, T) = 1 + expm1(-mu T) / (mu T), a doc reindexed every T days served at a uniform phase; "
               "hash freshness counts every edit, the other two only edits that change a fact.")

with tab_study:
    st.image("staleness_audit.png")
    st.write({k: f"{v * 100:+.1f} pts" for k, v in R["deltas"].items()})
    st.caption("Closed form under the declared model, checked by raw simulation; see evidence.txt.")
