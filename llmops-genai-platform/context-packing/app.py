"""Streamlit front end: your packing policies, the three numbers for each, and where the answers went.

Describe up to three packing policies as `budget, order, instruction` (order: rank / reorder / reverse;
instruction: top / both). Each is scored on the same declared query log: context recall (the dashboard),
effective recall, answer rate (the user), compliance, tokens, and the three failure classes. The banner says
whether the dashboard ranks the policies in the same order the answers do. The exact study from results.json is
on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import packing as P
import streamlit as st

st.set_page_config(page_title="Context packing", layout="wide")
R = json.load(open("results.json"))


@st.cache_data
def the_log():
    return P.make_log()


LOG = the_log()

st.title("The fact was in the context, and the model did not read it")
tab_use, tab_study = st.tabs(["Your packing policies", "The study"])

with tab_use:
    st.caption(f"Scored on the declared log: {P.Q:,} queries, {P.N_CAND} candidates each, hit@10 "
               f"{R['retriever_hit_at']['10']:.0%}. A policy is `budget, order, instruction`; order is rank (best first), "
               "reorder (best at both edges) or reverse (best last); instruction is top (once) or both (repeated before "
               "the question).")
    cols = st.columns(3)
    texts = [cols[i].text_input(f"Policy {i + 1}", v, key=f"policy{i}")
             for i, v in enumerate(("4000, rank, top", "12000, rank, top", "4000, reorder, both"))]
    try:
        policies = {}
        for i, t in enumerate(texts):
            if t.strip():
                policies[f"policy {i + 1}: {t.strip()}"] = P.parse_policy(t)
        if not policies:
            raise ValueError("give at least one policy")
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    ev = {k: P.evaluate(p, LOG) for k, p in policies.items()}
    rows = [{"policy": k, "context recall": f"{e['context_recall']:.1%}", "effective recall": f"{e['effective_recall']:.1%}",
             "answer rate": f"{e['answer_rate']:.1%}", "compliance": f"{e['compliance']:.1%}",
             "tokens / query": f"{e['tokens_per_query']:.0f}", "fail: retrieval miss": f"{e['fail_missing']:.1%}",
             "fail: present, unread": f"{e['fail_unread']:.1%}", "fail: instruction ignored": f"{e['fail_ignored']:.1%}"}
            for k, e in ev.items()]
    by_dash = sorted(ev, key=lambda k: -ev[k]["context_recall"])
    by_user = sorted(ev, key=lambda k: -ev[k]["answer_rate"])
    worst = max(ev.values(), key=lambda e: e["fail_unread"])
    if len(ev) > 1 and by_dash != by_user:
        st.error(f"RANKING FLIPS: context recall puts '{by_dash[0]}' first; the answer rate puts '{by_user[0]}' first "
                 f"({ev[by_user[0]]['answer_rate']:.1%} vs {ev[by_dash[0]]['answer_rate']:.1%} of answers).")
    elif worst["fail_unread"] > worst["fail_missing"]:
        st.warning(f"PRESENT BUT UNREAD is the largest failure class: {worst['fail_unread']:.1%} of queries had every "
                   f"needed fact in the window and lost it to position or length, vs {worst['fail_missing']:.1%} to "
                   "retrieval. Context recall counts the first group as a success.")
    else:
        st.success("CONSISTENT: the dashboard and the answer rate agree on the order, and retrieval misses outnumber "
                   "the facts that were present and unread. A bigger budget will move context recall; check the "
                   "answer rate before believing it.")
    st.table(rows)
    st.caption("read(u, L) is the quadratic through the three Lost-in-the-Middle points, sinking with window length; "
               "comply(d) falls with the tokens between the instruction and the question. Exact expectations over the "
               "declared log; see evidence.txt.")

with tab_study:
    st.image("packing_audit.png")
    st.write({k: f"{v * 100:+.1f} pts" for k, v in R["deltas"].items()})
    st.caption("Exact under the declared model, checked by raw Bernoulli simulation; see evidence.txt.")
