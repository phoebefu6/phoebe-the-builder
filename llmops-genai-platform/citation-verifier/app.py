"""Streamlit front end: check your own cited claims against the passages they cite.

Paste the passages one per line (an optional leading [n] is stripped; line order gives the numbers) and the
claims one per line, each citing with [n]. For every claim you get the five text-only verdicts, the coverage of
its words in the cited passage, and the disagreement that points at a wrong citation: in the context but not
in the cited passage, a number the passage does not hold, a polarity that differs. The exact study from
results.json is on the second tab.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import citation as T
import numpy as np
import streamlit as st

st.set_page_config(page_title="Citation verifier", layout="wide")
R = json.load(open("results.json"))
TINY = {"has_citation": "cite", "context_lexical": "ctx", "cited_lexical": "lex", "cited_lexical+numbers": "+num",
        "cited_lexical+numbers+negation": "+neg"}


def _default_claims() -> str:
    m = T.MODELS["v2 always cite"]
    rng = np.random.default_rng(7)
    lines = []
    for _ in range(14):
        i = int(rng.integers(0, len(T.FACTS)))
        e = T.EDITS[rng.choice(len(T.EDITS), p=m["edit"])]
        c = T.CITES[rng.choice(len(T.CITES), p=m["cite"])]
        lines.append(T.build(i, e, c, bool(rng.random() < m["para"]))["claim"])
    return "\n".join(lines)


st.title("The citation does not say that")
tab_use, tab_study = st.tabs(["Check your claims", "The study"])

with tab_use:
    left, right = st.columns(2)
    passages_text = left.text_area("Passages, one per line (numbered by line)",
                                   "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(T.PASSAGES)), height=300,
                                   key="passages")
    claims_text = right.text_area("Claims, one per line, citing with [n]", _default_claims(), height=300, key="claims")
    try:
        passages, claims = T.parse_claims(passages_text, claims_text)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    res = T.audit(passages, claims)
    wrong = [r for r in res if r["cited"] is not None and not r["checks"]["cited_lexical+numbers+negation"]]
    uncited = [r for r in res if r["cited"] is None]
    if wrong:
        st.error(f"CITATION DOES NOT SAY THAT: {len(wrong)} of {len(res)} claims fail against the passage they cite "
                 f"(coverage, a number, or polarity). The context-level check passes "
                 f"{sum(r['checks']['context_lexical'] for r in wrong)} of them.")
    elif uncited:
        st.warning(f"UNCITED: {len(uncited)} of {len(res)} claims carry no citation, so nothing can be checked.")
    else:
        st.success(f"LEXICALLY CONSISTENT: all {len(res)} claims match their cited passage on words, numbers and "
                   "polarity. Entity and scope are NOT checked by any lexical rule.")
    st.table([{"claim": r["claim"], "cites": f"[{r['cited'] + 1}]" if r["cited"] is not None else "-",
               "coverage": f"{r['coverage']:.2f}",
               "passes": ", ".join(TINY[k] for k, v in r["checks"].items() if v) or "none",
               "note": "; ".join(r["notes"])} for r in res])
    st.caption(f"Lexical threshold tau = {T.TAU}. Every rule here sees only text; what the cited passage *says* needs "
               "an entailment judgment (NLI model or LLM judge) - see the study tab for what the rules miss.")

with tab_study:
    st.image("citation_audit.png")
    st.write({k: f"{v * 100:+.1f} pts" for k, v in R["deltas"].items()})
    st.caption("Exact enumeration over every claim the declared generator can emit; see evidence.txt.")
