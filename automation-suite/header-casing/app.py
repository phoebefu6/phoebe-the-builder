"""Streamlit front end for the field-name audit.

Paste a block of header lines, pick a path, and the app shows what arrives. It
never shows the arrived message without showing which lookups can still find the
field in it.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import streamlit as st
from headers import (
    CORPUS,
    CORPUS_BY_NAME,
    HOPS,
    LOOKUPS,
    PATHS,
    REGISTRY_NAMES,
    Field,
    Message,
    Verdict,
    canonical_mismatches,
    deliver,
    deliver_all,
    environ_collisions,
    findings,
    go_canonical,
    is_token,
    lookup_audit,
    py_title,
    safe_form,
    turkish_breakage,
    verdict_counts,
    wire_cost,
    wsgi_key,
)

st.set_page_config(page_title="Header Casing", page_icon="🔤", layout="wide")

BADGE = {
    Verdict.PRESERVED: ("#e7f2e8", "#2f6b39",
                        "preserved - every field arrived, spelled as it was sent"),
    Verdict.RENORMALIZED: ("#e8eff2", "#2d5a68",
                           "renormalized - same meaning, different bytes on the wire"),
    Verdict.LOSSY: ("#fbeeda", "#8a5410",
                    "lossy - a field, a duplicate or an identity did not survive, "
                    "and nothing errored"),
    Verdict.REJECTED: ("#f9e3e0", "#a5291c",
                       "rejected - a hop must treat the message as malformed"),
}
SEV_ICON = {"blocking": "🔴", "silent": "🟠", "advisory": "🔵"}

st.title("A field name is not a string")
st.caption(
    "RFC 9110 5.1 makes field names case-insensitive. RFC 9113 8.2.1 makes them "
    "lowercase on an HTTP/2 wire, or the message is malformed. So the spelling that "
    "arrives is a property of the path, not of your code - and `headers[\"X-Foo\"]` "
    "is only correct on a path that changes nothing."
)


def parse_block(text: str) -> Tuple[List[Field], List[str]]:
    fields: List[Field] = []
    problems: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        if ":" not in line:
            problems.append(f"no colon: {line!r}")
            continue
        name, _, value = line.partition(":")
        if not is_token(name):
            problems.append(f"{name!r} is not a legal field name (RFC 9110 5.6.2)")
            continue
        fields.append(Field(name, value.strip()))
    return fields, problems


left, right = st.columns([0.44, 0.56], gap="large")

with left:
    st.subheader("The message")
    preset = st.selectbox("start from", ["(type your own)"] + [m.name for m in CORPUS],
                          index=2)
    default = "\n".join(
        f"{f.name}: {f.value}" for f in
        (CORPUS_BY_NAME[preset].fields if preset in CORPUS_BY_NAME
         else (Field("X-Request-ID", "abc123"), Field("X_Request_Id", "spoofed"),
               Field("Content-Type", "application/json")))
    )
    block = st.text_area("header lines, one per line", default, height=190)
    fields, problems = parse_block(block)
    for p in problems:
        st.error(p)
    wanted = st.text_input("field to look up afterwards",
                           fields[0].name if fields else "X-Request-ID")

if not fields:
    st.stop()

msg = Message("pasted", tuple(fields), "1.1")

with right:
    st.subheader("What each path delivers")
    rows = []
    for path, d in deliver_all(msg).items():
        got = lookup_audit(d, wanted)
        rows.append({
            "path": path,
            "verdict": d.verdict().value,
            "fields in": len(msg.fields),
            "fields out": 0 if d.arrived is None else len(d.arrived.fields),
            "lost": ", ".join(d.lost()) or "-",
            "h[name] finds it": "yes" if got["h[name]"] else "no",
            "case-folded finds it": "yes" if got["CaseInsensitiveDict"] else "no",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.divider()
st.subheader("One path, in detail")
path = st.selectbox("path", list(PATHS), index=list(PATHS).index("nginx-to-cgi"))
d = deliver(msg, path)
bg, fg, label = BADGE[d.verdict()]
st.markdown(
    f"<div style='background:{bg};color:{fg};padding:10px 14px;border-radius:6px;"
    f"font-weight:600'>{label}</div>",
    unsafe_allow_html=True,
)
hop_docs = {h.name: h.doc for h in HOPS}
st.caption(" → ".join(["client"] + [f"{h} ({hop_docs[h]})" for h in d.path] + ["app"])
           if d.path else "client → app, nothing in between")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**sent**")
    st.code("\n".join(f"{f.name}: {f.value}" for f in msg.fields), language="http")
with c2:
    st.markdown("**arrived**")
    if d.arrived is None:
        st.code("-- the message was treated as malformed --")
    else:
        st.code("\n".join(f"{f.name}: {f.value}" for f in d.arrived.fields),
                language="http")

fs = findings(d)
if fs:
    st.markdown("**findings**")
    for f in fs:
        st.write(f"{SEV_ICON[f.severity]} `{f.code}` {f.text}")
else:
    st.success("nothing happened to this message on this path.")

st.markdown(f"**Reading `{wanted}` back out**")
got = lookup_audit(d, wanted)
st.dataframe(
    pd.DataFrame([{"lookup": name, "what it does": doc,
                   "values found": len(got[name]),
                   "value": got[name][0] if got[name] else "-"}
                  for name, doc, _ in LOOKUPS]),
    hide_index=True, use_container_width=True,
)

st.divider()
c3, c4 = st.columns(2)
with c3:
    st.subheader("Advice, per name")
    for f in msg.fields:
        st.markdown(f"`{f.name}`")
        for bit in safe_form(f.name).split("; "):
            st.write(f"- {bit}")
with c4:
    st.subheader("The spelling functions")
    st.dataframe(
        pd.DataFrame([{"name": f.name, "identity": f.key,
                       "canonical": go_canonical(f.name),
                       "str.title()": py_title(f.name),
                       "CGI variable": wsgi_key(f.name)} for f in msg.fields]),
        hide_index=True, use_container_width=True,
    )
    h1, h2 = wire_cost(msg)
    st.caption(f"{h1} bytes as HTTP/1.1 field lines, {h2} modelled HPACK bytes "
               f"({1 - h2 / h1:.0%} smaller)")

st.divider()
st.subheader("Exhaustive searches over the registry")
counts = verdict_counts()
m1, m2, m3, m4 = st.columns(4)
m1.metric("field names searched", len(REGISTRY_NAMES))
m2.metric("CGI variable collisions", len(environ_collisions()))
m3.metric("respelled by canonicalisation", len(canonical_mismatches()))
m4.metric("broken by a tr_TR lowercase", len(turkish_breakage()))
st.caption(
    f"Across the bundled corpus: {counts[Verdict.PRESERVED]} preserved, "
    f"{counts[Verdict.RENORMALIZED]} renormalized, {counts[Verdict.LOSSY]} lossy, "
    f"{counts[Verdict.REJECTED]} rejected. The lossy column is the one that returns 200."
)
st.image("header_audit.png", use_container_width=True)
