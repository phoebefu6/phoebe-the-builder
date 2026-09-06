"""Render dsar_coverage.png - the two failure directions, side by side."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dsar import (
    MENTION,
    RETAIN,
    SUBJECT_EMAIL,
    build_corpus,
    coverage,
    erasure_plan,
    extract,
    naive_extract,
    naive_fk_sweep,
    resolve_identity,
    weak_link_cost,
)

corpus = build_corpus()
ident = resolve_identity(corpus, SUBJECT_EMAIL)
hits = extract(corpus, ident)
cov = coverage(corpus, hits)
naive = naive_extract(corpus, SUBJECT_EMAIL)
sweep = naive_fk_sweep(corpus, hits)
cost = weak_link_cost(corpus, ident)
plan = erasure_plan(hits, corpus)

INK = "#1d2433"
UNDER = "#c2410c"
OVER = "#1d4ed8"
OK = "#0f766e"
MUTED = "#94a3b8"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
fig.suptitle(
    "One subject access request: both failure directions are quantifiable",
    fontsize=14,
    fontweight="bold",
    color=INK,
    y=0.99,
)

# ---- Panel 1: under-collection, by mechanism -----------------------------------------
ax = axes[0]
missed = [h for h in hits if (h.table, h.pk) not in naive]
by_mech = {"plus-addressed\nmailbox": 0, "pre-login\nanon_id": 0, "free-text\nmention": 0}
for hit in missed:
    if hit.how == MENTION:
        by_mech["free-text\nmention"] += 1
    elif hit.table == "web_events":
        by_mech["pre-login\nanon_id"] += 1
    else:
        by_mech["plus-addressed\nmailbox"] += 1

labels = list(by_mech)
values = [by_mech[k] for k in labels]
bars = ax.bar(labels, values, color=UNDER, width=0.6)
ax.bar_label(bars, fmt="%d", padding=3, fontsize=11, fontweight="bold", color=INK)
ax.set_title(
    f"UNDER-COLLECTION\n{cov['naive']} rows found by the ordinary query, {cov['resolved']} actually held",
    fontsize=11,
    color=UNDER,
    fontweight="bold",
    pad=10,
)
ax.set_ylabel("rows the requester would never have seen")
ax.set_ylim(0, max(values) * 1.28)
ax.tick_params(labelsize=9)

# ---- Panel 2: over-collection and over-resolution ------------------------------------
ax = axes[1]
over_labels = [
    "reference rows\n(products)",
    "one reverse join\n(other people's items)",
    "weak name link\n(a stranger's rows)",
]
over_values = [
    sweep.get("products", 0),
    sweep.get("_reverse_order_items", 0),
    cost.get("orders", 0) + cost.get("payments", 0) + cost.get("order_items", 0),
]
bars = ax.barh(over_labels, over_values, color=[MUTED, OVER, OVER], height=0.55)
ax.bar_label(bars, fmt="%d", padding=4, fontsize=11, fontweight="bold", color=INK)
ax.set_title(
    f"OVER-COLLECTION\nrows a naive traversal adds - {sweep.get('_reverse_customers', 0)} other "
    "customers exposed",
    fontsize=11,
    color=OVER,
    fontweight="bold",
    pad=10,
)
ax.set_xlabel("rows wrongly included")
ax.set_xlim(0, max(over_values) * 1.3)
ax.invert_yaxis()
ax.tick_params(labelsize=9)

# ---- Panel 3: access is not erasure --------------------------------------------------
ax = axes[2]
counts: dict = {}
for action in plan:
    counts[action.action] = counts.get(action.action, 0) + 1
order = sorted(counts, key=lambda k: -counts[k])
colors = [UNDER if k == RETAIN else OK for k in order]
bars = ax.barh([k.replace(" - ", "\n") for k in order], [counts[k] for k in order], color=colors, height=0.55)
ax.bar_label(bars, fmt="%d", padding=4, fontsize=11, fontweight="bold", color=INK)
blocked = counts.get(RETAIN, 0)
ax.set_title(
    f"ACCESS IS NOT ERASURE\n{blocked} of {len(hits)} disclosed rows cannot be deleted",
    fontsize=11,
    color=INK,
    fontweight="bold",
    pad=10,
)
ax.set_xlabel("rows")
ax.set_xlim(0, max(counts.values()) * 1.3)
ax.invert_yaxis()
ax.tick_params(labelsize=9)

for ax in axes:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x" if ax is not axes[0] else "y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

fig.text(
    0.5,
    0.015,
    f"{cov['shared']} disclosed rows contain another living person and are redacted before the pack ships.",
    ha="center",
    fontsize=9.5,
    color=INK,
    style="italic",
)
fig.tight_layout(rect=(0, 0.045, 1, 0.94))
fig.savefig("dsar_coverage.png", dpi=150, bbox_inches="tight", facecolor="white")
print("wrote dsar_coverage.png")
