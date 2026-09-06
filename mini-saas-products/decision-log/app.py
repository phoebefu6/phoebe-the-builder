"""Score your own decision log, and see the rules disagree about it.

    streamlit run app.py
"""

from __future__ import annotations

from typing import List

import declog as D
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Decision Log Scorer", layout="wide")

st.title("A decision log is an instrument")
st.caption(
    "An instrument has a scoring rule, and three of the six below pay your team to "
    "misreport what they believe. Pick one deliberately."
)

tab_score, tab_lint, tab_rules = st.tabs(
    ["Score a log", "Lint a record", "The six rules"]
)

SAMPLE = "\n".join(
    f"{q},{y}" for q, y in [
        (0.9, 1), (0.7, 1), (0.6, 0), (0.8, 1), (0.95, 0), (0.5, 1),
        (0.3, 0), (0.75, 1), (0.85, 1), (0.4, 0), (0.65, 1), (0.2, 0),
    ]
)

# --------------------------------------------------------------------------
with tab_score:
    st.markdown(
        "One `probability,outcome` per line. Outcome is `1` if it happened, `0` if not."
    )
    raw = st.text_area("Your forecasts", value=SAMPLE, height=220)

    rows: List[tuple] = []
    bad_lines = []
    for i, line in enumerate(raw.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            q_s, y_s = line.split(",")
            q, y = float(q_s), float(y_s)
            if not (0.0 < q < 1.0) or y not in (0.0, 1.0):
                raise ValueError
            rows.append((q, y))
        except ValueError:
            bad_lines.append((i, line))

    if bad_lines:
        st.warning(
            "Skipped "
            + ", ".join(f"line {i} (`{t}`)" for i, t in bad_lines[:5])
            + ". A probability of exactly 0 or 1 is not a forecast - nothing can "
            "update it, and log loss is infinite there."
        )

    if len(rows) < 2:
        st.info("Give it at least two usable rows.")
    else:
        q = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])

        scores = {r.name: float(np.mean(r.loss(q, y))) for r in D.RULES}
        st.subheader("Your score, under each rule")
        st.dataframe(
            pd.DataFrame(
                {
                    "your loss": [scores[r.name] for r in D.RULES],
                    "proper?": [
                        "yes" if D.propriety(r.name)[0] else "NO - do not use"
                        for r in D.RULES
                    ],
                    "where you meet it": [r.seen_in for r in D.RULES],
                },
                index=[r.name for r in D.RULES],
            ),
            use_container_width=True,
        )

        m = D.murphy(q, y, bins=min(5, max(2, len(rows) // 3)))
        st.subheader("Murphy decomposition")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Brier", f"{m['brier']:.4f}")
        c2.metric("reliability", f"{m['reliability']:.4f}", help="lower is better")
        c3.metric("resolution", f"{m['resolution']:.4f}", help="higher is better")
        c4.metric("uncertainty", f"{m['uncertainty']:.4f}", help="a property of the events")

        if m["resolution"] < 0.01:
            st.error(
                "**Resolution near zero.** These forecasts barely separate the cases - "
                "they carry almost no information beyond the base rate. Recalibration "
                "will not help, because there is nothing to recalibrate. A forecaster "
                "who reports the base rate every time has perfect reliability and is "
                "useless."
            )
        elif m["reliability"] > m["resolution"]:
            st.warning(
                "Miscalibration is larger than the information content. Recalibration "
                "is worth doing here, and it is the cheaper of the two fixes."
            )
        else:
            st.success("Resolution exceeds miscalibration - these forecasts carry signal.")

        st.caption(
            f"n = {len(rows)}. For context, separating two genuinely different "
            f"forecasters by Brier score needs a median of "
            f"{int(np.median(list(D.power_matrix().values()))):,} decisions. "
            "A log this size can describe, not rank."
        )

# --------------------------------------------------------------------------
with tab_lint:
    st.markdown(
        "A record has to contain something that can turn out to be **wrong**. "
        "Four fields, all required."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        decision = st.text_input("Decision", "Migrate the warehouse to the new engine")
        claim = st.text_input("Claim", "query costs will fall")
        prob = st.text_input("Probability (strictly between 0 and 1)", "")
    with col_b:
        resolve_by = st.text_input("Resolve by (YYYY-MM-DD)", "Q4")
        metric = st.text_input("Metric", "")
        threshold = st.text_input("Threshold", "")

    rec = D.Record(
        "USER", decision, claim,
        float(prob) if prob.replace(".", "", 1).isdigit() and prob else None,
        resolve_by or None, metric or None, threshold or None,
    )
    checks = D.lint(rec)
    WHY = {
        "has_probability": "without it, a good decision that lost is indistinguishable "
                           "from a bad one that lost",
        "has_resolution_date": "without it, the record is remembered and never scored",
        "has_metric": "without it, two readers resolve it differently",
        "has_threshold": "without it, there is no fact of the matter",
    }
    st.subheader("scoreable" if D.resolvable(rec) else "NOT scoreable")
    for k, ok in checks.items():
        (st.success if ok else st.error)(
            f"**{k}** — {'present' if ok else 'missing: ' + WHY[k]}"
        )

    st.divider()
    st.markdown("**The reference corpus**")
    rep = D.resolvability_report()
    st.caption(
        f"{rep['resolvable']} of {rep['n']} records are scoreable. The corpus is written, "
        "not sampled, so that rate is a property of these records - the linter is the "
        "reusable part."
    )
    st.dataframe(
        pd.DataFrame(
            [{"id": r.id, "decision": r.decision, "claim": r.claim,
              "p": r.probability, "by": r.resolve_by,
              "scoreable": "yes" if D.resolvable(r) else "NO"} for r in D.RECORDS]
        ).set_index("id"),
        use_container_width=True, height=430,
    )

# --------------------------------------------------------------------------
with tab_rules:
    st.markdown(
        "**Proper** means the report that scores best is the report the forecaster "
        "actually believes. Computed here, not quoted: expected loss minimised over "
        "1001 candidate reports, for 99 true beliefs."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "proper": ["yes" if D.propriety(r.name)[0] else "NO" for r in D.RULES],
                "bounded": [r.bounded for r in D.RULES],
                "believe 0.55, report": [
                    D.optimal_report(r, 0.55) for r in D.RULES
                ],
                "believe 0.80, report": [
                    D.optimal_report(r, 0.80) for r in D.RULES
                ],
                "where you meet it": [r.seen_in for r in D.RULES],
            },
            index=[r.name for r in D.RULES],
        ),
        use_container_width=True,
    )
    st.markdown(
        """
`absolute` and `confidence_points` both return **1.00** for every belief above a coin flip.
A forecaster who believes 55% and reports 55% scores *worse* than one who claims certainty.
`threshold_01` has no single optimum at all - every report above 0.5 scores identically, so
it cannot see confidence.

**Winner under each rule, same six forecasters, same 4,000 events:**
"""
    )
    st.dataframe(
        pd.DataFrame(
            {
                "1st": [D.ranking(r.name)[0] for r in D.RULES],
                "last": [D.ranking(r.name)[-1] for r in D.RULES],
                "proper": ["yes" if D.propriety(r.name)[0] else "NO" for r in D.RULES],
            },
            index=[r.name for r in D.RULES],
        ),
        use_container_width=True,
    )
    st.markdown(
        """
`absolute` - "average error", the one that ends up in a spreadsheet - crowns the
**overconfident** forecaster. `confidence_points`, the in-house prediction game, crowns the
**underconfident** one. Two homebrew rules, two opposite wrong answers.

And a proper rule is still not the whole specification: **log loss ranks `noisy_expert` last,
below the forecaster that knows nothing**, because it is unbounded and a handful of confident
misses dominate the mean.
"""
    )
