"""Streamlit UI for sparkline-gen. Verdict first, picture second.

    streamlit run app.py

The layout is deliberate: you cannot see the rendered table until you have read what it is
and is not entitled to claim.
"""

from __future__ import annotations

import io
from typing import List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from sparkline import (
    SCALE_MODES,
    Geometry,
    Series,
    Style,
    audit_table,
    banked_width,
    banking_deg,
    build_path,
    endpoint_direction,
    escape,
    indexed_series,
    render_table,
    resolve_domain,
    sample_table,
    sparkline_svg,
    theil_sen_slope,
    trend_direction,
)

st.set_page_config(page_title="Sparkline Generator", page_icon="📈", layout="wide")

st.title("Sparkline generator")
st.caption(
    "Inline SVG trend marks for a table - with the four decisions that decide whether they "
    "tell the truth made explicit: scale, aspect ratio, gaps, and the time axis."
)

# --------------------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------------------

with st.sidebar:
    st.header("Data")
    src = st.radio("Source", ["Sample telemetry", "Upload CSV"], label_visibility="collapsed")

    rows: List[Series] = []
    if src == "Sample telemetry":
        rows = sample_table()
        st.caption(
            "Six rows, each carrying one lesson: two differing 1000x in level, one whose "
            "endpoints disagree with its trend, one with an outage gap, one with irregular "
            "reporting, one constant."
        )
    else:
        up = st.file_uploader("Wide CSV: first column = label, rest = the series", type="csv")
        if up is not None:
            df = pd.read_csv(up)
            label_col = df.columns[0]
            value_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
            if not value_cols:
                st.error("No numeric columns found after the label column.")
            for _, r in df.iterrows():
                vals: List[Optional[float]] = [
                    None if pd.isna(r[c]) else float(r[c]) for c in value_cols
                ]
                rows.append(Series(str(r[label_col]), vals))
            st.caption("%d rows x %d periods. Blank cells are treated as missing." % (len(rows), len(value_cols)))

    st.header("Scale")
    mode = st.selectbox(
        "Mode", list(SCALE_MODES), index=0, format_func=lambda m: m,
    )
    st.caption(SCALE_MODES[mode])

    st.header("Geometry")
    width = st.slider("Width (px)", 20, 400, 80, 5)
    height = st.slider("Height (px)", 10, 60, 20, 1)
    stroke = st.slider("Stroke (px)", 0.5, 4.0, 1.25, 0.25)
    dot = st.slider("Endpoint dot radius (0 = none)", 0.0, 4.0, 0.0, 0.5)

    st.header("Rendering")
    bridge = st.checkbox(
        "Bridge gaps (draw through missing values)",
        value=False,
        help="What a one-liner does by default. The bridge is drawn at full stroke weight and "
        "cannot be distinguished from measured data.",
    )
    show_zero = st.checkbox("Show a zero baseline when 0 is in the domain", value=False)
    precision = st.select_slider("Coordinate precision", [0, 1, 2, 3], value=1)

if not rows:
    st.info("Pick the sample data or upload a CSV to begin.")
    st.stop()

geom = Geometry(width=float(width), height=float(height), stroke=float(stroke), dot=float(dot))
style = Style()

# --------------------------------------------------------------------------------------
# The verdict, before the picture
# --------------------------------------------------------------------------------------

verdict = audit_table(rows, mode=mode, geom=geom, bridge_gaps=bridge)

if verdict.comparable:
    st.success("**COMPARABLE ACROSS ROWS** (mode=`%s`) - %s" % (mode, SCALE_MODES[mode]))
else:
    st.error(
        "**NOT COMPARABLE ACROSS ROWS** (mode=`%s`) - every row is scaled to its own min/max, "
        "so the pictures carry shape and nothing else. Two rows with identical shapes may "
        "differ in level by any factor." % mode
    )

for w in verdict.warnings:
    st.warning(w)
for nte in verdict.notes:
    st.caption("note: " + nte)

# --------------------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------------------

st.subheader("Rendered")
# components.html, not st.html: st.html sanitises the markup and strips <svg> entirely, which
# silently empties the one column this tool exists to produce. The component iframe does not.
components.html(
    render_table(
        rows, mode=mode, geom=geom, style=style, bridge_gaps=bridge,
        precision=int(precision), show_zero=show_zero,
        banner=False,  # the verdict is already printed above, in Streamlit's own callouts
    ),
    height=int(60 + len(rows) * (max(height, 20) + 14)),
    scrolling=True,
)

# --------------------------------------------------------------------------------------
# Per-row diagnostics
# --------------------------------------------------------------------------------------

st.subheader("What each row is actually doing")
st.caption(
    "The two trend columns are the point. Where they disagree, the sparkline's shape is "
    "reporting the endpoints and the endpoints are wrong."
)

diag = []
for s in rows:
    drawn = indexed_series(s) if (mode == "indexed" and s.present and s.present[0]) else s
    d = resolve_domain(drawn, mode, table=rows if mode in ("shared", "shared_zero", "indexed") else None)
    p = build_path(drawn, d, geom, bridge_gaps=bridge, precision=int(precision))
    e, t = endpoint_direction(s), trend_direction(s)
    arrow = {1: "up", -1: "down", 0: "flat"}
    diag.append(
        {
            "row": s.label,
            "n": len(s.values),
            "missing": s.n_missing,
            "endpoints say": arrow[e],
            "robust trend says": arrow[t],
            "agree": "yes" if e == t else "NO",
            "Theil-Sen slope": round(theil_sen_slope(s), 4),
            "rendered median slope": "%.0f deg" % banking_deg(p.points) if len(p.points) > 1 else "-",
            "banked width for 45 deg": (
                "%.0f px" % banked_width(drawn, d, geom) if len(drawn.present) > 1 else "-"
            ),
            "svg bytes": len(
                sparkline_svg(drawn, d, geom, style, bridge_gaps=bridge, precision=int(precision))
            ),
        }
    )
diag_df = pd.DataFrame(diag)
st.dataframe(diag_df, hide_index=True, use_container_width=True)

total = int(diag_df["svg bytes"].sum())
st.caption(
    "Total inline SVG for this table: %.1f KB. At 500 rows that is %.0f KB - larger than most "
    "pages' entire HTML, and the coordinates are only about a fifth of it. Past a few hundred "
    "rows, share the wrapper via `<defs>` or rasterise server-side."
    % (total / 1024, total / len(rows) * 500 / 1024)
)

# --------------------------------------------------------------------------------------
# Aspect ratio explorer
# --------------------------------------------------------------------------------------

with st.expander("Why the column width changes the story"):
    st.caption(
        "Cleveland's result: slope judgement is most accurate when the average absolute slope "
        "is near 45 degrees. A table cell sets that angle from its own geometry, not from the "
        "data - so the same series reads as flat in a wide column and volatile in a narrow one."
    )
    pick = st.selectbox("Row", [s.label for s in rows], key="aspect_row")
    s = next(x for x in rows if x.label == pick)
    d = resolve_domain(s, "per_row")
    ladder = []
    html_parts = []
    for w in (20, 40, 80, 160, 320):
        g = Geometry(width=float(w), height=float(height), stroke=float(stroke))
        p = build_path(s, d, g, bridge_gaps=bridge, precision=int(precision))
        deg = banking_deg(p.points) if len(p.points) > 1 else 0.0
        ladder.append(
            {
                "width": "%d px" % w,
                "median rendered slope": "%.0f deg" % deg,
                "reads as": (
                    "flat / stable" if deg < 15 else "gentle rise" if deg < 35
                    else "clear trend" if deg < 55 else "volatile / spiky"
                ),
            }
        )
        html_parts.append(
            "<div style='display:flex;align-items:center;gap:10px;margin:3px 0'>"
            "<code style='width:56px;color:#6b7280'>%d px</code>%s"
            "<span style='font:12px monospace;color:#6b7280'>%.0f deg</span></div>"
            % (w, sparkline_svg(s, d, g, style, bridge_gaps=bridge, precision=int(precision)), deg)
        )
    components.html(
        "<div style=\'font:13px -apple-system,system-ui,sans-serif\'>%s</div>" % "".join(html_parts),
        height=int(40 + 5 * (max(height, 20) + 14)),
        scrolling=True,
    )
    st.dataframe(pd.DataFrame(ladder), hide_index=True, use_container_width=True)
    if len(s.present) > 1:
        st.info(
            "Banked to 45 degrees, this row wants a **%.0f px** wide cell at %d px tall."
            % (banked_width(s, d, Geometry(width=80.0, height=float(height))), height)
        )

# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------

with st.expander("Export"):
    out = []
    for s in rows:
        drawn = indexed_series(s) if (mode == "indexed" and s.present and s.present[0]) else s
        d = resolve_domain(
            drawn, mode, table=rows if mode in ("shared", "shared_zero", "indexed") else None
        )
        out.append(
            {
                "label": s.label,
                "scale_mode": mode,
                "domain_lo": d.lo,
                "domain_hi": d.hi,
                "svg": sparkline_svg(
                    drawn, d, geom, style, bridge_gaps=bridge, precision=int(precision),
                    show_zero=show_zero,
                ),
            }
        )
    csv = pd.DataFrame(out).to_csv(index=False)
    st.download_button("Download sparklines.csv", csv, "sparklines.csv", "text/csv")
    st.caption(
        "The domain columns ship with the SVG on purpose. A sparkline separated from the "
        "domain it was drawn against is not reproducible and not readable."
    )
    st.code(out[0]["svg"], language="html")
