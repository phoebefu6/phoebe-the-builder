"""The six experiments the README quotes. Every one is seeded and runnable.

Each function returns a plain dict of measurements and prints a table. Nothing here reads a
file or a network - the data is generated so the numbers reproduce exactly.

Run all of them:  python3 evidence.py
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparkline import (
    Geometry,
    Series,
    Style,
    banked_width,
    banking_deg,
    build_path,
    endpoint_direction,
    indexed_series,
    resolve_domain,
    sparkline_svg,
    theil_sen_slope,
    trend_direction,
    y_position,
)

RULE = "-" * 86


# ======================================================================================
# 1. Per-row autoscale renders different data as the same picture
# ======================================================================================


def experiment_per_row_scale(n_rows: int = 8) -> Dict[str, object]:
    """Rows with identical shape and wildly different level, under each scale mode.

    The claim is not "per-row scaling exaggerates". It is stronger and it is checkable: the
    emitted path strings come out *byte-identical*, so no rendering, no anti-aliasing and no
    reader can recover the difference. Level information is not compressed, it is discarded.

    The shape uses eighths and the levels use powers of two so the collapse is exact rather
    than approximate - on powers of ten, float rounding at 1e7 splits the identical rows into
    a handful of paths that differ in the last coordinate digit, which is noise that would
    only obscure the point.
    """
    shape = np.array([0, 3, 2, 5, 4, 7, 6, 8]) / 8.0  # dyadic: exactly representable
    levels = np.array([2.0 ** (3 * k) for k in range(n_rows)])  # 1x .. 2,097,152x
    rows = [
        Series("row_%d (level %.3g)" % (i, lv), [float(lv * (1.0 + 0.5 * v)) for v in shape])
        for i, lv in enumerate(levels)
    ]
    n_same = len(rows)
    # One extra row with a genuinely different shape, as a control - a mode that collapses
    # everything to one path would be trivially "consistent", so we need a row that must differ.
    rows.append(Series("control (falling)", [float(1e3 * (1.5 - 0.5 * v)) for v in shape]))

    geom = Geometry(width=80, height=20)
    out: Dict[str, object] = {"levels": levels.tolist(), "n_same_shape": n_same}
    print(
        "1. %d ROWS, IDENTICAL SHAPE, LEVELS SPANNING %.3gx - do the pictures differ?\n"
        % (n_same, levels[-1] / levels[0])
    )
    print("%-14s %16s %18s %14s" % ("scale mode", "distinct paths", "control separated", "comparable"))
    print(RULE)
    for mode in ("per_row", "shared", "shared_zero", "indexed"):
        paths = []
        for s in rows:
            drawn = indexed_series(s) if mode == "indexed" else s
            table = [indexed_series(r) for r in rows] if mode == "indexed" else rows
            d = resolve_domain(drawn, mode, table=table)
            paths.append(build_path(drawn, d, geom).d)
        distinct = len(set(paths[:n_same]))
        control_ok = paths[-1] not in set(paths[:n_same])
        out[mode] = {"distinct_paths": distinct, "control_separated": control_ok}
        print(
            "%-14s %13d/%d %18s %14s"
            % (mode, distinct, n_same, "yes" if control_ok else "no",
               "yes" if mode != "per_row" else "NO")
        )
    print(RULE)
    print(
        "%d rows spanning %.3gx in level produce %d distinct path under per_row. The control\n"
        "row proves the renderer can still tell shapes apart - it is level, specifically, that\n"
        "is thrown away."
        % (n_same, levels[-1] / levels[0], out["per_row"]["distinct_paths"])
    )

    # What a shared scale costs: the small rows flatten.
    print("\n   what the comparable modes cost - rendered vertical extent, in pixels:\n")
    print("   %-26s %10s %10s %10s" % ("row", "per_row", "shared", "indexed"))
    print("   " + RULE[:60])
    extents: Dict[str, List[float]] = {}
    for mode in ("per_row", "shared", "indexed"):
        vals = []
        for s in rows:
            drawn = indexed_series(s) if mode == "indexed" else s
            table = [indexed_series(r) for r in rows] if mode == "indexed" else rows
            d = resolve_domain(drawn, mode, table=table)
            ys = [y_position(v, d, geom) for v in drawn.present]
            vals.append(max(ys) - min(ys))
        extents[mode] = vals
    for i, s in enumerate(rows):
        print(
            "   %-26s %10.2f %10.2f %10.2f"
            % (s.label[:26], extents["per_row"][i], extents["shared"][i], extents["indexed"][i])
        )
    print("   " + RULE[:60])
    small, big = extents["shared"][0], extents["shared"][n_same - 1]
    print(
        "   shared is honest and nearly unreadable: the smallest row gets %.2f px of vertical\n"
        "   extent against the largest row's %.2f, on a 20 px cell. indexed compares percent\n"
        "   change instead - every row gets %.2f px and the comparison still means something."
        % (small, big, extents["indexed"][0])
    )
    out["extents"] = extents
    out["shared_small_px"] = small
    out["shared_big_px"] = big
    return out


# ======================================================================================
# 2. The cell's aspect ratio, not the data, sets the perceived trend
# ======================================================================================


def experiment_aspect_ratio(seed: int = 3) -> Dict[str, object]:
    """The same series at five widths. Cleveland: slope judgement peaks near 45 degrees."""
    rng = np.random.default_rng(seed)
    n = 24
    t = np.arange(n)
    values = 100 + 0.9 * t + rng.normal(0, 2.2, n)
    s = Series("weekly_signups", list(map(float, values)))

    print("\n\n2. SAME SERIES, SIX CELL WIDTHS - what does the reader see?\n")
    print(
        "%-10s %8s %14s %14s %14s %s"
        % ("width", "height", "aspect", "median slope", "max slope", "reads as")
    )
    print(RULE)
    rows = []
    for w in (15.0, 30.0, 60.0, 120.0, 240.0, 480.0):
        geom = Geometry(width=w, height=20.0)
        d = resolve_domain(s, "per_row")
        p = build_path(s, d, geom)
        med = banking_deg(p.points)
        mx = max(abs(x) for x in _slopes(p.points))
        verdict = (
            "flat / stable" if med < 15 else
            "gentle rise" if med < 35 else
            "clear trend" if med < 55 else
            "volatile / spiky"
        )
        rows.append({"width": w, "median_deg": med, "max_deg": mx, "reads_as": verdict})
        print("%-10.0f %8.0f %14.2f %13.1f° %13.1f° %s" % (w, 20, w / 20, med, mx, verdict))
    print(RULE)
    geom = Geometry(width=80.0, height=20.0)
    d = resolve_domain(s, "per_row")
    bw = banked_width(s, d, geom, target_deg=45.0)
    p45 = build_path(s, d, Geometry(width=bw, height=20.0))
    print(
        "banked to 45°: width = %.1f px (aspect %.2f), achieved median slope %.1f°"
        % (bw, bw / 20.0, banking_deg(p45.points))
    )
    print(
        "The underlying data never changed. Median rendered slope runs %.0f° down to %.0f° -\n"
        "%r and %r are the same %d numbers in two column widths.\n"
        "Note the max-slope column: even at 480 px one segment still reaches %.0f°, so the\n"
        "spikiest bit of a 'flat' sparkline is doing most of the talking."
        % (
            rows[0]["median_deg"], rows[-1]["median_deg"],
            rows[0]["reads_as"], rows[-1]["reads_as"], n, rows[-1]["max_deg"],
        )
    )
    return {"rows": rows, "banked_width": bw, "banked_deg": banking_deg(p45.points)}


def _slopes(points: Sequence[Tuple[float, float]]) -> List[float]:
    from sparkline import segment_slopes_deg

    return segment_slopes_deg(points) or [0.0]


# ======================================================================================
# 3. A bridged gap is invented data drawn at full stroke weight
# ======================================================================================


def experiment_gaps() -> Dict[str, object]:
    """Hide a known dip, bridge it, and measure the pixels the renderer made up."""
    truth = [120.0, 124.0, 130.0, 240.0, 260.0, 205.0, 150.0, 128.0, 126.0, 124.0]
    observed: List[Optional[float]] = list(truth)
    for i in (3, 4, 5):  # the incident window: values exist, they were not collected
        observed[i] = None

    geom = Geometry(width=100.0, height=24.0)
    s_true = Series("p95 (truth)", [float(v) for v in truth])
    s_obs = Series("p95 (observed)", observed)

    # Both drawn against the SAME domain, so the comparison is only about the gap.
    d = resolve_domain(s_true, "per_row")
    p_true = build_path(s_true, d, geom)
    p_broken = build_path(s_obs, d, geom, bridge_gaps=False)
    p_bridged = build_path(s_obs, d, geom, bridge_gaps=True)

    # For each x where the bridge draws, how far is it from where the truth was?
    xs_true = {round(x, 3): y for x, y in p_true.points}
    dev = []
    for x, y in _interp_samples(p_bridged.points, sorted(xs_true)):
        if round(x, 3) in xs_true:
            dev.append(abs(y - xs_true[round(x, 3)]))
    max_dev = max(dev) if dev else 0.0

    ink_total = _path_length(p_bridged.points)
    ink_invented = _bridge_length(p_obs_runs(s_obs, d, geom))
    print("\n\n3. A GAP DRAWN THROUGH - three renderings of one incident\n")
    print("%-22s %10s %12s %14s %s" % ("rendering", "subpaths", "isolated", "bridged gaps", "shape"))
    print(RULE)
    print("%-22s %10d %12d %14d %s" % ("truth (all 10 pts)", p_true.n_subpaths, p_true.n_isolated, 0, "dip visible"))
    print("%-22s %10d %12d %14d %s" % ("observed, broken", p_broken.n_subpaths, p_broken.n_isolated, 0, "gap visible"))
    print("%-22s %10d %12d %14d %s" % ("observed, bridged", p_bridged.n_subpaths, p_bridged.n_isolated, p_bridged.bridged_gaps, "monotone rise"))
    print(RULE)
    print(
        "the bridge is %.1f px of the %.1f px drawn (%.0f%% of the ink is invented) and it\n"
        "misses the true path by up to %.1f px on a %.0f px tall cell (%.0f%% of the height)."
        % (
            ink_invented, ink_total, 100 * ink_invented / ink_total,
            max_dev, geom.height, 100 * max_dev / geom.height,
        )
    )
    print(
        "direction the reader infers over the window:  bridged = %s   truth = %s"
        % (
            "monotone UP",
            "UP then DOWN (peak at index 4, -%.0f%% by index 6)"
            % (100 * (truth[4] - truth[6]) / truth[4]),
        )
    )

    # And the single-observation case, which vanishes without the dot fallback.
    lone: List[Optional[float]] = [None] * 9
    lone[4] = 50.0
    s_lone = Series("one reading", lone)
    d2 = resolve_domain(Series("x", [0.0, 100.0]), "per_row")
    p_lone = build_path(s_lone, d2, geom)
    print(
        "\nisolated observation: %d subpath, %d dot, d=%r"
        % (p_lone.n_subpaths, p_lone.n_isolated, p_lone.d[:34] + "...")
    )
    print("   without the dot fallback a lone value strokes zero length and renders as nothing.")

    return {
        "max_dev_px": max_dev,
        "height_px": geom.height,
        "invented_frac": ink_invented / ink_total,
        "bridged_gaps": p_bridged.bridged_gaps,
        "broken_subpaths": p_broken.n_subpaths,
    }


def p_obs_runs(s: Series, d, geom) -> List[List[Tuple[float, float]]]:
    """The runs the broken renderer would produce - used to price the bridge segments."""
    from sparkline import x_positions

    xs = x_positions(s, geom)
    runs: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    for i, v in enumerate(s.values):
        if v is None:
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append((xs[i], y_position(float(v), d, geom)))
    if cur:
        runs.append(cur)
    return runs


def _bridge_length(runs: Sequence[Sequence[Tuple[float, float]]]) -> float:
    return sum(
        math.dist(runs[i][-1], runs[i + 1][0]) for i in range(len(runs) - 1)
    )


def _path_length(points: Sequence[Tuple[float, float]]) -> float:
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def _interp_samples(
    points: Sequence[Tuple[float, float]], xs: Sequence[float]
) -> List[Tuple[float, float]]:
    """Sample a polyline at given x positions."""
    out = []
    for x in xs:
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x0 <= x <= x1 and x1 > x0:
                out.append((x, y0 + (y1 - y0) * (x - x0) / (x1 - x0)))
                break
    return out


# ======================================================================================
# 4. Endpoint reading vs robust trend - and why neither is safe without the noise model
# ======================================================================================


def experiment_endpoint_vs_robust(trials: int = 4000, n: int = 24, seed: int = 7) -> Dict[str, object]:
    """How often does each trend reading recover the true direction?

    Run under three noise models, because the honest answer depends on which one you are in
    and a sparkline shows you neither:

      trend + iid noise   - endpoint reading uses 2 of n points; robust trend uses all n
      random walk + drift - last-minus-first is the efficient statistic; robust trend is not
      trend + one outlier - a single contaminated final point owns the endpoint reading
    """
    rng = np.random.default_rng(seed)
    # Tuned so the ENDPOINT reading lands near 70% in the first two models. That equalises
    # the difficulty across noise models, so the table isolates the estimator rather than the
    # signal-to-noise ratio. With a large drift both readings sit at 100% and prove nothing.
    drift = 0.10
    sigma = 3.0
    results: Dict[str, Dict[str, float]] = {}

    for model in ("trend+iid", "random_walk", "trend+outlier"):
        hit_end = hit_rob = 0
        for _ in range(trials):
            sign = 1.0 if rng.random() < 0.5 else -1.0
            t = np.arange(n)
            if model == "trend+iid":
                y = sign * drift * t + rng.normal(0, sigma, n)
            elif model == "random_walk":
                y = sign * drift * t + np.cumsum(rng.normal(0, sigma / math.sqrt(n), n))
            else:
                y = sign * drift * t + rng.normal(0, sigma * 0.35, n)
                y[-1] += -sign * 9.0 * sigma  # one bad final reading, against the trend
            s = Series("x", list(map(float, y)))
            hit_end += int(endpoint_direction(s) == (sign > 0) - (sign < 0))
            hit_rob += int(trend_direction(s) == (sign > 0) - (sign < 0))
        results[model] = {
            "endpoint": hit_end / trials,
            "robust": hit_rob / trials,
        }

    print("\n\n4. RECOVERING THE TRUE DIRECTION - %d trials, n=%d per series\n" % (trials, n))
    print("%-20s %18s %18s %14s" % ("noise model", "sign(last-first)", "Theil-Sen", "who wins"))
    print(RULE)
    for model, r in results.items():
        winner = "robust" if r["robust"] > r["endpoint"] + 0.005 else (
            "endpoint" if r["endpoint"] > r["robust"] + 0.005 else "tie"
        )
        print(
            "%-20s %17.1f%% %17.1f%% %14s"
            % (model, 100 * r["endpoint"], 100 * r["robust"], winner)
        )
    print(RULE)
    print(
        "Under trend+iid the endpoints throw away n-2 observations and lose %.0f points to a\n"
        "median-of-pairwise-slopes. Under a random walk last-minus-first is the efficient\n"
        "statistic and the robust reading gives back %.0f. Under a single contaminated final\n"
        "point the endpoint reading is %.0f%% accurate - it is not noisy, it is inverted.\n"
        "All three render as an %s in the same 80x20 cell, and nothing\n"
        "in the picture says which model you are in. That is the actual finding: a sparkline\n"
        "cannot tell you which trend estimator it is entitled to, so it should print one\n"
        "rather than imply one."
        % (
            100 * (results["trend+iid"]["robust"] - results["trend+iid"]["endpoint"]),
            100 * (results["random_walk"]["endpoint"] - results["random_walk"]["robust"]),
            100 * results["trend+outlier"]["endpoint"],
            "up-and-to-the-right line",
        )
    )
    return results


# ======================================================================================
# 5. Index spacing redraws an irregular series as a different series
# ======================================================================================


def experiment_time_axis() -> Dict[str, object]:
    """A monthly metric with a reporting gap, plotted by index and by time."""
    times = [0.0, 1.0, 2.0, 3.0, 9.0, 10.0]  # months; a six-month reporting hole
    values = [31.0, 33.0, 34.0, 36.0, 41.0, 42.0]
    s = Series("nps", [float(v) for v in values], times=times)
    geom = Geometry(width=90.0, height=22.0)
    d = resolve_domain(s, "per_row")

    p_time = build_path(s, d, geom, use_times=True)
    p_index = build_path(s, d, geom, use_times=False)

    from sparkline import segment_slopes_deg

    st, si = segment_slopes_deg(p_time.points), segment_slopes_deg(p_index.points)
    # The gap segment is index 3->4 in both.
    gap_t, gap_i = st[3], si[3]

    print("\n\n5. IRREGULAR TIME, PLOTTED TWO WAYS - the gap segment\n")
    print("%-18s %12s %14s %16s %16s" % ("x positions", "gap width px", "gap slope", "steepest seg", "median slope"))
    print(RULE)
    print(
        "%-18s %12.1f %13.1f° %15.1f° %15.1f°"
        % ("from `times`", p_time.points[4][0] - p_time.points[3][0], gap_t,
           max(abs(x) for x in st), banking_deg(p_time.points))
    )
    print(
        "%-18s %12.1f %13.1f° %15.1f° %15.1f°"
        % ("from index", p_index.points[4][0] - p_index.points[3][0], gap_i,
           max(abs(x) for x in si), banking_deg(p_index.points))
    )
    print(RULE)
    slope_time = theil_sen_slope(s, use_times=True)
    slope_index = theil_sen_slope(s, use_times=False)

    def rank(slopes: Sequence[float], i: int) -> str:
        order = sorted(range(len(slopes)), key=lambda k: abs(slopes[k]))
        pos = order.index(i)
        return "steepest" if pos == len(slopes) - 1 else (
            "shallowest" if pos == 0 else "%d of %d" % (pos + 1, len(slopes))
        )

    print(
        "The 6-month jump is drawn %.1fx wider on the time axis, so its slope falls from %.0f°\n"
        "to %.0f°: by index that segment is the %s of the chart, by time it is the %s.\n"
        "Trend per month: %.3f (time) vs %.3f (index) - reading positions off the index\n"
        "overstates the underlying rate by %.0f%%, because five of the six intervals are one\n"
        "month and the sixth is six."
        % (
            (p_time.points[4][0] - p_time.points[3][0]) / (p_index.points[4][0] - p_index.points[3][0]),
            gap_i, gap_t, rank(si, 3), rank(st, 3),
            slope_time, slope_index,
            100 * (slope_index / slope_time - 1),
        )
    )
    return {
        "gap_slope_time": gap_t,
        "gap_slope_index": gap_i,
        "slope_time": slope_time,
        "slope_index": slope_index,
        "rank_index": rank(si, 3),
        "rank_time": rank(st, 3),
    }


# ======================================================================================
# 6. SVG mechanics at 20 pixels tall
# ======================================================================================


def experiment_svg_mechanics(rows: int = 500) -> Dict[str, object]:
    """Clipping, stroke scaling, coordinate precision, and the page-weight budget."""
    s = Series("x", [10.0, 40.0, 25.0, 90.0, 55.0, 100.0, 70.0, 95.0])
    d = resolve_domain(s, "per_row")

    print("\n\n6. SVG MECHANICS\n")

    # (a) half-stroke clipping
    print("(a) a point at the top of the domain, with and without padding\n")
    print("    %-16s %8s %10s %12s %s" % ("stroke", "pad", "y at hi", "clipped px", "visible"))
    print("    " + RULE[:66])
    clip = []
    for stroke in (1.0, 1.25, 2.0, 3.0):
        g_pad = Geometry(width=80, height=20, stroke=stroke)
        y_pad = y_position(100.0, d, g_pad)
        lost = max(0.0, stroke / 2.0 - 0.0)  # with pad=0 the top half of the stroke exits
        clip.append({"stroke": stroke, "pad": g_pad.pad, "lost_px": lost})
        print(
            "    %-16.2f %8.3f %10.3f %12.3f %s"
            % (stroke, g_pad.pad, y_pad, lost, "%.0f%% of line thickness lost at pad=0" % (100 * lost / stroke))
        )
    print("    " + RULE[:66])
    print("    pad = stroke/2 costs %.0f%% of a 20px cell's height and is not optional." % (100 * 1.25 / 20))

    # (b) stroke scaling under a responsive viewBox
    print("\n(b) one 80x20 viewBox rendered into different column widths\n")
    print("    %-16s %12s %20s %20s" % ("rendered width", "scale", "stroke as drawn", "with non-scaling"))
    print("    " + RULE[:72])
    scaling = []
    for cw in (40.0, 80.0, 160.0, 320.0):
        k = cw / 80.0
        scaling.append({"cell_w": cw, "effective": 1.25 * k})
        print("    %-16.0f %12.2fx %19.2fpx %19.2fpx" % (cw, k, 1.25 * k, 1.25))
    print("    " + RULE[:72])
    print(
        "    the same 1.25px stroke renders %.2fpx to %.2fpx - an %.0fx difference in visual\n"
        "    weight from column width alone. vector-effect=\"non-scaling-stroke\" pins it."
        % (scaling[0]["effective"], scaling[-1]["effective"],
           scaling[-1]["effective"] / scaling[0]["effective"])
    )

    # (c) coordinate precision as a byte budget
    print("\n(c) coordinate precision, and where the bytes actually go\n")
    print(
        "    %-11s %8s %8s %14s %12s %s"
        % ("precision", "bytes", "d= only", "max err px", "x%d rows" % rows, "note")
    )
    print("    " + RULE[:78])
    geom = Geometry(width=80, height=20)
    exact = build_path(s, d, geom, precision=6)
    prec_rows = []
    for prec in (0, 1, 2, 3):
        svg = sparkline_svg(s, d, geom, precision=prec)
        p = build_path(s, d, geom, precision=prec)
        err = _max_coord_error(p.d, exact.d)
        kb = len(svg) * rows / 1024.0
        prec_rows.append(
            {"precision": prec, "bytes": len(svg), "d_bytes": len(p.d), "err_px": err, "kb": kb}
        )
        note = "invisible" if err < 0.5 else "visible kink" if err < 1.0 else "wrong"
        print(
            "    %-11d %8d %8d %14.3f %9.1f KB %s"
            % (prec, len(svg), len(p.d), err, kb, note)
        )
    print("    " + RULE[:78])
    best, worst = prec_rows[1], prec_rows[3]
    overhead = best["bytes"] - best["d_bytes"]
    print(
        "    precision=1 holds the error to %.2f px - sub-pixel on any display - and the whole\n"
        "    precision knob is worth only %.0f%% of the payload, because %d of %d bytes (%.0f%%)\n"
        "    are the wrapper: xmlns, viewBox, title, aria-label, stroke attributes."
        % (
            best["err_px"], 100 * (1 - best["bytes"] / worst["bytes"]),
            overhead, best["bytes"], 100 * overhead / best["bytes"],
        )
    )
    print(
        "    The real number is the total: %.0f KB of inline SVG for %d rows, which is larger\n"
        "    than most pages' entire HTML. Past a few hundred rows the fix is a shared <defs>\n"
        "    or server-side rasterisation, not another decimal place."
        % (best["kb"], rows)
    )

    # (d) escaping
    hostile = Series('Q1 "growth" & <margin>', [1.0, 2.0, 3.0])
    svg = sparkline_svg(hostile, resolve_domain(hostile, "per_row"), geom)
    ok = "<margin>" not in svg and "&lt;margin&gt;" in svg and svg.count("<title>") == 1
    print("\n(d) a row label taken from the data: %r" % hostile.label)
    print("    escaped into title/aria-label: %s" % ("yes" if ok else "NO - injection"))
    assert ok, "label escaping is broken"

    return {
        "clipping": clip,
        "scaling": scaling,
        "precision": prec_rows,
        "escaped": ok,
    }


def _max_coord_error(d_approx: str, d_exact: str) -> float:
    """Largest per-coordinate difference between two emitted path strings."""

    def nums(d: str) -> List[float]:
        out: List[float] = []
        for tok in d.replace("M", " ").replace("L", " ").replace("a", " ").split():
            for part in tok.split(","):
                try:
                    out.append(float(part))
                except ValueError:
                    pass
        return out

    a, b = nums(d_approx), nums(d_exact)
    if len(a) != len(b):
        return float("nan")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


# ======================================================================================


def run_all() -> Dict[str, object]:
    out = {}
    out["scale"] = experiment_per_row_scale()
    out["aspect"] = experiment_aspect_ratio()
    out["gaps"] = experiment_gaps()
    out["trend"] = experiment_endpoint_vs_robust()
    out["time"] = experiment_time_axis()
    out["svg"] = experiment_svg_mechanics()
    return out


if __name__ == "__main__":
    run_all()
