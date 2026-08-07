"""Tests for the sparkline core. Run: python3 test_sparkline.py

These hold the module to the specific claims the README makes, including the ones that are
easy to regress silently: half-stroke padding, gap breaks, out-of-domain clamping, and the
fact that a shared domain refuses to be computed without the table.
"""

from __future__ import annotations

import math

from sparkline import (
    COMPARABLE_MODES,
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
    segment_slopes_deg,
    sparkline_svg,
    theil_sen_slope,
    trend_direction,
    x_positions,
    y_position,
)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  PASS  %s" % name)
    else:
        FAIL += 1
        print("  FAIL  %s   %s" % (name, detail))


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# --------------------------------------------------------------------------------------
print("\ndomain resolution")
# --------------------------------------------------------------------------------------

s = Series("a", [10.0, 20.0, 30.0])
d = resolve_domain(s, "per_row")
check("per_row uses the row's own min/max", (d.lo, d.hi) == (10.0, 30.0), str(d))

table = [Series("a", [10.0, 20.0]), Series("b", [100.0, 400.0])]
d = resolve_domain(table[0], "shared", table=table)
check("shared spans the whole table", (d.lo, d.hi) == (10.0, 400.0), str(d))

check(
    "shared REFUSES to fall back to per_row without the table",
    raises(lambda: resolve_domain(table[0], "shared"), ValueError),
)

d = resolve_domain(Series("a", [5.0, 9.0]), "shared_zero", table=[Series("a", [5.0, 9.0])])
check("shared_zero forces 0 into the domain", d.lo == 0.0 and d.hi == 9.0, str(d))

d = resolve_domain(Series("flat", [7.0, 7.0, 7.0]), "per_row")
check("a constant row is flagged degenerate, not expanded", d.degenerate and d.span == 0.0, str(d))

d = resolve_domain(Series("empty", [None, None]), "per_row")
check("all-missing row yields a degenerate default domain", d.degenerate, str(d))

check("unknown mode raises", raises(lambda: resolve_domain(s, "nonsense"), ValueError))
check("per_row is not in COMPARABLE_MODES", "per_row" not in COMPARABLE_MODES)

# --------------------------------------------------------------------------------------
print("\nthe headline claim: per_row discards level")
# --------------------------------------------------------------------------------------

geom = Geometry(width=80, height=20)
small = Series("small", [1.0, 2.0, 1.5, 4.0])
big = Series("big", [1000.0, 2000.0, 1500.0, 4000.0])
p_small = build_path(small, resolve_domain(small, "per_row"), geom)
p_big = build_path(big, resolve_domain(big, "per_row"), geom)
check(
    "1x and 1000x rows emit byte-identical paths under per_row",
    p_small.d == p_big.d,
    "%r vs %r" % (p_small.d, p_big.d),
)

tbl = [small, big]
p_small_sh = build_path(small, resolve_domain(small, "shared", table=tbl), geom)
p_big_sh = build_path(big, resolve_domain(big, "shared", table=tbl), geom)
check("a shared domain separates them", p_small_sh.d != p_big_sh.d)

# --------------------------------------------------------------------------------------
print("\ngeometry")
# --------------------------------------------------------------------------------------

g = Geometry(width=80, height=20, stroke=1.25)
check("pad is half the stroke so the line is not clipped", abs(g.pad - 0.625) < 1e-12, str(g.pad))
g_dot = Geometry(width=80, height=20, stroke=1.25, dot=2.5)
check("an endpoint marker widens pad to its full radius", g_dot.pad == 2.5, str(g_dot.pad))

d = resolve_domain(s, "per_row")
y_hi, y_lo = y_position(30.0, d, g), y_position(10.0, d, g)
check("the domain max sits at the TOP (small y)", y_hi < y_lo, "%.3f vs %.3f" % (y_hi, y_lo))
check("the top of the stroke stays inside the viewBox", y_hi >= g.stroke / 2 - 1e-9, str(y_hi))
check("the bottom of the stroke stays inside", y_lo <= g.height - g.stroke / 2 + 1e-9, str(y_lo))
check(
    "out-of-domain values are clamped, not drawn outside the cell",
    y_position(9999.0, d, g) == y_hi and y_position(-9999.0, d, g) == y_lo,
)
check(
    "a degenerate domain draws at mid-height",
    abs(y_position(7.0, resolve_domain(Series("f", [7.0, 7.0]), "per_row"), g) - g.height / 2) < 1e-9,
)

xs = x_positions(Series("a", [1.0, 2.0, 3.0, 4.0, 5.0]), g)
gaps = [round(xs[i + 1] - xs[i], 9) for i in range(len(xs) - 1)]
check("no times -> equal index spacing", len(set(gaps)) == 1, str(gaps))
check("first and last x sit exactly on the padding", abs(xs[0] - g.pad) < 1e-9
      and abs(xs[-1] - (g.width - g.pad)) < 1e-9, "%.3f %.3f" % (xs[0], xs[-1]))

irregular = Series("a", [1.0, 2.0, 3.0], times=[0.0, 1.0, 9.0])
xs = x_positions(irregular, g)
check(
    "times produce proportional spacing",
    abs((xs[2] - xs[1]) / (xs[1] - xs[0]) - 8.0) < 1e-9,
    str(xs),
)
check(
    "use_times=False collapses it back to index spacing",
    abs(x_positions(irregular, g, use_times=False)[1] - (g.pad + g.inner_w / 2)) < 1e-9,
)
check("a single point is centred", abs(x_positions(Series("a", [1.0]), g)[0] - (g.pad + g.inner_w / 2)) < 1e-9)

check(
    "Series rejects times of the wrong length",
    raises(lambda: Series("a", [1.0, 2.0], times=[0.0]), ValueError),
)
check(
    "Series rejects non-increasing times",
    raises(lambda: Series("a", [1.0, 2.0], times=[1.0, 1.0]), ValueError),
)

# --------------------------------------------------------------------------------------
print("\ngaps")
# --------------------------------------------------------------------------------------

gapped = Series("g", [1.0, 2.0, None, None, 5.0, 6.0])
d = resolve_domain(gapped, "per_row")
p = build_path(gapped, d, geom)
check("a gap ends the subpath", p.n_subpaths == 2, str(p.n_subpaths))
check("two subpaths mean two M commands", p.d.count("M") == 2, p.d)
check("nothing is reported as bridged", p.bridged_gaps == 0)

pb = build_path(gapped, d, geom, bridge_gaps=True)
check("bridge_gaps=True joins into one subpath", pb.n_subpaths == 1 and pb.d.count("M") == 1)
check("and it says so", pb.bridged_gaps == 1, str(pb.bridged_gaps))

lone = Series("l", [None, 3.0, None])
pl = build_path(lone, resolve_domain(Series("x", [0.0, 6.0]), "per_row"), geom)
check("an isolated observation becomes a dot, not nothing", pl.n_isolated == 1 and len(pl.d) > 10, pl.d)
check("the dot is inside the single d attribute", pl.d.count("M") == 1 and " a" in pl.d)

check("missing values are counted", gapped.n_missing == 2 and len(gapped.present) == 4)

# --------------------------------------------------------------------------------------
print("\nbanking and aspect ratio")
# --------------------------------------------------------------------------------------

zig = Series("z", [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
d = resolve_domain(zig, "per_row")
b_narrow = banking_deg(build_path(zig, d, Geometry(width=40, height=20)).points)
b_wide = banking_deg(build_path(zig, d, Geometry(width=160, height=20)).points)
check("widening the cell flattens the rendered slope", b_wide < b_narrow, "%.2f vs %.2f" % (b_wide, b_narrow))
_ratio = math.tan(math.radians(b_narrow)) / math.tan(math.radians(b_wide))
# The expected ratio is inner_w, not width: padding is a fixed cost that does not scale, so
# 40->160 px multiplies the DRAWABLE width by 158.75/38.75 = 4.097, not by 4. Asserting 4.0
# here would be asserting that the padding does not exist.
_expected = Geometry(width=160, height=20).inner_w / Geometry(width=40, height=20).inner_w
check(
    "tan(slope) scales exactly with the drawable width, padding included",
    abs(_ratio - _expected) < 0.005,
    "%.4f vs expected %.4f" % (_ratio, _expected),
)

g0 = Geometry(width=80, height=20)
bw = banked_width(zig, d, g0, target_deg=45.0)
achieved = banking_deg(build_path(zig, d, Geometry(width=bw, height=20)).points)
check("banked_width round-trips to the target angle", abs(achieved - 45.0) < 0.5, "%.2f" % achieved)
achieved30 = banking_deg(
    build_path(zig, d, Geometry(width=banked_width(zig, d, g0, 30.0), height=20)).points
)
check("banked_width honours a non-45 target too", abs(achieved30 - 30.0) < 0.5, "%.2f" % achieved30)

rising = build_path(Series("r", [0.0, 1.0]), resolve_domain(Series("r", [0.0, 1.0]), "per_row"), g0)
check("a rising series has a POSITIVE rendered slope", segment_slopes_deg(rising.points)[0] > 0)

# --------------------------------------------------------------------------------------
print("\ntrend readings")
# --------------------------------------------------------------------------------------

check("endpoint_direction reads last minus first", endpoint_direction(Series("a", [5.0, 1.0])) == -1)
check("a flat series has no endpoint direction", endpoint_direction(Series("a", [2.0, 2.0])) == 0)
check(
    "Theil-Sen recovers an exact line's slope",
    abs(theil_sen_slope(Series("a", [0.0, 3.0, 6.0, 9.0])) - 3.0) < 1e-12,
)
outlier = Series("a", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -900.0])
check(
    "one contaminated final point flips the endpoint reading but not Theil-Sen",
    endpoint_direction(outlier) == -1 and trend_direction(outlier) == 1,
    "%d %d" % (endpoint_direction(outlier), trend_direction(outlier)),
)
check(
    "Theil-Sen uses real time spacing when it has it",
    abs(theil_sen_slope(Series("a", [0.0, 10.0], times=[0.0, 5.0])) - 2.0) < 1e-12,
)

# --------------------------------------------------------------------------------------
print("\nindexing")
# --------------------------------------------------------------------------------------

ix = indexed_series(Series("a", [200.0, 250.0, None, 300.0]))
check("indexed_series rebases to 1.0", ix.values[0] == 1.0 and ix.values[1] == 1.25, str(ix.values))
check("and preserves gaps", ix.values[2] is None and len(ix.values) == 4)
check(
    "indexing refuses a zero first value instead of dividing by it",
    raises(lambda: indexed_series(Series("a", [0.0, 5.0])), ValueError),
)

# --------------------------------------------------------------------------------------
print("\nSVG emission")
# --------------------------------------------------------------------------------------

svg = sparkline_svg(s, resolve_domain(s, "per_row"), g0)
check("emits a single well-formed svg element", svg.startswith("<svg") and svg.endswith("</svg>"))
check("carries a viewBox", 'viewBox="0 0 80 20"' in svg, svg[:90])
check("stroke is pinned by default", 'vector-effect="non-scaling-stroke"' in svg)
check(
    "and can be unpinned explicitly",
    "vector-effect" not in sparkline_svg(s, resolve_domain(s, "per_row"), g0, Style(non_scaling_stroke=False)),
)
check("has an accessible name", 'role="img"' in svg and "aria-label=" in svg and "<title>" in svg)
resp = sparkline_svg(s, resolve_domain(s, "per_row"), g0, responsive=True)
check("responsive mode drops width/height attributes", " width=" not in resp and " height=" not in resp)
check("non-responsive keeps them", ' width="80"' in svg)

zero_dom = resolve_domain(Series("z", [-5.0, 5.0]), "per_row")
check(
    "show_zero draws a baseline when 0 is in the domain",
    "<line" in sparkline_svg(Series("z", [-5.0, 5.0]), zero_dom, g0, show_zero=True),
)
check(
    "and draws none when 0 is outside it",
    "<line" not in sparkline_svg(s, resolve_domain(s, "per_row"), g0, show_zero=True),
)
check(
    "an endpoint dot is emitted only when requested",
    "<circle" in sparkline_svg(s, resolve_domain(s, "per_row"), Geometry(dot=2.0))
    and "<circle" not in svg,
)

check("higher precision costs bytes", len(build_path(s, resolve_domain(s, "per_row"), g0, precision=3).d)
      > len(build_path(s, resolve_domain(s, "per_row"), g0, precision=0).d))

# --------------------------------------------------------------------------------------
print("\nescaping")
# --------------------------------------------------------------------------------------

check("escape neutralises tags", escape("<b>") == "&lt;b&gt;")
check("escape does the ampersand FIRST", escape("&lt;") == "&amp;lt;", escape("&lt;"))
hostile = Series('</title><script>x</script>', [1.0, 2.0])
hsvg = sparkline_svg(hostile, resolve_domain(hostile, "per_row"), g0)
check("a hostile label cannot break out of <title>", "<script>" not in hsvg and hsvg.count("<title>") == 1)
_attr = hsvg.split('aria-label="')[1].split('"')[0]  # value ends at the next real quote
check(
    "a hostile label cannot break out of the attribute",
    not any(c in _attr for c in '<>"'),
    repr(_attr),
)
check("the label survives escaped rather than stripped", "script" in _attr and "&lt;" in _attr)

# --------------------------------------------------------------------------------------
print("\nthe audit")
# --------------------------------------------------------------------------------------

v = audit_table([Series("a", [1.0, 2.0]), Series("b", [1000.0, 2000.0])], mode="per_row")
check("per_row is reported as NOT comparable", not v.comparable)
check("a 1000x level spread is warned about", any("span" in w for w in v.warnings), str(v.warnings))

v2 = audit_table([Series("a", [1.0, 2.0]), Series("b", [1000.0, 2000.0])], mode="shared")
check("shared is reported as comparable", v2.comparable)
check("and the level spread is a note, not a warning", not any("span" in w for w in v2.warnings))

v3 = audit_table([Series("g", [1.0, None, 3.0])], mode="shared", bridge_gaps=True)
check("bridging is warned about", any("interpolated" in w for w in v3.warnings), str(v3.warnings))
v4 = audit_table([Series("g", [1.0, None, 3.0])], mode="shared", bridge_gaps=False)
check("breaking is only a note", not v4.warnings and any("missing" in n for n in v4.notes))

v5 = audit_table([outlier], mode="shared")
check("endpoint/robust disagreement is surfaced", any("disagreeing" in w for w in v5.warnings), str(v5.warnings))
check("and the count reads as singular for one row", any(w.startswith("1 row has") for w in v5.warnings), str(v5.warnings))
# A 4-point series is too short to outvote one bad point: with n=4 the contaminated pair
# slopes are half of all pairs, so the median flips too. Needs enough clean pairs.
_outlier2 = Series("b", [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, -400.0])
check("Theil-Sen needs enough clean pairs to outvote the outlier",
      trend_direction(_outlier2) == 1 and endpoint_direction(_outlier2) == -1)
_v5b = audit_table([outlier, _outlier2], mode="shared")
check("and plural for two", any(w.startswith("2 rows have") for w in _v5b.warnings), str(_v5b.warnings))

v6 = audit_table([Series("c", [4.0] * 6)], mode="shared")
check("a constant row is noted as drawn flat", any("constant" in n for n in v6.notes), str(v6.notes))

v7 = audit_table(sample_table(), mode="per_row")
check("the sample table trips the per_row warning", not v7.comparable and v7.warnings)
check("verdict text leads with the comparability line", v7.text().splitlines()[0].startswith("NOT COMPARABLE"))

# --------------------------------------------------------------------------------------
print("\ntable rendering")
# --------------------------------------------------------------------------------------

html = render_table(sample_table(), mode="shared")
check("one <tr> per series plus the header", html.count("<tr>") == len(sample_table()) + 1, str(html.count("<tr>")))
check("every row gets an svg", html.count("<svg") == len(sample_table()))
check("the domain drawn against is printed", "domain drawn against" in html)
check("the verdict banner is inline", "banner" in html)
check(
    "indexed mode renders without raising on a table containing gaps",
    "<svg" in render_table(sample_table(), mode="indexed"),
)
check(
    "a hostile label is escaped in the table too",
    "&lt;script&gt;" in render_table([hostile], mode="per_row"),
)

# --------------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("%d passed, %d failed" % (PASS, FAIL))
print("=" * 60)
raise SystemExit(1 if FAIL else 0)
