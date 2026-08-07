"""Sparkline generation with the scale, geometry and gap decisions made explicit.

A sparkline is a chart. Every failure mode of a chart applies to it - but at 60x18 pixels,
inside a table cell, nobody audits it. This module makes the four decisions that decide
whether a sparkline tells the truth into named, inspectable parameters:

  1. SCALE     - per-row autoscale destroys cross-row comparability (`resolve_domain`)
  2. GEOMETRY  - the cell's aspect ratio, not the data, sets perceived steepness (`banked_width`)
  3. GAPS      - a missing value drawn through is an invented trend (`build_path`)
  4. TIME      - unequal spacing plotted at equal x-intervals is a different series (`x_positions`)

Plus the SVG mechanics that silently corrupt output at this size: half-stroke clipping,
stroke scaling under a viewBox, coordinate precision as a byte budget, and label escaping.

No dependencies beyond the standard library. numpy/matplotlib are only used by the
experiment and chart modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Number = Optional[float]

# --------------------------------------------------------------------------------------
# Scale policies
# --------------------------------------------------------------------------------------

#: Every scale mode this module knows about, with the question each one answers.
SCALE_MODES: Dict[str, str] = {
    "per_row": "each row scaled to its own min/max - shape only, NOT comparable across rows",
    "shared": "one min/max across all rows - rows comparable, small rows may flatten",
    "shared_zero": "shared, with the domain forced to include 0 - level readable",
    "indexed": "each row divided by its own first value - comparable as percent change",
}

COMPARABLE_MODES = frozenset({"shared", "shared_zero", "indexed"})


@dataclass(frozen=True)
class Domain:
    """The vertical extent a sparkline is drawn against, and where it came from."""

    lo: float
    hi: float
    mode: str
    degenerate: bool = False

    @property
    def span(self) -> float:
        return self.hi - self.lo


@dataclass
class Series:
    """One row of a table: a label, values (None = missing), optional real time positions."""

    label: str
    values: Sequence[Number]
    times: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        if self.times is not None and len(self.times) != len(self.values):
            raise ValueError(
                "times has length %d but values has length %d"
                % (len(self.times), len(self.values))
            )
        if self.times is not None and any(
            self.times[i + 1] <= self.times[i] for i in range(len(self.times) - 1)
        ):
            raise ValueError("times must be strictly increasing")

    @property
    def present(self) -> List[float]:
        """Non-missing values, in order."""
        return [float(v) for v in self.values if v is not None]

    @property
    def n_missing(self) -> int:
        return sum(1 for v in self.values if v is None)


def resolve_domain(
    series: Series,
    mode: str = "per_row",
    table: Optional[Sequence[Series]] = None,
    pad_frac: float = 0.0,
) -> Domain:
    """Return the (lo, hi) this series should be drawn against under `mode`.

    `table` is required for the shared modes - that is the entire point of them, and
    silently falling back to per-row when it is missing is how a comparable table stops
    being comparable without anyone noticing. So it raises instead.
    """
    if mode not in SCALE_MODES:
        raise ValueError("unknown scale mode %r; expected one of %s" % (mode, sorted(SCALE_MODES)))

    if mode == "per_row":
        pool = series.present
    elif mode == "indexed":
        pool = [v for s in (table or [series]) for v in _index_to_first(s)]
    else:
        if table is None:
            raise ValueError(
                "mode=%r needs the whole table to compute a shared domain; "
                "pass table=[...] or use mode='per_row' and label it as non-comparable" % mode
            )
        pool = [v for s in table for v in s.present]

    if not pool:
        return Domain(0.0, 1.0, mode, degenerate=True)

    lo, hi = min(pool), max(pool)
    if mode == "shared_zero":
        lo, hi = min(lo, 0.0), max(hi, 0.0)

    if hi == lo:
        # A flat series has no extent. Manufacturing one by expanding the domain would
        # turn rounding noise into a visible trend, so it is flagged and drawn flat.
        return Domain(lo, hi, mode, degenerate=True)

    if pad_frac:
        pad = (hi - lo) * pad_frac
        lo, hi = lo - pad, hi + pad
    return Domain(lo, hi, mode)


def _index_to_first(series: Series) -> List[float]:
    """Values as a ratio to the first present value. Empty if that value is 0 or absent."""
    present = series.present
    if not present or present[0] == 0:
        return []
    base = present[0]
    return [float(v) / base for v in series.values if v is not None]


def indexed_series(series: Series) -> Series:
    """`series` rebased so its first present value is 1.0, preserving gaps and times."""
    present = series.present
    if not present or present[0] == 0:
        raise ValueError(
            "cannot index %r: first present value is %s"
            % (series.label, present[0] if present else "missing")
        )
    base = present[0]
    return Series(
        series.label,
        [None if v is None else float(v) / base for v in series.values],
        series.times,
    )


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Rendered pixel box. `pad` exists so a stroke at the domain edge is not half-clipped."""

    width: float = 80.0
    height: float = 20.0
    stroke: float = 1.25
    dot: float = 0.0  # endpoint marker radius; 0 = no marker

    @property
    def pad(self) -> float:
        # Half the stroke keeps the line inside the viewBox. A marker needs its full radius.
        return max(self.stroke / 2.0, self.dot)

    @property
    def inner_w(self) -> float:
        return max(self.width - 2 * self.pad, 1e-9)

    @property
    def inner_h(self) -> float:
        return max(self.height - 2 * self.pad, 1e-9)

    @property
    def aspect(self) -> float:
        return self.width / self.height


def x_positions(series: Series, geom: Geometry, use_times: bool = True) -> List[float]:
    """Horizontal pixel position per index.

    With `use_times=False` (or no times) points are equally spaced by index - which redraws
    an irregular series as a regular one. `experiment_time_axis` measures what that costs.
    """
    n = len(series.values)
    if n == 0:
        return []
    if n == 1:
        return [geom.pad + geom.inner_w / 2.0]

    if use_times and series.times is not None:
        t0, t1 = float(series.times[0]), float(series.times[-1])
        span = t1 - t0
        if span <= 0:
            return [geom.pad + geom.inner_w / 2.0] * n
        return [geom.pad + (float(t) - t0) / span * geom.inner_w for t in series.times]

    return [geom.pad + i / (n - 1) * geom.inner_w for i in range(n)]


def y_position(value: float, domain: Domain, geom: Geometry) -> float:
    """Vertical pixel position. SVG y grows downward, so the domain is flipped."""
    if domain.degenerate:
        return geom.pad + geom.inner_h / 2.0
    frac = (float(value) - domain.lo) / domain.span
    frac = min(max(frac, 0.0), 1.0)  # clamp: out-of-domain rows must not draw outside the cell
    return geom.pad + (1.0 - frac) * geom.inner_h


def segment_slopes_deg(points: Sequence[Tuple[float, float]]) -> List[float]:
    """Signed slope of each drawn segment, in degrees, in *rendered pixel space*.

    This is the quantity the reader's eye actually integrates. It depends on the cell's
    width and height, which is why the same data can read as flat or as a crisis.
    """
    out: List[float] = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx = x1 - x0
        if dx <= 0:
            continue
        out.append(math.degrees(math.atan2(-(y1 - y0), dx)))  # -dy: screen y is inverted
    return out


def banking_deg(points: Sequence[Tuple[float, float]]) -> float:
    """Median absolute rendered slope. Cleveland: slope judgement is best near 45 degrees."""
    slopes = [abs(s) for s in segment_slopes_deg(points)]
    return _median(slopes) if slopes else 0.0


def banked_width(series: Series, domain: Domain, geom: Geometry, target_deg: float = 45.0) -> float:
    """The width at which this series banks to `target_deg` at `geom`'s height.

    tan is monotone on [0, 90), so median(tan theta) == tan(median theta) exactly, and the
    solve is closed-form: horizontal spacing is proportional to width while vertical
    excursions are fixed by the height and the domain.
    """
    xs = x_positions(series, geom)
    ys = [None if v is None else y_position(v, domain, geom) for v in series.values]
    dys, dxs = [], []
    for i in range(len(ys) - 1):
        if ys[i] is None or ys[i + 1] is None:
            continue
        dys.append(abs(ys[i + 1] - ys[i]))
        dxs.append(xs[i + 1] - xs[i])
    if not dys or _median(dys) == 0:
        return geom.width
    # dx scales with inner_w, so the needed inner_w multiplies by the ratio of tangents.
    ratio = (_median(dys) / _median(dxs)) / math.tan(math.radians(target_deg))
    return geom.pad * 2 + geom.inner_w * ratio


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


# --------------------------------------------------------------------------------------
# Trend: what the endpoints say vs what the series does
# --------------------------------------------------------------------------------------


def endpoint_direction(series: Series) -> int:
    """sign(last - first). The statistic a sparkline invites, using 2 of n points."""
    present = series.present
    if len(present) < 2:
        return 0
    d = present[-1] - present[0]
    return (d > 0) - (d < 0)


def theil_sen_slope(series: Series, use_times: bool = True) -> float:
    """Median of all pairwise slopes - a breakdown-point-0.29 robust trend estimate."""
    pts = [
        (float(series.times[i]) if (use_times and series.times is not None) else float(i), float(v))
        for i, v in enumerate(series.values)
        if v is not None
    ]
    if len(pts) < 2:
        return 0.0
    slopes = [
        (pts[j][1] - pts[i][1]) / (pts[j][0] - pts[i][0])
        for i in range(len(pts))
        for j in range(i + 1, len(pts))
        if pts[j][0] != pts[i][0]
    ]
    return _median(slopes) if slopes else 0.0


def trend_direction(series: Series, use_times: bool = True) -> int:
    s = theil_sen_slope(series, use_times)
    return (s > 0) - (s < 0)


# --------------------------------------------------------------------------------------
# Path construction
# --------------------------------------------------------------------------------------


@dataclass
class Path:
    """A rendered path plus the accounting needed to describe what was drawn."""

    d: str
    points: List[Tuple[float, float]]
    n_subpaths: int
    n_isolated: int
    bridged_gaps: int = 0


def build_path(
    series: Series,
    domain: Domain,
    geom: Geometry,
    bridge_gaps: bool = False,
    precision: int = 2,
    use_times: bool = True,
) -> Path:
    """Turn a series into an SVG path `d`.

    With `bridge_gaps=False` (the default and the correct one) a missing value ends the
    subpath and the next present value starts a new `M`. The gap renders as a gap.

    With `bridge_gaps=True` - which is what every one-liner does, because it drops the NaNs
    and plots what is left - the two endpoints of the gap are joined by a straight line. That
    line is drawn at full stroke weight, indistinguishable from measured data, and it is
    always monotone, so a gap over a dip renders as a clean trend.

    A run of exactly one present value between two gaps cannot be stroked (a zero-length
    path draws nothing under `stroke-linecap: butt`), so it is emitted as a dot. Otherwise
    isolated observations vanish silently.
    """
    xs = x_positions(series, geom, use_times=use_times)
    pts: List[Optional[Tuple[float, float]]] = [
        None if v is None else (xs[i], y_position(float(v), domain, geom))
        for i, v in enumerate(series.values)
    ]

    runs: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    for p in pts:
        if p is None:
            if current:
                runs.append(current)
                current = []
        else:
            current.append(p)
    if current:
        runs.append(current)

    if bridge_gaps and len(runs) > 1:
        bridged = len(runs) - 1
        runs = [[p for run in runs for p in run]]
    else:
        bridged = 0

    parts: List[str] = []
    isolated = 0
    for run in runs:
        if len(run) == 1:
            x, y = run[0]
            r = max(geom.stroke / 2.0, 0.6)
            # A degenerate arc: two half-circles. Cheaper in bytes than a <circle> element
            # and it stays inside the single `d` attribute.
            parts.append(
                "M%s,%s a%s,%s 0 1,0 %s,0 a%s,%s 0 1,0 -%s,0"
                % (
                    _f(x - r, precision), _f(y, precision),
                    _f(r, precision), _f(r, precision), _f(2 * r, precision),
                    _f(r, precision), _f(r, precision), _f(2 * r, precision),
                )
            )
            isolated += 1
        else:
            head = "M%s,%s" % (_f(run[0][0], precision), _f(run[0][1], precision))
            tail = " ".join("%s,%s" % (_f(x, precision), _f(y, precision)) for x, y in run[1:])
            parts.append(head + " L" + tail)

    return Path(
        d=" ".join(parts),
        points=[p for run in runs for p in run],
        n_subpaths=len(runs),
        n_isolated=isolated,
        bridged_gaps=bridged,
    )


def _f(v: float, precision: int) -> str:
    """Format a coordinate. Trailing zeros and a bare `.0` are pure byte tax at this size."""
    s = ("%." + str(precision) + "f") % v
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


# --------------------------------------------------------------------------------------
# SVG emission
# --------------------------------------------------------------------------------------

_ESCAPES = (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&quot;"), ("'", "&#39;"))


def escape(text: object) -> str:
    """Escape for both attribute and text contexts. Labels come from data; data is hostile."""
    s = str(text)
    for a, b in _ESCAPES:
        s = s.replace(a, b)
    return s


@dataclass
class Style:
    color: str = "#1f4e79"
    band_color: str = "#e8eef5"
    dot_color: str = "#c0392b"
    zero_color: str = "#9aa5b1"
    non_scaling_stroke: bool = True


def sparkline_svg(
    series: Series,
    domain: Domain,
    geom: Geometry = Geometry(),
    style: Style = Style(),
    bridge_gaps: bool = False,
    precision: int = 2,
    use_times: bool = True,
    show_zero: bool = False,
    responsive: bool = False,
) -> str:
    """One inline SVG sparkline.

    `responsive=True` drops the width/height attributes so the SVG fills its cell. That is
    usually what you want in a table and it is also what makes `stroke-width` scale with the
    cell - a 1.25px line becomes 3px in a wide column and 0.4px in a narrow one, so the
    *same data* reads as bolder in one place than another. `vector-effect="non-scaling-stroke"`
    pins the stroke to device pixels and is the only reason responsive mode is safe.
    """
    path = build_path(
        series, domain, geom, bridge_gaps=bridge_gaps, precision=precision, use_times=use_times
    )

    dims = "" if responsive else ' width="%s" height="%s"' % (_f(geom.width, 2), _f(geom.height, 2))
    vec = ' vector-effect="non-scaling-stroke"' if style.non_scaling_stroke else ""
    title = escape("%s (%s)" % (series.label, domain.mode))

    body: List[str] = []
    if show_zero and not domain.degenerate and domain.lo <= 0.0 <= domain.hi:
        yz = _f(y_position(0.0, domain, geom), precision)
        body.append(
            '<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="0.5" '
            'stroke-dasharray="2 2"/>'
            % (_f(geom.pad, 2), yz, _f(geom.width - geom.pad, 2), yz, style.zero_color)
        )
    body.append(
        '<path d="%s" fill="none" stroke="%s" stroke-width="%s" stroke-linecap="round" '
        'stroke-linejoin="round"%s/>' % (path.d, style.color, _f(geom.stroke, 2), vec)
    )
    if geom.dot > 0 and path.points:
        x, y = path.points[-1]
        body.append(
            '<circle cx="%s" cy="%s" r="%s" fill="%s"/>'
            % (_f(x, precision), _f(y, precision), _f(geom.dot, 2), style.dot_color)
        )

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %s %s"%s '
        'role="img" aria-label="%s" preserveAspectRatio="none">'
        "<title>%s</title>%s</svg>"
        % (_f(geom.width, 2), _f(geom.height, 2), dims, title, title, "".join(body))
    )


# --------------------------------------------------------------------------------------
# Table rendering + the verdict
# --------------------------------------------------------------------------------------


@dataclass
class Verdict:
    """What the reader is entitled to conclude from the rendered table."""

    comparable: bool
    mode: str
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def text(self) -> str:
        head = (
            "COMPARABLE ACROSS ROWS (mode=%s)" % self.mode
            if self.comparable
            else "NOT COMPARABLE ACROSS ROWS (mode=%s) - shape only" % self.mode
        )
        lines = [head]
        lines += ["  WARNING: " + w for w in self.warnings]
        lines += ["  note:    " + n for n in self.notes]
        return "\n".join(lines)


def audit_table(
    table: Sequence[Series],
    mode: str = "per_row",
    geom: Geometry = Geometry(),
    bridge_gaps: bool = False,
) -> Verdict:
    """Check a table of series against the four decisions before anything is rendered."""
    comparable = mode in COMPARABLE_MODES
    v = Verdict(comparable=comparable, mode=mode)

    magnitudes = [max(s.present) for s in table if s.present]
    if magnitudes and min(magnitudes) > 0:
        spread = max(magnitudes) / min(magnitudes)
        if not comparable and spread >= 10:
            v.warnings.append(
                "row maxima span %.0fx (%.3g to %.3g) yet every row is autoscaled to fill the "
                "cell - two rows with identical shapes differ in level by that factor"
                % (spread, min(magnitudes), max(magnitudes))
            )
        if comparable and mode != "indexed" and spread >= 50:
            v.notes.append(
                "row maxima span %.0fx; under a shared domain the small rows will render as "
                "flat lines. mode='indexed' compares percent change instead" % spread
            )

    gapped = [s for s in table if s.n_missing]
    if gapped and bridge_gaps:
        v.warnings.append(
            "%d of %d rows %s missing values and bridge_gaps=True - the interpolated "
            "segments are drawn at full stroke weight and cannot be told from measurements"
            % (len(gapped), len(table), "has" if len(gapped) == 1 else "have")
        )
    elif gapped:
        v.notes.append(
            "%d of %d rows %s missing values, rendered as breaks in the line"
            % (len(gapped), len(table), "has" if len(gapped) == 1 else "have")
        )

    irregular = [s for s in table if s.times is not None and not _is_regular(s.times)]
    if irregular:
        v.notes.append(
            "%s irregular time spacing; positions are taken from `times`, not the index "
            "(pass use_times=False to see what index spacing would have drawn)"
            % _n(len(irregular), "row has", "rows have")
        )
    undated = [s for s in table if s.times is None and s.n_missing == 0]
    if len(undated) == len(table) and table:
        v.notes.append("no `times` given - x positions assume equally spaced observations")

    disagree = [s.label for s in table if _endpoint_disagrees(s)]
    if disagree:
        v.warnings.append(
            "%s sign(last - first) disagreeing with the robust trend: %s. The endpoints are "
            "the noisiest reading of a sparkline and the one it invites"
            % (_n(len(disagree), "row has", "rows have"),
               ", ".join(disagree[:4]) + (" ..." if len(disagree) > 4 else ""))
        )

    flat = [s.label for s in table if s.present and max(s.present) == min(s.present)]
    if flat:
        v.notes.append(
            "%s drawn as a flat mid-height line rather than expanded to fill the cell: %s"
            % (_n(len(flat), "constant row"), ", ".join(flat[:4]))
        )

    banks = []
    for s in table:
        if len(s.present) < 2:
            continue
        d = resolve_domain(s, mode if comparable else "per_row", table=table)
        p = build_path(s, d, geom)
        if len(p.points) >= 2:
            banks.append(banking_deg(p.points))
    if banks:
        med = _median(banks)
        if med < 15 or med > 75:
            v.notes.append(
                "median rendered slope is %.0f degrees at %.0fx%.0f; slope judgement is most "
                "accurate near 45. Try banked_width() for a cell that suits the data"
                % (med, geom.width, geom.height)
            )
    return v


def _n(count: int, noun: str, plural: Optional[str] = None) -> str:
    """"1 row" / "3 rows". Audit text is read by people; "1 rows" undercuts it."""
    return "%d %s" % (count, noun if count == 1 else (plural or noun + "s"))


def _is_regular(times: Sequence[float], tol: float = 1e-6) -> bool:
    if len(times) < 3:
        return True
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    return (max(gaps) - min(gaps)) <= tol * max(1.0, max(gaps))


def _endpoint_disagrees(series: Series) -> bool:
    e, t = endpoint_direction(series), trend_direction(series)
    return e != 0 and t != 0 and e != t


def render_table(
    table: Sequence[Series],
    mode: str = "shared",
    geom: Geometry = Geometry(),
    style: Style = Style(),
    bridge_gaps: bool = False,
    precision: int = 2,
    show_zero: bool = False,
    banner: bool = True,
) -> str:
    """A full HTML table with sparklines, a printed domain, and the audit verdict inline.

    The printed domain is not decoration. A shared-scale sparkline is unreadable without it -
    the reader has no way to know whether a flat line means "stable" or "small relative to
    row 3", and a per-row-scale table without the label is actively misleading.
    """
    verdict = audit_table(table, mode=mode, geom=geom, bridge_gaps=bridge_gaps)
    rows: List[str] = []
    for s in table:
        drawn = indexed_series(s) if mode == "indexed" and s.present and s.present[0] else s
        d = resolve_domain(drawn, mode, table=table)
        svg = sparkline_svg(
            drawn, d, geom, style, bridge_gaps=bridge_gaps, precision=precision,
            show_zero=show_zero,
        )
        last = drawn.present[-1] if drawn.present else float("nan")
        rows.append(
            "<tr><td>%s</td><td>%s</td><td class='num'>%.3g</td>"
            "<td class='num'>%.3g – %.3g</td></tr>"
            % (escape(s.label), svg, last, d.lo, d.hi)
        )

    banner_class = "ok" if verdict.comparable else "warn"
    detail = "".join("<li>%s</li>" % escape(x) for x in verdict.warnings + verdict.notes)
    banner_html = (
        "<div class='banner %s'><b>%s</b><ul>%s</ul></div>"
        % (banner_class, escape(verdict.text().splitlines()[0]), detail)
        if banner
        else ""
    )
    return (
        "<style>"
        "table.spark{border-collapse:collapse;font:13px -apple-system,system-ui,sans-serif}"
        "table.spark td,table.spark th{padding:4px 10px;border-bottom:1px solid #e5e7eb}"
        "table.spark td.num{text-align:right;font-variant-numeric:tabular-nums;color:#4b5563}"
        ".banner{font:12px ui-monospace,monospace;padding:6px 10px;margin-bottom:8px;"
        "border-radius:4px}.banner.ok{background:#ecfdf5;color:#065f46}"
        ".banner.warn{background:#fef2f2;color:#991b1b}"
        ".banner ul{margin:6px 0 0 16px;padding:0}"
        "</style>"
        "%s"
        "<table class='spark'><tr><th>row</th><th>trend</th><th>last</th>"
        "<th>domain drawn against</th></tr>%s</table>"
        % (banner_html, "".join(rows))
    )


# --------------------------------------------------------------------------------------
# Sample data
# --------------------------------------------------------------------------------------


def sample_table() -> List[Series]:
    """Six rows of plausible weekly SaaS telemetry, each carrying a different lesson.

    Deterministic - no RNG, so every number in the README is reproducible by reading this.
    """
    return [
        # Two rows with the SAME shape and a 1000x difference in level. Under per-row
        # autoscale they render as identical paths.
        Series("enterprise_mrr", [402.0, 404.0, 406.0, 408.0, 410.0, 412.0, 414.0, 416.0]),
        Series("self_serve_mrr", [0.402, 0.404, 0.406, 0.408, 0.410, 0.412, 0.414, 0.416]),
        # Endpoints say up; the series spent the whole quarter falling and bounced once.
        Series("active_seats", [980.0, 940.0, 900.0, 860.0, 820.0, 790.0, 760.0, 1010.0]),
        # An outage window: the values are unknown, not zero.
        Series("api_p95_ms", [120.0, 128.0, 133.0, None, None, 210.0, 205.0, 199.0]),
        # Irregular reporting - the last two points are 6 weeks apart, not 1.
        Series(
            "nps",
            [31.0, 33.0, 34.0, 36.0, 41.0],
            times=[0.0, 1.0, 2.0, 3.0, 9.0],
        ),
        # Constant. Any autoscale that expands the domain turns this into a story.
        Series("uptime_pct", [99.99] * 8),
    ]
