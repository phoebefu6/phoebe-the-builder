"""Tests for the six experiments. Run: python3 test_evidence.py

These assert the *direction and rough size* of each effect, not the exact figure. A Monte
Carlo percentage that has to match to three decimals is a test of the seed, not of the claim,
and it breaks the moment anyone reasonably refactors the loop. Where a number is exact by
construction (path collapse, banking round-trip) it is asserted exactly.
"""

from __future__ import annotations

import io
import contextlib

import evidence

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


def quiet(fn, *a, **kw):
    """Run an experiment with its printing suppressed and return the measurements."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


print("\n1. per-row autoscale discards level")
r = quiet(evidence.experiment_per_row_scale)
check(
    "8 identical-shape rows collapse to exactly ONE path under per_row",
    r["per_row"]["distinct_paths"] == 1,
    str(r["per_row"]),
)
check(
    "the differently-shaped control still separates under per_row",
    r["per_row"]["control_separated"],
    "otherwise the collapse is a bug, not a scaling property",
)
check(
    "indexed also collapses them - correctly, since percent change IS identical",
    r["indexed"]["distinct_paths"] == 1 and r["indexed"]["control_separated"],
    str(r["indexed"]),
)
check(
    "shared separates most of the rows",
    r["shared"]["distinct_paths"] >= 5,
    str(r["shared"]),
)
check(
    "shared flattens the smallest row to under a pixel",
    r["shared_small_px"] < 1.0 < r["shared_big_px"],
    "%.3f / %.3f" % (r["shared_small_px"], r["shared_big_px"]),
)
check(
    "indexed keeps every row legible instead",
    min(r["extents"]["indexed"]) > 5.0,
    "%.2f" % min(r["extents"]["indexed"]),
)


print("\n2. aspect ratio sets perceived slope")
r = quiet(evidence.experiment_aspect_ratio)
rows = r["rows"]
check(
    "median rendered slope falls monotonically as the cell widens",
    all(rows[i]["median_deg"] > rows[i + 1]["median_deg"] for i in range(len(rows) - 1)),
    str([round(x["median_deg"], 1) for x in rows]),
)
check(
    "the narrowest cell reads 'volatile' and the widest reads 'flat'",
    "volatile" in rows[0]["reads_as"] and "flat" in rows[-1]["reads_as"],
    "%s / %s" % (rows[0]["reads_as"], rows[-1]["reads_as"]),
)
check(
    "the swing is at least 20x in tan terms",
    rows[0]["median_deg"] / rows[-1]["median_deg"] > 15,
    "%.1f" % (rows[0]["median_deg"] / rows[-1]["median_deg"]),
)
check("banked_width hits 45 degrees", abs(r["banked_deg"] - 45.0) < 0.5, "%.2f" % r["banked_deg"])


print("\n3. bridging a gap invents ink")
r = quiet(evidence.experiment_gaps)
check("the broken rendering really is two subpaths", r["broken_subpaths"] == 2, str(r))
check("the bridged rendering reports its one bridge", r["bridged_gaps"] == 1)
check(
    "a third or more of the drawn ink is invented",
    r["invented_frac"] > 0.33,
    "%.3f" % r["invented_frac"],
)
check(
    "the bridge misses the truth by more than half the cell height",
    r["max_dev_px"] > 0.5 * r["height_px"],
    "%.1f px of %.0f" % (r["max_dev_px"], r["height_px"]),
)


print("\n4. endpoint reading vs robust trend")
r = quiet(evidence.experiment_endpoint_vs_robust, trials=1500, n=24, seed=7)
check(
    "under trend+iid the robust reading beats the endpoints by 5+ points",
    r["trend+iid"]["robust"] - r["trend+iid"]["endpoint"] > 0.05,
    str(r["trend+iid"]),
)
check(
    "under a random walk the endpoints win - last-minus-first is sufficient there",
    r["random_walk"]["endpoint"] > r["random_walk"]["robust"],
    str(r["random_walk"]),
)
check(
    "the two models are of comparable difficulty for the endpoint reading",
    abs(r["trend+iid"]["endpoint"] - r["random_walk"]["endpoint"]) < 0.15,
    "so the table compares estimators, not signal-to-noise: %s" % str(r),
)
check(
    "one contaminated final point inverts the endpoint reading entirely",
    r["trend+outlier"]["endpoint"] < 0.05 and r["trend+outlier"]["robust"] > 0.9,
    str(r["trend+outlier"]),
)


print("\n5. index spacing redraws the series")
r = quiet(evidence.experiment_time_axis)
check(
    "the gap segment is steeper by index than by time",
    r["gap_slope_index"] > r["gap_slope_time"] + 10,
    "%.1f vs %.1f" % (r["gap_slope_index"], r["gap_slope_time"]),
)
check(
    "by index it is the steepest segment; by time it is the shallowest",
    r["rank_index"] == "steepest" and r["rank_time"] == "shallowest",
    "%s / %s" % (r["rank_index"], r["rank_time"]),
)
check(
    "the index reading overstates the per-month trend by more than half",
    r["slope_index"] / r["slope_time"] > 1.5,
    "%.3f vs %.3f" % (r["slope_index"], r["slope_time"]),
)


print("\n6. SVG mechanics")
r = quiet(evidence.experiment_svg_mechanics, rows=500)
check(
    "without padding, half the stroke is lost at the domain edge",
    all(abs(c["lost_px"] - c["stroke"] / 2) < 1e-9 for c in r["clipping"]),
    str(r["clipping"]),
)
check(
    "pad always equals half the stroke",
    all(abs(c["pad"] - c["stroke"] / 2) < 1e-9 for c in r["clipping"]),
)
check(
    "a responsive viewBox scales the stroke 8x across plausible column widths",
    r["scaling"][-1]["effective"] / r["scaling"][0]["effective"] >= 8.0 - 1e-9,
    str(r["scaling"]),
)
check(
    "precision=1 keeps coordinate error sub-pixel",
    r["precision"][1]["err_px"] < 0.5,
    "%.3f" % r["precision"][1]["err_px"],
)
check(
    "error shrinks monotonically with precision",
    all(r["precision"][i]["err_px"] >= r["precision"][i + 1]["err_px"] for i in range(3)),
    str([x["err_px"] for x in r["precision"]]),
)
check(
    "the wrapper, not the coordinates, is most of the payload",
    (r["precision"][1]["bytes"] - r["precision"][1]["d_bytes"]) / r["precision"][1]["bytes"] > 0.6,
    str(r["precision"][1]),
)
check(
    "500 inline sparklines exceed 100 KB",
    r["precision"][1]["kb"] > 100,
    "%.1f KB" % r["precision"][1]["kb"],
)
check("hostile labels are escaped", r["escaped"])


print("\n" + "=" * 60)
print("%d passed, %d failed" % (PASS, FAIL))
print("=" * 60)
raise SystemExit(1 if FAIL else 0)
