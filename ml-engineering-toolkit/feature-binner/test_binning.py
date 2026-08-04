"""Tests for the five guarantees. Run: python3 test_binning.py"""

from __future__ import annotations

import math

import numpy as np

from binning import (
    MISSING,
    NUMERIC,
    SENTINEL_NO_BUREAU,
    SPECIAL,
    audit,
    build_dataset,
    fit,
    iv_band,
    null_iv,
    psi,
    refit_counts,
    sparse_bin_warning,
)

DATA = build_dataset()
Y = DATA["y"]
TR, HO = DATA["train_idx"], DATA["holdout_idx"]
F = DATA["features"]

checks = 0
failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if condition:
        print(f"  PASS  {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


print("\n1. Separation: missing and sentinel values never enter a numeric range")
me = fit(F["months_employed"][TR], Y[TR], feature="months_employed")
missing_bins = [b for b in me.bins if b.kind == MISSING]
check("missing gets its own bin", len(missing_bins) == 1)
check(
    "missing bin holds every NaN",
    missing_bins[0].n == int(np.isnan(F["months_employed"][TR]).sum()),
)
check(
    "missingness is predictive here, so keeping it matters",
    abs(me.woe(missing_bins[0])) > 0.15,
    f"woe={me.woe(missing_bins[0]):.3f}",
)

ni = fit(
    F["n_inquiries"][TR], Y[TR], feature="n_inquiries", specials=(SENTINEL_NO_BUREAU,)
)
special_bins = [b for b in ni.bins if b.kind == SPECIAL]
check("sentinel gets its own bin", len(special_bins) == 1)
check(
    "the sentinel value routes to its own bin, not the lowest numeric range",
    ni.bins[ni.bin_index(SENTINEL_NO_BUREAU)].kind == SPECIAL,
    f"routed to {ni.bins[ni.bin_index(SENTINEL_NO_BUREAU)].label}",
)
check(
    "no sentinel row was counted inside a numeric bin",
    sum(b.n for b in ni.bins if b.kind == NUMERIC)
    == int((F["n_inquiries"][TR] != SENTINEL_NO_BUREAU).sum()),
)
check(
    "sentinel group is the riskiest, as planted",
    special_bins[0].event_rate > max(b.event_rate for b in ni.bins if b.kind == NUMERIC),
)

print("\n2. Smoothed WOE: no infinities, and the constant is explicit")
x = np.concatenate([np.zeros(60), np.ones(60)])
y_sep = np.concatenate([np.zeros(60, dtype=int), np.ones(60, dtype=int)])

# The event floor is what normally prevents a zero-event bin, so drop it to 0 here in
# order to actually exercise the log(0) path the smoothing exists for.
sep = fit(x, y_sep, min_bin_events=0, min_bin_share=0.01, max_bins=2, monotone=False)
check("a zero-event bin survives with the floor removed", len(sep.bins) == 2, f"{len(sep.bins)} bins")
check(
    "perfectly separating feature yields finite WOE",
    all(math.isfinite(sep.woe(b)) for b in sep.bins),
    f"{[sep.woe(b) for b in sep.bins]}",
)
check("and finite IV", math.isfinite(sep.iv) and sep.iv > 0, f"iv={sep.iv}")
low = fit(x, y_sep, min_bin_events=0, min_bin_share=0.01, max_bins=2, monotone=False, smoothing=0.01)
check(
    "smoothing constant changes IV (so it must be declared, not hidden)",
    abs(low.iv - sep.iv) > 1e-6,
    f"{low.iv:.4f} vs {sep.iv:.4f}",
)
# And with the floor in place, the same feature collapses instead - which is the point.
floored = fit(x, y_sep, min_bin_events=1, min_bin_share=0.01, max_bins=2, monotone=False)
check(
    "with an event floor, a zero-event bin is merged away rather than smoothed",
    len(floored.bins) == 1,
    f"{len(floored.bins)} bins",
)

print("\n3. Constraints: size floors are respected")
u = fit(F["utilization"][TR], Y[TR], feature="utilization", max_bins=5, min_bin_share=0.05,
        min_bin_events=20)
n_num = sum(b.n for b in u.numeric_bins)
check("max_bins respected", len(u.numeric_bins) <= 5, f"{len(u.numeric_bins)}")
check(
    "every numeric bin meets the population floor",
    all(b.n >= 0.05 * n_num for b in u.numeric_bins),
    f"{[b.n for b in u.numeric_bins]} of {n_num}",
)
check(
    "every numeric bin meets the event floor",
    all(b.events >= 20 and b.nonevents >= 20 for b in u.numeric_bins),
    f"{[(b.events, b.nonevents) for b in u.numeric_bins]}",
)
check("no sparse-bin warning at these settings", sparse_bin_warning(u) is None)

print("\n4. Monotonicity: enforced when asked, and its cost is reported")
check("utilization is monotone as fitted", u.is_monotone() is not None)
free_age = fit(F["age"][TR], Y[TR], feature="age", monotone=False)
forced_age = fit(F["age"][TR], Y[TR], feature="age", monotone=True)
check("age is genuinely non-monotone unconstrained", free_age.is_monotone() is None)
check("forcing monotonicity succeeds", forced_age.is_monotone() is not None)
check(
    "and costs real IV on a U-shaped feature",
    free_age.iv - forced_age.iv > 0.02,
    f"free={free_age.iv:.4f} forced={forced_age.iv:.4f}",
)
check("the cost is stated in the notes", any("cost" in n for n in forced_age.notes))

print("\n5. Frozen cut points: applying a scheme elsewhere does not refit it")
ho_scheme = refit_counts(u, F["utilization"][HO], Y[HO])
check("cut points identical", ho_scheme.cuts == u.cuts, f"{ho_scheme.cuts} vs {u.cuts}")
check("bin count identical", len(ho_scheme.bins) == len(u.bins))
check(
    "all holdout rows land somewhere",
    sum(b.n for b in ho_scheme.bins) == len(HO),
    f"{sum(b.n for b in ho_scheme.bins)} vs {len(HO)}",
)
check("holdout IV differs from train IV", abs(ho_scheme.iv - u.iv) > 0)

print("\n6. Permutation null: calibrated, and it catches what IV bands do not")
noise_audit = audit(F["noise"][TR], Y[TR], F["noise"][HO], Y[HO], feature="noise")
check("pure noise is dropped", noise_audit.verdict.startswith("DROP"), noise_audit.verdict)
util_audit = audit(F["utilization"][TR], Y[TR], F["utilization"][HO], Y[HO], feature="utilization")
check("a real feature is not dropped", not util_audit.verdict.startswith("DROP"), util_audit.verdict)
check(
    "real feature clears its own null by a wide margin",
    util_audit.excess_iv > 20 * abs(noise_audit.excess_iv) or noise_audit.excess_iv <= 0,
    f"util excess={util_audit.excess_iv:.4f} noise excess={noise_audit.excess_iv:.4f}",
)

rng = np.random.default_rng(7)
false_positives = 0
trials = 20
for i in range(trials):
    xk = rng.normal(0, 1, 3000)
    yk = Y[TR][:3000]
    scheme = fit(xk, yk, max_bins=6, min_bin_events=20, min_bin_share=0.05)
    nulls = null_iv(xk, yk, n_permutations=40, seed=i, max_bins=6, min_bin_events=20,
                    min_bin_share=0.05)
    p = (np.sum(nulls >= scheme.iv) + 1) / (len(nulls) + 1)
    false_positives += p <= 0.05
check(
    "false-positive rate near nominal 5%",
    false_positives <= 4,
    f"{false_positives}/{trials} pure-noise columns flagged significant",
)

print("\n7. Raw IV inflates with bin count; that is the thing being guarded against")
xn = rng.normal(0, 1, 500)
yn = Y[TR][:500]
tight = fit(xn, yn, max_bins=6, min_bin_events=20, min_bin_share=0.05)
loose = fit(xn, yn, max_bins=20, min_bin_events=1, min_bin_share=0.002, monotone=False)
check("loose settings produce more bins", len(loose.bins) > len(tight.bins))
check(
    "and a much larger IV on the same noise column",
    loose.iv > 3 * tight.iv,
    f"tight={tight.iv:.4f} loose={loose.iv:.4f}",
)
check("loose scheme raises a sparse-bin warning", sparse_bin_warning(loose) is not None)

print("\n8. Transform and PSI")
woe_values = u.transform(F["utilization"][HO])
check("transform returns one WOE per row", len(woe_values) == len(HO))
check("transform output is finite", bool(np.isfinite(woe_values).all()))
check(
    "transform matches the bin table",
    abs(woe_values[0] - u.woe(u.bins[u.bin_index(F["utilization"][HO][0])])) < 1e-12,
)
h1 = F["utilization"][DATA["period"] == "H1"]
h2 = F["utilization"][DATA["period"] == "H2"]
scheme_h1 = fit(h1, Y[DATA["period"] == "H1"], feature="utilization")
drift = psi(scheme_h1, refit_counts(scheme_h1, h2, Y[DATA["period"] == "H2"]))
check("PSI detects the planted utilization drift", drift > 0.10, f"psi={drift:.4f}")
check("PSI against itself is ~0", psi(scheme_h1, scheme_h1) < 1e-9)

print("\n9. Determinism and bands")
check("same fit twice", fit(F["income"][TR], Y[TR]).cuts == fit(F["income"][TR], Y[TR]).cuts)
check("same dataset twice", bool((build_dataset()["y"] == build_dataset()["y"]).all()))
check("band boundaries", iv_band(0.01) == "unpredictive" and iv_band(0.2) == "medium")

print(f"\n{checks - len(failures)}/{checks} checks passed")
if failures:
    raise SystemExit("FAILED:\n" + "\n".join(failures))
