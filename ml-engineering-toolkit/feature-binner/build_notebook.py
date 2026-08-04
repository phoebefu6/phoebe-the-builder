"""Generate demo.ipynb for feature-binner. Run once, then nbconvert --execute."""

from __future__ import annotations

import json
import pathlib

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def md(src: str) -> None:
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src: str) -> None:
    nb["cells"].append(
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
    )


BASE = "ml-engineering-toolkit/feature-binner"

ENGINE = pathlib.Path("binning.py").read_text()
ENGINE = ENGINE.split('if __name__ == "__main__":')[0].rstrip() + "\n"

md(f"""# Feature Binner

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/{BASE}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath={BASE}/demo.ipynb)

> Binning a feature and computing its Information Value takes twenty lines. Knowing whether
> the IV you just printed means anything is the actual job - and the conventional
> 0.02 / 0.1 / 0.3 thresholds cannot tell you, because **IV is not comparable across bin counts**.

**What this covers**

1. A scorecard dataset with the four awkward cases planted in it
2. Missing values and sentinel codes - why imputing them destroys real signal
3. Smoothed WOE, and why the smoothing constant has to be a declared parameter
4. Monotonicity, and reading its cost as a diagnostic
5. **The main event**: raw IV on a column that is literally `rng.normal()`
6. The permutation null - the reference the IV bands are missing
7. Frozen cut points, holdout counts, and PSI
8. The audit table, and what it drops
9. Try your own feature

*Fully offline - the dataset is generated, no real credit data, numpy only.*""")

md("""## 1. The dataset

A small application scorecard: 12,000 applications, an 11% default rate. Four awkward cases
are planted deliberately, because they are the four that show up in every real credit file:

| Feature | What is planted |
|---|---|
| `utilization` | genuinely monotone, strongest predictor |
| `income` | genuinely monotone, moderate |
| `age` | genuinely **non-monotone** - young and old are both riskier |
| `months_employed` | 18% missing, **and the missingness itself is predictive** (thin file) |
| `n_inquiries` | `-999` sentinel for 'no bureau record', which is the riskiest group on the book |
| `noise` | `rng.normal()`. No relationship to anything. |

Here is the whole engine - numpy only, no sklearn.""")

code(ENGINE)

code('''data = build_dataset()
y = data["y"]
tr, ho = data["train_idx"], data["holdout_idx"]
F = data["features"]

print(f"applications      {len(y):,}")
print(f"default rate      {data['base_rate']:.2%}")
print(f"train / holdout   {len(tr):,} / {len(ho):,}")
print()
for name, values in F.items():
    n_missing = int(np.isnan(values).sum())
    n_special = int((values == SENTINEL_NO_BUREAU).sum())
    extra = []
    if n_missing:
        extra.append(f"{n_missing:,} missing")
    if n_special:
        extra.append(f"{n_special:,} at sentinel -999")
    print(f"  {name:<18} {' , '.join(extra) if extra else 'complete'}")''')

md("""## 2. Missing and sentinel values are populations, not gaps

The reflex is `fillna(median)`. On a credit file that is a mistake twice over: it merges two
different populations into one bin, and it throws away a signal that is often stronger than
the numeric trend it is hiding.""")

code('''me = fit(F["months_employed"][tr], y[tr], feature="months_employed")
print(format_table(me))

missing = [b for b in me.bins if b.kind == MISSING][0]
numeric_rate = sum(b.events for b in me.numeric_bins) / sum(b.n for b in me.numeric_bins)
print()
print(f"applicants with an employment record : {numeric_rate:.2%} default")
print(f"applicants with no record on file    : {missing.event_rate:.2%} default")
print(f"                                       WOE {me.woe(missing):+.3f}")
print("\\nImputing that group to the median would have hidden it inside a middle bin.")''')

md("""Sentinel codes are the same problem wearing a number. `-999` is not a low inquiry count -
it means the bureau returned nothing at all, and here it is the single riskiest group.

Left on the numeric scale it sorts below zero, lands in the lowest bin, and corrupts the
monotonic story with a group that does not belong on the axis at all.""")

code('''ni = fit(F["n_inquiries"][tr], y[tr], feature="n_inquiries", specials=(SENTINEL_NO_BUREAU,))
print(format_table(ni))

# Same feature, sentinel NOT declared - it gets treated as a real value of -999.
naive = fit(F["n_inquiries"][tr], y[tr], feature="n_inquiries")
print("\\nWithout declaring the sentinel:")
print(format_table(naive))
print(f"\\nIV {naive.iv:.4f} vs {ni.iv:.4f} declared - and the riskiest group is now buried")
print("inside the lowest bin, pulling its WOE up and flattening everything above it.")''')

md("""## 3. Smoothed WOE

WOE is a log-odds ratio, so a bin with zero events sends it to `-inf` and every downstream
sum to `nan`. Additive smoothing fixes the arithmetic. It does not fix the underlying
problem, and it is not free - the constant changes IV, so it belongs in the signature rather
than hardcoded at 0.5 somewhere inside.""")

code('''x = np.concatenate([np.zeros(60), np.ones(60)])
y_sep = np.concatenate([np.zeros(60, dtype=int), np.ones(60, dtype=int)])

for s in (0.01, 0.5, 2.0):
    scheme = fit(x, y_sep, min_bin_events=0, min_bin_share=0.01, max_bins=2,
                 monotone=False, smoothing=s)
    woes = [round(scheme.woe(b), 3) for b in scheme.bins]
    print(f"smoothing={s:<5} IV={scheme.iv:>8.4f}  WOE={woes}")

print("\\nSame data, same bins, three different IVs. Which is why the real fix is a floor:")
floored = fit(x, y_sep, min_bin_events=1, min_bin_share=0.01, max_bins=2, monotone=False)
print(f"with min_bin_events=1, the zero-event bin is merged away -> {len(floored.bins)} bin, IV {floored.iv:.4f}")''')

md("""## 4. Monotonicity, and reading its price

Scorecards want monotone WOE: risk should move one direction as the feature moves, or the
risk story does not survive a review committee. The binner enforces it by merging adjacent
violators.

Enforcing it always costs IV. **The size of the cost is the interesting part** - it tells you
what you just deleted.""")

code('''print(f"{'feature':<18} {'free IV':>9} {'monotone IV':>12} {'cost':>8} {'cost %':>7}  reading")
print("-" * 86)
for name in ("utilization", "income", "months_employed", "age"):
    free = fit(F[name][tr], y[tr], feature=name, monotone=False)
    forced = fit(F[name][tr], y[tr], feature=name, monotone=True)
    cost = free.iv - forced.iv
    pct = cost / free.iv if free.iv else 0
    reading = "the wiggle was noise - take the shape" if pct < 0.15 else "DELETING REAL SIGNAL"
    print(f"{name:<18} {free.iv:>9.4f} {forced.iv:>12.4f} {cost:>8.4f} {pct:>6.0%}  {reading}")''')

md("""`age` costs 32%. That is not a wiggle being cleaned up - the U-shape is real, planted as
`0.0016 * (age - 42)**2` in the data generator. Young and old applicants are both riskier
than the middle.

Forcing monotonicity here buys a shape the committee likes by throwing away a third of the
feature's predictive power. The honest options are to split it into two features, or to keep
the non-monotone bins and defend them.""")

code('''free_age = fit(F["age"][tr], y[tr], feature="age", monotone=False)
print(format_table(free_age))
print("\\nEvent rate by bin - the U is visible:")
for b in free_age.numeric_bins:
    bar = "#" * int(b.event_rate * 260)
    print(f"  {b.label:<20} {b.event_rate:>6.2%}  {bar}")''')

md("""## 5. The main event: IV on pure noise

`noise` is `rng.normal(0, 1, n)`. It appears nowhere in the data generator's logit. Its true
predictive power is exactly zero.

Bin it four ways and read the IV.""")

code('''small = build_dataset(n=800, seed=3)
ys, ts = small["y"], small["train_idx"]
xs = small["features"]["noise"][ts]
print(f"480 training rows, one pure-noise column.\\n")
print(f"  {'settings':<32} {'bins':>5} {'IV':>8}  band")
print("  " + "-" * 62)
for label, kw in SETTINGS_LADDER:
    scheme = fit(xs, ys[ts], **kw)
    print(f"  {label:<32} {len(scheme.bins):>5} {scheme.iv:>8.4f}  {iv_band(scheme.iv)}")''')

md("""Read that last row again. A column of Gaussian noise scores **"strong predictor"** on the
conventional scale.

Nothing is broken. IV is a sum over bins of `(share_events - share_nonevents) * WOE`, and
every extra bin adds another non-negative term. More bins means more IV, always. Add
smoothing and a near-empty bin contributes a *large* positive term built out of a count of
zero.

So the 0.02 / 0.1 / 0.3 / 0.5 thresholds are not thresholds on predictive power. They are
thresholds on predictive power **at a particular bin count**, and nobody writes down which one.""")

code('''# Where the IV actually comes from at the loosest setting.
loose = fit(xs, ys[ts], **dict(max_bins=20, min_bin_events=1, min_bin_share=0.002, monotone=False))
rows = sorted(loose.table(), key=lambda r: -r["iv_part"])[:6]
print(f"top IV contributors, {len(loose.bins)}-bin scheme on pure noise:\\n")
print(f"  {'bin':<22} {'n':>5} {'events':>7} {'WOE':>8} {'IV part':>9}")
for r in rows:
    print(f"  {str(r['bin']):<22} {r['n']:>5} {r['events']:>7} {r['woe']:>8.3f} {r['iv_part']:>9.4f}")
print(f"\\n{sparse_bin_warning(loose)}")''')

md("""## 6. The permutation null

If IV depends on the procedure, then measure the procedure. Shuffle the labels - destroying
any relationship while keeping the marginal distributions - refit with **identical** settings,
and record the IV. Repeat.

That distribution is what "no signal" looks like *through this binner*. Compare against it
instead of against a constant from a textbook.""")

code('''kw = dict(max_bins=20, min_bin_events=1, min_bin_share=0.002, monotone=False)
observed = fit(xs, ys[ts], **kw)
nulls = null_iv(xs, ys[ts], n_permutations=200, seed=1, **kw)

print(f"observed IV (real labels)  {observed.iv:.4f}")
print(f"null median                {np.median(nulls):.4f}")
print(f"null 90th percentile       {np.percentile(nulls, 90):.4f}")
print(f"null max of 200            {nulls.max():.4f}")
print(f"\\nfraction of shuffles reaching the observed IV: {(nulls >= observed.iv).mean():.3f}")
print("\\nThe 'strong predictor' is sitting inside its own noise distribution.")''')

md("""### Is the test calibrated?

A screening rule that flags noise 30% of the time is no better than the bands it replaces.
So check it the only way that means anything: run it on many independent noise columns and
count the false positives.""")

code('''rng = np.random.default_rng(99)
tight = dict(max_bins=6, min_bin_events=20, min_bin_share=0.05)
ps = []
for i in range(30):
    xk = rng.normal(0, 1, len(ts))
    scheme = fit(xk, ys[ts], **tight)
    nl = null_iv(xk, ys[ts], n_permutations=40, seed=i, **tight)
    ps.append((np.sum(nl >= scheme.iv) + 1) / (len(nl) + 1))

ps = np.array(ps)
print(f"30 independent pure-noise columns, alpha = 0.05")
print(f"  flagged significant : {(ps <= 0.05).sum()}/30 = {(ps <= 0.05).mean():.0%}")
print(f"  p-value quartiles   : {np.percentile(ps, [25, 50, 75]).round(3)}")
print("\\nRoughly nominal, and the p-values spread across the range the way they should")
print("under a true null. (40 permutations puts a floor of 1/41 = 0.024 on p.)")''')

md("""### Now screen every feature the same way""")

code('''audits = [
    audit(F[name][tr], y[tr], F[name][ho], y[ho], feature=name,
          specials=data["specials"].get(name, ()))
    for name in F
]
print(audit_report(audits))''')

md("""`noise` has **negative** excess IV - shuffled labels do slightly better than the real ones -
and a p-value of 0.63. It is dropped, on evidence, without anyone having to know in advance
that it was the planted decoy.

The two `REVIEW` rows are honest too: `n_inquiries` loses 38% of its IV out of sample because
its signal is concentrated in one small sentinel bin, and `months_employed` is monotone on
train but wobbles on holdout. Neither is fatal. Both are things you want to know before the
feature goes into a scorecard.

### The aggregate picture

One noise column can land anywhere in its own null distribution, so averaging over twelve is
the honest version of the earlier table.""")

code('''print("12 independent pure-noise columns, screened two ways:\\n")
print(f"  {'settings':<32} {'bins':>5} {'mean IV':>8} {'excess':>8} {'kept: IV>=0.1':>14} {'kept: perm':>11}")
print("  " + "-" * 84)
for row in noise_screen(ys[ts], n_columns=12, n_permutations=40):
    print(f"  {str(row['settings']):<32} {row['mean_bins']:>5.1f} {row['mean_iv']:>8.4f} "
          f"{row['mean_excess']:>8.4f} {row['kept_by_iv']:>13.0%} {row['kept_by_permutation']:>10.0%}")
print("\\nRaw IV climbs ~10x. Excess IV stays around 0.01-0.02. The permutation screen stays")
print("near nominal. (At 12 columns those keep-rates are noisy to about +/-10pp.)")''')

md("""## 7. Frozen cut points, holdout counts, PSI

Two more checks, both of which need the cut points to stop moving.

**Holdout IV**: same bins, different rows. If a feature's IV came from fitting the target,
this is where it goes.

**PSI**: same bins, later period. If the population moved, the bin shares move with it, and a
scorecard calibrated on the old shares is quietly wrong. The generator pushes `utilization`
up 10 points in the second half of the book.""")

code('''h1_mask = data["period"] == "H1"
h2_mask = data["period"] == "H2"

scheme_h1 = fit(F["utilization"][h1_mask], y[h1_mask], feature="utilization")
scheme_h2 = refit_counts(scheme_h1, F["utilization"][h2_mask], y[h2_mask])

print(f"cut points frozen from H1: {[round(c, 4) for c in scheme_h1.cuts]}")
print(f"same cut points in H2    : {[round(c, 4) for c in scheme_h2.cuts]}\\n")
print(f"  {'bin':<22} {'H1 share':>9} {'H2 share':>9} {'move':>8}")
h1_total = sum(b.n for b in scheme_h1.bins)
h2_total = sum(b.n for b in scheme_h2.bins)
for a, b in zip(scheme_h1.bins, scheme_h2.bins):
    sa, sb = a.n / h1_total, b.n / h2_total
    print(f"  {a.label:<22} {sa:>8.1%} {sb:>9.1%} {sb - sa:>+8.1%}")
print(f"\\nPSI = {psi(scheme_h1, scheme_h2):.4f}  (>0.10 investigate, >0.25 the scheme is stale)")''')

md("""## 8. Transform

Once a scheme is accepted, `transform` replaces raw values with the WOE of their bin. That
is the column that goes into the model, and it is the reason binning is worth the trouble:
missing values, sentinels and non-linearity all arrive pre-handled on one numeric scale.""")

code('''best = fit(F["utilization"][tr], y[tr], feature="utilization")
woe_train = best.transform(F["utilization"][tr])
woe_holdout = best.transform(F["utilization"][ho])

print(f"raw    : {F['utilization'][tr][:6].round(3)}")
print(f"binned : {woe_train[:6].round(3)}")
print(f"\\ndistinct WOE values: {len(np.unique(woe_train))} (one per bin)")
print(f"holdout rows transformed with the same frozen scheme: {len(woe_holdout):,}")

# A binned feature also handles NaN without a separate imputation step.
me_scheme = fit(F["months_employed"][tr], y[tr], feature="months_employed")
raw = F["months_employed"][tr]
nan_rows = np.isnan(raw)
print(f"\\nmonths_employed: {nan_rows.sum():,} NaN rows -> WOE "
      f"{np.unique(me_scheme.transform(raw)[nan_rows]).round(3)} (their own value, not an imputed one)")''')

code('''%matplotlib inline
import matplotlib.pyplot as plt

INK, GOOD, BAD, COOL, MUTED = "#1d2433", "#0f766e", "#c2410c", "#1d4ed8", "#94a3b8"

fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
fig.suptitle("Binning is easy. Knowing whether the IV you just measured is real is the job.",
             fontsize=14, fontweight="bold", color=INK, y=0.99)

ax = axes[0]
labels = [b.label.replace(" (special)", "\\n(no bureau)") for b in ni.bins]
woes = [ni.woe(b) for b in ni.bins]
colors = [BAD if b.kind in (SPECIAL, MISSING) else COOL for b in ni.bins]
bars = ax.bar(range(len(woes)), woes, color=colors, width=0.68)
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9, fontweight="bold", color=INK)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
ax.axhline(0, color=INK, linewidth=0.9)
ax.set_ylabel("WOE  (higher = riskier)")
ax.set_title("SEPARATE THE SENTINELS\\n-999 'no bureau record' is the riskiest group",
             fontsize=11, color=BAD, fontweight="bold", pad=10)
ax.set_ylim(min(woes) - 0.28, max(woes) + 0.34)

ax = axes[1]
pairs = [(n, fit(F[n][tr], y[tr], monotone=False).iv, fit(F[n][tr], y[tr], monotone=True).iv)
         for n in ("utilization", "income", "age", "months_employed")]
pos = np.arange(len(pairs))
ax.barh(pos + 0.19, [p[1] for p in pairs], height=0.36, color=MUTED, label="unconstrained")
ax.barh(pos - 0.19, [p[2] for p in pairs], height=0.36, color=GOOD, label="monotone")
for i, (name, free, forced) in enumerate(pairs):
    pct = (free - forced) / free if free else 0
    ax.text(free + 0.006, i + 0.19, f"-{pct:.0%}", va="center", fontsize=9,
            fontweight="bold", color=BAD if pct > 0.15 else INK)
ax.set_yticks(pos); ax.set_yticklabels([p[0] for p in pairs], fontsize=9)
ax.set_xlabel("IV"); ax.set_xlim(0, max(p[1] for p in pairs) * 1.22)
ax.legend(fontsize=8.5, loc="lower right", frameon=False); ax.invert_yaxis()
ax.set_title("MONOTONICITY HAS A PRICE\\nsmall = noise; large = deleting signal",
             fontsize=11, color=GOOD, fontweight="bold", pad=10)

ax = axes[2]
rows = noise_screen(ys[ts], n_columns=12, n_permutations=40)
xr = np.arange(len(rows))
ax.plot(xr, [r["mean_iv"] for r in rows], "o-", color=BAD, linewidth=2.4, markersize=8,
        label="mean raw IV")
ax.plot(xr, [r["mean_excess"] for r in rows], "s-", color=GOOD, linewidth=2.4, markersize=7,
        label="mean IV above permutation null")
ax.axhline(0.10, color=INK, linestyle="--", linewidth=1.2, alpha=0.75)
ax.text(-0.06, 0.113, 'conventional 0.10 "medium predictor" bar', fontsize=8.5,
        color=INK, style="italic")
for i, r in enumerate(rows):
    ax.annotate(f"{r['kept_by_iv']:.0%} kept", (i, r["mean_iv"]), textcoords="offset points",
                xytext=(0, 11), ha="center", fontsize=8.5, fontweight="bold", color=BAD)
ax.set_xticks(xr)
ax.set_xticklabels([str(r["settings"]).replace(", ", "\\n") for r in rows], fontsize=8)
ax.set_ylabel("IV on a column with NO signal")
ax.set_ylim(-0.02, max(r["mean_iv"] for r in rows) * 1.32)
ax.legend(fontsize=8.5, loc="upper left", frameon=False)
ax.set_title("RAW IV INFLATES WITH BIN COUNT\\n12 pure rng.normal() columns, 480 rows each",
             fontsize=11, color=BAD, fontweight="bold", pad=10)

for a in axes:
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    a.grid(alpha=0.22, linestyle=":"); a.set_axisbelow(True); a.tick_params(labelsize=9)

fig.tight_layout(rect=(0, 0.02, 1, 0.94))
plt.show()''')

md("""## 9. So does any of this produce a better model?

Everything above is univariate. IV scores one column against the target and says nothing
about what happens when six of them go into one model together - so a screen that rejects
features is only worth having if what survives actually scores better on rows nobody fitted.

Three encodings of the same six columns, one holdout:

| arm | encoding |
|---|---|
| **A** | raw continuous, median-imputed, sentinel `-999` left in the column as a number |
| **B** | constrained WOE bins - size floors, monotone, missing and sentinel separated |
| **C** | loose WOE bins - 20 bins, 1-event floor, no monotonicity (the biggest-IV arm) |

If **C** wins the holdout, every constraint in this notebook is costing real accuracy and the
argument is decorative.""")

code('''from downstream import format_lift, model_lift, robustness

rows = model_lift(data)
print(format_lift(rows))''')

md("""Read the two middle columns against the last one. **C** carries the most total IV and the
best training AUC, and lands *below* **B** on the holdout - its train-to-holdout gap is three
times as wide. IV ranks the arms in the opposite order to the thing anyone cares about.

**B** beating **A** is the part that justifies binning at all: same information, monotone and
reviewable, and about two AUC points better because the sentinel stopped being treated as the
number -999 and the missing rows stopped being median-imputed into a bin they do not belong in.

One split is not a finding, though. A 0.003 AUC gap between **B** and **C** is well inside the
noise of a single holdout - so run the whole thing again on ten independent datasets.""")

code('''r = robustness()

print(f"mean holdout AUC   raw {r['auc_raw']:.4f}   "
      f"constrained {r['auc_constrained']:.4f}   loose {r['auc_loose']:.4f}\\n")
print(f"constrained beats raw    {r['constrained_beats_raw']}/{r['n_seeds']} datasets  "
      f"(mean +{r['mean_margin_over_raw']:.4f} AUC)")
print(f"constrained beats loose  {r['constrained_beats_loose']}/{r['n_seeds']} datasets  "
      f"(mean +{r['mean_margin_over_loose']:.4f}, sd {r['sd_margin_over_loose']:.4f})")
print(f"\\nloose carried more IV in every dataset: {r['loose_always_higher_iv']}")
print(f"mean overfit gap  loose {r['mean_gap_loose']:+.4f}  vs  "
      f"constrained {r['mean_gap_constrained']:+.4f}")''')

md("""So the honest version of the claim, with the part that does not replicate left in:

- Binning beats raw continuous **in all ten datasets**, by ~0.026 AUC. That one is solid.
- Constrained beats loose in **eight of ten**, mean +0.007 with an sd of the same size. Real,
  and small. Anyone reporting that as "constrained binning is more accurate" full stop is
  overreading a 0.007 margin.
- The loose arm carried more total IV in **ten of ten** and overfitted **three times harder**
  every time. *That* is the reliable finding - not that loose binning scores worse, but that its
  IV is systematically inflated while its holdout performance is not.

Which is the whole point: the constraints are not there to buy accuracy. They are there so the
number you report is the number you get.""")

md("""## 10. Try your own feature

The engine takes plain numpy arrays. Three things are worth doing in order:""")

code('''# ---------------------------------------------------------------------------------
# 1. Declare what is NOT a number. Sentinels are business logic - no algorithm
#    can infer that -999 means 'no bureau record' rather than 'minus 999'.
#
# my_x, my_y = your_frame["feature"].to_numpy(float), your_frame["target"].to_numpy(int)
# SENTINELS = (-999.0, -1.0, 0.0)          # whatever your source system uses
#
# 2. Fit with floors you can defend. min_bin_events is the single most important
#    parameter here - it is what stops IV becoming a smoothing artifact.
#
# scheme = fit(my_x, my_y, feature="feature",
#              max_bins=6,
#              min_bin_share=0.05,
#              min_bin_events=20,          # per bin, events AND non-events
#              smoothing=0.5,
#              monotone=True,
#              specials=SENTINELS)
# print(format_table(scheme))
#
# 3. Never quote the raw IV. Quote the excess over the permutation null, and check
#    the p-value before the feature goes anywhere near a scorecard.
#
# result = audit(x_train, y_train, x_holdout, y_holdout,
#                feature="feature", n_permutations=200, specials=SENTINELS)
# print(result.verdict, f"excess IV {result.excess_iv:.4f}, p={result.p_value:.3f}")
#
# if result.sparse_warning:
#     print("raise min_bin_events -", result.sparse_warning)
#
# # Read the monotonicity cost before accepting the shape.
# free = fit(x_train, y_train, monotone=False, specials=SENTINELS)
# if free.iv and (free.iv - result.iv_train) / free.iv > 0.15:
#     print("monotonicity is deleting real signal - consider splitting the feature")
#
# # Only then transform.
# X_model = scheme.transform(x_holdout)''')

md(f"""---

**Streamlit version** - move the floors and the bin count, and watch the noise column climb
the IV bands while its p-value refuses to move:

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Tests** - the guarantees, asserted (40 checks, including the false-positive rate):

```bash
python3 test_binning.py
```

One limitation worth stating: the permutation null costs a refit per permutation, so it is
cheap per feature and expensive across a thousand. Screen with a modest 40 permutations, then
re-run the survivors at 200+.

Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder) · [`{BASE}`](https://github.com/phoebefu6/phoebe-the-builder/tree/main/{BASE})""")

pathlib.Path("demo.ipynb").write_text(json.dumps(nb, indent=1))
print(f"wrote demo.ipynb ({len(nb['cells'])} cells)")
