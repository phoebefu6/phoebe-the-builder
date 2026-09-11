"""The audit figure: four panels, each one a section of evidence.py recomputed live."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import intervals as iv
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, ACCENT, WARN, GOOD, GRID, BG = (
    "#1f2733", "#6b7684", "#2f6fdb", "#c8562b", "#2f8f5b", "#dfe4ea", "#ffffff")
H, N_TRAIN, CONF = 12, 168, 0.95

fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor=BG)
for ax in axes.ravel():
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)

# ------------------------------------------------ 1. the claim vs the frequency
ax = axes[0, 0]
runs = {n: iv.coverage_run(m, H, N_TRAIN, 500, np.random.default_rng(5), CONF)
        for n, m in iv.METHODS.items()}
names = list(runs)
cov = [runs[n]["hits"].mean() for n in names]
wid = [runs[n]["width"].mean() for n in names]
colors = [WARN if abs(c - CONF) > 0.03 else GOOD for c in cov]
ax.barh(range(len(names)), cov, color=colors, height=0.6)
ax.axvline(CONF, color=INK, ls="--", lw=1.8)
for i, (c, w) in enumerate(zip(cov, wid)):
    ax.text(c + 0.004, i, f"{c:.3f}   width {w:.1f}", va="center", color=INK, fontsize=9)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, color=MUTED, fontsize=9)
ax.set_xlim(0.80, 1.08)
ax.set_xlabel("measured coverage of a nominal 95% band", color=MUTED)
ax.set_title("The claim, and the frequency", color=INK, fontsize=12, loc="left")

# ------------------------------------------------ 2. where the sqrt(h) fan belongs
ax = axes[0, 1]
rng = np.random.default_rng(17)
errs_fit = np.zeros((500, H))
errs_rw = np.zeros((500, H))
for i in range(500):
    y, _ = iv.make_series(rng, N_TRAIN + H)
    errs_fit[i] = y[N_TRAIN:] - iv.predict(iv.fit_trend_season(y[:N_TRAIN]), H)
    rw = 100 + np.cumsum(rng.normal(0, 4.0, N_TRAIN + H))
    errs_rw[i] = rw[N_TRAIN:] - rw[N_TRAIN - 1]
h = np.arange(1, H + 1)
ax.plot(h, errs_fit.std(axis=0, ddof=1), color=ACCENT, marker="o", lw=1.8,
        label="fitted trend+season (direct h-step)")
ax.plot(h, errs_rw.std(axis=0, ddof=1), color=WARN, marker="s", lw=1.8,
        label="random walk + naive (accumulating)")
ax.plot(h, errs_fit.std(axis=0, ddof=1)[0] * np.sqrt(h), color=INK, ls="--", lw=1.5,
        label="the sqrt(h) fan everyone ships")
ax.set_xlabel("horizon h", color=MUTED)
ax.set_ylabel("true sd of the h-step error", color=MUTED)
ax.set_title("The fan is an assumption about the MODEL", color=INK, fontsize=12, loc="left")
ax.legend(frameon=False, fontsize=9)

# ------------------------------------------------ 3. marginal vs conditional
ax = axes[1, 0]
VOL = {"vol_period": 24, "vol_ratio": 9.0}
sel = ["gaussian + parameter term", "split conformal", "rolling h-step quantile"]
marg, quiet, loud = [], [], []
for n in sel:
    r = iv.coverage_run(iv.METHODS[n], H, N_TRAIN, 500, np.random.default_rng(23), CONF, **VOL)
    lo = r["sd"] < np.sqrt(r["sd"].min() * r["sd"].max())
    marg.append(float(r["hits"].mean()))
    quiet.append(float(r["hits"][lo].mean()))
    loud.append(float(r["hits"][~lo].mean()))
x = np.arange(len(sel))
ax.bar(x - 0.22, quiet, width=0.42, color=GOOD, label="in the quiet regime")
ax.bar(x + 0.22, loud, width=0.42, color=WARN, label="in the volatile regime")
for i, m in enumerate(marg):
    ax.plot([i - 0.45, i + 0.45], [m, m], color=INK, lw=2.2)
    ax.text(i, m + 0.012, f"marginal {m:.3f}", ha="center", color=INK, fontsize=9)
ax.axhline(CONF, color=MUTED, ls=":", lw=1.5)
ax.set_xticks(x)
ax.set_xticklabels([s.replace(" ", "\n", 1) for s in sel], color=MUTED, fontsize=9)
ax.set_ylim(0.5, 1.06)
ax.set_ylabel("coverage", color=MUTED)
ax.set_title("The guarantee is marginal; nobody stands in the margin",
             color=INK, fontsize=12, loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower left")

# ------------------------------------------------ 4. power of the coverage check
ax = axes[1, 1]
ms = np.unique(np.round(np.logspace(1.2, 4, 40)).astype(int))
for ratio, color, lab in ((1.0, GOOD, "independent test points"),
                          (1 / 12, ACCENT, "k_eff = 1/12 (rolling, step 12)"),
                          (1 / 20, WARN, "k_eff = 1/20 (rolling, step 1 - Day 172)")):
    ax.plot(ms, [iv.coverage_check_power(0.90, 0.95, m, ratio) for m in ms],
            color=color, lw=2.0, label=lab)
ax.axhline(0.8, color=INK, ls="--", lw=1.4)
ax.set_xscale("log")
ax.set_xlabel("test points the check is run on", color=MUTED)
ax.set_ylabel("power to catch a band covering 90% not 95%", color=MUTED)
ax.set_title("Why bad bands survive validation", color=INK, fontsize=12, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")

fig.suptitle("A prediction interval is a claim about a frequency, at a horizon, in a regime",
             color=INK, fontsize=15, x=0.02, ha="left", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("prediction_interval_audit.png", dpi=170, facecolor=BG)
fig.savefig("prediction_interval_audit.svg", facecolor=BG)
print("wrote prediction_interval_audit.png / .svg")
