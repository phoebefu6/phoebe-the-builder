"""The audit figure: four panels, each one a section of evidence.py recomputed live."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import backtest as bt
import matplotlib.pyplot as plt
import numpy as np

INK = "#1f2733"
MUTED = "#6b7684"
ACCENT = "#2f6fdb"
WARN = "#c8562b"
GOOD = "#2f8f5b"
GRID = "#dfe4ea"
BG = "#ffffff"

H, N, MIN_TRAIN = 12, 180, 96
fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor=BG)
for ax in axes.ravel():
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)

# ---------------------------------------------------------------- 1. the origin sweep
ax = axes[0, 0]
y = bt.make_series(np.random.default_rng(3), 400)
E = bt.rolling_origin(y, bt.f_trend_season, H, k=60, min_train=MIN_TRAIN)
per = np.array([bt.rmse(r[~np.isnan(r)]) for r in E])
ors = bt.origins(len(y), H, 60, MIN_TRAIN)
truth = bt.true_h_step_error(bt.f_trend_season, H, N - H, 600, np.random.default_rng(101))
ax.plot(ors, per, color=ACCENT, lw=1.6, marker="o", ms=3, label="RMSE if you split HERE")
ax.axhline(truth, color=INK, lw=1.8, ls="--", label=f"the honest number ({truth:.2f})")
ax.scatter([ors[int(np.argmin(per))]], [per.min()], color=GOOD, zorder=5, s=60)
ax.annotate(f"the number you could report\n{per.min():.2f} = {per.min()/truth*100:.0f}% of the truth",
            (ors[int(np.argmin(per))], per.min()), textcoords="offset points", xytext=(-58, 18),
            color=GOOD, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=GOOD, lw=1))
ax.set_ylim(per.min() - 0.35, per.max() + 0.35)
ax.set_title("One series, one model, 60 choices of origin", color=INK, fontsize=12, loc="left")
ax.set_xlabel("origin (last point of training)", color=MUTED)
ax.set_ylabel("12-step RMSE", color=MUTED)
ax.legend(frameon=False, fontsize=9)

# ---------------------------------------------------------------- 2. best-of-M
ax = axes[0, 1]
M, DELTA = 8, 2.0
rng = np.random.default_rng(21)
u = rng.normal(size=(M, H))
u = u / np.linalg.norm(u, axis=1, keepdims=True) * np.sqrt(H)


def cand(j):
    return lambda yy, h, p=bt.PERIOD: bt.f_trend_season(yy, h, p) + DELTA * u[j, :h]


rng = np.random.default_rng(22)
rep_win, fut_win, field = [], [], []
mat = []
for _ in range(400):
    yy = bt.make_series(rng, N + H)
    sp = np.array([bt.rmse(bt.single_split(yy[:N], cand(j), H)) for j in range(M)])
    mat.append(sp)
    j = int(np.argmin(sp))
    rep_win.append(sp[j])
    field.append(sp.mean())
    a, p = bt.forecast_at(yy, cand(j), N, H)
    fut_win.append(bt.rmse(a - p))
mat = np.array(mat)
cor = np.corrcoef(mat.T)
rho_bar = (cor.sum() - M) / (M * (M - 1))
pred = np.mean(field) + bt.expected_min_of_m(M, rho_bar) * mat.std(axis=0, ddof=1).mean()
bars = [np.mean(rep_win), np.mean(field), np.mean(fut_win)]
labels = ["winner's\nbacktest number", "the field's\nbacktest number", "winner's ACTUAL\nnext window"]
ax.bar(range(3), bars, color=[WARN, MUTED, ACCENT], width=0.6)
for i, v in enumerate(bars):
    ax.text(i, v + 0.05, f"{v:.3f}", ha="center", color=INK, fontsize=10)
ax.axhline(pred, color=GOOD, lw=1.6, ls=":",
           label=f"predicted winner = field + E[min of {M}]*sd\nat rho={rho_bar:.2f}: {pred:.3f}")
ax.set_xticks(range(3))
ax.set_xticklabels(labels, color=MUTED, fontsize=9)
ax.set_ylim(0, max(bars) * 1.25)
ax.set_ylabel("12-step RMSE", color=MUTED)
ax.set_title(f"{M} candidates that are equally good by construction", color=INK, fontsize=12, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")

# ---------------------------------------------------------------- 3. k_eff
ax = axes[1, 0]
ks, keffs = [], []
for k, step in [(20, 1), (20, 3), (20, 6), (20, 12), (8, 12)]:
    rng = np.random.default_rng(31)
    means, sdf = [], []
    for _ in range(200):
        yy = bt.make_series(rng, N + 60)
        Ei = bt.rolling_origin(yy, bt.f_trend_season, H, k=k, min_train=MIN_TRAIN, step=step)
        pr = np.array([bt.rmse(r[~np.isnan(r)]) for r in Ei])
        means.append(pr.mean())
        sdf.append(pr.std(ddof=1))
    keffs.append(bt.effective_folds(float(np.mean(sdf)), float(np.std(means, ddof=1))))
    ks.append(f"k={k}\nstep={step}")
ax.bar(range(len(ks)), keffs, color=ACCENT, width=0.6)
for i, (lab, v) in enumerate(zip(ks, keffs)):
    nominal = int(lab.split("=")[1].split("\n")[0])
    ax.plot([i - 0.3, i + 0.3], [nominal, nominal], color=WARN, lw=2)
    ax.text(i, v + 0.4, f"{v:.1f}", ha="center", color=INK, fontsize=10)
ax.set_xticks(range(len(ks)))
ax.set_xticklabels(ks, color=MUTED, fontsize=9)
ax.set_ylabel("independent folds actually present", color=MUTED)
ax.set_title("k_eff: what the folds are worth (red bar = the k you would quote)",
             color=INK, fontsize=12, loc="left")

# ---------------------------------------------------------------- 4. the scoreboard
ax = axes[1, 1]
rng = np.random.default_rng(71)
prot = {"single split (1)": [], "rolling k=5": [], "rolling k=20 (overlap 11/12)": [],
        "rolling step 12 (7 folds, no overlap)": []}
for _ in range(300):
    yy = bt.make_series(rng, N)
    prot["single split (1)"].append(bt.rmse(bt.single_split(yy, bt.f_trend_season, H)))
    for label, k, step in [("rolling k=5", 5, 1), ("rolling k=20 (overlap 11/12)", 20, 1),
                           ("rolling step 12 (7 folds, no overlap)", 20, 12)]:
        Ei = bt.rolling_origin(yy, bt.f_trend_season, H, k=k, min_train=MIN_TRAIN, step=step)
        prot[label].append(bt.rmse(Ei[~np.isnan(Ei)]))
ypos = np.arange(len(prot))
for i, (label, v) in enumerate(prot.items()):
    a = np.array(v)
    ax.plot([np.percentile(a, 5), np.percentile(a, 95)], [i, i], color=MUTED, lw=2)
    ax.scatter([a.mean()], [i], color=ACCENT, s=70, zorder=5)
    ax.text(np.percentile(a, 95) + 0.08, i,
            f"RMSE of\nthe estimate\n{np.sqrt(np.mean((a - truth)**2)):.3f}",
            va="center", color=INK, fontsize=8.5)
ax.axvline(truth, color=INK, ls="--", lw=1.6)
ax.text(truth, -0.7, f" the quantity all four claim to report: {truth:.3f}", color=INK, fontsize=9)
ax.set_yticks(ypos)
ax.set_yticklabels(list(prot), color=MUTED, fontsize=9)
ax.set_ylim(-1, len(prot) - 0.4)
ax.set_xlim(2.4, 7.4)
ax.set_xlabel("reported 12-step RMSE (dot = mean, bar = 5th-95th percentile)", color=MUTED)
ax.set_title("Every protocol as an estimator", color=INK, fontsize=12, loc="left")

fig.suptitle("A backtest number is an estimate, and the protocol decides how noisy an estimate",
             color=INK, fontsize=15, x=0.02, ha="left", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("forecast_backtest_audit.png", dpi=170, facecolor=BG)
fig.savefig("forecast_backtest_audit.svg", facecolor=BG)
print("wrote forecast_backtest_audit.png / .svg")
