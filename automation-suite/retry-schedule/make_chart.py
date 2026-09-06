"""The six-panel audit figure. Every value is computed at draw time.

Run: python3 make_chart.py [outfile.png]
"""

from __future__ import annotations

import sys
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import retry as R

INK = "#1b1b1f"
MUTED = "#8a8a94"
GRID = "#e3e3e8"
BAD = "#c0392b"
WARN = "#d98324"
OK = "#2d7d5a"
COOL = "#2f6f9f"
PAPER = "#fbfbfd"

FLEET, OUTAGE, CAPACITY = 500, 20.0, 50.0
BASE, CAP, ATTEMPTS = 0.1, 20.0, 10

COLOR = {
    "no_jitter": BAD,
    "fixed_interval": "#8e44ad",
    "equal_jitter": WARN,
    "full_jitter": COOL,
    "decorrelated_jitter": OK,
}


def _style(ax, title: str, subtitle: str = "") -> None:
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=11, weight="bold", loc="left", pad=14)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=MUTED,
                fontsize=8.2, va="bottom")


# ---------------------------------------------------------------------------


def panel_arrivals(ax, sims) -> None:
    """The herd itself: arrivals per second against the recovering service."""
    for name in ("no_jitter", "equal_jitter", "full_jitter"):
        sim = sims[name]
        edges, counts = sim.histogram(width=1.0, since=0.0, upto=80.0)
        ax.plot(edges, counts, color=COLOR[name], lw=1.6, label=name,
                drawstyle="steps-post")
    ax.axvspan(0, OUTAGE, color=BAD, alpha=0.06)
    ax.axhline(CAPACITY, color=INK, lw=1.0, ls="--")
    ax.text(78, CAPACITY * 1.25, f"capacity {CAPACITY:.0f} rps", color=INK,
            fontsize=7.5, ha="right")
    ax.text(OUTAGE / 2, 700, "dependency down", color=BAD, fontsize=7.5, ha="center")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(0, 3000)
    ax.set_xlabel("seconds since the fleet failed", fontsize=8, color=MUTED)
    ax.set_ylabel("arrivals per second", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    nj = sims["no_jitter"].recovery_peak_rps()
    _style(ax, "1. Backoff spaces the spikes. It does not shrink them.",
           f"{FLEET} clients fail together; un-jittered peak on recovery is "
           f"{nj:.0f} rps, {nj/CAPACITY:.0f}x capacity")


def panel_inversion(ax, sims) -> None:
    """Peak vs clients lost - the two objectives disagree."""
    names = R.POLICY_ORDER
    peaks = [sims[n].recovery_peak_rps() for n in names]
    lost = [sims[n].gave_up for n in names]
    # The two deterministic policies land on top of each other; offset their
    # labels rather than letting the figure lie about how many points there are.
    offsets = {"no_jitter": (0, 16), "fixed_interval": (0, -26),
               "equal_jitter": (0, 16), "full_jitter": (0, 16),
               "decorrelated_jitter": (0, 16)}
    for n, p, lost_pct in zip(names, peaks, lost):
        ax.scatter([p], [lost_pct], s=170, color=COLOR[n], zorder=3,
                   edgecolor="white", lw=1.4)
        ax.annotate(n, (p, lost_pct), textcoords="offset points", xytext=offsets[n],
                    ha="center", fontsize=7.5, color=INK)
    ax.axvline(CAPACITY, color=INK, lw=1.0, ls="--")
    ax.text(CAPACITY * 1.1, 200, "capacity", color=INK, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("peak arrival rate on the recovering service (rps, log)",
                  fontsize=8, color=MUTED)
    ax.set_ylabel("clients permanently failed", fontsize=8, color=MUTED)
    ax.set_ylim(-40, 560)
    fj, ej = sims["full_jitter"], sims["equal_jitter"]
    _style(ax, "2. The gentlest arrival process loses the most clients.",
           f"full_jitter peaks at {fj.recovery_peak_rps():.0f} rps and loses "
           f"{fj.gave_up}; equal_jitter peaks at {ej.recovery_peak_rps():.0f} "
           f"and loses {ej.gave_up}")


def panel_coverage(ax) -> None:
    """Worst case / mean / median reach against the outage that must be outlasted."""
    names = ["no_jitter", "equal_jitter", "full_jitter", "decorrelated_jitter"]
    ys = range(len(names))
    for y, n in zip(ys, names):
        s = R.Schedule(n, BASE, CAP, ATTEMPTS)
        med, _ = R.sampled_totals(s, n=4000, seed=11)
        ax.barh(y, s.worst_case_total(), height=0.55, color=COLOR[n], alpha=0.22)
        ax.barh(y, s.expected_total(), height=0.55, color=COLOR[n], alpha=0.55)
        ax.plot([med], [y], marker="|", ms=18, mew=2.2, color=INK)
        ax.text(s.worst_case_total() + 2, y, f"worst {s.worst_case_total():.0f}s",
                va="center", fontsize=7, color=MUTED)
    ax.axvline(OUTAGE, color=BAD, lw=1.4)
    ax.set_ylim(-0.6, 3.8)
    ax.text(OUTAGE + 2, 3.48, f"{OUTAGE:.0f}s outage to outlast",
            color=BAD, fontsize=7.5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("elapsed seconds before the client gives up "
                  "(pale = worst case, solid = mean, bar = median)",
                  fontsize=8, color=MUTED)
    _style(ax, "3. Jitter is drawn downward, so it halves your reach.",
           "same 10 attempts, same cap - full_jitter covers half the wall clock "
           "of the ladder it is drawn from")


def panel_floor(ax) -> None:
    """The cap is a load floor: predicted closed form vs measured process."""
    caps = [5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0]
    pred = [R.Schedule("full_jitter", BASE, c, ATTEMPTS).steady_state_rate(FLEET)
            for c in caps]
    meas: List[float] = []
    for c in caps:
        s = R.Schedule("full_jitter", BASE, c, max_attempts=80)
        sim = R.simulate(s, fleet=FLEET, outage_s=OUTAGE, capacity_rps=0.0,
                         seed=3, horizon_s=6 * c + 200)
        lo, hi = 3 * c, 5 * c
        meas.append(sum(1 for t in sim.arrivals if lo <= t < hi) / (hi - lo))
    ax.plot(caps, pred, color=COOL, lw=1.8, label="closed form  fleet / (cap/2)")
    ax.scatter(caps, meas, s=42, color=INK, zorder=3, label="measured arrivals")
    ax.axhline(CAPACITY, color=BAD, lw=1.2, ls="--")
    ax.text(120, CAPACITY * 1.1, f"capacity {CAPACITY:.0f} rps", color=BAD,
            fontsize=7.5, ha="right")
    ax.set_xlabel("cap (seconds)", fontsize=8, color=MUTED)
    ax.set_ylabel("steady-state load floor (rps)", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7.5)
    _style(ax, "4. The cap sets a floor that jitter cannot lower.",
           f"once the window stops widening the process stops thinning: "
           f"{2*FLEET:.0f}/cap rps, forever")


def panel_amplification(ax) -> None:
    stacks = [("browser", [3]), ("+ gateway", [3, 3]), ("+ service", [3, 3, 3]),
              ("+ db driver", [3, 3, 3, 2])]
    vals = [R.amplification(stack) for _, stack in stacks]
    bars = ax.bar([s for s, _ in stacks], vals,
                  color=[COOL, WARN, BAD, BAD], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v}x",
                ha="center", fontsize=9, color=INK, weight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1, 130)
    ax.set_ylabel("requests at the bottom service, per user click",
                  fontsize=8, color=MUTED)
    ax.tick_params(axis="x", labelsize=8)
    _style(ax, "5. Every retrying layer multiplies the one below it.",
           "three reasonable layers of 3 attempts is 27x; add a driver that "
           "retries twice and it is 54x")


def panel_findings(ax, sims) -> None:
    names = R.POLICY_ORDER
    sev = [("critical", BAD), ("warning", WARN), ("info", MUTED)]
    bottoms = [0.0] * len(names)
    verdicts = []
    for label, color in sev:
        vals = []
        for i, n in enumerate(names):
            v, fs = R.audit(sims[n].schedule, FLEET, OUTAGE, CAPACITY,
                            deadline_s=30.0, nested_layers=[3, 3], tick_s=1.0,
                            sim=sims[n])
            if label == "critical":
                verdicts.append(v)
            vals.append(sum(1 for f in fs if f.severity.value == label))
        ax.bar(names, vals, bottom=bottoms, color=color, width=0.62, label=label)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    vcol = {R.Verdict.HERDING: BAD, R.Verdict.BURSTY: WARN,
            R.Verdict.DISPERSED: OK}
    for i, (n, v) in enumerate(zip(names, verdicts)):
        ax.text(i, bottoms[i] + 0.25, v.value, ha="center", fontsize=7.8,
                color=vcol[v], weight="bold")
    ax.set_ylim(0, max(bottoms) + 1.6)
    ax.set_ylabel("findings", fontsize=8, color=MUTED)
    ax.tick_params(axis="x", labelsize=7.2, rotation=12)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    _style(ax, "6. Verdict is about the herd, not about whether clients recovered.",
           "`dispersed` and `clients gave up` are both true of full_jitter here - "
           "the audit reports both")


def main(out: str = "retry_audit.png") -> None:
    sims = R.compare(FLEET, OUTAGE, CAPACITY, BASE, CAP, ATTEMPTS)
    fig, axes = plt.subplots(3, 2, figsize=(15.5, 15.5))
    fig.patch.set_facecolor("white")
    panel_arrivals(axes[0][0], sims)
    panel_inversion(axes[0][1], sims)
    panel_coverage(axes[1][0])
    panel_floor(axes[1][1])
    panel_amplification(axes[2][0])
    panel_findings(axes[2][1], sims)

    fig.suptitle("Retry schedules: the delay is not the output. The arrival "
                 "process is.", x=0.06, y=0.985, ha="left", fontsize=16,
                 color=INK, weight="bold")
    fig.text(0.06, 0.963,
             f"{FLEET} clients fail together - {OUTAGE:.0f}s outage - "
             f"{CAPACITY:.0f} rps of capacity - base {BASE:g}s, cap {CAP:g}s, "
             f"{ATTEMPTS} attempts - shed load burns an attempt like failed load",
             ha="left", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=[0.02, 0.01, 0.99, 0.955])
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "retry_audit.png")
