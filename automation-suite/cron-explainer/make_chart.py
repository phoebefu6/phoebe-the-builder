"""Six panels: what the crontab line says, and when the job actually runs.

Every value plotted is computed at draw time from cron.py and evidence.py.
Nothing is hard-coded.

Run:  python3 make_chart.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import cron as C
import evidence as E

UTC = timezone.utc
INK = "#1c1c1e"
MUTED = "#8a8a8e"
GRID = "#e6e6ea"
PALE = "#f2f2f5"
MISREAD_C = "#c0392b"
TIMING_C = "#d98324"
PORT_C = "#3a6ea5"
OK_C = "#2e7d5b"

SEV_COLOR = {C.MISREAD: MISREAD_C, C.TIMING: TIMING_C, C.PORTABILITY: PORT_C}
YEAR = E.YEAR
TZ = E.TZ


def _style(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=24)
    if subtitle:
        ax.text(
            0, 1.045, subtitle, transform=ax.transAxes, fontsize=8.6, color=MUTED,
            va="bottom", ha="left",
        )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.2, length=3)


# ---------------------------------------------------------------- panel 1


def panel_union(ax) -> None:
    """Every day of 2026 that '0 0 13 * 5' fires, against the days you meant."""
    c = C.parse("0 0 13 * 5")
    start = datetime(YEAR, 1, 1)
    union = {d.date() for d in C._matching_days(c, start, 365)}
    inter = {d.date() for d in C._matching_days(c, start, 365, force_intersection=True)}

    for i in range(365):
        d = (start + timedelta(days=i)).date()
        col, h = PALE, 0.62
        if d in inter:
            col, h = MISREAD_C, 1.0
        elif d in union:
            col, h = TIMING_C, 0.85
        ax.add_patch(
            mpatches.Rectangle((i, (1 - h) / 2), 1.6, h, facecolor=col, edgecolor="none")
        )
        if d in inter:
            # Only three of these all year; mark them so they are findable.
            ax.plot([i], [1.12], marker="v", color=MISREAD_C, ms=6, zorder=4)
    ax.set_xlim(0, 365)
    ax.set_ylim(-0.4, 1.5)
    ax.set_yticks([])
    month_starts = [(datetime(YEAR, m, 1) - start).days for m in range(1, 13)]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.spines["left"].set_visible(False)

    u = E.union_evidence()
    ax.text(
        0, 1.30, f"{u['union_days']} firing days", fontsize=13, fontweight="bold",
        color=TIMING_C,
    )
    ax.text(
        95, 1.30, f"you meant {u['intersection_days']}", fontsize=13, fontweight="bold",
        color=MISREAD_C,
    )
    ax.text(
        215, 1.30, f"x{u['factor']} overrun", fontsize=13, fontweight="bold", color=INK,
    )
    _style(
        ax,
        "1.  '0 0 13 * 5' is not Friday the 13th",
        "both day fields restricted -> cron takes the union. Orange = it fires. "
        "Red = the days you actually wanted.",
    )


# ---------------------------------------------------------------- panel 2


def panel_step(ax) -> None:
    """'*/7' across one hour: eight gaps of 7, one of 4."""
    c = C.parse("*/7 * * * *")
    vals = list(c.minute.values)
    gaps = [b - a for a, b in zip(vals, vals[1:])] + [60 - vals[-1] + vals[0]]

    for i, (v, g) in enumerate(zip(vals, gaps)):
        short = g != 7
        ax.add_patch(
            mpatches.Rectangle(
                (v, 0.25), g, 0.5,
                facecolor=TIMING_C if short else PALE,
                edgecolor=TIMING_C if short else GRID, linewidth=1.0,
            )
        )
        ax.plot([v], [0.5], "o", color=INK, ms=4.5, zorder=3)
        if short:
            ax.annotate(
                f"{g} min, not 7",
                xy=(v + g / 2, 0.78), xytext=(v + g / 2 - 9, 1.18),
                fontsize=8.8, color=TIMING_C, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=TIMING_C, lw=1.1),
            )
    s = E.step_evidence()
    ax.text(
        0, -0.55,
        f"{s['runs_per_hour']} runs an hour, not 60/7 = 8.6. "
        f"Same shape on hours: '0 */5' fires {s['runs_per_day']}x a day with a "
        f"{max(s['hour_gaps'])}h and a {min(s['hour_gaps'])}h gap.",
        fontsize=8.6, color=MUTED,
    )
    ax.set_xlim(-1, 61)
    ax.set_ylim(-0.9, 1.5)
    ax.set_yticks([])
    ax.set_xticks([0, 7, 14, 21, 28, 35, 42, 49, 56, 60])
    ax.set_xlabel("minute of the hour", fontsize=8.4, color=MUTED)
    ax.spines["left"].set_visible(False)
    _style(
        ax, "2.  A step that does not divide its field",
        "the field wraps at 60; the step does not. The last interval of every cycle is short.",
    )


# ---------------------------------------------------------------- panel 3


def panel_dst_counts(ax) -> None:
    """'*/30 * * * *' - 'every 30 minutes' - counted on three real days."""
    d = E.dst_evidence()
    labels = ["ordinary day\n15 Jun", "clocks forward\n29 Mar", "clocks back\n25 Oct"]
    vals = [d["runs_normal_day"], d["runs_spring_day"], d["runs_fall_day"]]
    cols = [PALE, TIMING_C, TIMING_C]
    bars = ax.bar(labels, vals, color=cols, edgecolor=[GRID, TIMING_C, TIMING_C], width=0.58)
    for b, v in zip(bars, vals):
        delta = v - vals[0]
        cx = b.get_x() + b.get_width() / 2
        ax.text(cx, v - 5.0, str(v), ha="center", fontsize=15,
                fontweight="bold", color="white" if delta else INK)
        if delta:
            ax.text(cx, v - 9.6, f"{delta:+d} runs", ha="center", fontsize=10,
                    fontweight="bold", color="white")
    ax.axhline(vals[0], color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(-0.42, vals[0] + 1.1, f"{vals[0]} a day expected", fontsize=8.2,
            color=MUTED, ha="left")
    ax.set_ylim(0, max(vals) + 7)
    ax.set_ylabel("runs that day", fontsize=8.4, color=MUTED)
    _style(
        ax, "3.  'Every 30 minutes' is not 48 times a day",
        f"{TZ}. An interval job follows the wall clock, and twice a year the wall "
        "clock is not 24 hours long.",
    )


# ---------------------------------------------------------------- panel 4


def panel_fixed_vs_interval(ax) -> None:
    """Where a fixed-time job and an interval job land on the two transitions."""
    tz = C._zone(TZ)
    rows = [
        ("30 1 * * *", "fixed", "forward", datetime(YEAR, 3, 28, 12), datetime(YEAR, 3, 29)),
        ("*/30 * * * *", "interval", "forward", datetime(YEAR, 3, 29, 0, 15), datetime(YEAR, 3, 29)),
        ("30 1 * * *", "fixed", "back", datetime(YEAR, 10, 24, 12), datetime(YEAR, 10, 25)),
        ("*/30 * * * *", "interval", "back", datetime(YEAR, 10, 25, 0, 15), datetime(YEAR, 10, 25)),
    ]
    ylabels = []
    for y, (expr, kind, when, start, day) in enumerate(rows):
        c = C.parse(expr)
        got = [f for f in C.fires(c, start, 8, TZ) if f.kind != C.NORMAL][:2]
        ylabels.append(f"{expr}\n{kind}, clocks {when}")
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
        midnight = day.replace(tzinfo=tz).astimezone(UTC)
        for f in got:
            if f.instant is None:
                ax.plot([-0.35], [y], "x", color=MISREAD_C, ms=11, mew=2.4, zorder=3)
                ax.text(-0.2, y - 0.3, "never runs: 01:30 does not exist",
                        fontsize=7.8, color=MISREAD_C, va="center")
                continue
            local = f.instant.astimezone(tz)
            x = (f.instant - midnight).total_seconds() / 3600
            ax.plot([x], [y], "o", color=TIMING_C, ms=9, zorder=3)
            ax.text(x, y - 0.32, f"{local:%H:%M %Z}", fontsize=7.9, color=INK,
                    ha="center")
        if len(got) == 2 and all(f.instant for f in got):
            a, b = got[0].instant, got[1].instant
            xa = (a - midnight).total_seconds() / 3600
            xb = (b - midnight).total_seconds() / 3600
            ax.annotate("", xy=(xb, y + 0.22), xytext=(xa, y + 0.22),
                        arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.0))
            # The y axis is inverted, so a larger y renders lower on the page.
            ax.text((xa + xb) / 2, y + 0.44, "same wall clock, 1h apart",
                    fontsize=7.4, color=MUTED, ha="center", va="top")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(ylabels, fontsize=7.6)
    ax.set_xlim(-0.75, 3.2)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("hours of real elapsed time since local midnight", fontsize=8.4,
                  color=MUTED)
    ax.invert_yaxis()
    _style(
        ax, "4.  Same transition, two different behaviours",
        "Vixie compensates a fixed-time job and lets an interval job follow the clock. "
        "The crontab does not say which one you wrote.",
    )


# ---------------------------------------------------------------- panel 5


def panel_utc_drift(ax) -> None:
    """'0 9 * * *' on a UTC runner: the local hour moves twice a year."""
    tz = C._zone(TZ)
    start = datetime(YEAR, 1, 1)
    xs, utc_local, local_utc = [], [], []
    for i in range(0, 365, 2):
        d = start + timedelta(days=i)
        u = datetime(d.year, d.month, d.day, 9, tzinfo=UTC).astimezone(tz)
        l = datetime(d.year, d.month, d.day, 9).replace(tzinfo=tz).astimezone(UTC)
        xs.append(i)
        utc_local.append(u.hour + u.minute / 60)
        local_utc.append(l.hour + l.minute / 60)
    ax.step(xs, utc_local, where="post", color=TIMING_C, lw=2.2,
            label="'0 9 * * *' on a UTC runner -> local time")
    ax.step(xs, local_utc, where="post", color=PORT_C, lw=2.2,
            label="'0 9 * * *' on a local-time host -> UTC")
    ax.set_ylim(7.0, 11.6)
    ax.set_yticks([8, 9, 10])
    ax.set_yticklabels(["08:00", "09:00", "10:00"])
    month_starts = [(datetime(YEAR, m, 1) - start).days for m in range(1, 13)]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_xlim(0, 365)
    ax.legend(fontsize=7.8, frameon=False, loc="upper center", ncol=1,
              bbox_to_anchor=(0.5, 1.02))
    _style(
        ax, "5.  The hour you wrote is not the hour it runs",
        "GitHub Actions, EventBridge and Kubernetes CronJob read cron in UTC. "
        "Neither expression changes; the clock does.",
    )


# ---------------------------------------------------------------- panel 6


def panel_findings(ax) -> None:
    """Same ten lines, audited against two time lines."""
    z = E.zone_comparison()
    zones = [TZ, "UTC"]
    sev_counts: Dict[str, List[int]] = {s: [] for s in C.SEVERITIES}
    for zone in zones:
        counts = {s: 0 for s in C.SEVERITIES}
        for expr, _ in E.SAMPLE:
            for f in C.audit(C.parse(expr), zone, E.START):
                counts[f.severity] += 1
        for s in C.SEVERITIES:
            sev_counts[s].append(counts[s])

    left = [0.0, 0.0]
    for s in C.SEVERITIES:
        vals = sev_counts[s]
        ax.barh(zones, vals, left=left, color=SEV_COLOR[s], edgecolor="white",
                linewidth=1.2, height=0.42, label=s.title())
        for i, v in enumerate(vals):
            if v:
                ax.text(left[i] + v / 2, i, str(v), ha="center", va="center",
                        fontsize=9.5, color="white", fontweight="bold")
        left = [a + b for a, b in zip(left, vals)]

    for i, zone in enumerate(zones):
        v = z[zone]
        ax.text(left[i] + 0.45, i,
                f"  {v['clean_lines']}/{len(E.SAMPLE)} lines clean",
                va="center", fontsize=8.6, color=MUTED)
    ax.set_xlim(0, max(left) + 7)
    ax.invert_yaxis()
    ax.set_xlabel("findings across the same 10-line crontab", fontsize=8.4, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=3)
    _style(
        ax, "6.  The findings track the time line, not the text",
        "identical expressions, audited twice. Every one of the ten lines is valid "
        "cron and none of them errors.",
    )


def main(path: str = "cron_audit.png") -> str:
    fig, axes = plt.subplots(3, 2, figsize=(16.4, 13.6))
    fig.patch.set_facecolor("white")
    panel_union(axes[0][0])
    panel_step(axes[0][1])
    panel_dst_counts(axes[1][0])
    panel_fixed_vs_interval(axes[1][1])
    panel_utc_drift(axes[2][0])
    panel_findings(axes[2][1])
    fig.suptitle(
        "A cron line renders one meaning and schedules another",
        fontsize=16, fontweight="bold", color=INK, x=0.008, ha="left", y=0.995,
    )
    fig.text(
        0.008, 0.968,
        f"Day 141 - cron-explainer. Every value computed at draw time from cron.py. "
        f"Zone: {TZ}, year {YEAR}.",
        fontsize=9, color=MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.subplots_adjust(hspace=0.62, wspace=0.2)
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main()
