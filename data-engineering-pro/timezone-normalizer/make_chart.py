"""Six-panel figure. ``python3 make_chart.py`` writes both PNGs."""

from __future__ import annotations

import datetime as dt
from typing import Dict, List
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from evidence import (
    exp_ambiguous,
    exp_day_bucketing,
    exp_identifiers,
    exp_nonexistent,
    exp_offset_vs_zone,
    exp_sub_hour,
)
from tznorm import UTC, build_session_log, ground_truth, tzdata_version

INK = "#1c1c1c"
MUTED = "#8a8a8a"
GOOD = "#2f6f52"
BAD = "#b23a3a"
WARM = "#c98a2b"
COOL = "#3a6ea5"
PAPER = "#faf8f4"
SHADE = "#ede6d8"

NY = ZoneInfo("America/New_York")
ANCHOR = dt.datetime(2024, 11, 3, 4, 0, tzinfo=UTC)


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)


def _mins(instant: dt.datetime) -> float:
    return (instant - ANCHOR).total_seconds() / 60


# --------------------------------------------------------------------------


def panel_fallback(ax: plt.Axes) -> None:
    """The wall clock reads 01:00-01:59 twice. Draw both passes on a real time axis."""
    truth = ground_truth()
    a = exp_ambiguous(verbose=False)

    # the repeated hour: 05:00Z-06:00Z and 06:00Z-07:00Z are both local 01:xx
    ax.axvspan(_mins(ANCHOR + dt.timedelta(hours=1)), _mins(ANCHOR + dt.timedelta(hours=3)),
               color=SHADE, zorder=0)
    ax.axvline(_mins(dt.datetime(2024, 11, 3, 6, 0, tzinfo=UTC)), color=INK, lw=1.0, ls="--")
    ax.text(
        _mins(dt.datetime(2024, 11, 3, 6, 0, tzinfo=UTC)) + 3,
        2.62,
        "clocks go back",
        fontsize=6.5,
        color=INK,
    )

    sessions = ["S-101", "S-104", "S-105"]
    for i, sid in enumerate(sessions):
        y = 2 - i
        o, c = truth[(sid, "open")], truth[(sid, "close")]
        assert o is not None and c is not None
        ax.plot([_mins(o), _mins(c)], [y + 0.16, y + 0.16], color=GOOD, lw=5, solid_capstyle="butt")
        ax.text(_mins(c) + 4, y + 0.16, f"{a['results'][sid]['truth']:.0f}m", fontsize=6.8,
                color=GOOD, va="center")

        # what fold=0 reconstructs: both endpoints forced into the first pass
        ro = o if sid == "S-101" else dt.datetime.combine(
            o.astimezone(NY).date(), o.astimezone(NY).time(), tzinfo=NY
        ).replace(fold=0).astimezone(UTC)
        rc = c if sid == "S-101" else dt.datetime.combine(
            c.astimezone(NY).date(), c.astimezone(NY).time(), tzinfo=NY
        ).replace(fold=0).astimezone(UTC)
        d = a["results"][sid]["earlier"]
        colour = BAD if d != a["results"][sid]["truth"] else GOOD
        ax.annotate(
            "",
            xy=(_mins(rc), y - 0.16),
            xytext=(_mins(ro), y - 0.16),
            arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.2, shrinkA=0, shrinkB=0),
        )
        ax.text(max(_mins(rc), _mins(ro)) + 5, y - 0.30, f"{d:.0f}m", fontsize=6.8,
                color=colour, va="center")

    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(sessions, fontsize=7.5, fontfamily="monospace")
    ticks = [i * 30 for i in range(7)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [(ANCHOR + dt.timedelta(minutes=t)).astimezone(NY).strftime("%H:%M") for t in ticks],
        fontsize=7,
    )
    ax.set_xlabel("local wall clock, New York, 2024-11-03", fontsize=8)
    ax.set_xlim(-8, 190)
    ax.set_ylim(-0.6, 2.75)
    ax.set_title("A  the hour that happens twice", fontsize=9.5, color=INK, loc="left", pad=16)
    ax.legend(
        handles=[Patch(color=GOOD, label="true instants"), Patch(color=BAD, label="fold=0 reading")],
        fontsize=6.5, loc="lower right", bbox_to_anchor=(1.02, 1.005), ncol=2, frameon=False,
    )
    ax.text(
        0.0, -0.30,
        "Shaded band = the wall clock reading 01:00-01:59, which happens twice.\n"
        "S-105's reconstruction runs backwards: it closes 40 minutes before it opens.",
        transform=ax.transAxes, fontsize=6.8, color=MUTED, va="top",
    )
    _style(ax)


def panel_gaps(ax: plt.Axes) -> None:
    b = exp_nonexistent(verbose=False)
    labels = [c["zone"].split("/")[-1].replace("_", " ") for c in b["cases"]]
    widths = []
    for c in b["cases"]:
        widths.append((c["fold0_utc"] - c["fold1_utc"]).total_seconds() / 60)
    widths = [abs(w) for w in widths]
    bars = ax.barh(labels[::-1], widths[::-1], color=[BAD, WARM, BAD][::-1], height=0.5)
    for bar, w in zip(bars, widths[::-1]):
        ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.0f} min",
                va="center", fontsize=7.5, color=INK)
    ax.set_xlim(0, 80)
    ax.set_xlabel("width of the skipped window", fontsize=8)
    ax.set_title("B  the hour that never happens", fontsize=9.5, color=INK, loc="left")
    ax.tick_params(labelsize=7.5)
    ax.text(
        0.0, -0.30,
        "Every one of these wall-clock times is accepted without an exception.\n"
        "Lord Howe's gap is thirty minutes - a check for '02:xx' never sees it.",
        transform=ax.transAxes, fontsize=6.8, color=MUTED, va="top",
    )
    _style(ax)


def panel_recovery(ax: plt.Axes) -> None:
    c = exp_offset_vs_zone(verbose=False)
    sids = list(c["recovered"])
    x = np.arange(len(sids))
    w = 0.26
    wall = [c["recovered"][s]["wall"] for s in sids]
    api = [c["recovered"][s]["api"] for s in sids]
    truth = [c["recovered"][s]["truth"] for s in sids]
    ax.bar(x - w, wall, w, color=BAD, label="wall clock only")
    ax.bar(x, api, w, color=COOL, label="wall clock + offset")
    ax.bar(x + w, truth, w, color=GOOD, label="truth")
    for xi, vals in zip(x, zip(wall, api, truth)):
        for dx, v in zip((-w, 0, w), vals):
            ax.text(xi + dx, v + (3 if v >= 0 else -9), f"{v:.0f}", ha="center", fontsize=7,
                    color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sids, fontsize=7.5, fontfamily="monospace")
    ax.set_ylabel("session duration (minutes)", fontsize=8)
    ax.set_ylim(-55, 100)
    ax.set_title("C  six characters of offset fix it", fontsize=9.5, color=INK, loc="left", pad=16)
    ax.legend(fontsize=6.4, loc="lower right", bbox_to_anchor=(1.02, 1.005), ncol=3, frameon=False)
    ax.text(
        0.0, -0.30,
        "The blue bars are exact. `01:30-04:00` and `01:50-05:00` are different\n"
        "instants and say so; `01:30` and `01:50` alone cannot.",
        transform=ax.transAxes, fontsize=6.8, color=MUTED, va="top",
    )
    _style(ax)


def panel_aliases(ax: plt.Axes) -> None:
    d = exp_identifiers(verbose=False)
    by_zone = d["by_zone"]
    groups = d["groups"]
    canon = {z: g[0] for g in groups for z in g}
    zones = sorted(by_zone, key=lambda z: -by_zone[z])
    colours = [WARM if z in canon else COOL for z in zones]
    bars = ax.barh([z.split("/")[-1].replace("_", " ") for z in zones][::-1],
                   [by_zone[z] for z in zones][::-1], color=colours[::-1], height=0.6)
    merged_val = sum(by_zone[z] for z in canon)
    for bar, z in zip(bars, zones[::-1]):
        ax.text(by_zone[z] + 25, bar.get_y() + bar.get_height() / 2, f"{by_zone[z]:,.0f}",
                va="center", fontsize=6.8, color=INK)
    if canon:
        ys = [i for i, z in enumerate(zones[::-1]) if z in canon]
        ax.plot([merged_val, merged_val], [min(ys) - 0.35, max(ys) + 0.35], color=BAD, lw=1.4,
                ls="--")
        ax.text(merged_val + 25, (min(ys) + max(ys)) / 2, f"actually\n{merged_val:,.0f}",
                fontsize=6.8, color=BAD, va="center")
    ax.set_xlim(0, 2400)
    ax.set_xlabel("revenue, grouped by the zone column", fontsize=8)
    ax.set_title("D  two names, one place", fontsize=9.5, color=INK, loc="left")
    ax.tick_params(labelsize=7.2)
    ax.text(
        0.0, -0.30,
        "Asia/Calcutta and Asia/Kolkata are the same zone. Every row converts\n"
        "correctly; the GROUP BY splits one office into two smaller ones.",
        transform=ax.transAxes, fontsize=6.8, color=MUTED, va="top",
    )
    _style(ax)


def panel_days(ax: plt.Axes) -> None:
    e = exp_day_bucketing(verbose=False)
    days = sorted(set(list(e["utc_rev"]) + list(e["local_rev"])))
    x = np.arange(len(days))
    w = 0.38
    u = [e["utc_rev"].get(d, 0.0) for d in days]
    l = [e["local_rev"].get(d, 0.0) for d in days]
    ax.bar(x - w / 2, u, w, color=COOL, label="by UTC day")
    ax.bar(x + w / 2, l, w, color=WARM, label="by local day")
    for xi, (uu, ll) in zip(x, zip(u, l)):
        if uu != ll:
            ax.text(xi, max(uu, ll) + 60, f"{ll - uu:+,.0f}", ha="center", fontsize=6.8, color=BAD)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%m-%d") for d in days], fontsize=7)
    ax.set_ylabel("revenue", fontsize=8)
    ax.set_title("E  which day did it happen on", fontsize=9.5, color=INK, loc="left", pad=16)
    ax.legend(fontsize=6.5, loc="lower right", bbox_to_anchor=(1.02, 1.005), ncol=2, frameon=False)
    ax.text(
        0.0, -0.28,
        f"{len(e['moved'])} of 24 events land on a different calendar day. The totals still\n"
        "reconcile exactly, which is why this one survives audits.",
        transform=ax.transAxes, fontsize=6.8, color=MUTED, va="top",
    )
    _style(ax)


def panel_offsets(ax: plt.Axes) -> None:
    f = exp_sub_hour(verbose=False)
    table = sorted(f["table"], key=lambda t: t["offset"].total_seconds())
    labels = [t["zone"].split("/")[-1].replace("_", " ") for t in table]
    vals = [t["offset"].total_seconds() / 3600 for t in table]
    colours = [COOL if t["whole_hour"] else BAD for t in table]
    for h in range(-6, 15):
        ax.axvline(h, color="#ddd6c8", lw=0.7, zorder=0)
    ax.hlines(range(len(table)), 0, vals, color=colours, lw=1.4, zorder=2)
    ax.scatter(vals, range(len(table)), color=colours, s=42, zorder=3)
    for i, (t, v) in enumerate(zip(table, vals)):
        tag = f"{v:+.2f}"
        if t["dst_shift"] and t["dst_shift"] != 60:
            tag += f"   DST {t['dst_shift']:.0f}m"
        ax.text(v + 0.35, i, tag, fontsize=6.8, va="center", color=INK)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlim(-7, 19)
    ax.set_xlabel("offset from UTC (hours) - grey lines are whole hours", fontsize=8)
    ax.set_title("F  offsets are not whole hours", fontsize=9.5, color=INK, loc="left")
    ax.text(
        0.0, -0.30,
        f"{len(f['odd'])} of 7 miss the grid entirely. Lord Howe's clocks change by thirty\n"
        "minutes, not sixty - so even its DST step lands off the hour.",
        transform=ax.transAxes, fontsize=6.8, color=MUTED, va="top",
    )
    _style(ax)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), facecolor=PAPER)
    fig.suptitle(
        "A local timestamp is not a point in time. Six ways the conversion is decided for you.",
        fontsize=13, color=INK, x=0.012, ha="left", y=0.985,
    )
    panel_fallback(axes[0][0])
    panel_gaps(axes[0][1])
    panel_recovery(axes[0][2])
    panel_aliases(axes[1][0])
    panel_days(axes[1][1])
    panel_offsets(axes[1][2])
    fig.text(0.012, 0.012, f"resolved against {tzdata_version()}", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    fig.subplots_adjust(hspace=0.62, wspace=0.30)
    fig.savefig("tz_audit.png", dpi=300, facecolor=PAPER)
    fig.savefig("tz_audit_nb.png", dpi=110, facecolor=PAPER)
    print("wrote tz_audit.png (300 dpi) and tz_audit_nb.png (110 dpi)")


if __name__ == "__main__":
    main()
