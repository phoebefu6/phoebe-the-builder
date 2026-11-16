"""dupkeys.png / dupkeys.svg: left, which value each reader returns per fixture; right, what each reader makes of the
2,000-line order feed's 46 refunds."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

INK, MUTED = "#1f2933", "#6b7684"
COL = {"last": "#e3efe6", "ok": "#e3efe6", "first": "#c0392b", "refuse": "#f2c94c", "other": "#8e44ad",
       "WRONG": "#c0392b"}


def main() -> None:
    res = json.load(open("results.json"))["study"]
    m, fd = res["matrix"], res["feed"]
    readers, fx = list(m), list(next(iter(m.values())))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for i, r in enumerate(readers):
        for j, f in enumerate(fx):
            v = m[r][f]
            c = COL[v.split(" ")[0]]
            a1.add_patch(Rectangle((j, i), 0.94, 0.86, color=c, lw=0))
            a1.text(j + 0.47, i + 0.43, v, ha="center", va="center", fontsize=7.5,
                    color="white" if c in ("#c0392b", "#8e44ad") else INK)
    a1.set_xlim(0, len(fx))
    a1.set_ylim(len(readers), -0.1)
    a1.set_xticks([j + 0.47 for j in range(len(fx))], [f.replace(" (", "\n(") for f in fx], fontsize=8)
    a1.xaxis.tick_top()
    a1.set_yticks([i + 0.43 for i in range(len(readers))], readers, fontsize=8.5)
    a1.tick_params(length=0)
    for s in a1.spines.values():
        s.set_visible(False)
    a1.set_title('{"a":1,"a":2}: last wins in 7 readers, first in 3, 4 refuse', loc="left", fontsize=10,
                 color=INK, pad=34)
    t = fd["_truth"]
    vals = [fd[r]["refunds"] for r in readers]
    cols = ["#f2c94c" if fd[r]["raised"] else ("#7fb38f" if v == t["refunds"] else "#c0392b")
            for r, v in zip(readers, vals)]
    a2.barh(range(len(readers)), vals, color=cols, height=0.7)
    for i, r in enumerate(readers):
        lab = "refused whole file" if fd[r]["raised"] else (f"{vals[i]} (${fd[r]['amount']:,.0f})" if vals[i]
                                                             else "0 - no error")
        a2.text(max(vals[i], 0) + 0.8, i, lab, va="center", fontsize=7.5, color=INK)
    a2.axvline(t["refunds"], color=MUTED, lw=0.8, ls="--")
    a2.set_ylim(len(readers) - 0.5, -0.5)
    a2.set_yticks([])
    a2.set_xlim(0, t["refunds"] * 1.6)
    a2.set_xlabel(f"refunds seen of {t['refunds']} (dashed = intent)", fontsize=8.5, color=MUTED)
    for s in ("top", "right", "left"):
        a2.spines[s].set_visible(False)
    a2.set_title(f"Order feed: {t['refunds']} refund patches, ${t['amount']:,.0f}", loc="left", fontsize=10,
                 color=INK, pad=34)
    fig.text(0.01, 0.015, "green = the later value (or correct)   red = the earlier value, or the wrong column   "
             "yellow = refused   purple = neither value intact.  All data fictional, seeded Generator(215).",
             fontsize=8, color=MUTED)
    fig.suptitle("Which value wins? One object with a repeated key through 14 JSON readers", x=0.01, ha="left",
                 fontsize=12, color=INK)
    plt.tight_layout(rect=(0, 0.035, 1, 0.97))
    plt.savefig("dupkeys.png", dpi=150)
    plt.savefig("dupkeys.svg")
    print("wrote dupkeys.png / dupkeys.svg")


if __name__ == "__main__":
    main()
