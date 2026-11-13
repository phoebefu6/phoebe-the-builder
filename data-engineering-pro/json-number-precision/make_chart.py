"""precision.png / precision.svg: left, which readers get big IDs wrong in which column context; right, whether the
documented fix survives the same five columns."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import explore as E  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

INK, MUTED = "#1f2933", "#6b7684"
COL = {"ok": "#e3efe6", "wrong": "#c0392b", "raised": "#f2c94c"}


def grid(ax, rows, data, title: str) -> None:
    ctx = list(E.CONTEXTS)
    for i, k in enumerate(rows):
        for j, c in enumerate(ctx):
            d = data[k][c]
            state, label = d
            ax.add_patch(Rectangle((j, i), 0.94, 0.88, color=COL[state], lw=0))
            ax.text(j + 0.47, i + 0.44, label, ha="center", va="center", fontsize=7.5,
                    color="white" if state == "wrong" else INK)
    ax.set_xlim(0, len(ctx))
    ax.set_ylim(len(rows), -0.1)
    ax.set_xticks([j + 0.47 for j in range(len(ctx))], ctx, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.44 for i in range(len(rows))], rows, fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=24)


def state(d: dict, n: int) -> tuple:
    if d["raised"]:
        return ("raised", "refused")
    return ("ok", "exact") if d["wrong"] == 0 else ("wrong", f"{d['wrong']:,} wrong")


def main() -> None:
    res = json.load(open("results.json"))["study"]
    rd = res["readers"]
    left = {k: {c: state({"raised": any(rd[k][c][g]["raised"] for g in ("boundary", "snowflake")),
                          "wrong": sum(rd[k][c][g]["wrong"] for g in ("boundary", "snowflake"))}, 0)
                for c in E.CONTEXTS} for k in E.READERS}
    right = {k: {c: state(res["fixes"][k][c], 0) for c in E.CONTEXTS} for k in E.FIXES}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1, 1]})
    grid(a1, list(E.READERS), left, "4,000 IDs past 2**53: one odd row decides the column type")
    grid(a2, list(E.FIXES), right, "The documented fix, same five columns (2,000 snowflake IDs)")
    fig.text(0.01, 0.02, "red = wrong with no error or warning    yellow = the reader refused the file    "
             "green = every ID exact.  Column context = 2,000 IDs plus one extra row. All IDs fictional.",
             fontsize=8, color=MUTED)
    fig.suptitle("The ID lost its last digits: 64-bit IDs through 9 JSON readers, compared to the integer sent",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    plt.savefig("precision.png", dpi=150)
    plt.savefig("precision.svg")
    print("wrote precision.png / precision.svg")


if __name__ == "__main__":
    main()
