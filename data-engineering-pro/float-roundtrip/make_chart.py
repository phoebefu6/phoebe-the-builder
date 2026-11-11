"""roundtrip.png / roundtrip.svg: left, how much of each value group comes back changed; right, the computed
column across 30 more worlds next to what a typed-amount fixture shows."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import explore as E  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, BLUE, PALE, RED = "#1f2933", "#6b7684", "#1f4e79", "#a9c8e4", "#c0392b"


def main() -> None:
    res = json.load(open("results.json"))
    wr, rng = res["study"]["writers"], res["seeds"]["range"]
    names = list(E.WRITERS)[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})
    marks = {"entered": ("o", PALE), "computed": ("s", BLUE), "measured": ("^", MUTED)}
    for g, (m, c) in marks.items():
        xs = [100 * (wr[k][g]["changed"] + wr[k][g]["lost"]) / E.N for k in names]
        a1.scatter(xs, range(len(names)), marker=m, s=46, color=c, label=g, zorder=3, edgecolor="white", lw=0.6)
    a1.set_yticks(range(len(names)), names, fontsize=9)
    a1.set_xlim(-4, 104)
    a1.set_xlabel("% of values that came back with different bits", fontsize=9)
    a1.grid(axis="x", color="#e4e7eb", zorder=0)
    a1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.92), fontsize=8, frameon=False, title="value group", title_fontsize=8)
    a1.set_title("Typed amounts survive almost everything; computed ones do not", loc="left", fontsize=10,
                 color=INK)
    bad = [k for k in E.WRITERS if rng[k][1] > 0]
    for i, k in enumerate(bad[::-1]):
        lo, hi = rng[k]
        a2.plot([100 * lo, 100 * hi], [i, i], color=BLUE, lw=5, solid_capstyle="round")
        a2.scatter([100 * (1 - wr[k]["entered"]["exact"] / E.N)], [i], marker="o", s=46, color=PALE,
                   edgecolor=INK, lw=0.6, zorder=3)
        a2.text(100 * hi + 2, i, f"{100 * hi:.0f}%", va="center", fontsize=8, color=INK)
    a2.set_yticks(range(len(bad)), bad[::-1], fontsize=9)
    a2.set_xlim(-4, 112)
    a2.set_xlabel("% of computed join keys that no longer match (bar: 30 worlds)\n"
                  "circle: what a fixture of typed amounts shows", fontsize=9)
    a2.grid(axis="x", color="#e4e7eb", zorder=0)
    a2.set_title("The ledger total does not move; the keys do", loc="left", fontsize=10, color=INK)
    for a in (a1, a2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("One float64 column, 15 writers, read back and compared bit for bit (all values fictional)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    plt.tight_layout()
    plt.savefig("roundtrip.png", dpi=150)
    plt.savefig("roundtrip.svg")
    print("wrote roundtrip.png / roundtrip.svg")


if __name__ == "__main__":
    main()
