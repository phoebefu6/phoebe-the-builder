"""Render type_benchmark.png - the three strategies on the same 17 columns."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from type_infer import run_benchmark

BUCKETS = ["exact", "wide", "untyped", "unsafe", "lossy"]
COLORS = {"exact": "#1b7f5f", "wide": "#5b8ac7", "untyped": "#9aa0a6",
          "unsafe": "#d98317", "lossy": "#b3312c"}
LABEL = {
    "exact": "exact - matches the hand-written answer",
    "wide": "wide - safe, just wider than necessary",
    "untyped": "untyped - text where a safe type existed",
    "unsafe": "unsafe - round-trips but semantically unproven",
    "lossy": "lossy - would corrupt a real value",
}


def main(path: str = "type_benchmark.png") -> None:
    bench = run_benchmark()
    names = list(bench)
    counts = {b: [sum(1 for v in bench[n].values() if v == b) for n in names] for b in BUCKETS}
    total = len(next(iter(bench.values())))

    fig, ax = plt.subplots(figsize=(11, 3.9))
    left = [0.0] * len(names)
    for b in BUCKETS:
        ax.barh(names, counts[b], left=left, color=COLORS[b], label=LABEL[b], height=0.55)
        for i, (v, left_edge) in enumerate(zip(counts[b], left)):
            if v:
                ax.text(left_edge + v / 2, i, str(v), ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")
        left = [left_edge + v for left_edge, v in zip(left, counts[b])]

    ax.set_xlim(0, total)
    ax.set_xlabel(f"columns (of {total})")
    ax.set_title("Same 17 columns, three ways to type them", fontsize=13, fontweight="bold", pad=14)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
