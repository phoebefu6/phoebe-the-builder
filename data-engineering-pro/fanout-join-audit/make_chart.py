"""audit.png / audit.svg: left, what each join does to the report; right, which check fires on it."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import explore as E  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, RED, BLUE, GREEN = "#1f2933", "#6b7684", "#c0392b", "#1f4e79", "#2e7d4f"


def main() -> None:
    r = json.load(open("results.json"))["scenarios"]
    names = E.SCENARIOS[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.35]})
    for i, k in enumerate(names):
        v = r[k]
        a1.barh(i + 0.2, 100 * v["rows"], height=0.36, color="#a9c8e4", label="rows" if i == 0 else None)
        a1.barh(i - 0.2, 100 * v["worst"], height=0.36, color=BLUE, label="worst report cell" if i == 0 else None)
    a1.axvline(0, color=MUTED, lw=1)
    a1.set_yticks(range(len(names)), names, fontsize=9)
    a1.set_xlabel("change vs the correct join, %", fontsize=9)
    a1.legend(loc="lower right", fontsize=8, frameon=False)
    a1.set_title("Rows moved a little; report cells moved a lot", loc="left", fontsize=10, color=INK)
    for i, k in enumerate(names):
        for j, c in enumerate(E.CHECKS):
            fire = r[k]["fires"][c]
            broken = k not in ("clean", "orphan_dups")
            if fire and broken:
                col, txt = GREEN, "caught"
            elif fire:
                col, txt = RED, "false alarm"
            elif broken:
                col, txt = RED, "MISSED"
            else:
                col, txt = "#e9edf1", "quiet"
            a2.add_patch(plt.Rectangle((j, i - 0.42), 0.94, 0.84, color=col, alpha=0.85 if col != "#e9edf1" else 1))
            a2.text(j + 0.47, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if col != "#e9edf1" else MUTED)
    a2.set_xlim(0, len(E.CHECKS))
    a2.set_ylim(-0.6, len(names) - 0.4)
    a2.set_xticks([j + 0.47 for j in range(len(E.CHECKS))],
                  [c.replace("_", "\n") for c in E.CHECKS], fontsize=8)
    a2.set_yticks(range(len(names)), names, fontsize=9)
    a2.tick_params(length=0)
    for s in a2.spines.values():
        s.set_visible(False)
    a2.set_title("Only the grain check (pk_preserved) is right on every join", loc="left", fontsize=10, color=INK)
    fig.suptitle("Join fan-out, checked before the aggregate (fictional shop, 20,000 orders, seed 212)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    plt.tight_layout()
    fig.savefig("audit.png", dpi=150)
    fig.savefig("audit.svg")
    print("wrote audit.png, audit.svg")


if __name__ == "__main__":
    main()
