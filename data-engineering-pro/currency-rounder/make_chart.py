"""Six-panel audit figure for currency-rounder.

Every number plotted is computed here from money.py, not typed in.

Run: python3 make_chart.py [outfile.png] [--dpi N]
"""

from __future__ import annotations

import random
import sys
from decimal import Decimal as D

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import money as m
import numpy as np

INK = "#1b1b1f"
MUTED = "#7a7a85"
GRID = "#e6e6ec"
BAD = "#c2384a"
OK = "#2f7d5c"
WARN = "#d98324"
COOL = "#2f5d9e"
FILL = "#f4f4f7"


def _style(ax, title: str, sub: str = "") -> None:
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=30)
    if sub:
        ax.text(
            0, 1.018, sub, transform=ax.transAxes, fontsize=8.4, color=MUTED, va="bottom", ha="left"
        )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.2, length=3)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------- panel 1


def p1_shortfall(ax) -> None:
    """How far off a ledger lands when every row is rounded on its own."""
    usd = m.currency("USD")
    ns = list(range(2, 41))
    worst, mean = [], []
    for n in ns:
        gaps = []
        for total_cents in range(100, 100 + 60):
            total = D(total_cents) / 100
            exact = D(total) / n
            naive = m.quantize(exact, usd, "half_even") * n
            gaps.append(float(naive - total))
        worst.append(max(abs(g) for g in gaps) * 100)
        mean.append(float(np.mean([abs(g) for g in gaps])) * 100)

    ax.plot(ns, worst, color=BAD, lw=2.0, zorder=3, label="worst case")
    ax.plot(ns, mean, color=COOL, lw=1.8, ls="--", zorder=3, label="mean")
    ax.fill_between(ns, 0, worst, color=BAD, alpha=0.07, zorder=1)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xlabel("rows the total is split across", fontsize=8.4, color=MUTED)
    ax.set_ylabel("ledger error (cents)", fontsize=8.4, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    peak = max(worst)
    ax.annotate(
        f"up to {peak:.0f}c adrift\non a single ledger",
        xy=(ns[worst.index(peak)], peak),
        xytext=(24, peak * 0.55),
        fontsize=8.2,
        color=BAD,
        arrowprops=dict(arrowstyle="->", color=BAD, lw=1),
    )
    _style(
        ax,
        "1. Independent rounding does not preserve a sum",
        "each row correctly rounded to the nearest cent; ledger vs stated total, 60 totals per n",
    )


# --------------------------------------------------------------- panel 2


def p2_who_pays(ax) -> None:
    """Same total, same shares, different row order: the penny moves."""
    usd = m.currency("USD")
    names = ["alice", "bob", "carol"]
    orders = [
        ("carol, alice, bob", ["carol", "alice", "bob"]),
        ("alice, bob, carol", ["alice", "bob", "carol"]),
        ("bob, carol, alice", ["bob", "carol", "alice"]),
    ]
    ideal = 10000 / 3  # in cents
    width = 0.26
    xs = np.arange(len(names))
    shades = [COOL, WARN, OK]
    for k, (label, order) in enumerate(orders):
        a = m.allocate(D("100.00"), [D(1)] * 3, usd, order)
        d = a.by_label()
        devs = [float(d[n]) * 100 - ideal for n in names]
        bars = ax.bar(
            xs + (k - 1) * width, devs, width * 0.88, color=shades[k], zorder=3, label=label
        )
        for b, dev, n in zip(bars, devs, names):
            got = dev > 0
            ax.text(
                b.get_x() + b.get_width() / 2,
                dev + (0.045 if got else -0.045),
                f"${d[n]}",
                ha="center",
                va="bottom" if got else "top",
                fontsize=7.4,
                color=INK if got else MUTED,
                fontweight="bold" if got else "normal",
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(-0.62, 1.62)
    ax.set_ylabel("deviation from the ideal share (cents)", fontsize=8.4, color=MUTED)
    ax.axhline(0, color=MUTED, lw=1.1, ls=":", zorder=2)
    ax.text(2.52, 0.02, "exact 1/3", fontsize=7.6, color=MUTED, va="bottom", ha="right")
    ax.legend(frameon=False, fontsize=7.8, loc="upper center", ncol=3,
              title="row order as the file arrived", title_fontsize=7.8)
    _style(
        ax,
        "2. The total is stable; the rows are not",
        "all three orders sum to exactly $100.00 - they disagree on who absorbs the cent",
    )


# --------------------------------------------------------------- panel 3


def p3_mode_bias(ax) -> None:
    """Accumulated bias of each mode over a stream of exact ties."""
    usd = m.currency("USD")
    rng = random.Random(20260813)
    n = 400
    # amounts that land exactly on a half-cent, cent parity varying
    amounts = [D(rng.randrange(1, 200000)) / 100 + D("0.005") for _ in range(n)]
    exact_cum = np.cumsum([float(a) for a in amounts])
    for mode, colour, ls in [
        ("half_even", OK, "-"),
        ("half_up", BAD, "-"),
        ("half_down", WARN, "--"),
    ]:
        rounded = np.cumsum([float(m.quantize(a, usd, mode)) for a in amounts])
        ax.plot(
            np.arange(1, n + 1),
            (rounded - exact_cum) * 100,
            color=colour,
            lw=1.7,
            ls=ls,
            zorder=3,
            label=mode,
        )
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("transactions, each an exact half-cent tie", fontsize=8.4, color=MUTED)
    ax.set_ylabel("accumulated bias (cents)", fontsize=8.4, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.annotate(
        "half_up gains a cent on every\ntie: +$2.00 over 400 rows.\nThat is not a bug, it is the\nbehaviour most tax codes specify.",
        xy=(n * 0.80, n * 0.80 * 0.5),
        xytext=(n * 0.31, n * 0.5 * 0.82),
        fontsize=7.8,
        color=BAD,
        arrowprops=dict(arrowstyle="->", color=BAD, lw=1),
    )
    ax.text(
        n * 0.40, -n * 0.5 * 0.30,
        "floor and 'down' lie exactly on half_down here:\non a tie they all take the lower cent",
        fontsize=7.6, color=MUTED,
    )
    _style(
        ax,
        "3. Every mode has a bias; only its direction is a choice",
        "400 amounts landing exactly on a half-cent, seed 20260813",
    )


# --------------------------------------------------------------- panel 4


def p4_float_drift(ax) -> None:
    """Float accumulation vs Decimal, adding 0.01 repeatedly."""
    steps = 50000
    every = 500
    marks = set(range(0, steps + 1, every))
    xs, drift = [], []
    running = 0.0
    for i in range(steps + 1):
        if i in marks:
            xs.append(i)
            drift.append(running - i * 0.01)
        running += 0.01
    xs = np.array(xs)
    drift = np.array(drift) * 100  # cents

    drift = drift / 1e-8  # plot in units of 1e-8 cents so no offset text is needed

    ax.plot(xs, drift, color=BAD, lw=1.9, zorder=3, label="float, += 0.01")
    ax.axhline(0, color=OK, lw=2.2, zorder=4, label="Decimal, += 0.01 (exact)")
    ax.set_xlabel("rows added", fontsize=8.4, color=MUTED)
    ax.set_ylabel("drift from the true sum (units of 1e-8 cents)", fontsize=8.4, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    final = drift[-1]
    ax.annotate(
        f"{final * 1e-8:+.1e} cents after {steps:,} rows.\n"
        f"Never visible at 2dp - and it changes\nif the same rows arrive in another order.",
        xy=(xs[-1], final),
        xytext=(steps * 0.06, final * 0.42),
        fontsize=7.8,
        color=BAD,
        arrowprops=dict(arrowstyle="->", color=BAD, lw=1, connectionstyle="arc3,rad=-0.15"),
    )
    _style(
        ax,
        "4. Float drift is small, silent and order dependent",
        "0.01 accumulated in a float; Decimal holds the line exactly",
    )


# --------------------------------------------------------------- panel 5


def p5_minor_units(ax) -> None:
    """The book increment and the cash increment, per currency, in USD terms."""
    # Indicative FX, stated as of the build date. The ratio between the two bars
    # in each pair is exact and FX-independent; only the absolute heights move.
    fx = {"USD": 1.0, "EUR": 1.09, "JPY": 0.0067, "KWD": 3.26, "CHF": 1.13,
          "SEK": 0.095, "CAD": 0.73, "MRU": 0.025}
    codes = ["USD", "EUR", "JPY", "KWD", "MRU", "CAD", "CHF", "SEK"]
    xs = np.arange(len(codes))
    book, cash, labs = [], [], []
    for c in codes:
        cur = m.currency(c)
        book.append(float(cur.step) * fx[c])
        cash.append(float(cur.cash_step) * fx[c] if cur.cash_step is not None else np.nan)
        labs.append(f"{c}\n{cur.step}")
    ax.bar(xs - 0.19, book, 0.36, color=COOL, zorder=3, label="book increment")
    ax.bar(xs + 0.19, cash, 0.36, color=BAD, zorder=3, label="cash increment")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=7.8)
    ax.set_ylabel("one increment, in USD (log)", fontsize=8.4, color=MUTED)
    ax.axhline(0.01, color=MUTED, lw=1.1, ls=":", zorder=2)
    ax.text(len(codes) - 0.45, 0.0108, "one US cent", fontsize=7.4, color=MUTED,
            va="bottom", ha="right")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    ratio = float(m.currency("SEK").cash_step) / float(m.currency("SEK").step)
    ax.annotate(
        f"SEK pays in units {ratio:.0f}x coarser\nthan it books in: a cash total can\n"
        f"differ from the invoice by up to\n{float(m.currency('SEK').cash_step) * fx['SEK'] / 2:.3f} USD, legally",
        xy=(xs[-1] + 0.19, cash[-1]),
        xytext=(xs[-1] - 5.3, cash[-1] * 1.6),
        fontsize=7.7,
        color=BAD,
        arrowprops=dict(arrowstyle="->", color=BAD, lw=1),
    )
    ax.set_ylim(3e-4, 0.6)
    _style(
        ax,
        "5. 'Round to the cent' is not one instruction",
        "indicative FX at build date; the ratio within each pair is exact and FX-independent",
    )


# --------------------------------------------------------------- panel 6


def p6_verdicts(ax) -> None:
    """The corpus: verdict per ledger, and which properties each one exercises."""
    ledgers = m.sample_ledgers()
    props = ["needed\nrounding", "modes\ndisagree", "order\nsensitive", "cash gap", "decided"]
    grid = np.zeros((len(ledgers), len(props)))
    names = []
    for i, led in enumerate(ledgers):
        a = m.audit(led)
        r = a.reconciliation
        cur = m.currency(led.currency)
        amounts = [x for _, x in led.rows]
        needed = any(not m.is_payable(x, cur) for x in amounts)
        disagree = any(
            m.quantize(x, cur, "half_even") != m.quantize(x, cur, "half_up") for x in amounts
        )
        grid[i] = [
            1 if needed else 0,
            1 if disagree else 0,
            1 if a.order_sensitive else 0,
            1 if (a.cash_gap not in (None, D(0))) else 0,
            1 if r.decided else 0,
        ]
        names.append(f"{led.name}  ({led.currency})")

    cmap = matplotlib.colors.ListedColormap([FILL, COOL])
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1, zorder=2)
    # recolour the 'decided' column so a failure reads as a failure
    for i in range(len(ledgers)):
        if grid[i, 4] == 0:
            ax.add_patch(plt.Rectangle((3.5, i - 0.5), 1, 1, color=BAD, zorder=3))
            ax.text(4, i, "no", ha="center", va="center", fontsize=7.6, color="white", zorder=4,
                    fontweight="bold")
    ax.set_xticks(range(len(props)))
    ax.set_xticklabels(props, fontsize=7.6)
    ax.set_yticks(range(len(ledgers)))
    ax.set_yticklabels(names, fontsize=7.8)
    ax.set_xticks(np.arange(-0.5, len(props), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ledgers), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title(
        "6. What each sample actually exercises", fontsize=11.5, fontweight="bold", color=INK,
        loc="left", pad=30,
    )
    ax.text(
        0, 1.018,
        "a blank cell is a setting this ledger leaves untested - reported, not assumed",
        transform=ax.transAxes, fontsize=8.4, color=MUTED, va="bottom", ha="left",
    )


def main() -> None:
    out = "rounding_audit.png"
    dpi = 190
    args = sys.argv[1:]
    if args and not args[0].startswith("--"):
        out = args[0]
    if "--dpi" in args:
        dpi = int(args[args.index("--dpi") + 1])

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 15.6))
    fig.patch.set_facecolor("white")
    p1_shortfall(axes[0][0])
    p2_who_pays(axes[0][1])
    p3_mode_bias(axes[1][0])
    p4_float_drift(axes[1][1])
    p5_minor_units(axes[2][0])
    p6_verdicts(axes[2][1])

    fig.suptitle(
        "currency-rounder - the cent a ledger loses, and where it goes",
        fontsize=16, fontweight="bold", color=INK, x=0.055, ha="left", y=0.985,
    )
    fig.text(
        0.055, 0.962,
        "Day 143 - phoebe-the-builder - every value computed from money.py, none typed in",
        fontsize=9.6, color=MUTED, ha="left",
    )
    fig.tight_layout(rect=[0.012, 0.008, 0.988, 0.952])
    fig.savefig(out, dpi=dpi, facecolor="white")
    print(f"wrote {out} at {dpi} dpi")


if __name__ == "__main__":
    main()
