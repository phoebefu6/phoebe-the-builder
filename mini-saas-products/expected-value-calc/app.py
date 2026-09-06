"""Move the ranges, watch the recommendation flip.

    streamlit run app.py
"""

from __future__ import annotations

import evcalc as E
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Expected Value", layout="wide")

st.title("Expected value is a number")
st.caption(
    "It is not a decision. Move the ranges below and watch how little it takes to "
    "reverse the answer."
)

tab_decide, tab_repeat, tab_info = st.tabs(
    ["The decision", "When the bet repeats", "Worth finding out first"]
)

# --------------------------------------------------------------------------
with tab_decide:
    st.markdown(
        "Ranges are **P10 / most-likely / P90** - what people can actually give you. "
        "The most-likely value is the number that gets typed into a spreadsheet, and "
        "for a skewed range it is not the mean."
    )
    cols = st.columns(4)
    ranges = {}
    for col, i in zip(cols, E.INPUTS):
        with col:
            st.caption(f"**{i.name}** · {i.unit}")
            lo = st.number_input("P10", value=float(i.low), key=f"lo{i.name}")
            mid = st.number_input("most likely", value=float(i.mid), key=f"m{i.name}")
            hi = st.number_input("P90", value=float(i.high), key=f"hi{i.name}")
            ranges[i.name] = (lo, mid, hi)

    bad = [k for k, (lo, mid, hi) in ranges.items() if not lo < mid < hi]
    if bad:
        st.error(f"P10 < most-likely < P90 is required. Check: {', '.join(bad)}")
        st.stop()

    @st.cache_data(show_spinner=False)
    def run(spec: tuple, n: int = 60_000) -> dict:
        rng = np.random.default_rng(E.RNG_SEED)
        draws = {
            name: E.Input(name, lo, mid, hi, "").draw(rng, n)
            for name, (lo, mid, hi) in spec
        }
        out = {o: E.value_of_option(o, **draws) for o in E.OPTIONS}
        out.update({f"input:{k}": v for k, v in draws.items()})
        return out

    sims = run(tuple(sorted((k, v) for k, v in ranges.items())))
    mids = {k: np.array([v[1]]) for k, v in ranges.items()}
    typed = {o: float(E.value_of_option(o, **mids)[0]) for o in E.OPTIONS}
    ev = {o: float(sims[o].mean()) for o in E.OPTIONS}
    best_ev = max(E.OPTIONS, key=lambda o: ev[o])
    p_build = float(np.mean(sims["build"] > sims["buy"]))

    st.subheader("What the spreadsheet says, and what the simulation says")
    st.dataframe(
        pd.DataFrame(
            {
                "typed estimate": [typed[o] for o in E.OPTIONS],
                "expected value": [ev[o] for o in E.OPTIONS],
                "P10": [float(np.quantile(sims[o], 0.1)) for o in E.OPTIONS],
                "median": [float(np.median(sims[o])) for o in E.OPTIONS],
                "P90": [float(np.quantile(sims[o], 0.9)) for o in E.OPTIONS],
                "P(loss)": [float(np.mean(sims[o] < 0)) for o in E.OPTIONS],
            },
            index=list(E.OPTIONS),
        ).style.format({"P(loss)": "{:.1%}"}, precision=0),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Highest expected value", best_ev)
    c2.metric("P(build beats buy)", f"{p_build:.1%}")
    c3.metric("EV gap", f"{ev['build'] - ev['buy']:,.0f}")

    if (ev["build"] > ev["buy"]) != (p_build > 0.5):
        winner = "build" if ev["build"] > ev["buy"] else "buy"
        loser = "buy" if winner == "build" else "build"
        st.error(
            f"**These disagree.** `{winner}` has the higher expected value and wins "
            f"{(p_build if winner == 'build' else 1 - p_build):.1%} of the time. Expected "
            f"value ranks the mean; a longer tail on `{winner}` lifts its average while "
            f"most individual futures land below `{loser}`. Which one matters depends on "
            "whether this decision repeats - see the next tab."
        )
    else:
        st.success(
            "Expected value and most-likely-to-win agree here. Nudge a range and watch "
            "how quickly they stop agreeing."
        )

    st.subheader("What would have to be true")
    rows = []
    for i in E.INPUTS:
        lo, mid, hi = ranges[i.name]

        def gap(x: float, name: str = i.name) -> float:
            args = {k: np.array([v[1]]) for k, v in ranges.items()}
            args[name] = np.array([x])
            return float(E.value_of_option("build", **args)[0]
                         - E.value_of_option("buy", **args)[0])

        if gap(lo) * gap(hi) > 0:
            rows.append({"input": i.name, "typed": mid, "flips at": None,
                         "note": "never flips inside the plausible range"})
            continue
        a, b = lo, hi
        for _ in range(120):
            m = (a + b) / 2
            if gap(a) * gap(m) <= 0:
                b = m
            else:
                a = m
        sw = (a + b) / 2
        rows.append({"input": i.name, "typed": mid, "flips at": sw,
                     "note": f"{abs(sw - mid) / max(abs(mid), 1e-9):.0%} from the estimate"})
    st.dataframe(pd.DataFrame(rows).set_index("input"), use_container_width=True)
    st.caption(
        "This is the line worth carrying out of the meeting. A point estimate invites "
        "agreement; a switching point invites somebody to go and check."
    )

    st.subheader("The distribution the single number came from")
    chart = pd.DataFrame({o: sims[o] for o in ("build", "buy")}).sample(
        4000, random_state=0
    )
    st.bar_chart(
        pd.DataFrame({
            "build": np.histogram(chart["build"], bins=60,
                                  range=(-600_000, 900_000))[0],
            "buy": np.histogram(chart["buy"], bins=60, range=(-600_000, 900_000))[0],
        }, index=np.round(np.linspace(-600_000, 900_000, 60)).astype(int)),
        height=300,
    )

# --------------------------------------------------------------------------
with tab_repeat:
    st.markdown(
        f"A gamble: **{E.P_UP:.0%}** chance of **×{E.UP}**, otherwise **×{E.DOWN}**. "
        "The average multiplier per round is above 1, so expected value says take it "
        "with everything you have."
    )
    stake = st.slider("Fraction of the bankroll staked each round", 0.02, 1.0, 1.0, 0.02)
    rounds = st.slider("Rounds", 20, 500, 250, 10)
    res = E.trajectories(rounds=rounds, fraction=stake)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average multiplier / round", f"{E.ensemble_growth():.4f}")
    c2.metric("What one run experiences", f"{E.time_average_growth():.4f}")
    c3.metric("Mean final wealth", f"{res['mean']:,.2f}x")
    c4.metric("Median final wealth",
              f"{res['median']:.2e}x" if res["median"] < 0.01
              else f"{res['median']:,.2f}x")

    st.caption(
        f"{res['p_below_start']:.1%} of runs end below where they started; "
        f"{res['p_ruin_99pct']:.1%} lose 99% of the stake."
    )
    if res["median"] < 1.0 < res["mean"]:
        st.error(
            f"**The average rises and the runs fall.** Mean {res['mean']:,.0f}x, median "
            f"{res['median']:.2e}x. When payoffs multiply, the arithmetic mean is carried "
            "by a vanishing set of paths nobody is on. Maximising expected value per "
            f"round is the instruction that produced this. Maximising expected *log* "
            f"wealth gives a stake of **{E.kelly_fraction():.0%}**."
        )
    elif abs(stake - E.kelly_fraction()) < 0.03:
        st.success(
            f"This is the Kelly stake ({E.kelly_fraction():.0%}) - it maximises the "
            "growth rate a single run actually experiences."
        )

    st.dataframe(
        pd.DataFrame(E.sizing_comparison(rounds=rounds)).style.format({
            "fraction": "{:.3f}", "mean": "{:,.2f}", "median": "{:,.4f}",
            "p_below_start": "{:.1%}", "p_ruin_99pct": "{:.1%}",
        }),
        use_container_width=True, hide_index=True,
    )

# --------------------------------------------------------------------------
with tab_info:
    v = E.evpi()
    st.markdown(
        "**Expected value of perfect information** - what the decision would be worth "
        "if you knew the answer, minus what it is worth now. It is the ceiling on any "
        "study, pilot or spike, and it is computable *before* commissioning one."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Best without information", f"{v['best_without_information']:,.0f}")
    c2.metric("If the future were known", f"{v['with_perfect_information']:,.0f}")
    c3.metric("EVPI", f"{v['evpi']:,.0f}",
              delta=f"{v['evpi'] / v['best_without_information']:.0%} of the decision")

    info = {k: val for k, val in E.information_value().items() if not k.startswith("_")}
    st.subheader("What is one input worth?")
    st.dataframe(
        pd.DataFrame({"most a study could be worth": info}).sort_values(
            "most a study could be worth", ascending=False
        ).style.format(precision=0),
        use_container_width=True,
    )
    top = max(info, key=info.get)
    worst = min(info, key=info.get)
    st.warning(
        f"A study of **{top}** is worth up to **{info[top]:,.0f}**. The same study of "
        f"**{worst}** is worth **{info[worst]:,.0f}** - it cannot repay a single "
        f"afternoon. Both would be proposed in the same meeting with the same "
        f"seriousness.\n\nThe parts sum to {sum(info.values()):,.0f} against an EVPI of "
        f"{v['evpi']:,.0f}: information is **not additive**. The second study only pays "
        "where the first left the decision open."
    )
