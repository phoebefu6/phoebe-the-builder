"""Cost of Delay - score the schedule your ordering produces.

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import codelay as C

st.set_page_config(page_title="Cost of Delay", page_icon="⏱", layout="wide")

st.title("An ordering is not a schedule")
st.caption(
    "Every prioritisation method emits an *order*. What you pay is the **schedule** "
    "that order produces. This prices both, over every one of the 362,880 orderings "
    "of a nine-item backlog. Cost of delay in $k."
)

items = C.backlog()
lin = C.linearised(items)


@st.cache_data(show_spinner="Pricing every ordering...")
def _sweeps():
    return (C.sweep(items), C.sweep(lin), C.all_costs(items),
            C.sweep(items, edges=C.PRECEDENCE))


sr, sl, costs, sp = _sweeps()

# ------------------------------------------------------------------- the backlog
st.subheader("The backlog")
st.caption(
    "`duration` is calendar weeks for one team - what delay is paid in. "
    "`person_weeks` is effort - what estimates are given in. RICE divides by the "
    "second; the schedule is built from the first."
)
st.dataframe(pd.DataFrame([{
    "": k,
    "item": it.name,
    "duration (wks)": it.duration,
    "person-weeks": it.person_weeks,
    "CoD shape": it.cod.kind,
    "rate at week 0": round(it.cod.rate(0.0), 1),
    "mean rate": round(it.cod.mean_rate(), 1),
    "total if last (wk 40)": round(it.cod.cum(C.HORIZON), 1),
    "RICE": round(it.rice),
} for k, it in sorted(items.items())]), hide_index=True, width="stretch")

st.info(
    f"**`soc2-evidence` is quoted at 0/week and is the most expensive item in the "
    f"backlog** ({items['B'].cod.cum(40):.0f} if it lands last). "
    f"**`onboarding-revamp` is quoted at 70/week** - twice anything else - "
    f"and tops out at {items['D'].cod.cum(40):.0f}, because its window saturates. "
    "A rate at one instant cannot represent a cost over an interval."
)

# ---------------------------------------------------------------- the scoreboard
st.subheader("Score the schedule, not the list")
rows = []
for name, f in C.ORDERINGS.items():
    o = f(items)
    v = C.cost_of(o, items)
    rows.append({
        "method": name,
        "ordering": "".join(o),
        "delay cost": round(v, 1),
        "vs optimum": f"+{100 * (v / sr['best'] - 1):.1f}%",
        "percentile": f"{100 * C.percentile_of(costs, v):.1f}%",
        "beaten by a shuffle": "yes" if v > sr["mean"] else "",
        "soc2 ships wk": int(C.completions(o, items)["B"]),
    })
rows.sort(key=lambda r: r["delay cost"])
rows.insert(0, {"method": "EXHAUSTIVE OPTIMUM", "ordering": "".join(sr["best_order"]),
                "delay cost": round(sr["best"], 1), "vs optimum": "-",
                "percentile": "0.0%", "beaten by a shuffle": "",
                "soc2 ships wk": int(C.completions(sr["best_order"], items)["B"])})
rows.append({"method": "random (mean of all)", "ordering": "-",
             "delay cost": round(sr["mean"], 1),
             "vs optimum": f"+{100 * (sr['mean'] / sr['best'] - 1):.1f}%",
             "percentile": "-", "beaten by a shuffle": "",
             "soc2 ships wk": 0})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

losers = [r["method"] for r in rows if r["beaten by a shuffle"]]
st.error(
    f"**{len(losers)} of {len(C.ORDERINGS)} orderings cost more than the average of "
    f"all {len(costs):,} orderings**: {', '.join(losers)}. "
    "The percentile column is exact, not sampled - the enumeration is the population."
)

# ----------------------------------------------------------------- Smith's rule
st.subheader("WSJF is optimal, and it needs four conditions")
c1, c2 = st.columns(2)
cd3l = C.order_cd3_mean(lin)
cd3r = C.order_cd3_mean(items)
with c1:
    st.markdown("**Linearised backlog** — Smith's rule (1956) applies")
    st.metric("exhaustive optimum", f"{sl['best']:.4f}")
    st.metric("CD3 = weight ÷ duration", f"{C.cost_of(cd3l, lin):.4f}",
              delta=f"{C.cost_of(cd3l, lin) - sl['best']:.4f} gap", delta_color="off")
    st.caption(f"Same ordering: `{''.join(cd3l)}` — provably optimal, no search.")
with c2:
    st.markdown("**Real cost shapes** — nothing else changed")
    st.metric("exhaustive optimum", f"{sr['best']:.1f}")
    st.metric("CD3", f"{C.cost_of(cd3r, items):.1f}",
              delta=f"+{C.cost_of(cd3r, items) - sr['best']:.1f} gap "
                    f"(+{100 * (C.cost_of(cd3r, items) / sr['best'] - 1):.1f}%)",
              delta_color="inverse")
    st.caption(f"Optimum `{''.join(sr['best_order'])}` vs CD3 `{''.join(cd3r)}`.")
st.caption(
    "The theorem's four conditions: linear delay cost, one machine, no deadlines, "
    "no precedence. This backlog violates all four."
)

# ------------------------------------------------------------ one method, three
st.subheader("'CD3' does not name an ordering")
st.caption(
    "The formula is cost of delay ÷ duration. 'Cost of delay' is one number pulled "
    "out of a room, and there are three honest ways to pull it."
)
a, b, p = (C.order_cd3_initial(items), C.order_cd3_mean(items),
           C.order_cd3_peak(items))
st.dataframe(pd.DataFrame([
    {"elicitation": "what a week costs us right now", "ordering": "".join(a),
     "delay cost": round(C.cost_of(a, items), 1),
     "percentile": f"{100 * C.percentile_of(costs, C.cost_of(a, items)):.1f}%"},
    {"elicitation": "averaged over the planning window", "ordering": "".join(b),
     "delay cost": round(C.cost_of(b, items), 1),
     "percentile": f"{100 * C.percentile_of(costs, C.cost_of(b, items)):.1f}%"},
    {"elicitation": "the worst week in the window", "ordering": "".join(p),
     "delay cost": round(C.cost_of(p, items), 1),
     "percentile": f"{100 * C.percentile_of(costs, C.cost_of(p, items)):.1f}%"},
]), hide_index=True, width="stretch")
npairs = len(items) * (len(items) - 1) // 2
st.warning(
    f"The same named method, applied in good faith, disagrees with itself about "
    f"**{C.kendall_distance(a, b)} of {npairs} pairs** and spans the "
    f"**{100 * C.percentile_of(costs, C.cost_of(a, items)):.0f}th to the "
    f"{100 * C.percentile_of(costs, C.cost_of(b, items)):.1f}th percentile**. "
    "The elicitation is the decision, not the method name."
)

# ------------------------------------------------------------------- the date
st.subheader("Nobody schedules to the date")
date = items["B"].cod.t_break
dr = [{"method": n, "position": f(items).index("B") + 1,
       "ships week": int(C.completions(f(items), items)["B"]),
       "vs date": int(C.completions(f(items), items)["B"] - date),
       "pays": round(items["B"].cod.cum(C.completions(f(items), items)["B"]), 1)}
      for n, f in C.ORDERINGS.items()]
dr.append({"method": "OPTIMUM", "position": sr["best_order"].index("B") + 1,
           "ships week": int(C.completions(sr["best_order"], items)["B"]),
           "vs date": int(C.completions(sr["best_order"], items)["B"] - date),
           "pays": round(items["B"].cod.cum(C.completions(sr["best_order"], items)["B"]), 1)})
dr.sort(key=lambda r: r["ships week"])
st.dataframe(pd.DataFrame(dr), hide_index=True, width="stretch")
st.success(
    f"The optimum ships it week "
    f"{C.completions(sr['best_order'], items)['B']:.0f} — one week of slack, pays "
    f"nothing. Two methods ship it week 40 and pay "
    f"{items['B'].cod.cum(40):.0f}; three ship it week 4, "
    f"{date - 4:.0f} weeks early, and send four weeks of queue ahead of everything "
    "that was actually bleeding. A date is a constraint, not a high score."
)

# ------------------------------------------------------- the sensitivity knob
st.subheader("The rank is noise. The cost is not.")
c1, c2 = st.columns([1, 2])
with c1:
    sigma = st.slider("lognormal error on every duration estimate (sigma)",
                      0.05, 1.00, 0.35, 0.05)
    trials = st.select_slider("trials", [200, 500, 1000, 2000], value=1000)
res = C.noise_sweep(items, sigma, trials)
with c2:
    m1, m2, m3 = st.columns(3)
    m1.metric("ranking changed", f"{100 * res['reorder_rate']:.1f}%")
    m2.metric("mean realised cost", f"{res['mean']:.1f}",
              delta=f"+{res['mean'] - res['truth_cost']:.1f} vs true durations",
              delta_color="inverse")
    m3.metric("CD3-to-optimum method gap",
              f"{C.cost_of(cd3r, items) - sr['best']:.1f}")
    st.caption(
        f"Rank on the estimate, pay on the truth. p90 {res['p90']:.1f}, "
        f"worst {res['max']:.1f}. Random-shuffle mean is {sr['mean']:.1f}, so a "
        "badly-estimated CD3 still beats a hat — but the *order* it produced is "
        "not reproducible, and the *cost* nearly is."
    )

# ------------------------------------------------------------------- footer
st.divider()
st.markdown(
    f"""
**What to actually do on Monday**

- Write cost of delay as **a rate, a date, and a shape** — not one number per item.
- Score the **schedule** your order implies and publish that number next to the order.
- Elicit cost of delay **the same way every time**; the convention is worth more
  than the method.
- Treat fixed dates as **constraints**, not high scores.
- Do **not** build the parallel-assignment optimiser: it is worth
  {100 * (C.parallel_cost(cd3l, lin, 2) / C.optimal_two_team_assignment(lin)['best'] - 1):.2f}%
  here. Precedence costs {sp['best'] - sr['best']:.1f} at the optimum, while
  ranking as if it were absent and repairing afterwards costs
  {C.cost_of(C.repair_precedence(cd3r, C.PRECEDENCE), items) - sp['best']:.1f}.

Day 158 of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder).
Reproduce every number: `python3 evidence.py` · `python3 -m pytest test_codelay.py -q`
"""
)
