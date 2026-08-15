"""Streamlit front end for the retry-schedule audit.

Run: streamlit run app.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import retry as R

st.set_page_config(page_title="Retry Schedule Audit", layout="wide")

st.title("Retry schedules and the herd they create")
st.caption(
    "A backoff function returns a delay. Pick a policy and a fleet size, and "
    "this reports what the *fleet* does to the dependency when all of it fails "
    "at once - which is the only time retries matter."
)

with st.sidebar:
    st.header("Fleet")
    fleet = st.slider("clients failing together", 10, 2000, 500, step=10)
    outage = st.slider("outage length (s)", 1.0, 120.0, 20.0, step=1.0)
    capacity = st.slider("capacity once recovered (rps)", 1.0, 500.0, 50.0, step=1.0)

    st.header("Schedule")
    base = st.select_slider("base delay (s)",
                            options=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0], value=0.1)
    cap = st.slider("cap (s)", 1.0, 300.0, 20.0, step=1.0)
    attempts = st.slider("max attempts", 1, 30, 10)

    st.header("Context")
    deadline = st.slider("caller deadline (s)", 1.0, 300.0, 30.0, step=1.0)
    tick = st.select_slider("scheduler tick (s)",
                            options=[0.0, 0.01, 0.1, 1.0, 5.0, 60.0], value=1.0)
    layers_txt = st.text_input("nested retry layers (comma separated)", "3, 3")
    seed = st.number_input("seed", 0, 10_000, 7)

try:
    layers = [int(x) for x in layers_txt.split(",") if x.strip()]
except ValueError:
    layers = []
    st.sidebar.error("layers must be integers")

sims = {}
for name in R.POLICY_ORDER:
    s = R.Schedule(name, base, cap, attempts)
    sims[name] = R.simulate(s, fleet=fleet, outage_s=outage,
                            capacity_rps=capacity, seed=int(seed))

VCOLOR = {"herding": "#c0392b", "bursty": "#d98324", "dispersed": "#2d7d5a"}

rows = []
verdicts = {}
findings = {}
for name, sim in sims.items():
    v, fs = R.audit(sim.schedule, fleet, outage, capacity,
                    deadline_s=deadline, nested_layers=layers or None,
                    tick_s=tick or None, seed=int(seed), sim=sim)
    verdicts[name] = v
    findings[name] = fs
    ct = sim.completion_time()
    rows.append({
        "policy": name,
        "verdict": v.value,
        "peak rps (recovery)": round(sim.recovery_peak_rps()),
        "requests": sim.total_requests(),
        "sent while down": sim.wasted_requests(),
        "recovered": sim.succeeded,
        "gave up": sim.gave_up,
        "cleared at (s)": round(ct, 1) if ct else None,
        "E[elapsed] (s)": round(sim.schedule.expected_total(), 1),
    })

df = pd.DataFrame(rows)
st.subheader("Every policy through the same outage")
st.dataframe(df, use_container_width=True, hide_index=True)

best = min(rows, key=lambda r: (r["gave up"], r["peak rps (recovery)"]))
lowest_peak = min(rows, key=lambda r: r["peak rps (recovery)"])
if best["policy"] != lowest_peak["policy"]:
    st.warning(
        f"**{lowest_peak['policy']}** has the lowest peak "
        f"({lowest_peak['peak rps (recovery)']} rps) but loses "
        f"{lowest_peak['gave up']} clients. **{best['policy']}** peaks at "
        f"{best['peak rps (recovery)']} rps and loses {best['gave up']}. Peak "
        f"and recovery are different objectives; a schedule tuned on the first "
        f"can lose on the second."
    )

# -- arrivals -----------------------------------------------------------------

st.subheader("Arrivals against the dependency")
picked = st.multiselect("policies to plot", R.POLICY_ORDER,
                        default=["no_jitter", "equal_jitter", "full_jitter"])
fig, ax = plt.subplots(figsize=(11, 3.6))
upto = max(outage * 3, 60.0)
for name in picked:
    edges, counts = sims[name].histogram(width=1.0, since=0.0, upto=upto)
    ax.plot(edges, counts, lw=1.5, label=name, drawstyle="steps-post")
ax.axvspan(0, outage, color="#c0392b", alpha=0.07)
ax.axhline(capacity, color="#1b1b1f", ls="--", lw=1.0)
ax.set_yscale("symlog", linthresh=10)
ax.set_ylim(0, max(6 * fleet, 100))
ax.set_xlabel("seconds since the fleet failed")
ax.set_ylabel("arrivals per second")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, color="#e3e3e8", lw=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
st.pyplot(fig, use_container_width=True)

# -- the floor ----------------------------------------------------------------

c1, c2 = st.columns(2)
with c1:
    st.subheader("The load floor the cap sets")
    floor = R.Schedule("full_jitter", base, cap, attempts).steady_state_rate(fleet)
    st.metric("steady-state arrivals, full jitter", f"{floor:.0f} rps",
              delta=f"{floor - capacity:+.0f} vs capacity",
              delta_color="inverse")
    st.caption(
        f"Once `base * 2^n` reaches the {cap:g}s cap the jitter window stops "
        f"widening, so the arrival process stops thinning. The floor is "
        f"`fleet / (cap/2)` = {fleet}/{cap/2:g}. Jitter changes its shape, not "
        f"its height - only a smaller retrying fleet lowers it."
    )
with c2:
    st.subheader("Nested retry amplification")
    if layers:
        amp = R.amplification(layers)
        st.metric("requests at the bottom service, per user click", f"{amp}x")
        st.caption(
            f"Layers {layers} multiply. Each one was individually reasonable; "
            f"the product is what the unhealthy dependency actually receives."
        )
    else:
        st.caption("Enter retry counts per layer in the sidebar.")

# -- findings -----------------------------------------------------------------

st.subheader("Findings")
policy = st.selectbox("policy", R.POLICY_ORDER, index=3)
v = verdicts[policy]
st.markdown(
    f"### Verdict: <span style='color:{VCOLOR[v.value]}'>{v.value}</span>",
    unsafe_allow_html=True)
st.caption(
    "The verdict describes the *arrival process*: whether the retries take the "
    "recovering service down again. It is not a statement about whether the "
    "clients recovered - `dispersed` with clients giving up is a real and "
    "common combination, and both are reported."
)
ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
for f in findings[policy]:
    st.markdown(f"{ICON[f.severity.value]} **{f.code}** - {f.message}")
    st.caption(f.detail)

with st.expander("What actually fixes it"):
    st.markdown(
        "- **Jitter fixes the shape, not the volume.** The floor is `fleet/cap`; "
        "jitter only decides whether that load arrives as a spike or a hum.\n"
        "- **Cap the fleet, not just the delay.** A client-side retry budget "
        "(gRPC `retryThrottling`, a token bucket that only refills on success) "
        "reduces the numerator, which is the only term that matters.\n"
        "- **Retry at one layer.** Amplification is a product; pick the layer "
        "with the context to decide and make every other layer pass the error "
        "through.\n"
        "- **Size the budget against the median, not the mean.** For "
        "decorrelated jitter the two differ by nearly half.\n"
        "- **Check the jitter survives your scheduler.** A window narrower than "
        "the timer tick is quantised away and the fleet re-synchronises."
    )
