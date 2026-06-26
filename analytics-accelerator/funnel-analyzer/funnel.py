"""Funnel Analyzer — core logic.

Answers "where are we losing users?" from raw event data. Given an event log
(one row per user-event) and an ordered list of funnel steps, it computes how
many users reach each step, the conversion vs the previous step and vs the top,
and flags the single biggest drop-off — the step worth fixing first.

Users only count at a step if they also completed every earlier step (a strict,
ordered funnel), so the numbers are monotonically non-increasing and honest.

Pure pandas — no external services or API keys, so it runs standalone in a
notebook or CI. (The Streamlit app adds an interactive Plotly funnel on top.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FunnelStep:
    step: str
    users: int
    pct_of_top: float          # conversion from the first step
    step_conversion: float     # conversion from the previous step
    drop_off: int              # users lost vs previous step
    drop_off_pct: float        # % lost vs previous step


@dataclass
class FunnelResult:
    steps: List[FunnelStep]
    total_users: int
    overall_conversion: float          # last step / first step
    biggest_drop: Optional[str]        # step name with the largest % drop
    biggest_drop_pct: float
    segment_col: Optional[str] = None
    # segment -> overall conversion
    by_segment: Dict[str, float] = field(default_factory=dict)


def compute_funnel(
    events: pd.DataFrame,
    steps: List[str],
    user_col: str = "user_id",
    event_col: str = "event",
) -> FunnelResult:
    """Strict ordered funnel: a user counts at step k only if they hit steps 0..k.

    "Hit" is order-agnostic within the log (we use set membership per user), which
    is the common product-analytics definition for a conversion funnel.
    """
    if len(steps) < 2:
        raise ValueError("A funnel needs at least 2 steps.")
    missing = [s for s in steps if s not in set(events[event_col].unique())]
    # Missing steps are allowed (they'll just show 0 users) but warn-worthy.

    # For each user, the set of events they performed
    user_events = events.groupby(user_col)[event_col].agg(set)

    counts: List[int] = []
    reached_prev = pd.Series(True, index=user_events.index)
    for s in steps:
        did_step = user_events.apply(lambda evs, step=s: step in evs)
        reached = reached_prev & did_step
        counts.append(int(reached.sum()))
        reached_prev = reached  # must keep all earlier steps too

    top = counts[0] if counts[0] > 0 else 1
    funnel_steps: List[FunnelStep] = []
    biggest_drop = None
    biggest_drop_pct = 0.0
    for i, (name, n) in enumerate(zip(steps, counts)):
        prev = counts[i - 1] if i > 0 else n
        step_conv = (n / prev * 100) if prev > 0 else 0.0
        drop = (prev - n) if i > 0 else 0
        drop_pct = (drop / prev * 100) if (i > 0 and prev > 0) else 0.0
        if i > 0 and drop_pct > biggest_drop_pct:
            biggest_drop_pct = drop_pct
            biggest_drop = name
        funnel_steps.append(
            FunnelStep(
                step=name,
                users=n,
                pct_of_top=round(n / top * 100, 1),
                step_conversion=round(step_conv, 1),
                drop_off=drop,
                drop_off_pct=round(drop_pct, 1),
            )
        )

    overall = round(counts[-1] / top * 100, 1)
    _ = missing  # reserved for surfacing in the UI
    return FunnelResult(
        steps=funnel_steps,
        total_users=int(user_events.shape[0]),
        overall_conversion=overall,
        biggest_drop=biggest_drop,
        biggest_drop_pct=round(biggest_drop_pct, 1),
    )


def compute_funnel_by_segment(
    events: pd.DataFrame,
    steps: List[str],
    segment_col: str,
    user_col: str = "user_id",
    event_col: str = "event",
) -> FunnelResult:
    """Overall funnel plus per-segment overall conversion, for comparison."""
    base = compute_funnel(events, steps, user_col, event_col)
    # Map each user to a single segment (first seen)
    user_seg = events.groupby(user_col)[segment_col].first()
    by_seg: Dict[str, float] = {}
    for seg, users in user_seg.groupby(user_seg):
        seg_events = events[events[user_col].isin(users.index)]
        try:
            r = compute_funnel(seg_events, steps, user_col, event_col)
            by_seg[str(seg)] = r.overall_conversion
        except ValueError:
            continue
    base.segment_col = segment_col
    base.by_segment = dict(sorted(by_seg.items(), key=lambda kv: kv[1], reverse=True))
    return base


def steps_to_frame(result: FunnelResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": s.step,
                "users": s.users,
                "% of top": s.pct_of_top,
                "step conversion %": s.step_conversion,
                "dropped": s.drop_off,
                "drop-off %": s.drop_off_pct,
            }
            for s in result.steps
        ]
    )


def sample_events(n_users: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Deterministic event log for a 5-step e-commerce funnel with a clear leak."""
    rng = np.random.default_rng(random_state)
    steps = ["visit", "view_product", "add_to_cart", "checkout", "purchase"]
    # Per-step pass-through probabilities (the big leak is at checkout)
    pass_prob = [1.0, 0.62, 0.55, 0.40, 0.75]
    devices = rng.choice(["mobile", "desktop"], size=n_users, p=[0.6, 0.4])

    rows = []
    base_ts = pd.Timestamp("2026-01-01")
    for uid in range(n_users):
        device = devices[uid]
        # Mobile converts a bit worse at checkout
        reached = True
        for i, step in enumerate(steps):
            p = pass_prob[i]
            if step == "checkout" and device == "mobile":
                p *= 0.85
            if i == 0:
                do_step = True
            else:
                do_step = reached and (rng.random() < p)
            if do_step:
                rows.append(
                    {
                        "user_id": uid,
                        "event": step,
                        "device": device,
                        "ts": base_ts
                        + pd.Timedelta(minutes=int(rng.integers(0, 100000))),
                    }
                )
            else:
                reached = False
    return pd.DataFrame(rows)


DEFAULT_STEPS = ["visit", "view_product", "add_to_cart", "checkout", "purchase"]
