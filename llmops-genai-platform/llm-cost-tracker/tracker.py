from __future__ import annotations

# LLM Cost & Token Tracker - core logic.
#
# Surprise API bills happen because token usage is invisible until the invoice
# lands. This module logs every call (model, prompt/completion tokens, cost,
# tag), estimates tokens offline when a provider count isn't handy, and rolls
# usage up by model, tag, and day - with a monthly budget check. Fully
# offline: rates live in an editable table, no network calls.
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Pricing table - USD per 1,000,000 tokens (input, output).
# EXAMPLE rates for illustration - edit to match your actual contract.
# --------------------------------------------------------------------------

PRICING: Dict[str, Dict[str, float]] = {
    "claude-opus":   {"in": 15.00, "out": 75.00},
    "claude-sonnet": {"in": 3.00,  "out": 15.00},
    "claude-haiku":  {"in": 0.80,  "out": 4.00},
    "gpt-large":     {"in": 5.00,  "out": 15.00},
    "gpt-mini":      {"in": 0.15,  "out": 0.60},
}


def estimate_tokens(text: str) -> int:
    """Rough offline token estimate (~4 chars/token). Use provider counts when
    available; this is a fallback for logging calls you didn't instrument."""
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def call_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Cost in USD for a single call given token counts."""
    if model not in PRICING:
        raise KeyError(f"unknown model '{model}'. Known: {sorted(PRICING)}")
    rate = PRICING[model]
    return round(
        prompt_tokens / 1_000_000 * rate["in"]
        + completion_tokens / 1_000_000 * rate["out"],
        6,
    )


@dataclass
class CallRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    tag: str = "default"  # feature / team / route label
    day: str = "2026-07-01"  # ISO date - passed in (offline: no clock)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CostTracker:
    """Append-only log of LLM calls with rollups and a budget check."""

    def __init__(self, monthly_budget_usd: Optional[float] = None) -> None:
        self.records: List[CallRecord] = []
        self.monthly_budget_usd = monthly_budget_usd

    def log(
        self,
        model: str,
        prompt: str | int,
        completion: str | int,
        tag: str = "default",
        day: str = "2026-07-01",
    ) -> CallRecord:
        """Log a call. prompt/completion may be raw text (estimated) or an int
        token count (used as-is)."""
        pt = prompt if isinstance(prompt, int) else estimate_tokens(prompt)
        ct = completion if isinstance(completion, int) else estimate_tokens(completion)
        rec = CallRecord(model, pt, ct, call_cost(model, pt, ct), tag, day)
        self.records.append(rec)
        return rec

    # ---- rollups ----------------------------------------------------------

    @property
    def total_cost(self) -> float:
        return round(sum(r.cost for r in self.records), 4)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    def _rollup(self, key: str) -> List[dict]:
        agg: Dict[str, Dict[str, float]] = {}
        for r in self.records:
            k = getattr(r, key)
            a = agg.setdefault(k, {"calls": 0, "tokens": 0, "cost": 0.0})
            a["calls"] += 1
            a["tokens"] += r.total_tokens
            a["cost"] += r.cost
        rows = [
            {key: k, "calls": int(v["calls"]), "tokens": int(v["tokens"]),
             "cost_usd": round(v["cost"], 4)}
            for k, v in agg.items()
        ]
        rows.sort(key=lambda x: x["cost_usd"], reverse=True)
        return rows

    def by_model(self) -> List[dict]:
        return self._rollup("model")

    def by_tag(self) -> List[dict]:
        return self._rollup("tag")

    def by_day(self) -> List[dict]:
        rows = self._rollup("day")
        rows.sort(key=lambda x: x["day"])
        return rows

    def budget_status(self) -> dict:
        """Where total spend sits against the monthly budget."""
        spent = self.total_cost
        if self.monthly_budget_usd is None:
            return {"budget": None, "spent": spent, "pct": None, "over": False}
        pct = round(spent / self.monthly_budget_usd * 100, 1) if self.monthly_budget_usd else 0.0
        return {
            "budget": self.monthly_budget_usd,
            "spent": spent,
            "pct": pct,
            "over": spent > self.monthly_budget_usd,
            "remaining": round(self.monthly_budget_usd - spent, 4),
        }


# --------------------------------------------------------------------------
# Sample traffic for the demo / smoke test
# --------------------------------------------------------------------------


def sample_tracker() -> CostTracker:
    t = CostTracker(monthly_budget_usd=50.0)
    # (model, prompt_tokens, completion_tokens, tag, day, repeat)
    traffic = [
        ("claude-opus", 4000, 1200, "contract-analysis", "2026-07-01", 12),
        ("claude-sonnet", 1500, 600, "chat-support", "2026-07-01", 40),
        ("claude-haiku", 800, 200, "classify-tickets", "2026-07-02", 200),
        ("gpt-large", 2000, 800, "summarize", "2026-07-02", 25),
        ("claude-opus", 4000, 1500, "contract-analysis", "2026-07-03", 10),
        ("gpt-mini", 500, 150, "autocomplete", "2026-07-03", 500),
        ("claude-sonnet", 1500, 700, "chat-support", "2026-07-03", 60),
    ]
    for model, pt, ct, tag, day, n in traffic:
        for _ in range(n):
            t.log(model, pt, ct, tag=tag, day=day)
    return t


if __name__ == "__main__":
    t = sample_tracker()
    print(f"Total: {t.total_tokens:,} tokens · ${t.total_cost}")
    print("\nBy model:")
    for r in t.by_model():
        print(f"  {r['model']:15} {r['calls']:>5} calls  ${r['cost_usd']:>8}")
    print("\nBy tag:")
    for r in t.by_tag():
        print(f"  {r['tag']:20} ${r['cost_usd']:>8}")
    b = t.budget_status()
    print(f"\nBudget: ${b['spent']} / ${b['budget']} ({b['pct']}%)"
          f" {'⚠️ OVER' if b['over'] else 'ok'}")
