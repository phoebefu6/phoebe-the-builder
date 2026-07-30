from __future__ import annotations

# Token & Cost Estimator - core logic.
#
# Day 85 (llm-cost-tracker) answers "what did we spend?" after the calls.
# This module answers the question you actually get asked in the design review:
# "what will this feature cost per month, and on which model?" - BEFORE any
# integration is written.
#
# You describe the workload once (prompt size, RAG context, expected output,
# monthly call volume, retry rate, cache hit rate) and it projects monthly cost
# across every candidate model, finds the break-even volume between two models,
# and sweeps sensitivity so you know which lever actually moves the bill.
#
# Fully offline: no tokenizer downloads, no network calls. Rates are an
# editable table - replace with your contract.
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Pricing table - USD per 1,000,000 tokens (input, output).
# EXAMPLE rates for illustration only. Edit to match your actual contract.
# --------------------------------------------------------------------------

PRICING: Dict[str, Dict[str, float]] = {
    "premium-large": {"in": 15.00, "out": 75.00},
    "balanced-mid": {"in": 3.00, "out": 15.00},
    "fast-small": {"in": 0.80, "out": 4.00},
    "budget-nano": {"in": 0.15, "out": 0.60},
}

# Characters per token, by content type. English prose sits near 4.0; code and
# JSON are denser in punctuation so they tokenize worse (more tokens per char).
CHARS_PER_TOKEN: Dict[str, float] = {
    "prose": 4.0,
    "code": 3.0,
    "json": 2.8,
    "cjk": 1.5,
}


def estimate_tokens(text: str, kind: str = "prose") -> int:
    """Estimate token count for `text` without loading a tokenizer.

    Deliberately heuristic: a design-time estimate wants to be right within
    ~10-15%, not exact. Exact counts come from the provider at runtime (that's
    Day 85's job). Unknown `kind` falls back to prose.
    """
    if not text:
        return 0
    cpt = CHARS_PER_TOKEN.get(kind, CHARS_PER_TOKEN["prose"])
    return max(1, math.ceil(len(text) / cpt))


@dataclass
class Workload:
    """One feature's LLM traffic, described at design time.

    All token fields are PER CALL. `calls_per_month` is the volume driver.
    """

    name: str
    system_tokens: int = 0
    user_tokens: int = 0
    context_tokens: int = 0  # retrieved RAG chunks, few-shot examples, tool schemas
    output_tokens: int = 0
    calls_per_month: int = 0
    retry_rate: float = 0.0  # 0.15 => 15% of calls are retried once
    cache_hit_rate: float = 0.0  # fraction served from cache at ~zero cost
    kind: str = "prose"

    def __post_init__(self) -> None:
        # Edge case that silently wrecks a forecast: a rate given as a percent
        # (15) instead of a fraction (0.15) would inflate cost 100x. Catch it
        # loudly rather than shipping a wrong number into a budget deck.
        rates = (("retry_rate", self.retry_rate), ("cache_hit_rate", self.cache_hit_rate))
        for label, rate in rates:
            if not 0.0 <= rate <= 1.0:
                raise ValueError(
                    f"{self.name}: {label}={rate} must be a fraction in [0, 1] "
                    f"(use 0.15 for 15%, not 15)"
                )

    @property
    def input_tokens(self) -> int:
        return self.system_tokens + self.user_tokens + self.context_tokens

    @property
    def billed_calls(self) -> float:
        """Calls that actually reach the provider.

        Cache hits cost nothing; retries are extra billed attempts on the
        misses. Order matters - retrying a cache hit is free.
        """
        misses = self.calls_per_month * (1.0 - self.cache_hit_rate)
        return misses * (1.0 + self.retry_rate)


def call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of a single call in USD."""
    if model not in PRICING:
        raise KeyError(f"unknown model {model!r}; known: {sorted(PRICING)}")
    p = PRICING[model]
    return (input_tokens * p["in"] + output_tokens * p["out"]) / 1_000_000


def project(workload: Workload, model: str) -> Dict[str, float]:
    """Project one workload on one model."""
    per_call = call_cost(model, workload.input_tokens, workload.output_tokens)
    billed = workload.billed_calls
    monthly = per_call * billed
    in_cost = workload.input_tokens * PRICING[model]["in"]
    out_cost = workload.output_tokens * PRICING[model]["out"]
    in_share = in_cost / max(1e-12, in_cost + out_cost)
    return {
        "workload": workload.name,
        "model": model,
        "input_tokens": workload.input_tokens,
        "output_tokens": workload.output_tokens,
        "per_call_usd": round(per_call, 6),
        "billed_calls": round(billed, 1),
        "monthly_usd": round(monthly, 2),
        "annual_usd": round(monthly * 12, 2),
        "input_cost_share": round(in_share, 3),
    }


def compare_models(
    workload: Workload, models: Optional[List[str]] = None
) -> List[Dict[str, float]]:
    """Project one workload across candidate models, cheapest first."""
    models = models or list(PRICING)
    rows = [project(workload, m) for m in models]
    return sorted(rows, key=lambda r: r["monthly_usd"])


def portfolio(workloads: List[Workload], model: str) -> Dict[str, float]:
    """Roll several workloads up onto one model - the whole-feature bill."""
    rows = [project(w, model) for w in workloads]
    total = round(sum(r["monthly_usd"] for r in rows), 2)
    biggest = max(rows, key=lambda r: r["monthly_usd"]) if rows else None
    return {
        "model": model,
        "monthly_usd": total,
        "annual_usd": round(total * 12, 2),
        "rows": rows,
        "top_line_item": biggest["workload"] if biggest else None,
        "top_line_share": round(biggest["monthly_usd"] / total, 3) if biggest and total else 0.0,
    }


def break_even_volume(
    workload: Workload, cheap_model: str, premium_model: str, monthly_budget: float
) -> Dict[str, float]:
    """How many calls/month each model buys for a fixed budget.

    The number that ends the "can we just use the big model?" argument: budget
    divided by per-call cost, on the same workload shape.
    """
    out = {}
    for label, model in (("cheap", cheap_model), ("premium", premium_model)):
        per_call = call_cost(model, workload.input_tokens, workload.output_tokens)
        per_billed = per_call * (1.0 + workload.retry_rate)
        # calls the budget covers, grossing back up for cache hits (which are free)
        billable = monthly_budget / per_billed if per_billed else float("inf")
        covered = billable / max(1e-12, 1.0 - workload.cache_hit_rate)
        out[label] = {
            "model": model,
            "per_call_usd": round(per_call, 6),
            "calls_covered": int(covered),
        }
    ratio = (
        out["premium"]["per_call_usd"] / out["cheap"]["per_call_usd"]
        if out["cheap"]["per_call_usd"]
        else float("inf")
    )
    out["premium_multiple"] = round(ratio, 1)
    out["monthly_budget"] = monthly_budget
    return out


def sensitivity(
    workload: Workload,
    model: str,
    lever: str,
    factors: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0),
) -> List[Dict[str, float]]:
    """Scale one lever and re-project - shows which knob actually moves the bill.

    Levers: 'volume', 'output_tokens', 'context_tokens', 'cache_hit_rate'.
    """
    valid = {"volume", "output_tokens", "context_tokens", "cache_hit_rate"}
    if lever not in valid:
        raise ValueError(f"lever must be one of {sorted(valid)}")
    base = project(workload, model)["monthly_usd"]
    rows = []
    for f in factors:
        kwargs = dict(
            name=workload.name,
            system_tokens=workload.system_tokens,
            user_tokens=workload.user_tokens,
            context_tokens=workload.context_tokens,
            output_tokens=workload.output_tokens,
            calls_per_month=workload.calls_per_month,
            retry_rate=workload.retry_rate,
            cache_hit_rate=workload.cache_hit_rate,
            kind=workload.kind,
        )
        if lever == "volume":
            kwargs["calls_per_month"] = int(workload.calls_per_month * f)
        elif lever == "cache_hit_rate":
            # a *fraction* can't scale past 1.0 - clamp instead of raising
            kwargs["cache_hit_rate"] = min(1.0, workload.cache_hit_rate * f)
        else:
            kwargs[lever] = int(kwargs[lever] * f)
        monthly = project(Workload(**kwargs), model)["monthly_usd"]
        rows.append(
            {
                "lever": lever,
                "factor": f,
                "monthly_usd": monthly,
                "delta_usd": round(monthly - base, 2),
                "delta_pct": round((monthly - base) / base * 100, 1) if base else 0.0,
            }
        )
    return rows


def sample_workloads() -> List[Workload]:
    """A realistic three-feature GenAI product, priced before it is built."""
    return [
        Workload(
            name="support-rag-bot",
            system_tokens=600,
            user_tokens=120,
            context_tokens=4_000,  # 8 retrieved chunks
            output_tokens=350,
            calls_per_month=40_000,
            retry_rate=0.05,
            cache_hit_rate=0.30,
        ),
        Workload(
            name="ticket-classifier",
            system_tokens=400,
            user_tokens=250,
            context_tokens=0,
            output_tokens=20,  # just a label
            calls_per_month=200_000,
            retry_rate=0.02,
            cache_hit_rate=0.10,
            kind="json",
        ),
        Workload(
            name="monthly-report-writer",
            system_tokens=900,
            user_tokens=6_000,
            context_tokens=12_000,
            output_tokens=3_000,
            calls_per_month=300,
            retry_rate=0.10,
            cache_hit_rate=0.0,
        ),
    ]


def main() -> None:
    workloads = sample_workloads()
    print("=" * 74)
    print("TOKEN & COST ESTIMATOR - pre-flight forecast (example rates)")
    print("=" * 74)

    for w in workloads:
        print(f"\n{w.name}  |  {w.input_tokens:,} in / {w.output_tokens:,} out per call")
        print(f"  {w.calls_per_month:,} calls/mo, {w.cache_hit_rate:.0%} cached, "
              f"{w.retry_rate:.0%} retried -> {w.billed_calls:,.0f} billed calls")
        for r in compare_models(w):
            print(f"    {r['model']:<14} ${r['per_call_usd']:.6f}/call   "
                  f"${r['monthly_usd']:>10,.2f}/mo   ${r['annual_usd']:>11,.2f}/yr")

    print("\n" + "-" * 74)
    print("WHOLE-PRODUCT BILL")
    for m in PRICING:
        p = portfolio(workloads, m)
        print(f"  {m:<14} ${p['monthly_usd']:>10,.2f}/mo   "
              f"top line item: {p['top_line_item']} ({p['top_line_share']:.0%})")

    print("\n" + "-" * 74)
    print("BREAK-EVEN on a $500/mo budget (support-rag-bot)")
    be = break_even_volume(workloads[0], "fast-small", "premium-large", 500.0)
    print(f"  {be['cheap']['model']:<14} {be['cheap']['calls_covered']:>9,} calls/mo")
    print(f"  {be['premium']['model']:<14} {be['premium']['calls_covered']:>9,} calls/mo "
          f"({be['premium_multiple']}x the per-call price)")

    print("\n" + "-" * 74)
    print("SENSITIVITY on balanced-mid (support-rag-bot)")
    for lever in ("volume", "output_tokens", "context_tokens", "cache_hit_rate"):
        rows = sensitivity(workloads[0], "balanced-mid", lever, factors=(0.5, 1.0, 2.0))
        spread = rows[-1]["monthly_usd"] - rows[0]["monthly_usd"]
        print(f"  {lever:<16} 0.5x=${rows[0]['monthly_usd']:>9,.2f}  "
              f"2x=${rows[-1]['monthly_usd']:>9,.2f}  spread=${spread:>9,.2f}")


if __name__ == "__main__":
    main()
