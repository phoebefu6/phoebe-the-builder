from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class Metric:
    name: str
    value: float
    prior: float
    unit: str = ""
    higher_is_better: bool = True

    @property
    def delta(self) -> float:
        return self.value - self.prior

    @property
    def pct_change(self) -> Optional[float]:
        if self.prior == 0:
            return None
        return (self.value - self.prior) / abs(self.prior) * 100.0

    @property
    def improved(self) -> bool:
        return self.delta >= 0 if self.higher_is_better else self.delta <= 0

    def fmt(self, v: float) -> str:
        if self.unit == "$":
            return f"${v:,.0f}"
        if self.unit == "%":
            return f"{v:.1f}%"
        return f"{v:,.0f}{self.unit}"

    def trend_arrow(self) -> str:
        if self.delta == 0:
            return "→"
        return "↑" if self.delta > 0 else "↓"


@dataclass
class ReportContext:
    title: str
    period: str  # e.g. "June 2026"
    metrics: List[Metric]
    notes: str = ""


@dataclass
class Section:
    name: str
    role: str  # the "agent" persona responsible for this section
    body: str


@dataclass
class Report:
    context: ReportContext
    sections: List[Section]

    def to_markdown(self) -> str:
        lines = [f"# {self.context.title}", f"*{self.context.period}*", ""]
        lines.append("## Contents")
        for s in self.sections:
            anchor = s.name.lower().replace(" ", "-")
            lines.append(f"- [{s.name}](#{anchor})")
        lines.append("")
        for s in self.sections:
            lines.append(f"## {s.name}")
            lines.append(f"*Prepared by: {s.role}*")
            lines.append("")
            lines.append(s.body)
            lines.append("")
        return "\n".join(lines).strip() + "\n"


# --- Section agents: each has a distinct lens over the same data ---

@dataclass
class SectionAgent:
    name: str
    role: str
    generator: Callable[[ReportContext], str]


def _pct_str(m: Metric) -> str:
    p = m.pct_change
    if p is None:
        return "n/a"
    return f"{p:+.1f}%"


def _exec_summary(ctx: ReportContext) -> str:
    improved = [m for m in ctx.metrics if m.improved]
    declined = [m for m in ctx.metrics if not m.improved]
    lead = ctx.metrics[0] if ctx.metrics else None
    parts = [f"In {ctx.period}, {len(improved)} of {len(ctx.metrics)} key metrics moved in the right direction."]
    if lead is not None:
        parts.append(
            f"{lead.name} landed at {lead.fmt(lead.value)} ({_pct_str(lead)} vs. prior period)."
        )
    if declined:
        parts.append(
            "Watch items: " + ", ".join(m.name for m in declined) + "."
        )
    return " ".join(parts)


def _metrics_table(ctx: ReportContext) -> str:
    rows = ["| Metric | Current | Prior | Change | Trend |", "|---|---|---|---|---|"]
    for m in ctx.metrics:
        rows.append(
            f"| {m.name} | {m.fmt(m.value)} | {m.fmt(m.prior)} | {_pct_str(m)} | {m.trend_arrow()} |"
        )
    return "\n".join(rows)


def _highlights(ctx: ReportContext) -> str:
    wins = sorted(
        (m for m in ctx.metrics if m.improved),
        key=lambda m: abs(m.pct_change or 0),
        reverse=True,
    )
    if not wins:
        return "No metrics improved this period. See Risks & Watch Items."
    return "\n".join(
        f"- **{m.name}** {m.trend_arrow()} {m.fmt(m.value)} ({_pct_str(m)}) — strongest positive mover."
        if i == 0
        else f"- {m.name} {m.trend_arrow()} {m.fmt(m.value)} ({_pct_str(m)})"
        for i, m in enumerate(wins)
    )


def _risks(ctx: ReportContext) -> str:
    losses = sorted(
        (m for m in ctx.metrics if not m.improved),
        key=lambda m: abs(m.pct_change or 0),
        reverse=True,
    )
    if not losses:
        return "No material risks detected this period. All tracked metrics held or improved."
    return "\n".join(
        f"- **{m.name}** {m.trend_arrow()} {m.fmt(m.value)} ({_pct_str(m)}) — needs attention."
        for m in losses
    )


def _next_steps(ctx: ReportContext) -> str:
    losses = [m for m in ctx.metrics if not m.improved]
    steps: List[str] = []
    for m in losses:
        steps.append(f"- Assign an owner to investigate the {m.name} decline and report back next period.")
    if not steps:
        steps.append("- Maintain current playbook; set a stretch target for the top-performing metric.")
    if ctx.notes:
        steps.append(f"- Follow up on noted context: {ctx.notes}")
    return "\n".join(steps)


DEFAULT_AGENTS: List[SectionAgent] = [
    SectionAgent("Executive Summary", "Chief of Staff", _exec_summary),
    SectionAgent("Key Metrics", "Data Analyst", _metrics_table),
    SectionAgent("Highlights", "Growth Lead", _highlights),
    SectionAgent("Risks & Watch Items", "Risk Officer", _risks),
    SectionAgent("Next Steps", "Strategy Lead", _next_steps),
]


def _polish_with_claude(agent: SectionAgent, ctx: ReportContext, draft: str) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        metrics_blob = "; ".join(
            f"{m.name} {m.fmt(m.value)} ({_pct_str(m)})" for m in ctx.metrics
        )
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": (
                    f"You are the {agent.role} writing the '{agent.name}' section of the "
                    f"{ctx.period} report. Metrics: {metrics_blob}. "
                    f"Tighten this draft into crisp, executive-ready prose. Keep any markdown "
                    f"tables and bullet lists intact. Draft:\n{draft}"
                ),
            }],
        )
        return response.content[0].text.strip()
    except Exception:
        return None


def generate_report(
    ctx: ReportContext,
    agents: Optional[List[SectionAgent]] = None,
    use_claude: bool = True,
) -> Report:
    """Multi-agent report builder: each section agent drafts its section from the shared
    context, then (optionally) Claude polishes each draft. A coordinator assembles the
    templated report in a fixed order."""
    agents = agents or DEFAULT_AGENTS
    sections: List[Section] = []
    for agent in agents:
        draft = agent.generator(ctx)
        polished = _polish_with_claude(agent, ctx, draft) if use_claude else None
        sections.append(Section(name=agent.name, role=agent.role, body=polished or draft))
    return Report(context=ctx, sections=sections)


SAMPLE_CONTEXT = ReportContext(
    title="Monthly Business Review",
    period="June 2026",
    metrics=[
        Metric("Revenue", 128_400, 112_000, unit="$", higher_is_better=True),
        Metric("Active Users", 8_920, 8_100, higher_is_better=True),
        Metric("Churn Rate", 4.8, 3.9, unit="%", higher_is_better=False),
        Metric("NPS", 47, 44, higher_is_better=True),
        Metric("Support Tickets", 610, 720, higher_is_better=False),
    ],
    notes="New onboarding flow shipped mid-month.",
)
