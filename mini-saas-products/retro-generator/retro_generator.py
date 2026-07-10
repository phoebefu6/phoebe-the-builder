from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SprintData:
    sprint_name: str
    planned_points: int
    completed_points: int
    carryover_items: int
    bugs_found: int
    incidents: int
    team_mood: float           # 1-5 average
    prev_completed_points: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        return self.completed_points / self.planned_points if self.planned_points else 0.0

    @property
    def velocity_delta(self) -> int:
        return self.completed_points - self.prev_completed_points


@dataclass
class Retro:
    sprint_name: str
    format: str
    sections: Dict[str, List[str]]
    action_items: List[str]
    narrative: str = ""

    def to_markdown(self) -> str:
        lines = [f"# Retrospective — {self.sprint_name}", f"*Format: {self.format}*", ""]
        if self.narrative:
            lines += ["> " + self.narrative, ""]
        for title, items in self.sections.items():
            lines.append(f"## {title}")
            lines += [f"- {it}" for it in items] or ["- (nothing noted)"]
            lines.append("")
        lines.append("## Action Items")
        lines += [f"- [ ] {a}" for a in self.action_items] or ["- [ ] (none)"]
        return "\n".join(lines).strip() + "\n"


def _observations(d: SprintData) -> Dict[str, List[str]]:
    """Turn sprint metrics into went-well / needs-improvement observations."""
    good, bad = [], []
    cr = d.completion_rate
    if cr >= 0.9:
        good.append(f"Hit {cr:.0%} of planned points ({d.completed_points}/{d.planned_points}) — strong delivery.")
    elif cr < 0.7:
        bad.append(f"Only {cr:.0%} of planned points completed — scope or estimation needs a look.")
    else:
        good.append(f"Completed {cr:.0%} of planned points — steady.")

    if d.velocity_delta > 0:
        good.append(f"Velocity up {d.velocity_delta} points vs last sprint.")
    elif d.velocity_delta < 0:
        bad.append(f"Velocity down {abs(d.velocity_delta)} points vs last sprint.")

    if d.carryover_items == 0:
        good.append("No carryover items — clean sprint boundary.")
    elif d.carryover_items >= 3:
        bad.append(f"{d.carryover_items} items carried over — likely over-committed.")

    if d.bugs_found >= 5:
        bad.append(f"{d.bugs_found} bugs found — quality gate may need tightening.")
    if d.incidents > 0:
        bad.append(f"{d.incidents} production incident(s) — add to follow-up.")
    if d.team_mood >= 4:
        good.append(f"Team mood high ({d.team_mood}/5).")
    elif d.team_mood <= 2.5:
        bad.append(f"Team mood low ({d.team_mood}/5) — check in on load and blockers.")
    return {"good": good, "bad": bad}


def _action_items(d: SprintData, obs: Dict[str, List[str]]) -> List[str]:
    actions = []
    if d.completion_rate < 0.7:
        actions.append("Reduce next sprint commitment by ~15% and re-estimate the largest stories.")
    if d.carryover_items >= 3:
        actions.append("Break carryover items into smaller stories before pulling them in.")
    if d.bugs_found >= 5:
        actions.append("Add a definition-of-done checklist and a mid-sprint bug triage.")
    if d.incidents > 0:
        actions.append("Schedule a blameless post-incident review for each production incident.")
    if d.team_mood <= 2.5:
        actions.append("Hold 1:1s to surface blockers; protect focus time next sprint.")
    if not actions:
        actions.append("Keep the current cadence; pick one small process experiment to try.")
    return actions


_FORMATS = {
    "start-stop-continue": ["Start", "Stop", "Continue"],
    "went-well-improve": ["What went well", "What to improve"],
    "4Ls": ["Liked", "Learned", "Lacked", "Longed for"],
}


def _map_to_format(fmt: str, obs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    good, bad = obs["good"], obs["bad"]
    if fmt == "start-stop-continue":
        return {"Start": [a.replace("Add ", "Start ") for a in bad], "Stop": [], "Continue": good}
    if fmt == "4Ls":
        return {"Liked": good, "Learned": bad[:1], "Lacked": bad[1:], "Longed for": []}
    return {"What went well": good, "What to improve": bad}


def _claude_narrative(d: SprintData, obs: Dict[str, List[str]]) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    f"Write a 2-sentence, encouraging retro summary for sprint '{d.sprint_name}'. "
                    f"Went well: {obs['good']}. To improve: {obs['bad']}."
                ),
            }],
        )
        return resp.content[0].text.strip()
    except Exception:
        return None


def generate_retro(d: SprintData, fmt: str = "start-stop-continue", use_claude: bool = True) -> Retro:
    if fmt not in _FORMATS:
        fmt = "start-stop-continue"
    obs = _observations(d)
    sections = _map_to_format(fmt, obs)
    actions = _action_items(d, obs)
    narrative = (_claude_narrative(d, obs) if use_claude else None) or ""
    if d.notes:
        sections.setdefault("Team notes", []).extend(d.notes)
    return Retro(sprint_name=d.sprint_name, format=fmt, sections=sections,
                 action_items=actions, narrative=narrative)


SAMPLE_SPRINT = SprintData(
    sprint_name="Sprint 24",
    planned_points=40,
    completed_points=31,
    carryover_items=4,
    bugs_found=6,
    incidents=1,
    team_mood=3.2,
    prev_completed_points=35,
    notes=["New CI pipeline saved review time.", "Onboarding of two contractors slowed pairing."],
)
