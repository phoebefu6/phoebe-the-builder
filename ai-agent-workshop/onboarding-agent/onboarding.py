from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Set


@dataclass
class ChecklistItem:
    item_id: str
    phase: str  # "Day 1" | "Week 1" | "Month 1" | "Month 3"
    title: str
    expected_by_day: int  # days since start date
    depends_on: Optional[str] = None
    roles: Optional[List[str]] = None  # None = applies to all roles


@dataclass
class NewHire:
    name: str
    role: str
    start_date: date
    completed_ids: Set[str] = field(default_factory=set)


@dataclass
class StepStatus:
    item: ChecklistItem
    status: str  # completed | overdue | due_soon | blocked | upcoming
    nudge: str = ""


DEFAULT_CHECKLIST: List[ChecklistItem] = [
    ChecklistItem("it_setup", "Day 1", "Provision laptop, email, and SSO access", 0),
    ChecklistItem("hr_paperwork", "Day 1", "Complete HR paperwork and tax forms", 0),
    ChecklistItem("welcome_meeting", "Day 1", "Welcome meeting with manager", 0),
    ChecklistItem("tool_access", "Week 1", "Get access to Slack, Jira/Linear, and internal wiki", 1, depends_on="it_setup"),
    ChecklistItem("team_intro", "Week 1", "1:1 intros with each teammate", 3),
    ChecklistItem("first_task", "Week 1", "Assigned first small task/ticket", 5, depends_on="tool_access"),
    ChecklistItem("codebase_walkthrough", "Week 1", "Codebase/architecture walkthrough", 5, roles=["Engineer", "Data Scientist"]),
    ChecklistItem("sales_playbook", "Week 1", "Sales playbook and CRM training", 5, roles=["Sales"]),
    ChecklistItem("30_day_review", "Month 1", "30-day check-in with manager", 30),
    ChecklistItem("first_project", "Month 1", "Ship first real project/deliverable", 30, depends_on="first_task"),
    ChecklistItem("60_day_review", "Month 3", "60-day check-in with manager", 60),
    ChecklistItem("90_day_review", "Month 3", "90-day performance review", 90, depends_on="60_day_review"),
]


def _rule_based_nudge(item: ChecklistItem, status: str) -> str:
    if status == "overdue":
        return f"Overdue: '{item.title}' was expected by day {item.expected_by_day}. Follow up today."
    if status == "due_soon":
        return f"Due soon: '{item.title}' expected by day {item.expected_by_day}."
    if status == "blocked":
        return f"Blocked: '{item.title}' depends on an incomplete step."
    return ""


def _call_claude_nudge(item: ChecklistItem, status: str, hire_name: str) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=80,
            messages=[{
                "role": "user",
                "content": f"Write a one-sentence nudge message for a manager about {hire_name}'s onboarding step '{item.title}' which is currently {status}.",
            }],
        )
        return response.content[0].text.strip()
    except Exception:
        return None


def evaluate_onboarding(hire: NewHire, checklist: Optional[List[ChecklistItem]] = None, today: Optional[date] = None) -> List[StepStatus]:
    """Multi-step agent: walk the checklist in order, resolving dependencies and deadlines against today's date."""
    checklist = checklist or DEFAULT_CHECKLIST
    today = today or date.today()
    days_elapsed = (today - hire.start_date).days

    applicable_items = [item for item in checklist if item.roles is None or hire.role in item.roles]
    results: List[StepStatus] = []

    for item in applicable_items:
        if item.item_id in hire.completed_ids:
            results.append(StepStatus(item, "completed"))
            continue

        if item.depends_on and item.depends_on not in hire.completed_ids:
            dep_applicable = any(i.item_id == item.depends_on for i in applicable_items)
            if dep_applicable:
                status = "blocked"
                nudge = _call_claude_nudge(item, status, hire.name) or _rule_based_nudge(item, status)
                results.append(StepStatus(item, status, nudge))
                continue

        if days_elapsed > item.expected_by_day:
            status = "overdue"
        elif item.expected_by_day - days_elapsed <= 3:
            status = "due_soon"
        else:
            status = "upcoming"

        nudge = _call_claude_nudge(item, status, hire.name) or _rule_based_nudge(item, status) if status in ("overdue", "due_soon") else ""
        results.append(StepStatus(item, status, nudge))

    return results


def progress_summary(statuses: List[StepStatus]) -> Dict[str, int]:
    summary = {"completed": 0, "overdue": 0, "due_soon": 0, "blocked": 0, "upcoming": 0}
    for s in statuses:
        summary[s.status] += 1
    return summary
