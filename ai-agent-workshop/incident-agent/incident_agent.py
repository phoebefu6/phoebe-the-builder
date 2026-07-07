from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional


@dataclass
class RunbookStep:
    step_id: str
    description: str
    auto: bool  # True = agent executes automatically, False = requires human ack
    action: Optional[Callable[[], "tuple[bool, str]"]] = None
    critical: bool = False  # if this step fails on a critical step, escalate immediately


@dataclass
class Runbook:
    incident_type: str
    steps: List[RunbookStep]
    escalation_policy: List[str]  # ordered on-call chain, e.g. ["primary-oncall", "team-lead", "eng-manager"]


@dataclass
class Incident:
    incident_type: str
    severity: str  # P1 | P2 | P3
    detected_at: datetime
    context: dict = field(default_factory=dict)


@dataclass
class StepExecution:
    step: RunbookStep
    status: str  # done | failed | awaiting_human | skipped
    output: str
    timestamp: datetime


@dataclass
class IncidentReport:
    incident: Incident
    executions: List[StepExecution]
    escalated: bool
    escalated_to: Optional[str]
    summary: str


# --- simulated runbook actions (mock infra checks/remediations) ---

def _check_service_health(context: dict) -> "tuple[bool, str]":
    healthy = context.get("service_healthy", True)
    return (healthy, "service responded 200 OK" if healthy else "service health check failed: connection refused")


def _restart_service(context: dict) -> "tuple[bool, str]":
    recovers = context.get("restart_recovers", True)
    return (recovers, "service restarted, health check passing" if recovers else "restart completed but service still unhealthy")


def _check_disk_usage(context: dict) -> "tuple[bool, str]":
    usage = context.get("disk_usage_pct", 50)
    ok = usage < 90
    return (ok, f"disk usage at {usage}%" + ("" if ok else " — above 90% threshold"))


def _clear_temp_files(context: dict) -> "tuple[bool, str]":
    freed = context.get("cleanup_frees_space", True)
    return (freed, "cleanup freed sufficient space" if freed else "cleanup ran but insufficient space freed")


def _check_query_latency(context: dict) -> "tuple[bool, str]":
    latency_ms = context.get("query_latency_ms", 50)
    ok = latency_ms < 500
    return (ok, f"query latency at {latency_ms}ms" + ("" if ok else " — above 500ms threshold"))


def _kill_long_running_queries(context: dict) -> "tuple[bool, str]":
    resolved = context.get("kill_queries_resolves", True)
    return (resolved, "long-running queries terminated, latency normalized" if resolved else "queries killed but latency still elevated")


RUNBOOKS = {
    # Detection steps (check_*) are expected to fail — that's the trigger for the incident.
    # Only a failing remediation step (critical=True) should page the on-call chain.
    "service_down": Runbook(
        incident_type="service_down",
        steps=[
            RunbookStep("check_health", "Check service health endpoint", auto=True, action=_check_service_health, critical=False),
            RunbookStep("restart", "Restart the service", auto=True, action=_restart_service, critical=True),
            RunbookStep("notify_users", "Notify affected users via status page", auto=False),
            RunbookStep("postmortem", "Schedule postmortem", auto=False),
        ],
        escalation_policy=["primary-oncall", "team-lead", "eng-manager"],
    ),
    "disk_space_critical": Runbook(
        incident_type="disk_space_critical",
        steps=[
            RunbookStep("check_disk", "Check disk usage", auto=True, action=_check_disk_usage, critical=False),
            RunbookStep("cleanup", "Clear temp files and old logs", auto=True, action=_clear_temp_files, critical=True),
            RunbookStep("resize_volume", "Request volume resize", auto=False),
        ],
        escalation_policy=["primary-oncall", "infra-lead"],
    ),
    "database_high_latency": Runbook(
        incident_type="database_high_latency",
        steps=[
            RunbookStep("check_latency", "Check query latency", auto=True, action=_check_query_latency, critical=False),
            RunbookStep("kill_queries", "Kill long-running queries", auto=True, action=_kill_long_running_queries, critical=True),
            RunbookStep("scale_replicas", "Scale read replicas", auto=False),
            RunbookStep("notify_users", "Notify affected users via status page", auto=False),
        ],
        escalation_policy=["primary-oncall", "dba-oncall", "team-lead"],
    ),
}


def _rule_based_summary(incident: Incident, executions: List[StepExecution], escalated: bool, escalated_to: Optional[str]) -> str:
    done = sum(1 for e in executions if e.status == "done")
    failed = sum(1 for e in executions if e.status == "failed")
    awaiting = sum(1 for e in executions if e.status == "awaiting_human")
    base = f"{incident.incident_type} ({incident.severity}): {done} step(s) completed, {failed} failed, {awaiting} awaiting human action."
    if escalated:
        return base + f" Escalated to {escalated_to} after repeated critical-step failure."
    return base + " No escalation needed."


def _call_claude_summary(incident: Incident, executions: List[StepExecution]) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        log = "\n".join(f"- {e.step.description}: {e.status} ({e.output})" for e in executions)
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": f"Write a 2-sentence incident summary for on-call, given this runbook execution log for a {incident.severity} {incident.incident_type}:\n{log}",
            }],
        )
        return response.content[0].text.strip()
    except Exception:
        return None


def execute_runbook(incident: Incident, runbook: Optional[Runbook] = None) -> IncidentReport:
    """Agent loop: run each step in order. Auto steps execute immediately; manual steps pause for human ack.
    A critical auto step that fails triggers escalation to the next person in the on-call chain."""
    runbook = runbook or RUNBOOKS[incident.incident_type]
    executions: List[StepExecution] = []
    escalated = False
    escalated_to: Optional[str] = None

    for step in runbook.steps:
        now = datetime.now()
        if step.auto and step.action:
            success, output = step.action(incident.context)
            status = "done" if success else "failed"
            executions.append(StepExecution(step, status, output, now))
            if not success and step.critical and not escalated:
                escalated = True
                escalated_to = runbook.escalation_policy[0] if runbook.escalation_policy else None
        else:
            executions.append(StepExecution(step, "awaiting_human", "requires manual acknowledgment", now))

    summary = _call_claude_summary(incident, executions) or _rule_based_summary(incident, executions, escalated, escalated_to)
    return IncidentReport(incident=incident, executions=executions, escalated=escalated, escalated_to=escalated_to, summary=summary)
