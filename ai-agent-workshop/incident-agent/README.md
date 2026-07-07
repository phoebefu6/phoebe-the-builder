# Incident Response Agent

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ai-agent-workshop/incident-agent/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ai-agent-workshop/incident-agent/demo.ipynb)

> Incident playbooks exist but nobody follows them — this agent executes the runbook step-by-step and escalates when a remediation step fails.

## Business Impact
- **Before:** During an incident, an on-call engineer scrambles through a wiki runbook under pressure, skips steps, and forgets to page the right person.
- **After:** The agent walks the runbook in order, auto-runs the checks and remediations it can, pauses for the human-only steps, and pages the on-call chain the moment a critical remediation fails.
- **Estimated ROI:** ~15-30 min faster mean-time-to-resolution per incident + consistent, auditable execution every time.

## Tech Stack
Python 3.10+, Streamlit, pandas, matplotlib. Optional Claude (Anthropic SDK) for a natural-language incident summary — falls back to a rule-based summary with no API key.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Runbooks** — each incident type (`service_down`, `disk_space_critical`, `database_high_latency`) has an ordered list of steps. Each step is either `auto` (agent executes) or manual (pauses for human acknowledgment), and can be flagged `critical`.
2. **Agent loop** — `execute_runbook()` runs each step in order. Auto steps run their action against the incident context; manual steps are logged as `awaiting_human`.
3. **Escalation** — if a `critical` remediation step fails, the agent immediately pages the first person in the on-call chain (`escalation_policy`) and marks the incident escalated.
4. **Summary** — Claude writes a 2-sentence on-call summary if `ANTHROPIC_API_KEY` is set; otherwise a deterministic rule-based summary is used.

Detection steps (the `check_*` steps) are *expected* to fail — that's the trigger for the incident. Only a failing critical remediation step pages the on-call chain, avoiding alert fatigue.

## Learning Connection
Built while studying Agentic AI, incident-response runbooks, and agent evaluation (Month 5: AI Agent Workshop).
Applies: agent step-execution loops, human-in-the-loop gating, and escalation policies.

## Impact Note
- **Who benefits:** On-call engineers, SREs, and platform teams who have runbooks but inconsistent execution under pressure.
- **Potential risks:** Auto-remediation acting on real infra needs guardrails and dry-run modes; this demo uses simulated actions only. Never wire auto-remediation to production without approval gates and rollback.
