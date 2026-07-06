# Onboarding Checklist Agent

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ai-agent-workshop/onboarding-agent/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ai-agent-workshop/onboarding-agent/demo.ipynb)

> "New hires miss steps, ramp-up takes 3 months" — a multi-step agent that resolves onboarding step dependencies and flags what's overdue or blocked before it cascades.

## Business Impact
- **Before:** flat checklists miss that step B depends on step A; a missed Week-1 step silently blocks everything downstream until someone notices at the 90-day review
- **After:** agent walks the checklist, resolves role filters and dependencies, and flags overdue/blocked steps with a nudge
- **Estimated ROI:** faster time-to-productive by catching blocked chains in week 1, not month 3

## Tech Stack
Python, Claude API (Anthropic SDK, optional), Streamlit, Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `ANTHROPIC_API_KEY` for Claude-written nudge messages (falls back to rule-based nudges otherwise).

## Learning Connection
Built while studying Agentic AI on LinkedIn Learning.
Applies: multi-step agent reasoning — resolving dependencies before classifying status, not just checking each item in isolation.

## Impact Note
- **Who benefits:** HR/People Ops and hiring managers tracking new-hire ramp-up
- **Potential risks:** checklist template is generic — dependency chains and role filters should be reviewed per company before relying on "blocked" status to drive real follow-up
