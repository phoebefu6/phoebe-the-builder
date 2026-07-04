# Email Draft Agent

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ai-agent-workshop/email-agent/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ai-agent-workshop/email-agent/demo.ipynb)

> "I spend 2 hours daily on email" — an agent that drafts replies, checks its own work against the original ask, and revises before handing it back.

## Business Impact
- **Before:** manually re-reading each email to make sure every question gets answered, drafting from scratch each time
- **After:** agent produces a ready-to-edit draft with a visible self-critique trace, so you review instead of write
- **Estimated ROI:** ~1 hour/day saved on routine email triage

## Tech Stack
Python, Claude API (Anthropic SDK), agent loop (draft → self-critique → revise), Streamlit, Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `ANTHROPIC_API_KEY` as an environment variable to swap the template-based agent for real Claude-generated drafts. Without it, the agent falls back to a rule-based draft + critique loop so the demo runs standalone.

## Learning Connection
Built while studying Agentic AI on LinkedIn Learning.
Applies: the reflection pattern (generate → self-evaluate → revise) used in CrewAI/LangGraph-style agent loops.

## Impact Note
- **Who benefits:** anyone triaging a high volume of routine email (support, sales, ops)
- **Potential risks:** agent may misread nuance or tone in sensitive emails; always human-review before sending, especially for anything involving commitments or client-facing promises
