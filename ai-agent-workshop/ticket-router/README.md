# Customer Ticket Router

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ai-agent-workshop/ticket-router/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ai-agent-workshop/ticket-router/demo.ipynb)

> "Tickets go to the wrong team 40% of the time" — an agent that routes tickets by weighted keyword signal, and escalates to human triage instead of guessing when confidence is low.

## Business Impact
- **Before:** tickets misrouted, bounce between teams, slower time-to-resolution
- **After:** every ticket scored across teams with subject-weighted signal; ambiguous tickets go to human-triage instead of a wrong-but-confident guess
- **Estimated ROI:** fewer bounced tickets, faster first-response time

## Tech Stack
Python, Claude API (Anthropic SDK, optional), Streamlit, Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `ANTHROPIC_API_KEY` to use Claude for classification instead of the weighted keyword matcher.

## Learning Connection
Built while studying Agentic AI on LinkedIn Learning.
Applies: classification with a confidence threshold and an explicit "don't know" path — routing to human-triage instead of forcing a low-confidence prediction.

## Impact Note
- **Who benefits:** support/ops teams triaging a shared inbox across multiple specialist teams
- **Potential risks:** keyword-based classification can miss nuance (sarcasm, multi-issue tickets); low-confidence threshold is the safety valve, tune it per team's tolerance for misroutes vs manual review volume
