# Data Pipeline Monitor Agent

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ai-agent-workshop/pipeline-monitor-agent/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ai-agent-workshop/pipeline-monitor-agent/demo.ipynb)

> "We discover pipeline failures from angry users" — an agent that catches failures, slow runs, row-count drops, and silent staleness before anyone downstream notices.

## Business Impact
- **Before:** a pipeline silently stops running and nobody knows until a stakeholder asks why the dashboard is stale
- **After:** agent checks every run against its own job's history and flags failures, 2x-slow runs, row-count drops, and jobs overdue on their schedule
- **Estimated ROI:** failures caught in minutes instead of days, avoiding downstream incidents and trust erosion

## Tech Stack
Python, Claude API (Anthropic SDK, optional), Streamlit, Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `ANTHROPIC_API_KEY` for Claude-written recommended actions (falls back to rule-based actions otherwise).

## Learning Connection
Built while studying Monitoring and Observability on LinkedIn Learning.
Applies: per-job adaptive thresholds and the "silent failure" staleness check — core observability patterns beyond simple pass/fail alerting.

## Impact Note
- **Who benefits:** data engineers and analysts who depend on pipelines running on schedule
- **Potential risks:** thresholds (2x median duration, 50% row-count drop) are heuristics — tune per job to avoid alert fatigue on genuinely variable workloads
