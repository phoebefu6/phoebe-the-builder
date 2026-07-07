# Report Generation Agent

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ai-agent-workshop/report-agent/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ai-agent-workshop/report-agent/demo.ipynb)

> Monthly reports take 2 days of copy-paste — a team of role-specialized agents turns a table of metrics into a ready-to-send report in seconds.

## Business Impact
- **Before:** An analyst spends ~2 days each month copy-pasting numbers into a template, writing the same narrative by hand, and manually deciding what counts as a win vs. a risk.
- **After:** Paste your metrics, hit generate. Five agents each own one section, the coordinator assembles a templated report, and you download `report.md`.
- **Estimated ROI:** ~2 days/month saved per report owner + consistent structure and never-missed risk flags.

## Tech Stack
Python 3.10+, Streamlit, pandas, matplotlib. Optional Claude (Anthropic SDK) to polish each section into executive prose — falls back to deterministic rule-based prose with no API key.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works — Multi-Agent by Section
Instead of one prompt doing everything, each report section is owned by a role-specialized **section agent**. All agents read the same shared `ReportContext`; a coordinator runs them in a fixed order and assembles the output.

| Section | Agent (role) | Auto-derived from |
|---|---|---|
| Executive Summary | Chief of Staff | win/loss count + lead metric |
| Key Metrics | Data Analyst | full metrics table with trends |
| Highlights | Growth Lead | improved metrics, ranked by magnitude |
| Risks & Watch Items | Risk Officer | declined metrics, ranked |
| Next Steps | Strategy Lead | one action per declining metric |

The key modeling detail is `higher_is_better` on each `Metric`. Metrics like churn rate and support tickets are inverted, so a **decrease** is correctly scored as a win and an **increase** correctly lands in Risks & Watch Items — not the other way around.

## Learning Connection
Built while studying Agentic AI and multi-agent orchestration (Month 5: AI Agent Workshop).
Applies: role-specialized agents over shared context, fixed-order coordination, template assembly, and human-editable inputs.

## Impact Note
- **Who benefits:** Analysts, chiefs of staff, and team leads who produce recurring metric reports (monthly business reviews, board updates, sprint reports).
- **Potential risks:** Auto-generated narrative can sound confident about noisy or wrong data — always sanity-check the source metrics. The rule-based summaries describe *what* moved, not *why*; causal claims still need a human.
