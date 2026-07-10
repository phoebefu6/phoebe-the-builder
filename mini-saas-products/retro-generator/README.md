# Sprint Retrospective Generator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/mini-saas-products/retro-generator/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=mini-saas-products/retro-generator/demo.ipynb)

> Retros are unstructured and repetitive — feed in the sprint numbers and get grounded observations and concrete action items in your team's format.

## Business Impact
- **Before:** Retros run on memory and the loudest voice; they drift into venting and produce no action items.
- **After:** The retro is grounded in the sprint's actual metrics, mapped to observations and specific actions, in a consistent format every sprint.
- **Estimated ROI:** Faster, more honest retros that actually produce follow-through — and comparable notes sprint over sprint.

## Tech Stack
Python 3.10+, Streamlit, matplotlib. Rule-based observation + action engine (runs offline). Optional Claude (Anthropic SDK) for an encouraging narrative summary. Supports Start/Stop/Continue, went-well/improve, and 4Ls formats.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Metrics in** — completion rate, velocity vs. last sprint, carryover, bugs, incidents, team mood, notes.
2. **Observations** — rules translate each metric into a "went well" or "to improve" note (e.g. <70% completion → estimation flag; ≥3 carryover → over-commitment).
3. **Action items** — each problem maps to a concrete action (cut commitment, split stories, add DoD checklist, run post-incident review).
4. **Format** — the same observations are framed as Start/Stop/Continue, 4Ls, or went-well/improve.
5. **Narrative (optional)** — Claude writes a 2-sentence encouraging summary when a key is set.

The sample sprint (31/40 points, velocity down 4, 4 carryovers, 6 bugs, 1 incident, mood 3.2) produces a "steady but over-committed" retro with actions to cut commitment, split carryover, tighten the quality gate, and run a post-incident review.

## Learning Connection
Built while studying agile facilitation and templated generation (Month 6: Mini SaaS Products).
Applies: metrics-driven observations, action-item derivation, and multi-format templating.

## Impact Note
- **Who benefits:** Scrum masters, team leads, and agile teams tired of shapeless retros.
- **Potential risks:** Metrics don't capture everything — a "green" sprint can still hide burnout or hidden tech debt. Use the generated retro as a starting point for discussion, not a replacement for the team talking.
