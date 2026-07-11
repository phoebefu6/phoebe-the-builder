# User Story Generator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/mini-saas-products/user-story-gen/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=mini-saas-products/user-story-gen/demo.ipynb)

> Writing good user stories takes practice — paste raw feature ideas and get INVEST-scored stories with Given/When/Then acceptance criteria.

## Business Impact
- **Before:** Feature ideas arrive as one-liners in Slack; turning each into a proper story with acceptance criteria takes 10–15 minutes and quality varies by who writes it.
- **After:** A whole intake list becomes a scored, consistently formatted backlog in seconds — with specific flags on what to sharpen before sprint planning.
- **Estimated ROI:** ~2 hours per sprint of story-writing plus fewer mid-sprint "wait, what does done mean?" moments.

## Tech Stack
Python 3.10+, Streamlit, pandas, matplotlib. Rule-based parsing + feature-type detection + acceptance-criteria templates (runs fully offline, no API keys).

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Parse** — ideas already in "As a X, I want Y, so that Z" form are split apart; raw one-liners get filler stripped ("users should be able to…") and possessives normalized.
2. **Detect feature type** — keyword rules classify each idea (auth, search, crud, notification, report, upload, integration, general); the type selects the default persona, benefit, and acceptance-criteria template.
3. **Acceptance criteria** — 3 Given/When/Then criteria per story, always including the unhappy path (invalid input, failure, empty state).
4. **INVEST score** — six checks worth ~17 points each: Independent (no bundled "and"), Negotiable (no implementation words), Valuable (real benefit given), Estimable (concrete enough to size), Small (thin slice), Testable (no vague quality words). Each failure produces a concrete fix suggestion.
5. **Export** — download the full backlog as Markdown, ready to paste into Jira/Linear/Notion.

The sample input (6 messy ideas) produces scores from 100/100 (already well-formed) down to 33/100 ("make the app easy and fast for everyone" — flagged on 4 of 6 checks).

## Learning Connection
Built while studying Digital Product Management and Design Thinking (Month 6: Mini SaaS Products).
Applies: INVEST criteria, Given/When/Then acceptance-criteria templating, and rule-based text parsing.

## Impact Note
- **Who benefits:** Product managers, tech leads, and founders triaging raw feature requests into a sprint-ready backlog.
- **Potential risks:** Templated criteria can create false confidence — generated stories are a disciplined first pass, not a substitute for talking to users; teams should edit the defaults, not ship them blind.
