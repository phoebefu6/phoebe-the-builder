# Feature Prioritization Tool

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/mini-saas-products/feature-prioritizer/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=mini-saas-products/feature-prioritizer/demo.ipynb)

> We argue about priorities without data — RICE turns "I feel strongly about X" into a rankable, defensible number.

## Business Impact
- **Before:** Roadmap fights are won by whoever argues loudest or outranks the room, not by expected value.
- **After:** Every feature gets a RICE score and a value/effort quadrant. The ranking is explicit, so debate moves to the *inputs* (which are checkable) instead of egos.
- **Estimated ROI:** Fewer wasted build cycles on low-value work; a defensible roadmap you can show stakeholders.

## Tech Stack
Python 3.10+, Streamlit, pandas, matplotlib. Pure-Python RICE + quadrant logic (runs offline). Optional Claude (Anthropic SDK) estimates RICE inputs from a plain-English feature description.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **RICE** — `(Reach × Impact × Confidence%) ÷ Effort`. Reach = users/quarter, Impact on a fixed scale (massive=3 … minimal=0.25), Confidence = % (the honesty knob), Effort = person-months (the denominator, so cheap wins rise).
2. **Ranking** — features sorted by RICE, highest first.
3. **Value/effort quadrant** — value proxy = the RICE numerator; split at the medians into Quick win / Big bet / Fill-in / Time sink, so you know *why* something ranks where it does.
4. **Auto-estimate (optional)** — with `ANTHROPIC_API_KEY`, Claude proposes RICE inputs for a feature description as a starting point.

In the sample backlog, Two-factor auth and Dark mode (high reach, low effort) top the list as Quick wins, while the Mobile app — huge reach and massive impact — is correctly dragged down by 12 person-months and 60% confidence into Big bet territory.

## Learning Connection
Built while studying product management and prioritization frameworks (Month 6: Mini SaaS Products).
Applies: RICE scoring, value/effort quadrants, and data-driven roadmap decisions.

## Impact Note
- **Who benefits:** PMs and founders who need to defend a roadmap and cut low-value work.
- **Potential risks:** RICE is only as good as its inputs — inflated reach or optimistic confidence produces a confident-looking but wrong ranking. Use it to structure the conversation, not to replace judgment; revisit estimates as real data arrives.
