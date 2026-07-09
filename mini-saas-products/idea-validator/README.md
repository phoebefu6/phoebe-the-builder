# Startup Idea Validator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/mini-saas-products/idea-validator/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=mini-saas-products/idea-validator/demo.ipynb)

> We build before we validate — this scores an idea across five dimensions, builds a Lean Canvas, and hands you the cheapest experiment to test your riskiest assumption first. First build of Month 6.

## Business Impact
- **Before:** Teams fall in love with an idea and spend months building before checking whether anyone has the problem or will pay.
- **After:** A five-minute scorecard produces a blunt verdict, surfaces the two weakest assumptions, and prescribes a cheap experiment for each — so you spend days de-risking instead of months building.
- **Estimated ROI:** Avoids the single most expensive startup mistake: building the wrong thing. One prevented dead-end pays for itself.

## Tech Stack
Python 3.10+, Streamlit, numpy, matplotlib. Applies the Lean Canvas + riskiest-assumption method offline. Optional Claude (Anthropic SDK) acts as a skeptical pre-seed investor that fills the canvas and scores the idea.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Five dimensions** — Problem Severity, Market Size, Differentiation, Feasibility, Willingness to Pay, each scored 1-5 (self-scored via sliders, or scored by a Claude "skeptical investor" prompt).
2. **Verdict** — a deliberately harsh function of the average score: most ideas land at "promising but unproven" or lower until validated.
3. **Riskiest assumptions** — the two lowest-scoring dimensions. Lean Startup's core move is to test what's most likely to kill you *first*.
4. **Cheapest experiments** — each dimension maps to a specific, cheap validation (problem interviews, bottom-up TAM, a comparison sheet, a 1-day spike, a fake-door pre-sale page) — none of which require building the product.
5. **Lean Canvas** — all nine blocks, scaffolded with guiding questions offline and filled with real content when Claude is enabled.

The sample idea (an AI meal-plan app for allergy-conscious parents) scores strong on problem/market but thin on differentiation and willingness-to-pay — so the tool tells you to run a comparison sheet and a fake-door pre-sale before writing product code.

## Learning Connection
Built while studying Design Thinking and lean product discovery (Month 6: Mini SaaS Products).
Applies: Lean Canvas, validation scorecards, and riskiest-assumption-first experiment design.

## Impact Note
- **Who benefits:** Founders, PMs, and intrapreneurs deciding whether an idea is worth building.
- **Potential risks:** Self-scoring is subjective — an optimistic founder inflates every dimension, and the tool can only be as honest as the inputs. Use it to structure a decision, not to replace real customer evidence; the experiments are the point, not the score.
