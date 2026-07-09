# Agent Evaluation Dashboard

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ai-agent-workshop/agent-eval-dashboard/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ai-agent-workshop/agent-eval-dashboard/demo.ipynb)

> "We don't know if our agents are actually good." Run a fixed eval suite against two agent versions and get pass rate, quality, latency, and per-category regressions — numbers instead of vibes. **Month 5 capstone.**

## Business Impact
- **Before:** Agent quality is judged by gut feel. A "small prompt tweak" silently breaks refusals or tone, and nobody notices until a customer does.
- **After:** Every version runs the same eval suite. You see pass rate by category and a hard regression gate: ship the candidate only if nothing went backward.
- **Estimated ROI:** Turns agent releases from risky guesswork into a measurable, gated process — the difference between shipping confidently and shipping blind.

## Tech Stack
Python 3.10+, Streamlit, pandas, matplotlib. Scoring is offline (keyword-recall quality proxy + tool-correctness + latency SLA). Optional Claude (Anthropic SDK) as an LLM-as-judge for graded quality.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit dashboard:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Eval suite** — a fixed list of `EvalCase`s (prompt, expected keywords, expected tool, latency SLA) across categories: factual, tool_use, refusal, tone.
2. **Traces** — the agent's recorded runs (`AgentTrace`: output, latency, tokens, tool calls).
3. **Scorers** — each case is scored on quality (keyword recall, or a Claude judge), tool correctness, and latency SLA, then reduced to pass/fail (quality ≥ 0.6, within SLA, no wrong tool).
4. **Aggregate** — overall pass rate, avg quality, avg latency, tokens, plus a per-category breakdown.
5. **Regression gate** — compare v1 → v2 by category; flag any category that went backward. "No category regressed" is a release rule you can enforce in CI.

The sample suite ships a baseline (`v1`) and candidate (`v2`): v1 leaks a customer address, calls the wrong password-reset tool, and answers a frustrated customer coldly and slowly (33% pass rate). v2 fixes all three (100%) — so the dashboard shows a clear, category-localized improvement.

## Learning Connection
Built while studying agent evaluation and MLOps (Month 5: AI Agent Workshop — capstone).
Applies: eval-suite design, LLM-as-judge vs. rule-based scoring, per-category aggregation, and regression gating.

## Impact Note
- **Who benefits:** Teams shipping LLM agents who need to know a new version is better, not just different.
- **Potential risks:** Keyword-recall quality is a coarse proxy — a response can hit the keywords and still be wrong; use the Claude judge and human review for high-stakes categories. Evals are only as good as the suite: a category you don't test is a category you can't gate.
