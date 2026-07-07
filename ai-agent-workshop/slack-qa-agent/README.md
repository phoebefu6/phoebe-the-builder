# Slack Q&A Agent

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ai-agent-workshop/slack-qa-agent/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ai-agent-workshop/slack-qa-agent/demo.ipynb)

> People ask the same questions in #general every day — this agent answers them from a knowledge base via RAG, and escalates anything it isn't confident about to a human instead of guessing.

## Business Impact
- **Before:** The same "how much PTO do I get / what's the wifi password / when's payday" questions get asked in #general daily, and a person re-answers each one.
- **After:** The agent retrieves the answer from the knowledge base, replies with a citation, and only pings a human for questions it genuinely can't answer.
- **Estimated ROI:** Deflects the long tail of repetitive questions + gives consistent, sourced answers — and never invents policy.

## Tech Stack
Python 3.10+, Streamlit, pandas, matplotlib. RAG retrieval is a hand-rolled TF-IDF cosine index (no external embedding API or vector DB). Optional Claude (Anthropic SDK) synthesizes a friendly grounded answer; optional Slack API (`SLACK_BOT_TOKEN`) posts the reply. Both are optional — the whole loop runs offline.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works — RAG with a Confidence Gate
1. **Retrieve** — `TfidfRetriever` turns each KB doc and the incoming question into TF-IDF vectors and ranks docs by cosine similarity.
2. **Gate** — the agent compares the top retrieval score to a threshold (default `0.35`). This is the most important part: it's what stops the bot from confidently answering a question it has no source for.
3. **Answer or escalate** — above the threshold, it replies with the grounded answer and a citation (Claude-synthesized if `ANTHROPIC_API_KEY` is set, otherwise the source doc directly). Below it, it routes the question to a human channel and suggests adding a doc.
4. **Post** — `post_to_slack()` sends the reply via the Slack Web API when `SLACK_BOT_TOKEN` is set, and dry-runs otherwise.

The threshold was calibrated against real in-scope vs. out-of-scope questions so there's a clean gap between "answer" and "escalate" — the demo notebook charts exactly that gap. Writing KB docs in the *users'* vocabulary (e.g. "vacation" and "payday", not just "PTO" and "payroll") is what makes lexical retrieval reliable on a small corpus.

## Learning Connection
Built while studying RAG, retrieval, and agent evaluation (Month 5: AI Agent Workshop).
Applies: TF-IDF retrieval, grounded generation with citations, and confidence-gated human escalation.

## Impact Note
- **Who benefits:** IT, People Ops, and support teams drowning in repetitive workplace questions.
- **Potential risks:** A stale or wrong KB produces confidently wrong answers — keep docs current and always cite the source so humans can verify. The confidence gate reduces but does not eliminate bad answers; keep the escalation path and monitor what gets escalated to find KB gaps.
