# Prompt Registry & Versioning

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/llmops-genai-platform/prompt-registry/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=llmops-genai-platform/prompt-registry/demo.ipynb)

> Prompts get pasted into random `.py` files and Slack threads - no history, no diff, no rollback. This gives them the same discipline as code and models.

## Business Impact
- **Before:** Prompts scattered across the codebase; a well-meaning "improvement" silently breaks answers and nobody can say what changed or roll back.
- **After:** Every prompt is a versioned artifact - committed, hashed, diffable, with exactly one version promoted to production and one-line rollback.
- **Estimated ROI:** ~2-3 hours/week saved per team hunting for "the current prompt" and debugging un-tracked prompt edits; far fewer silent regressions in production.

## Tech Stack
Python (stdlib `difflib` + `hashlib` + `json` + `re`) · pandas · matplotlib · Streamlit · Docker

No API keys, no database, no external services - a file-backed store you can drop into any repo.

## What it does
- **Versioned commits** - each prompt commit becomes v1, v2, v3... with a sha256 content hash
- **Idempotent** - committing identical text never creates a phantom version
- **Unified diff** between any two versions, so prompt edits are reviewable like code
- **Variable extraction** - detects the `{placeholders}` a prompt expects
- **Safe render** - refuses to fill a prompt when a required variable is missing (no un-replaced `{typo}` shipping to the model)
- **Stage promotion** - `draft → staging → production` with a single-production-per-prompt invariant; rollback = promote the previous version

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with outputs (history table, unified diff, evolution chart), or click the Colab/Binder badges above to run it live.

For the Streamlit app (Browse · Commit · Diff · Render tabs):
```bash
pip install -r requirements.txt
streamlit run app.py
```

Core library usage:
```python
from registry import PromptRegistry

reg = PromptRegistry("./_registry")
reg.commit("support_triage", "Answer the question: {question}")
reg.commit("support_triage", "You are a support agent. Answer: {question}")
print(reg.diff("support_triage", 1, 2))
reg.promote("support_triage", 2, "production")
print(reg.render("support_triage", {"question": "How do I reset my password?"}))
```

## Learning Connection
Built while studying **LLMOps / prompt operations** (Month 9 of the FDE roadmap: RAG evaluation, guardrails, prompt ops, cost control).
Applies: treating prompts as versioned artifacts - the foundation every downstream LLMOps capability (eval, cost tracking, A/B testing) hangs off of.

## Impact Note
- **Who benefits:** Applied-AI / platform teams running LLM features who currently keep prompts as inline strings.
- **Potential risks:** This is a local, single-user file store - not concurrency-safe or access-controlled. For a shared team registry, back it with a real database and add auth/audit before trusting it in production.

---
*Day 81 / 120 - part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder), Month 9: LLMOps & GenAI Platform.*
