# Contract Clause Extractor

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/document-intelligence/contract-extractor/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=document-intelligence/contract-extractor/demo.ipynb)

> Turn a 3-day legal read into a 30-second clause + risk triage.

## Business Impact
- **Before:** A reviewer reads each contract end to end (1-3 days) just to locate termination, liability, payment, and auto-renewal terms - and easily misses what's *absent*.
- **After:** Paste text or upload a PDF; get every key clause flagged by risk band, party names, effective date, and a list of missing expected clauses - in seconds.
- **Estimated ROI:** ~6-10 hours/week saved on first-pass contract triage.

## Tech Stack
Python · regex heuristics (zero-cost first pass) · Claude API (optional, for paraphrased clauses) · pypdf · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Upload a contract PDF or paste text. Set an `ANTHROPIC_API_KEY` in the sidebar to upgrade from regex to Claude extraction.

## How it works
1. **Heuristic pass (default):** scans each sentence against cue patterns for 8 clause types (Termination, Liability, Confidentiality, Payment, Governing Law, Auto-Renewal, IP, Warranty), assigns a risk band, and flags any expected clause that is missing entirely.
2. **LLM pass (optional):** with an API key, Claude catches paraphrased clauses the regex misses - same `ContractReview` output shape, so the UI and charts are unchanged.

## Learning Connection
Built while studying **Prompt Engineering & structured extraction** (Anthropic).
Applies: schema-constrained JSON extraction, heuristic-before-LLM cost discipline, and graceful fallback when no API key is present.

## Impact Note
- **Who benefits:** legal ops, procurement, founders reviewing vendor agreements without in-house counsel.
- **Potential risks:** **not legal advice.** A regex/LLM pass is a triage aid, not a substitute for a lawyer. Risk flags are heuristics and can miss context-specific liabilities; a human must review anything flagged - and anything not flagged.
