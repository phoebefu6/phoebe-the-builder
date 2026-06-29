# Meeting Notes Summarizer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/document-intelligence/meeting-summarizer/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=document-intelligence/meeting-summarizer/demo.ipynb)

> "Meetings produce no actionable output" — paste the transcript, get TL;DR + decisions + action items + open questions.

## Business Impact
- **Before:** Meetings end with no written follow-up; decisions and owners get forgotten.
- **After:** Paste a transcript, get structured output in seconds — TL;DR, decisions, action items, open questions.
- **Estimated ROI:** Saves ~15-20 min of manual note-cleanup per meeting; reduces dropped follow-ups.

## Tech Stack
Python, Streamlit, Anthropic API (optional), regex-based heuristic fallback

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Learning Connection
Built while studying Prompt Engineering and structured-output techniques (Anthropic track).
Applies: forcing structured JSON output from an LLM, and a no-LLM heuristic fallback for offline/cost-free use.

## Impact Note
- **Who benefits:** Anyone running recurring meetings — PMs, managers, founders — who needs a written record without manual note-taking.
- **Potential risks:** Heuristic mode is cue-word matching, not true comprehension — it can mislabel casual agreement as a "decision." Always show the source sentence so users can sanity-check.
