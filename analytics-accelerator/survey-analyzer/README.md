# Survey Results Analyzer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-accelerator/survey-analyzer/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-accelerator/survey-analyzer/demo.ipynb)

> Turn a raw survey export into a type-aware summary in seconds — NPS, Likert scores, choice breakdowns, and open-text sentiment.

## Business Impact
- **Before:** An analyst spends most of a day building pivot tables, hand-coding NPS, and skim-reading hundreds of free-text comments.
- **After:** Upload the CSV, get NPS + per-question summaries + open-text sentiment and themes instantly.
- **Estimated ROI:** ~6-7 hours saved per survey wave.

## How it works
The analyzer classifies every column on its own into one of five types and summarizes each the right way:

| Type | Detection | Summary |
|------|-----------|---------|
| **NPS** | 0-10 numeric, high ceiling | NPS score + promoter / passive / detractor split |
| **Numeric** | >90% parses as numbers | mean / median / min / max |
| **Likert** | matches a Strongly disagree → Strongly agree vocab | mean 1-5 score + distribution |
| **Single-choice** | few short distinct values | top category + counts |
| **Open-text** | everything else | lexicon sentiment % + top themes + samples |

No API keys, no model download — pandas plus a small built-in lexicon, so it runs standalone in CI or a notebook.

## Tech Stack
Python · pandas · Streamlit · matplotlib · seaborn · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Click **Use sample survey** in the sidebar to try it with no data of your own.

## Learning Connection
Built while studying **Streamlit** and **AWS Cloud Technical Essentials** (Month 3 of the FDE track).
Applies: type inference over messy tabular data, NPS math, lightweight NLP sentiment, and Streamlit layout with expanders + metrics.

## Impact Note
- **Who benefits:** analysts, PMs, and CX teams who run recurring surveys.
- **Potential risks:** the lexicon sentiment is intentionally simple — it flags tone direction, not nuance (sarcasm, negation, mixed sentiment). Treat open-text sentiment as a triage signal, not a verdict, and read flagged comments before acting.
