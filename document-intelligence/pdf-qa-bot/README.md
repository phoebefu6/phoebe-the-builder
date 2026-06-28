# PDF Q&A Bot

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/document-intelligence/pdf-qa-bot/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=document-intelligence/pdf-qa-bot/demo.ipynb)

> "Nobody reads the 200-page policy manual" — upload it, ask questions, get cited answers.

## Business Impact
- **Before:** Employees ping HR/legal for answers buried in a long PDF, or skip reading it entirely.
- **After:** Upload the PDF, ask in plain English, get a page-cited answer in seconds.
- **Estimated ROI:** Cuts policy lookup time from minutes/hours of searching to seconds; fewer repeat HR/legal tickets.

## Tech Stack
Python, Streamlit, pypdf, scikit-learn (TF-IDF retrieval), Anthropic API (optional synthesis)

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Learning Connection
Built while studying Prompt Engineering, RAG, and Vector DBs (Anthropic track).
Applies: chunking strategy, retrieval ranking, extractive fallback when no LLM key is present.

## Impact Note
- **Who benefits:** Employees, HR, legal, support teams fielding repeat questions on long documents.
- **Potential risks:** TF-IDF retrieval can miss semantically-related-but-differently-worded content; no API key means answers are extractive only, not synthesized — always show source page so users can verify.
