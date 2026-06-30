# FAQ Generator from Docs

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/document-intelligence/faq-generator/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=document-intelligence/faq-generator/demo.ipynb)

> Stop answering the same question daily — turn your docs into a ready FAQ.

## Business Impact
- **Before:** Support reps re-type the same answers daily, even though the docs already cover them. New FAQ pages are written from scratch.
- **After:** Paste your docs; get a grounded question/answer FAQ in seconds, exportable as Markdown — every answer pulled straight from the source.
- **Estimated ROI:** ~5-8 hours/week of repeat support replies deflected.

## Tech Stack
Python · RAG loop (chunk → retrieve → ground) · token-overlap retriever (no vector DB needed) · Claude API (optional) · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Paste docs, generate the FAQ, expand any entry to see the grounding chunk, and download the whole thing as `FAQ.md`. Set an `ANTHROPIC_API_KEY` in the sidebar to upgrade phrasing.

## How it works
1. **Chunk** the doc into self-contained passages (blank-line / heading split, length-capped).
2. **Retrieve** the best passage per topic with a length-normalized token-overlap score — a transparent stand-in for a vector DB.
3. **Ground** the answer: heuristic mode turns each section heading into a question answered verbatim from the retrieved chunk; **Claude mode** writes the questions customers actually ask while still answering only from the doc.

## Learning Connection
Built while studying **RAG & Vector DBs** (Anthropic).
Applies: the chunk → retrieve → ground pipeline, and the discipline of grounding every answer in source text instead of letting the model free-associate.

## Impact Note
- **Who benefits:** support teams, docs/DevRel, founders standing up a help center fast.
- **Potential risks:** the FAQ is only as accurate as the source doc — stale docs produce stale answers. Heuristic mode mirrors doc wording exactly; a human should review tone and completeness before publishing.
