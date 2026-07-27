# Named Entity Extractor

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-science-cookbook/ner-extractor/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-science-cookbook/ner-extractor/demo.ipynb)

> "Pull the names, orgs, money, and dates out of this text" — structured entities, no heavy model download.

## Business Impact
- **Before:** Key facts (who, which company, how much, when) are trapped in prose and re-typed by hand.
- **After:** Entities are extracted as structured, typed fields — highlighted in the text and grouped for export.
- **Estimated ROI:** faster data entry from documents; a building block for search, KGs, and monitoring.

## Tech Stack
Python · regex + capitalization rules (zero dependencies) · Claude API (optional, higher recall) · Streamlit · matplotlib

## Demo
**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered, or use the badges above.
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app highlights entities inline and groups them by type. Set an `ANTHROPIC_API_KEY` for Claude-powered NER.

## How it works
1. **Regex** for the regular shapes — MONEY, DATE, EMAIL, PERCENT (high precision).
2. **ORG** via a capitalized run followed by a legal suffix (Inc, LLC, Corp, Labs, University…).
3. **PERSON** via a capitalized multi-word heuristic that skips stopwords and already-tagged spans.
4. **Group** by label; **Claude mode** adds locations, roles, and suffix-less orgs.

## Learning Connection
Built while studying **information extraction & NER**. Applies: precision-first rules for structured entities, and knowing which entity types need a model vs a regex.

## Impact Note
- **Who benefits:** analysts, ops, and anyone digitizing facts from documents.
- **Potential risks:** the heuristic PERSON detector has **medium recall/precision** — it misses lowercase or single-token names and can mis-tag capitalized non-names (headings, product names). Use Claude mode or a proper NER model when accuracy is critical, and never auto-act on extracted PII without review.
