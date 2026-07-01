# Competitive Intel Summarizer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/document-intelligence/competitive-intel/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=document-intelligence/competitive-intel/demo.ipynb)

> Turn scattered competitor notes into one comparison matrix + strategic takeaways.

## Business Impact
- **Before:** "Competitor tracking" is a graveyard of screenshots and Slack links; nobody can answer "where's our white space?"
- **After:** Paste competitor copy; get an aligned feature matrix, structured profiles, and strategic takeaways — a review-ready artifact in seconds.
- **Estimated ROI:** hours per competitive review, and a living matrix instead of stale decks.

## Tech Stack
Python · regex feature/pricing extraction against a fixed lexicon · pandas matrix · Claude API (optional synthesis) · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Enter N competitors with their site copy or your notes; get the matrix, profiles, and takeaways. Set an `ANTHROPIC_API_KEY` for Claude-synthesized takeaways.

## How it works
1. **Extract** each competitor's profile — pricing, features (scored against a fixed lexicon so columns align), target market, positioning.
2. **Pivot** into a feature × competitor matrix (only features at least one has).
3. **Derive** takeaways: white space (features nobody offers), feature leader, pricing transparency.
4. **Claude mode** infers strategy the regex can't — e.g. a free tier feeding per-seat paid plans as land-and-expand.

## Learning Connection
Built while studying **structured extraction & synthesis** (Anthropic Prompt Engineering).
Applies: aligning heterogeneous inputs to a common schema, and grounding every takeaway in an auditable cell of the matrix.

## Impact Note
- **Who benefits:** product, marketing, founders running competitive reviews.
- **Potential risks:** extraction only sees what's in the pasted text — missing features read as absent, so a thin blurb can misrank a rival. Treat the matrix as a starting draft, verify against primary sources, and respect sites' terms when gathering copy.
