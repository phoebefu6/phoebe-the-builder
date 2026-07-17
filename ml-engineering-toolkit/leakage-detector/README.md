# Data Leakage Detector

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ml-engineering-toolkit/leakage-detector/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ml-engineering-toolkit/leakage-detector/demo.ipynb)

> The model scores 0.99 in cross-validation and falls apart in production. That gap is usually leakage.

A small, dependency-light toolkit that runs a set of independent heuristic checks for the two classic failure modes behind "great CV, terrible in prod": **target leakage** (a feature secretly encodes the answer) and **train/test leakage** (the same rows sit in both splits, so validation is optimistic).

## Business Impact

- **Before** - A team ships a churn model that scored 0.98 AUC in validation. In production it barely beats a coin flip. Weeks of retraining, lost trust, and a stalled roadmap while everyone hunts for the cause.
- **After** - Point the detector at your train/test split before you celebrate the score. It flags the leaky feature, the duplicated rows, and the id column in seconds, with a clear high/medium/low verdict.
- **ROI** - Catching one leaky feature before launch saves a full model rebuild cycle (typically 2-4 engineer-weeks) and, more importantly, protects the credibility of the whole ML program with stakeholders.

## Tech Stack

- Python 3.11, pandas, numpy
- Rank-based AUC and correlation heuristics (no heavy dependencies in the core)
- Streamlit UI with a color-coded findings table and severity chart
- scikit-learn available for extension
- Docker + GitHub Actions CI

## Demo

**Notebook:** open [`demo.ipynb`](./demo.ipynb) (or use the Colab / Binder badges above) for a full walk-through - it builds a leaky dataset, runs each check, explains what it caught, then contrasts with a clean dataset.

**Streamlit app:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pick the leaky demo, the clean demo, or upload your own train + test CSVs and choose a target column.

**Command line:**

```bash
python leakage.py   # runs run_all on the leaky demo, then the clean demo
```

## Learning Connection

This is an **ML Engineering** project. It applies leakage detection and honest validation discipline: the score you trust in a notebook has to survive contact with production. The checks encode the questions a careful reviewer asks before believing any offline metric - is a feature suspiciously predictive, did rows leak across the split, does a column just encode row identity, is the split even balanced.

## Impact Note

- **Who benefits** - ML engineers, data scientists, and reviewers who need a fast sanity check before trusting a validation score.
- **Risks** - These are heuristics. They can **false-positive on legitimately strong features** (a genuinely dominant predictor can trip the correlation or single-feature AUC check). Treat every finding as a prompt for human review, not an automatic verdict. Never drop a column on the tool's say-so alone.
