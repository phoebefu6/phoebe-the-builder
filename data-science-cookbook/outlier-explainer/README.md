# Outlier Explainer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-science-cookbook/outlier-explainer/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-science-cookbook/outlier-explainer/demo.ipynb)

> "Which rows are weird, and why?" — Isolation Forest finds the anomalies; per-feature z-scores explain each one.

## Business Impact
- **Before:** Anomaly detectors flag rows with an opaque score nobody can act on.
- **After:** Every flagged row comes with a reason — the feature(s) that make it extreme and in which direction — so analysts can triage immediately.
- **Estimated ROI:** faster fraud/quality triage; anomaly alerts that get acted on instead of ignored.

## Tech Stack
Python · scikit-learn (IsolationForest) · z-score reason attribution · Streamlit · matplotlib

## Demo
**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered, or use the badges above.
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. **Detect** — Isolation Forest scores each row's anomalousness (unsupervised, any dimensionality).
2. **Explain** — for each flagged row, compute per-feature z-scores and surface the most extreme ones with direction (high/low).
3. **Summarize** — which features drive outliers across the dataset, and the overall outlier rate.

## Learning Connection
Built while studying **anomaly detection & explainability**. Applies: pairing an unsupervised detector with an attribution layer so flags are actionable, not just scored.

## Impact Note
- **Who benefits:** fraud, data-quality, ops, and monitoring teams.
- **Potential risks:** z-score attribution assumes roughly unimodal features — it can mislead on multimodal or categorical-encoded data, and Isolation Forest's `contamination` is an assumption, not a measurement. An outlier is *statistically* unusual, not necessarily *wrong*; always review before acting.
