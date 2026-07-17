# Feature Importance Explainer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ml-engineering-toolkit/feature-importance/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ml-engineering-toolkit/feature-importance/demo.ipynb)

> "Stakeholders don't trust the model." One importance number is easy to doubt — three that agree is evidence.

## Business Impact
- **Before:** A model ships as a black box. The business asks "why did it flag this customer?" and the DS team hand-waves at a single feature-importance bar chart that changes every retrain.
- **After:** Every feature is scored three independent ways (impurity, permutation, drop-column) and labelled **trusted / noise / review**. Consensus is the trust signal; disagreement is the flag to investigate.
- **Estimated ROI:** ~3 hours saved per model review, and a defensible answer when a stakeholder pushes back.

## Tech Stack
Python, scikit-learn (`permutation_importance`, RandomForest), pandas, matplotlib, Streamlit. No SHAP dependency — runs in any environment.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
Three importance methods, each with a different blind spot, cross-check each other:

| Method | What it measures | Blind spot |
|--------|------------------|------------|
| **Impurity** | Mean decrease in impurity from the tree's splits | Biased toward high-cardinality features |
| **Permutation** | AUC lost when one column is shuffled | Splits credit across correlated features |
| **Drop-column** | AUC lost when the model is retrained without the column | Expensive (one refit per feature) |

A feature that ranks top-half by **all three** is `trusted`. Bottom-half by all three is `noise` (safe to drop). Disagreement is `review` — usually a redundant/correlated feature a human should decide on.

## Learning Connection
Built while studying **ML Engineering & Explainability** (MLOps track).
Applies: model introspection, permutation vs. drop-column importance, communicating model behaviour to non-technical stakeholders.

## Impact Note
- **Who benefits:** Data scientists defending a model in review; stakeholders who need to trust it before it ships.
- **Potential risks:** Importance is not causation — a "trusted" feature drives *predictions*, not necessarily outcomes. Drop-column can mislead on highly correlated features (both look droppable). Always pair with domain review.
