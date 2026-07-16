# Model Card Generator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ml-engineering-toolkit/model-card-gen/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ml-engineering-toolkit/model-card-gen/demo.ipynb)

> "No model documentation." — a model with no card is a black box: nobody knows what it's for, what it trained on, or where it breaks.

Auto-generates a **Google-style Model Card** in Markdown from any trained scikit-learn model + a held-out test set. Introspects everything the estimator can tell us (algorithm, hyperparameters, task type) and computes performance **overall and per-slice** — then prompts for the human context only you have (intended use, data provenance, ethics) so the card is never silently blank.

## Business Impact
- **Before:** Models ship undocumented; downstream teams reverse-engineer intent, and fairness gaps go unnoticed until an incident.
- **After:** One function call produces a committable `MODEL_CARD.md` — intended use, training data, metrics table, per-group performance, limitations, ethics — in seconds.
- **Estimated ROI:** Standardizes model documentation across the team; makes the per-slice fairness check a default step instead of an afterthought.

## Tech Stack
Python · scikit-learn (`is_classifier`, metrics) · pandas · matplotlib · Streamlit · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — trains a churn model, scores it by region, generates and exports the card, with a per-slice performance chart. Or click the Colab/Binder badges above.

Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Fill the human-authored fields, watch the card render live, download `MODEL_CARD.md`.

CLI check:
```bash
python model_card.py
```

## How It Works
1. `compute_metrics()` — auto-detects classification vs regression (`is_classifier`) and picks the right metric set; breaks metrics down by any aligned slice series (e.g. a sensitive attribute held out of the feature matrix).
2. `generate_model_card()` — introspects the estimator name + hyperparameters, fills the quantitative sections, and merges your human-authored intended-use / training-data / limitations / ethics text.
3. Ends every card with a **"human review required before publishing"** footer — generation is a first draft, not a sign-off.

**Edge cases handled:** slice series with a length mismatch is skipped (card still generates); binary-only ROC-AUC guarded; regression auto-switches to RMSE/MAE/R².

## Learning Connection
Built while studying **responsible ML / model governance** in the ML Engineering track.
Applies: model documentation standards (Model Cards, Mitchell et al.), sliced evaluation for fairness, estimator introspection.

## Impact Note
- **Who benefits:** ML engineers and governance/risk teams who need consistent model documentation.
- **Potential risks:** A generated card is a **draft** — the auto-filled metrics are only as honest as the test set, and the human sections must be reviewed. Don't treat card generation as a fairness sign-off; the slice table flags gaps but doesn't remediate them.
