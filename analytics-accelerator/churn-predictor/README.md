# Churn Predictor

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-accelerator/churn-predictor/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-accelerator/churn-predictor/demo.ipynb)

> Spot at-risk customers *before* they leave — ranked risk scores, churn drivers, and an honest model-quality check.

## Business Impact
- **Before:** Churn is discovered after the customer is already gone — too late to save the account.
- **After:** A ranked at-risk list every time the data refreshes, plus the drivers behind churn so retention knows *where* to intervene.
- **Estimated ROI:** Even a few percentage points of saved churn compounds — retention is far cheaper than acquisition.

## How it works
1. **Auto-detect the label** — finds the binary churn column (handles `churned` 0/1, yes/no, true/false) and drops ID-like columns.
2. **Train + honestly evaluate** — standardizes features, fits a `GradientBoostingClassifier`, and reports AUC / precision / recall on a held-out 25% so the metrics reflect *new* customers.
3. **Explain** — feature importances surface what actually drives churn.
4. **Score** — every customer gets a churn probability and a Low / Medium / High risk band, sorted into an outreach list.

Pure scikit-learn + pandas — no API keys, runs standalone in a notebook or CI.

## Tech Stack
Python · scikit-learn (GradientBoosting, ROC AUC, train/test split) · pandas · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Click **Use sample customer base** in the sidebar to try it (800 customers, ~21% churn). The sample reaches AUC ≈ 0.74.

## Learning Connection
Built while studying **Streamlit** and **AWS Cloud Technical Essentials** (Month 3 of the FDE track).
Applies: supervised classification, train/test discipline, threshold tuning (precision vs recall), feature importance, and probability calibration into action-ready risk bands.

## Impact Note
- **Who benefits:** retention, CRM, and customer-success teams who need an early-warning list.
- **Potential risks:** importance ≠ causation — a driver flagged by the model isn't proven to *cause* churn, so validate before acting. Tune the threshold to your cost trade-off (missing a churner vs over-contacting a happy customer), and never feed the model protected attributes (age, gender, race) or proxies for them.
