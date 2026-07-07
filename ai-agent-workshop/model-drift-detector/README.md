# Model Drift Detector

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ai-agent-workshop/model-drift-detector/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ai-agent-workshop/model-drift-detector/demo.ipynb)

> Models degrade silently in production — this watches a live window against the training reference via PSI and alerts before accuracy quietly rots.

## Business Impact
- **Before:** A model ships, quietly drifts as the world changes, and nobody notices until a business metric tanks or a customer complains — often weeks later.
- **After:** Feature and prediction distributions are checked against the training reference every run; a significant shift fires an alert with the exact feature that moved.
- **Estimated ROI:** Catches degradation *before* ground-truth labels arrive (which is often days/weeks late), cutting silent-failure windows from weeks to hours.

## Tech Stack
Python 3.10+, numpy, pandas, Streamlit, matplotlib. Population Stability Index (PSI) built from scratch — no ML monitoring SaaS required. Optional webhook alerting via `ALERT_WEBHOOK_URL` (Slack-style JSON).

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **PSI** — for each numeric feature, bin the reference (training) sample into quantiles and measure how much the production sample's mass redistributed across those bins. An epsilon guards the `log` against empty bins.
2. **Bands** — PSI < 0.10 = no shift, 0.10–0.25 = moderate (watch), ≥ 0.25 = significant (alert). These are the standard credit-risk thresholds.
3. **Feature + prediction drift** — the same PSI runs on the model's own prediction/score column, giving a **label-free** proxy for concept drift: you can catch degradation before ground-truth labels ever arrive.
4. **Alerting** — if any feature or the prediction crosses the significant band, `emit_alert()` posts to `ALERT_WEBHOOK_URL` (or dry-runs), naming exactly which feature drifted.

The sample generator simulates a silently degrading model: `income` shifts (new market) and the `score` output drifts with it, while `age` and `tenure_months` stay stable — so you can see the detector flag only what actually moved.

## Learning Connection
Built while studying MLOps, monitoring, and observability (Month 5: AI Agent Workshop).
Applies: Population Stability Index, data vs. prediction (concept) drift, and threshold-based alerting.

## Impact Note
- **Who benefits:** ML engineers and data scientists responsible for models in production without a full monitoring stack.
- **Potential risks:** PSI flags *that* a distribution moved, not *why* or whether it hurts the metric you care about — always confirm against real performance once labels arrive, and tune bins/thresholds per feature. A drift alert is a prompt to investigate, not proof the model is broken.
