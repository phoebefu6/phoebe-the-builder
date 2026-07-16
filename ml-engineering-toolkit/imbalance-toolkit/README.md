# Class Imbalance Toolkit

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ml-engineering-toolkit/imbalance-toolkit/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ml-engineering-toolkit/imbalance-toolkit/demo.ipynb)

> "Our fraud model ignores the minority class." — a classifier that predicts "not fraud" every time scores 98% accuracy and catches zero fraud.

Compares four ways to handle skewed classification data — **Baseline**, **Class Weights**, **SMOTE**, **Random Undersample** — and ranks them by the metric that actually matters on rare-class problems: **minority-class recall** (plus PR-AUC), not accuracy.

## Business Impact
- **Before:** Team ships the highest-accuracy model; it quietly misses ~half the fraud because accuracy rewards ignoring the rare class.
- **After:** One run shows the trade-off table — the rebalanced model catches **89% of fraud vs 53% baseline** (on the sample data), and you pick the recall/precision point that fits your review budget.
- **Estimated ROI:** Cuts the "why didn't the model catch this?" fire-drill; standard rebalancing evaluation in one function instead of an ad-hoc notebook per project.

## Tech Stack
Python · scikit-learn · imbalanced-learn (SMOTE, RandomUnderSampler, imblearn Pipeline) · pandas · matplotlib · Streamlit · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs and a recall-vs-precision chart, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Upload any CSV (binary 0/1 target) or use the built-in fraud sample, pick the target column, and hit **Compare strategies**.

Quick CLI check:
```bash
python imbalance.py
```

## How It Works
1. `imbalance_report()` — quantifies skew (minority %, imbalance ratio).
2. `compare_strategies()` — trains all four strategies on the *same* stratified split; SMOTE/undersampling sit inside an `imblearn` Pipeline so resampling never leaks into the test set.
3. Scores each on recall, precision, F1, PR-AUC, ROC-AUC + a confusion-matrix breakdown, sorted by recall.
4. `recommend()` — one-line pick with the recall lift over baseline and the precision cost.

**Edge case handled:** if the minority class is too sparse to stratify or to run SMOTE (< 2 samples), those paths are skipped gracefully instead of crashing the run.

## Learning Connection
Built while studying **class imbalance handling** in the ML Engineering track.
Applies: resampling vs cost-sensitive learning, leakage-safe pipelines, and choosing evaluation metrics for rare-class problems.

## Impact Note
- **Who benefits:** ML engineers / data scientists building fraud, churn, defect, or disease classifiers on skewed data.
- **Potential risks:** Higher recall comes with more false positives — on the sample data precision drops to ~20%. Don't deploy a rebalanced model without a downstream review capacity or a tuned decision threshold; and SMOTE on categorical/leaky features can create unrealistic samples.
