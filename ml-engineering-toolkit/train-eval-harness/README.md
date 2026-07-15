# Train/Eval Leaderboard

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ml-engineering-toolkit/train-eval-harness/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ml-engineering-toolkit/train-eval-harness/demo.ipynb)

> Model selection is ad-hoc - train a few classifiers, eyeball one accuracy number, ship whichever looked best on one lucky split. This harness cross-validates a whole roster of models on identical folds, scores every fold on several metrics, and ranks them with a mean ± std leaderboard.

## Business Impact
- **Before:** Model comparison happens in a scratch notebook. Someone reports "the random forest got 82%" from a single train/test split, with no baseline, no error bars, and no reproducible fold seed. A model that won by luck on one split gets shipped.
- **After:** One call runs stratified k-fold CV over every model through an identical, leakage-safe pipeline, scores Accuracy / ROC AUC / F1 / Precision / Recall per fold, and returns a ranked table with mean ± std - plus a majority-class baseline so the no-skill floor is always visible.
- **Estimated ROI:** ~1-2 hrs of hand-rolled comparison code per modeling project, and the far more expensive class of "we shipped the model that got lucky on one split" mistakes.

## Tech Stack
Python, scikit-learn (`cross_validate`, `StratifiedKFold`, `Pipeline`, `ColumnTransformer`, `DummyClassifier`, LogisticRegression / DecisionTree / RandomForest / GradientBoosting), pandas, numpy, Streamlit, matplotlib. No API keys, runs anywhere.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with the ranked leaderboard, the baseline pinned at AUC 0.5, and the per-fold spread boxplot. Or click the Colab/Binder badges above to run it live.

For the Streamlit app (upload a CSV, pick the target, download the leaderboard):
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
Every model is wrapped in the **same** preprocessing pipeline (median-impute + scale numerics, most-frequent-impute + one-hot categoricals), and that whole pipeline is what gets cross-validated - so scaler and encoder statistics are learned per fold and never leak across the split boundary.

| Piece | Choice | Why |
|-------|--------|-----|
| Splitter | `StratifiedKFold(shuffle, seed)` | class balance preserved in every fold; reproducible |
| Baseline | `DummyClassifier(most_frequent)` | pins the no-skill floor - AUC lands at 0.5 |
| Metrics | Accuracy, ROC AUC, F1, Precision, Recall | one number hides the precision/recall trade-off |
| Ranking | mean of the first metric, `± std` reported | rank on the mean, judge on the spread |
| Safety | folds auto-capped to the rarest class count | won't ask for more folds than the data supports |

The result is two frames: a ranked `leaderboard` (mean/std per metric + fit time) and a tidy `fold_detail` (model, metric, fold, score) that drives the per-fold boxplot. Point it at any DataFrame with a binary target - preprocessing, CV, scoring, and ranking are automatic.

## Learning Connection
Built while studying the ML Engineering Toolkit track (sklearn pipelines, cross-validation, model documentation, MLOps).
Applies: `cross_validate` with multiple scorers, stratified CV, leakage-safe pipeline composition, baseline discipline, reading mean vs. variance in model comparison.

## Impact Note
- **Who benefits:** Data scientists and ML engineers choosing between candidate models on tabular data who want a reproducible, baseline-anchored comparison instead of a single-split eyeball.
- **Potential risks:** A leaderboard ranks - it does not validate. Cross-validation still leaks if you engineered features on the full dataset *before* handing it in, and the synthetic churn data is illustrative only. Watch the metric you rank on (accuracy flatters the majority class on imbalanced targets - prefer ROC AUC or F1 there), and treat a narrow winning margin inside the ± std band as a tie, not a victory.
