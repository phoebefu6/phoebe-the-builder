# Hyperparameter Tuner

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ml-engineering-toolkit/hyperparam-tuner/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ml-engineering-toolkit/hyperparam-tuner/demo.ipynb)

> We grid-search by hand - nudge `n_estimators`, re-run a cell, keep whatever looked better. This wraps an Optuna TPE search around a cross-validated objective, so tuning is a budgeted, reproducible search that reports its lift over the model's out-of-the-box defaults.

## Business Impact
- **Before:** Tuning is a person editing hyperparameters in a notebook and eyeballing one split. It's slow, unreproducible, overfits to that split, and there's no honest baseline - "I tuned it" with no number for how much it actually helped.
- **After:** Define the space once. An Optuna TPE sampler proposes configurations, each scored by 5-fold CV, and the tool reports the **tuned score AND the lift over defaults** plus a convergence curve. Reproducible (seeded), budgeted (you set the trial count), and it tells you when to stop.
- **Estimated ROI:** Hours of manual grid-poking per model replaced by a fixed budget; more importantly, an honest lift number that stops you tuning a model whose defaults were already fine.

## Tech Stack
Python, Optuna (TPE sampler), scikit-learn (`cross_val_score`, `StratifiedKFold`, `Pipeline`, `ColumnTransformer`, `RandomForestClassifier`), pandas, numpy, Streamlit, matplotlib. No API keys, runs anywhere.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with the default baseline, the tuned lift, the winning params, and the optimization-history curve. Or click the Colab/Binder badges above to run it live.

For the Streamlit app (upload a CSV, set a trial budget, watch the search beat the defaults):
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
The RandomForest is wrapped in a shared impute→scale→one-hot preprocessor, and the whole pipeline is cross-validated per configuration - so fold statistics never leak into the score being optimized.

| Piece | Choice | Why |
|-------|--------|-----|
| Baseline | model defaults, same CV | the honest reference every lift is measured against |
| Sampler | Optuna `TPESampler(seed)` | learns from prior trials; reproducible, beats blind grid |
| Objective | 5-fold stratified CV ROC AUC | optimize generalization, not one lucky split |
| Space | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | edit `_suggest()` to widen/narrow |
| Report | tuned score, **lift over defaults**, best params, history | know *how much* tuning bought you, and when it converged |

Output is a result dict (`default_score`, `best_score`, `lift`, `best_params`, `n_trials`) plus a tidy `history` frame (`trial`, `value`, `best_so_far`) that drives the convergence plot.

## Learning Connection
Built while studying the ML Engineering Toolkit track (Optuna, sklearn pipelines, cross-validation, MLOps).
Applies: Bayesian/TPE hyperparameter search, CV-in-the-loop objectives, baseline-anchored improvement measurement, reading a convergence curve to set a budget.

## Impact Note
- **Who benefits:** Data scientists and ML engineers who want reproducible tuning with an honest baseline instead of hand-edited notebook cells.
- **Potential risks:** More trials = more chances to overfit the CV folds; hold out a final untouched test set before trusting the tuned score. The lift is only as meaningful as the search space and the CV setup - a leaky preprocessing step upstream, or a space that's too narrow, will mislead. If the lift over defaults is tiny, that's a signal to stop tuning and improve features, not to spend a bigger budget.
