# Probability Calibration Checker

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ml-engineering-toolkit/calibration-checker/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ml-engineering-toolkit/calibration-checker/demo.ipynb)

> A "0.9" from your model should mean it happens 90% of the time. Often it doesn't.

Reliability curve + Brier score + Expected Calibration Error (ECE), comparing an
uncalibrated base learner against **Platt (sigmoid)** and **isotonic** recalibration.

## Business Impact

**Before.** A risk model outputs "0.9 churn" and the retention team treats it as
a near-certainty. In reality those 0.9 accounts churn ~65% of the time - the model
is over-confident. Budgets, expected-value math, and threshold decisions are all
built on numbers that do not mean what they say. Nobody notices because AUC looks
fine (ranking is fine; the *probabilities* are wrong).

**After.** The reliability curve exposes the gap in one chart, ECE and Brier put a
single number on it, and a Platt or isotonic wrapper maps the scores back onto
reality - without retraining the underlying model.

**ROI.** Trustworthy probabilities let downstream teams do expected-value math
(cost * P(event)) correctly. On a churn book where you spend retention dollars in
proportion to predicted risk, cutting calibration error typically recovers
mis-allocated spend and prevents both over- and under-investing in the wrong
accounts - a one-file, no-retrain fix that unblocks every decision layered on top.

## Tech Stack

- **scikit-learn** - `make_classification`, `GaussianNB` (a deliberately
  over-confident base learner), `CalibratedClassifierCV` (sigmoid + isotonic),
  `calibration_curve`, `brier_score_loss`, `log_loss`, `roc_auc_score`
- **ECE** - computed from scratch: sample-weighted mean |confidence - accuracy|
  across probability bins
- **matplotlib** - reliability diagram
- **Streamlit** - interactive app
- **pandas / numpy** - scoring table and math

## Demo

Notebook: [demo.ipynb](demo.ipynb) - or open it in
[Colab](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ml-engineering-toolkit/calibration-checker/demo.ipynb)
/ [Binder](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ml-engineering-toolkit/calibration-checker/demo.ipynb).

Run the core script:

```bash
pip install -r requirements.txt
python calibration.py          # prints the metric table, sorted by ECE
```

Launch the Streamlit app:

```bash
streamlit run app.py
```

Or with Docker:

```bash
docker build -t calibration-checker .
docker run -p 8501:8501 calibration-checker
```

## Learning Connection

**ML Engineering / MLOps.** Calibration is the difference between a model that
*ranks* well and one whose outputs can be *trusted as probabilities*. This project
applies the standard toolkit - reliability curves, the Brier score, and Expected
Calibration Error - and the two canonical fixes (Platt scaling and isotonic
regression via `CalibratedClassifierCV`). It is the check every classifier feeding
a decision or an expected-value calculation should pass before it ships.

## Impact Note

**Who benefits.** Anyone consuming model probabilities as numbers, not just
rankings: risk / churn / fraud / pricing / credit teams, and any pipeline that
multiplies a cost by a predicted probability.

**Risks.** Calibration is itself a fitted model. On **small data**, isotonic
regression in particular can overfit the calibration set and look great in-sample
while being worse out-of-fold - always calibrate on held-out data (as
`CalibratedClassifierCV` does with `cv`), and prefer the simpler sigmoid map when
data is scarce. A well-calibrated *aggregate* can still hide subgroup miscalibration,
so check calibration within important segments, not just overall.
