# Batch Scoring Service

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ml-engineering-toolkit/batch-scorer/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ml-engineering-toolkit/batch-scorer/demo.ipynb)

> Scoring new data shouldn't be a manual copy-paste ritual.

## Business Impact
- **Before:** A trained model sits in a notebook. Every new batch of records is scored by hand - copy the rows in, re-run scattered cells, hope the columns line up, paste the outputs back into a spreadsheet. Slow, error-prone, and impossible to hand off.
- **After:** A trained model is persisted once as a versioned bundle. New data - a CSV or a DataFrame - is scored with a single call (or a single upload in the app), returning both a probability `score` and a 0/1 `prediction`, with schema mismatches caught up front instead of producing silently-wrong numbers.
- **Estimated ROI:** For a team scoring a few batches a week, this collapses a ~30-minute manual ritual into seconds and removes the class of "wrong column order" errors entirely - roughly 2-4 analyst hours reclaimed per week, plus far fewer bad-scoring incidents.

## Tech Stack
- **Python 3.11** with type hints throughout
- **scikit-learn** - `RandomForestClassifier` for the churn-style demo model
- **joblib** - model + schema + threshold persisted as one bundle
- **pandas / numpy** - batch inference and schema reconciliation
- **Streamlit** - upload-and-download scoring UI
- **matplotlib** - score-distribution and flagged-count charts
- **Docker** + **GitHub Actions** - containerized run and CI smoke test

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Or score from the command line:
```bash
python scorer.py        # trains, scores sample data, prints a summary
```

## Learning Connection
Built while studying ML Engineering (MLOps track). Applies: model persistence, batch inference, schema reconciliation.

## Impact Note
- **Who benefits:** Data analysts and ML engineers who need to run a trained model over fresh batches without babysitting a notebook, and the downstream teams (retention, ops) who consume the flagged rows.
- **Potential risks:** The demo model is trained on synthetic data - real deployments must validate against ground-truth outcomes, monitor for drift, and calibrate the threshold to the business cost of false positives vs false negatives. Scores are probabilities, not certainties; flagged accounts warrant human review before any customer-facing action.
