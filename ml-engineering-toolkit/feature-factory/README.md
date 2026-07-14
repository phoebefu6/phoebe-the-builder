# Feature Factory

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ml-engineering-toolkit/feature-factory/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ml-engineering-toolkit/feature-factory/demo.ipynb)

> Every new ML project you rewrite the same impute / scale / one-hot boilerplate, get the column lists slightly wrong, and leak test statistics into training. Feature Factory inspects a table, infers each column's role, and hands you a fitted, reusable sklearn `ColumnTransformer` in one call.

## Business Impact
- **Before:** Feature prep is copy-pasted per project. Someone forgets to scale, one-hots a 5,000-category id column into a memory bomb, or fits the scaler on the full dataset and quietly leaks the test set into training.
- **After:** Point it at a DataFrame. It classifies every column (numeric / categorical / binary / datetime / drop), builds the matching transformer, and fits it with no leakage - stats learned on `fit`, replayed on `transform`. Unseen categories at score time are absorbed, not crashed on.
- **Estimated ROI:** ~1-2 hrs of feature-prep boilerplate saved per project start, plus the silent-leakage bugs that never ship.

## Tech Stack
Python, scikit-learn (`ColumnTransformer`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`), pandas, numpy, Streamlit, matplotlib. No API keys, runs anywhere.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with the inferred plan, feature matrix, an unseen-data score, and the expansion chart. Or click the Colab/Binder badges above to run it live.

For the Streamlit app (upload a CSV, review/edit the plan, download the feature matrix):
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
An `infer_plan` pass classifies each column by dtype + cardinality, then `build_transformer` assembles the matching sklearn pieces:

| Role | Rule | Treatment |
|------|------|-----------|
| `numeric` | numeric dtype | median impute + standardize |
| `categorical` | low-cardinality object | most-frequent impute + one-hot (`handle_unknown='ignore'`) |
| `binary` | 2 unique values | one-hot with a single reference column |
| `datetime` | datetime dtype | expand to year / month / day / dayofweek / hour |
| `drop` | id-like, high-cardinality, or the target | excluded from features |

The plan is a plain, inspectable object - override any role before building. The result is a vanilla `ColumnTransformer`, so it drops straight into a `Pipeline([..., ('model', clf)])`, `joblib.dump`s cleanly, and gives you identical feature logic across training, batch scoring, and a live API.

The key point the demo makes: the **same fitted transformer** turns a fresh batch - even one containing a category the model never saw - into the exact same feature columns, instead of crashing your 3am scoring job.

## Learning Connection
Built while studying the ML Engineering Toolkit track (sklearn pipelines, data-centric ML, leakage prevention, MLOps feature reuse).
Applies: `ColumnTransformer`/`Pipeline` composition, dtype-driven feature planning, train/serve skew and leakage prevention, reusable feature specs.

## Impact Note
- **Who benefits:** Data scientists and ML engineers starting a new modeling project on tabular data who want a leakage-safe feature baseline in one call.
- **Potential risks:** Role inference is heuristic - it can mislabel a numeric-coded category (a zip code) as numeric, or drop a high-cardinality column you actually wanted (target-encode instead). Always eyeball the inferred plan and override before trusting it. It builds features, it does not select or validate them; pair it with feature-importance and cross-validation before shipping a model.
