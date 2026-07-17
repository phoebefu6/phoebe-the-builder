# Mini Model Registry

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ml-engineering-toolkit/model-registry/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ml-engineering-toolkit/model-registry/demo.ipynb)

> "Which model is actually in production?" should have a one-word answer.

A tiny, file-backed model registry that solves one specific pain: **"We lose track of model versions."**
Versioned local artifact store + a JSON metadata index + stage promotion with a
single-production-per-model invariant.

## Business Impact

- **Before:** Model files scattered as `model_final.pkl`, `model_final_v2.pkl`,
  `model_REALLY_final.pkl`. Nobody can say which one is live, what its metrics
  were, or which params trained it. Rollback is a guessing game.
- **After:** Every model is a numbered version with a content hash, its metrics,
  its params, and a stage (`staging` / `production` / `archived`). Exactly one
  version can be in production at a time - promotion auto-archives the previous one.
- **ROI:** Minutes instead of hours to answer "what's live and how good is it?",
  and safe, auditable rollbacks. Removes a whole class of "wrong model shipped"
  incidents at near-zero cost (no service to run, just a folder + a JSON file).

## Tech Stack

- Versioned local store: joblib artifacts under `_registry/artifacts/<name>/v<N>.joblib`
- Metadata: `index.json` (list of version records)
- Integrity: `hashlib.sha256` content hash (first 12 chars) of the artifact bytes
- Stage promotion: `staging` -> `production` -> `archived`, single-production invariant
- UI: Streamlit + matplotlib; models trained with scikit-learn
- pandas for the registry table view

## Demo

Notebook walkthrough: [`demo.ipynb`](./demo.ipynb) (also runnable via the Colab /
Binder badges above).

Run the Streamlit app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or run the core logic demo directly:

```bash
python registry.py
```

## Learning Connection

This is an **MLOps** primitive. It applies three ideas that scale up to any
production system:

- **Model versioning** - every train is an immutable, numbered artifact.
- **Artifact hashing** - a content hash proves two "v3"s are byte-identical (or not).
- **Stage promotion** - a controlled lifecycle (`staging` -> `production` ->
  `archived`) with an enforced invariant, the same pattern MLflow / SageMaker
  Model Registry use for stage transitions.

## Impact Note

- **Who benefits:** small teams and solo builders who need model lineage and a
  clear "what's in production" answer without standing up a registry service.
- **Risks / limits:** this is a **mini / local** registry meant for learning and
  small projects. It is **not** a substitute for MLflow, SageMaker Model
  Registry, or Vertex AI in production - no concurrency control, no access
  control, no remote artifact store, no lineage to training data. Treat the
  `_registry/` folder as ephemeral demo state.
