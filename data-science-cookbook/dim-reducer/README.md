# PCA / t-SNE Explorer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-science-cookbook/dim-reducer/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-science-cookbook/dim-reducer/demo.ipynb)

> "Too many features to see" — project high-dimensional data down to 2D and actually look at its structure.

## Business Impact
- **Before:** High-dimensional data is a black box; clusters, gradients, and outliers are invisible in a table.
- **After:** PCA/t-SNE project it to 2D so structure is visible — and PCA reports how much information the 2D view kept.
- **Estimated ROI:** faster EDA, earlier discovery of segments and anomalies, better feature intuition before modeling.

## Tech Stack
Python · scikit-learn (PCA, t-SNE) · StandardScaler · scree analysis · Streamlit · matplotlib

## Demo
**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered, or use the badges above.
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. **Standardize** features (both methods are scale-sensitive).
2. **PCA** → 2D, with per-component explained variance and a scree plot.
3. **t-SNE** → 2D for non-linear cluster separation (perplexity-tunable).
4. **Color** the projection by a label to check whether groups separate.

## Learning Connection
Built while studying **dimensionality reduction**. Applies: linear vs non-linear projection, explained variance for "how many dimensions do I need," and the trap of reading distances off a t-SNE plot.

## Impact Note
- **Who benefits:** data scientists doing EDA on wide datasets.
- **Potential risks:** **t-SNE distances and cluster sizes are not meaningful** — only local neighborhoods are; don't infer "these two clusters are far apart." t-SNE is also stochastic and perplexity-sensitive; PCA is the reproducible default.
