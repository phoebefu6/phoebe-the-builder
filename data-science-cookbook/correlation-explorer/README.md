# Correlation & Multicollinearity Explorer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-science-cookbook/correlation-explorer/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-science-cookbook/correlation-explorer/demo.ipynb)

> "Which features relate?" — a correlation heatmap plus VIF to catch the redundant, collinear features that quietly wreck a model.

## Business Impact
- **Before:** Feature sets carry hidden redundancy; regression coefficients come out unstable or backwards and nobody knows why.
- **After:** A correlation heatmap surfaces related pairs, and VIF exposes multicollinearity a pairwise view can't see — with greedy drop suggestions.
- **Estimated ROI:** more stable, interpretable models; less time chasing nonsensical coefficients.

## Tech Stack
Python · pandas correlation · VIF via numpy least-squares (R² of each feature on the rest) · greedy drop selector · Streamlit · matplotlib

## Demo
**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered, or use the badges above.
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. **Correlation matrix** (Pearson/Spearman) + a list of pairs above a threshold.
2. **VIF** per feature: `1/(1-R²)` from regressing it on all others — catches joint collinearity (e.g. BMI derived from height & weight).
3. **Greedy drops**: remove the highest-VIF feature, recompute, repeat until under threshold.

## Learning Connection
Built while studying **feature selection & regression diagnostics**. Applies: why pairwise correlation misses multicollinearity, and VIF as the honest multivariate check.

## Impact Note
- **Who benefits:** data scientists building regressions/linear models.
- **Potential risks:** VIF matters most for *linear* models — tree ensembles tolerate collinearity, so don't drop features reflexively. The greedy drop keeps the lowest-VIF feature, which isn't always the most business-meaningful one; review before removing.
