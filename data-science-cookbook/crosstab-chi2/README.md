# Crosstab & Chi-Square Tool

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-science-cookbook/crosstab-chi2/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-science-cookbook/crosstab-chi2/demo.ipynb)

> "Compare groups in survey data" — contingency table, chi-square test, effect size, and the exact cells driving the difference.

**The final build of the 120-project portfolio.**

## Business Impact
- **Before:** Survey crosstabs get eyeballed; people call a difference "significant" from raw counts, or report a p-value with no sense of strength.
- **After:** A proper analysis — chi-square (is it real?), Cramér's V (how strong?), and standardized residuals (where's the difference?) — with a plain-English summary.
- **Estimated ROI:** correct, defensible survey conclusions instead of anecdote.

## Tech Stack
Python · scipy.stats chi-square · Cramér's V effect size · standardized residuals · Streamlit · matplotlib

## Demo
**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered, or use the badges above.
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. **Crosstab** two categorical variables into a contingency table.
2. **Chi-square** test of independence vs expected counts (significance).
3. **Cramér's V** effect size (strength — because significance ≠ importance).
4. **Standardized residuals** identify the cells with far more/fewer responses than expected — the story behind the p-value.

## Learning Connection
Built while studying **categorical data analysis**. Applies: the whether/how-much/where triad, and the discipline of reporting effect size alongside significance.

## Impact Note
- **Who benefits:** researchers, PMs, analysts working with survey/categorical data.
- **Potential risks:** chi-square needs adequate expected cell counts (≥5) — use Fisher's exact test for small samples; with many cells, some residuals cross ±2 by chance (multiple comparisons). And association is not causation — a significant crosstab is a starting hypothesis, not a conclusion.

---

*Day 120 of 120 — the final build. From a Day-1 CSV loader to a full data/AI toolkit across data engineering, ML, LLMOps, governance, analytics engineering, and data science.*
