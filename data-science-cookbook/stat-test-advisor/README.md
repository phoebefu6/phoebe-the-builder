# Statistical Test Advisor

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-science-cookbook/stat-test-advisor/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-science-cookbook/stat-test-advisor/demo.ipynb)

> "Which statistical test do I use?" — describe the data, get the right test (parametric or not), and run it.

Opens **Month 12: Data Science Cookbook.**

## Business Impact
- **Before:** People reach for a t-test by reflex, ignore assumptions, and report p-values from the wrong test.
- **After:** A decision tree picks the correct test from the question shape, auto-checks normality to choose parametric vs non-parametric, and runs it with an effect size and plain-English conclusion.
- **Estimated ROI:** fewer invalid analyses; correct tests without a stats refresher every time.

## Tech Stack
Python · scipy.stats · Shapiro-Wilk normality gate · decision-tree recommender · Streamlit · matplotlib

## Demo
**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered, or use the badges above.
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. **Classify** the question: 2 groups / 3+ groups / two categories / correlation.
2. **Check assumptions** — Shapiro-Wilk decides parametric vs non-parametric.
3. **Recommend** the test with its reason and assumptions.
4. **Run** it (t-test, Mann-Whitney, ANOVA, Kruskal-Wallis, chi-square, Pearson/Spearman) and report statistic, p-value, effect size, and conclusion.

## Learning Connection
Built while studying **statistics & hypothesis testing**. Applies: assumption-aware test selection and separating recommendation from execution.

## Impact Note
- **Who benefits:** analysts, PMs running experiments, anyone doing ad-hoc stats.
- **Potential risks:** picking the right test doesn't validate the study design — confounding, multiple comparisons, and p-hacking still apply. Significance ≠ importance; always read the effect size.
