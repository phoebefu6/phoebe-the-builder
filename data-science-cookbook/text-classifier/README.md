# Text Classification Trainer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-science-cookbook/text-classifier/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-science-cookbook/text-classifier/demo.ipynb)

> "Auto-tag our support tickets" — train a TF-IDF + logistic-regression classifier and route new text by category.

## Business Impact
- **Before:** Tickets are triaged by hand; routing is slow and inconsistent.
- **After:** A trained classifier tags incoming text (billing / technical / account) with a confidence score — cheap, offline, and interpretable.
- **Estimated ROI:** faster routing, fewer misrouted tickets, and an auditable model you can trust.

## Tech Stack
Python · scikit-learn (TF-IDF + LogisticRegression pipeline) · held-out evaluation · coefficient-based interpretability · Streamlit · matplotlib

## Demo
**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered, or use the badges above.
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. **Vectorize** with TF-IDF (uni+bigrams, stop words removed).
2. **Train** a logistic regression on a stratified split.
3. **Evaluate** on held-out data (accuracy + confusion matrix).
4. **Interpret** — surface the top weighted words per class.
5. **Predict** new text with class probabilities.

## Learning Connection
Built while studying **text classification & interpretable NLP baselines**. Applies: when a simple TF-IDF + linear model beats an LLM (cost, latency, transparency), and reading model coefficients as explanations.

## Impact Note
- **Who benefits:** support, ops, and anyone routing high-volume text.
- **Potential risks:** small labeled sets overstate accuracy (few test samples) and miss vocabulary the model never saw. Add an "uncertain → human" confidence threshold, retrain as data grows, and never auto-close/route critical tickets on a low-confidence tag.
