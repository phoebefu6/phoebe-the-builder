# Recommendation Engine

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-science-cookbook/recommender/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-science-cookbook/recommender/demo.ipynb)

> "We have no 'you may also like'" — item-item collaborative filtering, with an explanation for every recommendation.

## Business Impact
- **Before:** No related-items rail, no personalization; users don't discover more of what they'd like.
- **After:** "Customers who liked X also liked Y" plus personalized recommendations — each with a plain-English reason.
- **Estimated ROI:** higher engagement, cross-sell, and session depth from relevant suggestions.

## Tech Stack
Python · numpy/pandas · item-item cosine similarity · similarity-weighted scoring · explanation trace · Streamlit · matplotlib

## Demo
**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered, or use the badges above.
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. **Matrix** — users × items with ratings/engagement.
2. **Similarity** — cosine between item columns (co-engaged items are similar).
3. **Recommend** — score each unseen item by similarity to the user's history, weighted by their ratings.
4. **Explain** — surface the user's items most responsible for each recommendation.

## Learning Connection
Built while studying **recommender systems**. Applies: item-item CF as the pragmatic first recommender, cosine similarity on interactions, and explainability by construction.

## Impact Note
- **Who benefits:** product, growth, e-commerce, content teams.
- **Potential risks:** CF has a **cold-start** problem — brand-new items with no interactions can't be recommended, and popular items dominate (popularity bias). Blend with content features for new items, normalize for popularity, and watch for feedback loops that narrow what users ever see.
