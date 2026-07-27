# Topic Modeling Tool

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-science-cookbook/topic-modeler/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-science-cookbook/topic-modeler/demo.ipynb)

> "What are these documents about?" — discover the themes in a text pile automatically, no labels required.

## Business Impact
- **Before:** Thousands of tickets/reviews/docs and no idea what's in them; reading a sample and guessing.
- **After:** Unsupervised topic modeling surfaces recurring themes and tags every document, turning an undifferentiated pile into countable, filterable categories.
- **Estimated ROI:** hours of manual reading replaced; emerging issues spotted by tracking topic prevalence.

## Tech Stack
Python · scikit-learn (NMF over TF-IDF) · per-topic top words + doc-topic assignment · Streamlit · matplotlib

## Demo
**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered, or use the badges above.
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. **Vectorize** documents with TF-IDF (English stop words removed).
2. **Factorize** with NMF into topics × words and docs × topics.
3. **Read** the top words of each topic as its theme.
4. **Label** each document with its dominant topic and confidence.

## Learning Connection
Built while studying **topic modeling & unsupervised NLP**. Applies: **why NMF over TF-IDF beats LDA on short texts** (LDA's generative model assumes long documents), and reading a factorization as themes.

## Impact Note
- **Who benefits:** support, product, DevRel, research — anyone with unlabeled text at volume.
- **Potential risks:** topics are statistical, not semantic — top words need human interpretation, and a bad topic count produces muddled or split themes. Validate topic coherence and don't over-trust auto-labels for anything decision-critical.
