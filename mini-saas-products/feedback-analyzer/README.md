# User Feedback Analyzer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/mini-saas-products/feedback-analyzer/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=mini-saas-products/feedback-analyzer/demo.ipynb)

> We have 10K reviews and no insights — this turns raw feedback into a sentiment split, ranked complaints, ranked praises, and the one thing to fix first.

## Business Impact
- **Before:** Reviews pile up in the App Store / G2 / support inbox. Nobody reads all of them, so decisions are driven by the loudest single complaint.
- **After:** Every review is scored and mined for themes in seconds. You see the sentiment split and a frequency-ranked complaint list — a backlog, not anecdotes.
- **Estimated ROI:** Cuts a full day of manual review-reading to seconds and points engineering at the highest-frequency pain first.

## Tech Stack
Python 3.10+, Streamlit, pandas, matplotlib. Sentiment is a lexicon scorer with a negation rule (zero ML deps, runs offline). Optional Claude (Anthropic SDK) writes a PM-ready summary.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Sentiment** — lexicon of positive/negative words plus one rule: a negator within the prior two tokens flips polarity, so "not good" scores negative. Each review gets a label (positive/neutral/negative) and a score in [-1, 1].
2. **Distribution + NPS-like** — the sentiment split and a promoter-minus-detractor proxy (% positive − % negative).
3. **Theme mining** — top content unigrams + bigrams, mined *separately* within negative and positive reviews. Stopwords, sentiment words, and negators are stripped so themes are nouns ("battery", "customer service"), not "good".
4. **Insight** — a plain-English takeaway (Claude-written if `ANTHROPIC_API_KEY` is set, else rule-based) that leads with the top complaint to fix.

The split tells you *how many* are unhappy; separating themes by sentiment tells you *why* — negative themes become the bug backlog, positive themes become marketing copy.

## Learning Connection
Built while studying NLP and product discovery (Month 6: Mini SaaS Products).
Applies: lexicon sentiment, negation handling, n-gram theme extraction, complaint ranking.

## Impact Note
- **Who benefits:** PMs, support leads, and founders drowning in unread feedback.
- **Potential risks:** Lexicon sentiment misses sarcasm and domain slang — for high-stakes decisions, validate on a labeled sample or swap in a transformer model (the theme-mining structure stays the same). Frequency ≠ importance: a rare complaint can still be a churn driver, so read the negative reviews, don't just count them.
