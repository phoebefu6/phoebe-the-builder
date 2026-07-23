# Business Glossary Manager

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-quality-governance/business-glossary/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-quality-governance/business-glossary/demo.ipynb)

> "Active user" means five different things to five teams, so every metric argument starts from zero - this gives each business term one owned, versioned definition and flags the governance gaps that let ambiguity creep back in.

## Business Impact
- **Before:** Every dashboard review reopens the same fight over what a metric means. Definitions live in someone's head, a stale wiki, and three conflicting Slack threads. Nobody owns them.
- **After:** One glossary holds each term with an owner, definition, status, synonyms, related links, and the exact data assets it governs. The validator surfaces gaps - ownerless terms, colliding synonyms, retired-but-still-live terms - before they cause the next argument.
- **Estimated ROI:** Kills the recurring "what does this number mean" cycle (hours/week across analytics + business teams) and de-risks metric migrations by flagging deprecated terms still wired to live tables.

## What it flags
- **Ownerless terms** - nobody accountable for keeping the definition true (high).
- **No definition** - a named term that says nothing (high).
- **Synonym collisions** - the same synonym pointing at two different terms, the exact "active user" trap (high).
- **Deprecated but still linked** - a retired term still governing live assets; consumers may still trust a definition you dropped (high).
- **Broken related references** - a related-term link to a term that does not exist (medium).
- **Orphaned terms** - no linked assets and no related terms; likely unused or disconnected from the model (low).

Every finding is a typed, explainable `Issue` (`issue_type`, `severity`, `term`, `message`) - a prioritized review queue, not a black box.

## Tech Stack
Python, pandas, Streamlit, matplotlib. Fully offline - no API keys, no external services. Standard library dataclasses for the core model.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

CLI quick look (prints the glossary summary + the validation issues table):
```bash
python glossary.py
```

## Learning Connection
Built on the data governance / semantic-layer arc: a business glossary is the human-readable half of a semantic layer, the layer where metric definitions become governed, owned assets instead of tribal knowledge. It pairs with data contracts and the metric layer - the glossary says what a term means and who owns it; the contract and models enforce how it is computed.

## Impact Note
- **Who benefits:** data stewards and glossary owners, analytics engineers building the semantic/metric layer, and business teams who need one trustworthy definition to argue from.
- **Potential risks:** a glossary is only as good as its owners. The validator flags process gaps - it does not decide what a metric should mean, and it cannot tell you a definition is wrong, only that one is missing, ownerless, or ambiguous. Fixing the meaning is a stewardship conversation, not a tool output. A clean validation report is necessary, not sufficient.
