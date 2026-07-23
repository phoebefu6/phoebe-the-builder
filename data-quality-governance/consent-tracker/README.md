[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-quality-governance/consent-tracker/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-quality-governance/consent-tracker/demo.ipynb)

# Consent & Purpose Tracker

> We process personal data for purposes the user never agreed to - and can't prove otherwise when asked.

## Business Impact
- **Before:** Consent lives in one system, processing happens in another, and nobody can answer "what's our lawful basis for this?" without a manual, days-long trawl. When a subject or regulator asks, the org is exposed.
- **After:** Every processing activity is cross-checked against the consent log in seconds. Each gap is flagged by type - no-consent, withdrawn, expired, purpose-mismatch - with a plain-English reason a reviewer can act on, sorted by severity.
- **Estimated ROI:** ~1-2 days saved per data-subject-access or audit request, plus early catch of processing that would otherwise become a reportable breach or a fine.

## What it flags
- **No consent** - processing a purpose with nothing on file for that subject.
- **Withdrawn** - the subject pulled consent (latest record wins) yet processing continues.
- **Expired** - consent lapsed before the audit date but the data is still in use.
- **Purpose-mismatch** - data gathered under basis A (e.g. analytics) reused for purpose B (e.g. personalization); the purpose-limitation principle.

## Tech Stack
Python, pandas, numpy, Streamlit, matplotlib. Fully offline - no API keys, no external services.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

CLI quick look (prints findings by severity + the compliance summary for the built-in sample):
```bash
python tracker.py
```

## Learning Connection
Built while studying consent management and the purpose-limitation principle on the Data Quality & Governance arc - the PDPA-to-GDPR consent/lawful-basis line. Applies: treating consent as a time-ordered log (grant / withdraw / re-grant), lawful-basis validation, and explainable, steward-first compliance alerting. Reproducibility is enforced with a fixed `AS_OF` date so expiry checks never drift with the wall clock.

## Impact Note
- **Who benefits:** DPOs, privacy engineers, data stewards, and anyone who has to prove a lawful basis on demand.
- **Potential risks:** findings are a compliance review queue, not legal advice. The tool surfaces and explains gaps; it does not make the lawful-basis determination. Legal owns that call - a flagged item may have a valid basis the tool can't see (e.g. a legitimate-interest assessment recorded elsewhere), and a clean item may still warrant scrutiny. Never stop or greenlight processing on the tool's word alone.
