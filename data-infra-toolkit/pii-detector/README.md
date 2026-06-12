# PII Detector and Masker

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-infra-toolkit/pii-detector/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-infra-toolkit/pii-detector/demo.ipynb)

> "We accidentally share sensitive data in staging" — scan CSV data for PII and mask it before it leaves production.

## Business Impact
- **Before:** Data engineers manually review exports for sensitive data, miss PII in free-text fields, staging databases contain real SSNs/emails
- **After:** Automated scan detects 7 PII types across all columns in seconds, smart masking preserves data utility while removing identifiers
- **Estimated ROI:** 3-5 hours/week saved per data team, eliminates compliance incident risk

## Tech Stack
Python, pandas, regex, Streamlit, matplotlib, seaborn, Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs (heatmaps, masked tables, risk scores), or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## PII Types Detected
| Type | Example | Masking Strategy |
|------|---------|-----------------|
| Email | `alice@acme.com` | `a████@acme.com` |
| Phone (US) | `(555) 123-4567` | `███████4567` |
| SSN | `123-45-6789` | `XXX-XX-6789` |
| Credit Card | `4111-1111-1111-1111` | `████████████1111` |
| IP Address | `192.168.1.100` | `██████████████` |
| Date of Birth | `03/15/1990` | `██████████` |
| US Zip Code | `94102-3456` | `██████████` |

## Learning Connection
Built while studying Data Engineer Career Track (DS365) and IBM DE Certificate.
Applies: Data governance (PII detection, data privacy & GDPR/CCPA compliance), regex pattern matching, data quality automation.

## Impact Note
- **Who benefits:** Data engineers, analytics teams, compliance officers — anyone who shares data across environments
- **Potential risks:** Regex-based detection has false positives/negatives; should not be sole compliance mechanism. Always pair with manual review for regulated data.
