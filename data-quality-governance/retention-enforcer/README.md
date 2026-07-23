# Data Retention Enforcer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-quality-governance/retention-enforcer/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-quality-governance/retention-enforcer/demo.ipynb)

> We keep everything forever - storage bloats and we are one audit away from a finding for holding personal data past its purpose. Score every record against its retention policy and get an explainable action plan for human review.

## Business Impact
- **Before:** Data lives forever by default. Nobody can say which records are past their purpose until an auditor asks - and then it is a manual, panicked hunt across systems.
- **After:** Every record is scored against the policy for its data_class. The engine produces a triaged action plan - what is past retention (with the action due), what is approaching expiry, what is within policy - each row explainable by class, age, policy, and verdict.
- **Estimated ROI:** Weeks of audit-prep collapse to a re-runnable report; storage-at-stake is quantified up front; the odds of an over-retention finding drop because expiry is visible before it happens, not after.

## Tech Stack
Python, pandas, numpy, Streamlit, matplotlib. Fully offline - no API keys, no external services. Ages are computed against a fixed `AS_OF` baseline so every run is reproducible.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app (upload records, edit policies, view the plan by action):
```bash
pip install -r requirements.txt
streamlit run app.py
```

CLI quick look (prints the action plan and rollup for the built-in sample):
```bash
python enforcer.py
```

## Learning Connection
Built while studying data retention and minimization on the Data Quality & Governance arc. Applies the storage-limitation principle that runs through PDPA and GDPR - personal data should be kept no longer than necessary for the purpose it was collected. This tool turns that principle into a concrete, explainable schedule: policy per data_class, age against a fixed baseline, verdict per record.

## Impact Note
- **Who benefits:** data protection officers, data stewards, and platform owners who need to prove - not assert - that data is not held past its purpose.
- **Potential risks:** the engine PLANS actions for human review; it never auto-deletes or auto-anonymizes. Retention limits are policy decisions, not code defaults, and a record may carry a legal hold that overrides its schedule. Legal owns the policy; treat the output as a prioritized review queue and confirm holds and business need before any disposal.
