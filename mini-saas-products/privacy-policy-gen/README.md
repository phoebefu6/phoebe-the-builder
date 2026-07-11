# Privacy Policy Generator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/mini-saas-products/privacy-policy-gen/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=mini-saas-products/privacy-policy-gen/demo.ipynb)

> Legal templates cost money and are still generic — answer a short questionnaire and get a GDPR/CCPA-aware policy draft plus a checklist of the compliance work the document can't do for you.

## Business Impact
- **Before:** Founders copy a competitor's policy or pay for a template that doesn't match what they actually collect — and still miss the operational requirements (consent banner, Do-Not-Sell link, DPAs).
- **After:** A policy that reflects your real data practices in minutes, with hard warnings for legally mandatory actions beyond the document.
- **Estimated ROI:** A defensible first draft plus a compliance to-do list — lawyer review time drops from drafting to reviewing.

## Tech Stack
Python 3.10+, Streamlit. Dataclass-driven template engine with conditional GDPR/CCPA/COPPA sections. Runs fully offline, no API keys.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Profile** — a `CompanyProfile` captures the questionnaire: data categories, purposes, cookies/analytics, third-party processors, data selling, regions served (EU → GDPR, California → CCPA/CPRA), children, retention.
2. **Conditional sections** — GDPR rights (access, erasure, portability…) appear only if you serve the EU; CCPA rights (know, delete, opt out…) only for California; "Do Not Sell or Share" language only if you sell data; COPPA language only for children's products.
3. **Compliance checklist** — the honest layer: ✅ what the policy covers, 🔧 what you must set up (consent banner, DPAs, Records of Processing), ⚠️ hard legal requirements (homepage Do-Not-Sell link, parental consent).
4. **Export** — download as Markdown, ready for your site and your lawyer.

Three sample scenarios (EU+CA SaaS that sells data, US-only bootstrap, kids app) produce three legally distinct documents from the same engine.

## Learning Connection
Built while studying Digital Product Management (Month 6: Mini SaaS Products).
Applies: regulation-driven conditional templating, GDPR/CCPA/COPPA requirement mapping, and honest-scope product design (the checklist says what the tool can't do).

## Impact Note
- **Who benefits:** Founders and indie hackers shipping their first product; PMs sanity-checking whether the current policy matches actual data practices.
- **Potential risks:** A generated policy is a starting draft, not legal advice — publishing it unreviewed could create false compliance confidence; the tool states this in the output itself.
