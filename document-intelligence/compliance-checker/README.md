# Compliance Checker

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/document-intelligence/compliance-checker/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=document-intelligence/compliance-checker/demo.ipynb)

> Run a document against a policy ruleset before the auditor does.

## Business Impact
- **Before:** Auditors keep finding the same policy violations because nobody checks documents before they ship.
- **After:** Policy is encoded as code (require/forbid rules with severity); any document is scored against it at authoring time, violations flagged with evidence.
- **Estimated ROI:** fewer audit findings, faster reviews, a versionable single source of truth for "the policy."

## Tech Stack
Python · dataclass rules engine (require/forbid + severity) · weighted compliance score · Claude API (optional semantic judgment) · Streamlit · pandas · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Pick the good/bad sample or paste your own; get a compliance score and per-rule findings with evidence. Set an `ANTHROPIC_API_KEY` for meaning-based checking.

## How it works
1. **Model** policy as a ruleset — each rule has a pattern, a mode (`require` = must be present, `forbid` = must be absent), a severity, and a category.
2. **Run** every rule over the document (deterministic regex).
3. **Score** 0-100, penalizing violations weighted by severity (critical ≫ low).
4. **Claude mode** judges each rule by meaning — catching negations ("we *never* store plaintext passwords") and paraphrased compliance the regex can't, using the same human-authored ruleset.

## Learning Connection
Built while studying **rules-engine design + LLM judgment** (Anthropic Prompt Engineering).
Applies: policy-as-code, deterministic-first with LLM escalation for gray areas, and severity-weighted scoring.

## Impact Note
- **Who benefits:** compliance, legal, security, and data-governance teams; anyone shipping policies/SOPs/contracts.
- **Potential risks:** regex mode is literal — it can false-positive on negations and miss paraphrased requirements, so a passing score is **not** a legal guarantee. Treat it as a pre-audit smoke test; a qualified human owns the final compliance sign-off.
