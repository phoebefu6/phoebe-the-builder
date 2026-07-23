# Data Access Auditor

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-quality-governance/access-auditor/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-quality-governance/access-auditor/demo.ipynb)

> "Who can see the PII table?" takes a week of digging, so over-privileged and stale access piles up unnoticed - ingest your access grants and get triaged, explainable findings for a least-privilege review in seconds.

## Business Impact
- **Before:** Answering "who can access this restricted dataset, and should they?" means exporting grants, cross-checking roles by hand, and hoping nobody offboarded left a door open. Reviews slip, and over-privilege accretes.
- **After:** Every grant is audited in seconds against five governance rules; each finding names the user, the dataset, and WHY it fired, sorted by severity so the access review starts from evidence.
- **Estimated ROI:** turns a multi-day access review into a same-morning triage - roughly 1-2 days saved per quarterly review cycle, plus stale/orphaned grants caught before they become an audit finding or a breach path.

## What it flags
- **Over-privileged access** - a non-privileged role holding write/admin on restricted data (violates least privilege). Admin escalates to high.
- **Stale access** - a grant unused for `STALE_DAYS` (default 90); never-used grants count from their grant date. Stale on restricted data escalates.
- **Sensitive-data exposure** - more than `EXPOSURE_MAX_USERS` distinct users on one restricted dataset - a dataset-level signal to shrink the audience.
- **Orphaned grants** - roles that policy says should never touch restricted data (contractor, intern, guest, vendor), often leftovers from a role change or offboarding gap.
- **Segregation-of-duties conflicts** - one user holding a conflicting role pair (e.g. `data_engineer` + `auditor`), so a single person both performs and checks an action.

The policy bar (`STALE_DAYS`, `EXPOSURE_MAX_USERS`, `PRIVILEGED_ROLES`, `DISALLOWED_ON_RESTRICTED`, `SOD_CONFLICTS`) lives as constants at the top of `auditor.py` - tune them to your own governance policy.

## Tech Stack
Python, pandas, numpy, Streamlit, matplotlib. Fully offline - no API keys, no external services.

## Demo

**[Run the interactive demo notebook -](demo.ipynb)** - pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

CLI quick look (prints the findings table + summary for the built-in sample):
```bash
python auditor.py
```

## Learning Connection
Built while working through the data governance / least-privilege / access-review arc of the Data Quality & Governance Suite.
Applies: least-privilege as an auditable rule set, explainable governance findings, sensitivity-tiered risk (public -> restricted), and steward-first triage over black-box scoring.

## Impact Note
- **Who benefits:** data protection officers, security and governance teams, data stewards, and anyone who owns access to a sensitive table and has to defend it in an audit.
- **Potential risks:** these findings are hygiene signals for a human access review, NOT automatic revocation. A flagged grant may be legitimate (a genuinely privileged owner, a seasonal-but-valid grant), and an unflagged one can still be wrong - the rules are heuristics, not policy truth. Expect false positives. Always confirm with the data owner before removing any access; never revoke on the tool's word alone.
