# Accessibility Checker

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/mini-saas-products/accessibility-checker/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=mini-saas-products/accessibility-checker/demo.ipynb)

> We don't test for accessibility until the end — paste HTML and get WCAG-referenced findings, severity levels, and a 0-100 score in seconds.

## Business Impact
- **Before:** Accessibility is a launch-week audit; issues found late cost 10x to fix, or ship anyway.
- **After:** Every page (or PR) gets an instant score with specific WCAG-cited fixes — accessibility shifts left into the build loop.
- **Estimated ROI:** Catches unlabeled inputs, missing alt text, and keyboard traps before review — hours of audit remediation avoided per release.

## Tech Stack
Python 3.10+ (stdlib `html.parser` — zero parsing dependencies), Streamlit, pandas, matplotlib. 13 automated checks mapped to WCAG success criteria. Runs fully offline.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **One parser pass** — a stdlib `HTMLParser` subclass collects images, links, buttons, form controls, headings, label targets, tabindex, and inline event handlers.
2. **13 rules fire** — each finding carries a rule id, severity (critical/serious/moderate/minor), the offending element, a plain-English message, and the WCAG criterion (e.g. missing alt → 1.1.1, unlabeled input → 3.3.2, positive tabindex → 2.4.3, `onclick` on a `<div>` → 2.1.1).
3. **Score** — 100 minus severity-weighted penalties (critical 10, serious 6, moderate 3, minor 1), plus a letter grade.
4. **Report** — findings grouped by severity as Markdown, ready for a PR comment or ticket.

The sample page with 12 planted issues scores 24/100 (F); the fixed version scores 100/100 (A).

## Learning Connection
Built while studying Design Thinking and Digital Product Management (Month 6: Mini SaaS Products).
Applies: WCAG success criteria, severity-weighted quality scoring, and single-pass HTML analysis with the stdlib.

## Impact Note
- **Who benefits:** Frontend developers, PMs, and QA teams who want accessibility feedback during the build, not after it.
- **Potential risks:** Automated checks catch only ~30-40% of WCAG issues — a passing score must not be treated as full compliance; keyboard and screen-reader testing remain essential.
