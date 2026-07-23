# Column Anomaly Detector

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-quality-governance/anomaly-detector/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-quality-governance/anomaly-detector/demo.ipynb)

> Bad values slip into columns and nobody notices until a dashboard breaks - scan every column and get triaged, explainable alerts before they do damage.

## Business Impact
- **Before:** A steward eyeballs spreadsheets or waits for a downstream report to look "off," then hunts for the bad row by hand.
- **After:** Every column is scanned in seconds; each finding names WHICH method flagged it and WHY, sorted by severity for triage.
- **Estimated ROI:** ~3-4 hours/week saved per analyst, plus caught-before-it-ships errors that would otherwise corrupt reports.

## What it flags
- **Numeric outliers** via three complementary methods - z-score (Gaussian), IQR / Tukey fences (distribution-free), and MAD modified z-score (robust to extreme values). A value flagged by 2+ methods is escalated to high severity.
- **Null spikes** - a column-level null rate above the tolerance bar (default 20%).
- **Rare categories** - typos, junk codes, or leaked new values below the 1% frequency bar. Identifier-like columns (emails, IDs) are auto-skipped so they don't flag everything.

Thresholds (`Z_THRESH`, `IQR_MULT`, `MAD_THRESH`, `NULL_RATE_WARN`, `RARE_CATEGORY_FRAC`) are constants at the top of `detector.py` - tune them to your own quality bar.

## Tech Stack
Python, pandas, numpy, Streamlit, matplotlib. Fully offline - no API keys, no external services.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

CLI quick look (prints the triage table for the built-in sample):
```bash
python detector.py
```

## Learning Connection
Built while studying data quality and profiling on the Data Quality & Governance arc.
Applies: robust outlier statistics (z-score vs IQR vs MAD), explainable data-quality alerting, and steward-first triage design.

## Impact Note
- **Who benefits:** data stewards, analytics engineers, and anyone who owns a table feeding a dashboard.
- **Potential risks:** thresholds are heuristics, not truth - a flagged value may be legitimate (a real large order), and an unflagged one may still be wrong. Treat findings as a prioritized review queue, not an auto-delete list. Never drop rows on the tool's word alone.
