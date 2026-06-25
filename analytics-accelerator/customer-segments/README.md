# Customer Segmentation Tool

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-accelerator/customer-segments/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-accelerator/customer-segments/demo.ipynb)

> Find your natural customer groups with KMeans — auto-picks the number of segments, names each one, and shows how distinct they really are.

## Business Impact
- **Before:** Customer groups are guessed in a workshop full of sticky notes — subjective, stale the moment the data changes.
- **After:** Upload the customer table, get data-driven segments with named profiles and a quality score, re-runnable whenever data refreshes.
- **Estimated ROI:** Turns a multi-day analyst exercise into a one-click, repeatable run.

## How it works
1. **Select features** — picks the numeric behavioral columns automatically and ignores ID-like ones (so `customer_id` is never clustered as behavior).
2. **Standardize** — scales features so dollars and counts are comparable.
3. **Auto-pick k** — scores k = 2…8 by silhouette and keeps the tightest, best-separated split (or force your own k).
4. **Cluster + profile** — runs KMeans, then names each segment from how it deviates from average (*High-value loyalists*, *At-risk / lapsing*, *Mainstream / mid-tier*).

Pure scikit-learn + pandas — no API keys, runs standalone in a notebook or CI.

## Tech Stack
Python · scikit-learn (KMeans, silhouette) · pandas · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Click **Use sample customer base** in the sidebar to try it (300 customers, 3 hidden groups).

## Learning Connection
Built while studying **Streamlit** and **AWS Cloud Technical Essentials** (Month 3 of the FDE track).
Applies: unsupervised learning (KMeans), feature scaling, model selection via silhouette, and turning cluster output into stakeholder-readable profiles.

## Impact Note
- **Who benefits:** marketing, CRM, and growth teams that need to target by segment.
- **Potential risks:** KMeans assumes roughly round, similarly-sized clusters and is sensitive to feature scaling and outliers. The auto-k silhouette pick is a heuristic, not gospel — sanity-check the segment profiles, and never use segments to justify treating protected groups differently.
