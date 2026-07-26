# Scheduled Report Sender

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-engineering-bi/report-scheduler/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-engineering-bi/report-scheduler/demo.ipynb)

> Weekly reports sent by hand — this defines a report once on a cron and previews every scheduled send (a dry run that never emails).

## Business Impact
- **Before:** Someone rebuilds the same digest every Monday and pastes it into an email. Manual, forgettable, inconsistent.
- **After:** Each report is defined as data (query + cron + recipients); the body renders itself and a dry-run plan shows exactly what would go out this week.
- **Estimated ROI:** hours/week of recurring report toil removed, plus no more forgotten or malformed sends.

## Tech Stack
Python · minimal cron parser + next-run calculator · Markdown report renderer · dry-run send-plan builder · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Browse configured reports, see next fire times, preview the rendered digest, and view a dry-run send plan for the next N days.

## How it works
1. **Define** each report: metric + group-by, a 5-field cron schedule, recipients.
2. **Parse** the cron (`*`, `*/n`, ranges, lists) and scan forward for the next fire times.
3. **Render** the report body as Markdown from the source data.
4. **Preview** — a dry-run send plan lists every send in the coming window. **It never sends email.**

## Learning Connection
Built while studying **BI automation & scheduling**.
Applies: encoding a manual routine as data, computing schedules from cron, and deliberately keeping the side-effecting step (email send) explicit rather than automatic.

## Impact Note
- **Who benefits:** analysts and ops who own recurring reports.
- **Potential risks:** this build **does not send email** by design — sending on a schedule is a real side effect that should be owned by a reviewed deploy and logged for audit. Wire `render_report` to your provider deliberately, and guard recipient lists so a misconfigured report can't blast the wrong audience.
