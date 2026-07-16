<div align="center">

# 🛠️ phoebe-the-builder

### *I see problems as opportunities. I build solutions. I ship - every single day.*

**A new mini AI/data product every day, converging into one governed data platform.**

[![Builds](https://img.shields.io/badge/builds-75%20%2F%20120-2ea44f?style=for-the-badge)](https://phoebefu6.github.io/phoebe-the-builder/)
[![Streak](https://img.shields.io/badge/daily%20streak-75-ff6b35?style=for-the-badge)](TRACKER.md)
[![Notebooks](https://img.shields.io/badge/runnable%20notebooks-75-f37726?style=for-the-badge&logo=jupyter&logoColor=white)](https://phoebefu6.github.io/phoebe-the-builder/)

### 🌐 [**Explore the live One Data Platform →**](https://phoebefu6.github.io/phoebe-the-builder/)

</div>

---

Most portfolios show a handful of polished projects. This one shows **the habit**: pick a
real, painful problem a data team actually has, ship a working tool for it in ~30 minutes,
and publish it - code, a one-click runnable notebook, and the thinking behind it. Then do it
again tomorrow.

Every build below is a **[Colab / Binder](https://phoebefu6.github.io/phoebe-the-builder/)-runnable
notebook** plus a real app (Streamlit, CLI, or FastAPI). No dead demos. Click any project and
run it in your browser in under a minute.

> **The arc:** 120 builds converging into **One Data Platform** - a single governed home for a
> whole data team (DA / DE / DS / AI). The [live homepage](https://phoebefu6.github.io/phoebe-the-builder/)
> catalogs every build; the wiki walks the design decisions.

<div align="center">

**75 shipped · 6 product lines · 2 portfolios · every one runnable in the browser**

</div>

---

# 📦 Portfolio 1 - The Foundation *(60 / 60 ✅)*

*Sixty products across the full data lifecycle: infra → automation → analytics → documents → agents → product.*

## 🗄️ Data Infrastructure

| Project | The pain it kills |
|---------|-------------------|
| [CSV → PostgreSQL Loader](data-infra-toolkit/csv-loader/) | Messy manual data imports into production databases |
| [Schema Diff Tool](data-infra-toolkit/schema-diff/) | Breaking migration changes that hit prod silently |
| [API → Warehouse Connector](data-infra-toolkit/api-warehouse-connector/) | Hand-exporting from 5 SaaS tools every week |
| [Data Freshness Monitor](data-infra-toolkit/data-freshness-monitor/) | Dashboards quietly serving stale data |
| [PII Detector & Masker](data-infra-toolkit/pii-detector/) | Sensitive data leaking into staging |
| [ERD Generator](data-infra-toolkit/erd-generator/) | No database diagrams, drawn by hand |
| [dbt Model Generator](data-infra-toolkit/dbt-model-generator/) | Writing dbt models from scratch, over and over |
| [Data Lineage Visualizer](data-infra-toolkit/data-lineage-viz/) | Not knowing what breaks when you change a table |
| [GX Config Generator](data-infra-toolkit/gx-config-generator/) | Setting up data validation by hand |
| [DB Health Dashboard](data-infra-toolkit/db-health-dashboard/) | Zero visibility into database performance |

## ⚙️ Operations & Automation

| Project | The pain it kills |
|---------|-------------------|
| [CSV Cleaner](automation-suite/csv-cleaner/) | Messy data exports, standardized in one command |
| [JSON Schema Validator](automation-suite/json-validator/) | Broken API payloads caught before incidents |
| [Auto-README Generator](automation-suite/auto-readme/) | Undocumented repos |
| [Cron Job Monitor](automation-suite/cron-monitor/) | Silent batch-job failures |
| [Env Variable Checker](automation-suite/env-checker/) | Deploys that die on a missing env var |
| [Log Parser & Alerter](automation-suite/log-parser/) | Grepping logs by hand |
| [SQL Query Formatter](automation-suite/sql-formatter/) | Unreadable spaghetti SQL |
| [Markdown → PDF Report](automation-suite/md-to-pdf/) | Hand-formatting weekly reports |
| [Duplicate File Finder](automation-suite/duplicate-finder/) | 50GB of duplicate files on the shared drive |
| [Daily Standup Bot](automation-suite/standup-bot/) | Standup notes scattered across Slack |

## 📊 Analytics & Insights

| Project | The pain it kills |
|---------|-------------------|
| [Auto-EDA Dashboard](analytics-accelerator/auto-eda/) | Profiling every new dataset in seconds, not hours |
| [KPI Tracker](analytics-accelerator/kpi-tracker/) | The Monday "what are our numbers?" email |
| [A/B Test Calculator](analytics-accelerator/ab-test-calc/) | Eyeballing significance instead of computing it |
| [Survey Analyzer](analytics-accelerator/survey-analyzer/) | Survey results that take a full day to process |
| [Customer Segmentation](analytics-accelerator/customer-segments/) | Not knowing your natural customer groups |
| [Churn Predictor](analytics-accelerator/churn-predictor/) | Finding out customers left *after* they're gone |
| [Sales Forecast](analytics-accelerator/sales-forecast/) | Forecasts trapped in someone's spreadsheet |
| [Data Quality Scorecard](analytics-accelerator/data-quality-scorecard/) | Not knowing how bad your data actually is |
| [Funnel Analyzer](analytics-accelerator/funnel-analyzer/) | Can't see where users drop off |
| [Cohort Analysis](analytics-accelerator/cohort-analysis/) | Retention analysis that used to take 2 days |

## 📄 Document & Knowledge Tools

| Project | The pain it kills |
|---------|-------------------|
| [PDF Q&A Bot](document-intelligence/pdf-qa-bot/) | Nobody reads the 200-page manual |
| [Meeting Summarizer](document-intelligence/meeting-summarizer/) | Meetings with no actionable output |
| [Contract Extractor](document-intelligence/contract-extractor/) | Legal reviews that take 3 days per contract |
| [FAQ Generator](document-intelligence/faq-generator/) | Support answering the same question daily |
| [Semantic Search](document-intelligence/semantic-search/) | Ctrl+F that doesn't understand what you mean |
| [Resume Screener](document-intelligence/resume-screener/) | HR reading resumes 40 hrs/week |
| [Competitive Intel](document-intelligence/competitive-intel/) | No systematic competitor tracking |
| [Data Dictionary Generator](document-intelligence/data-dict-gen/) | Nobody knows what the columns mean |
| [Compliance Checker](document-intelligence/compliance-checker/) | Auditors finding policy violations |
| [Knowledge Base Builder](document-intelligence/knowledge-base/) | Institutional knowledge walking out the door |

## 🤖 Intelligent Agents

| Project | The pain it kills |
|---------|-------------------|
| [Email Draft Agent](ai-agent-workshop/email-agent/) | 2 hours a day lost to email |
| [Code Review Agent](ai-agent-workshop/code-review-agent/) | PRs sitting unreviewed for days |
| [Pipeline Monitor Agent](ai-agent-workshop/pipeline-monitor-agent/) | Learning of pipeline failures from angry users |
| [Ticket Router](ai-agent-workshop/ticket-router/) | Tickets hitting the wrong team 40% of the time |
| [Onboarding Agent](ai-agent-workshop/onboarding-agent/) | New hires missing steps, 3-month ramp-up |
| [Incident Response Agent](ai-agent-workshop/incident-agent/) | Runbooks that exist but nobody follows |
| [Report Generation Agent](ai-agent-workshop/report-agent/) | Monthly reports = 2 days of copy-paste |
| [Slack Q&A Agent](ai-agent-workshop/slack-qa-agent/) | The same questions in #general every day |
| [Model Drift Detector](ai-agent-workshop/model-drift-detector/) | Models degrading silently in production |
| [Agent Eval Dashboard](ai-agent-workshop/agent-eval-dashboard/) | No idea if your agents are actually good |

## 🚀 Product & Strategy Tools

| Project | The pain it kills |
|---------|-------------------|
| [Idea Validator](mini-saas-products/idea-validator/) | Building before validating |
| [Feedback Analyzer](mini-saas-products/feedback-analyzer/) | 10K reviews and no insights |
| [Feature Prioritizer](mini-saas-products/feature-prioritizer/) | Arguing priorities without data |
| [Retro Generator](mini-saas-products/retro-generator/) | Unstructured, repetitive retros |
| [OKR Tracker](mini-saas-products/okr-tracker/) | OKRs set and forgotten |
| [Roadmap Visualizer](mini-saas-products/roadmap-viz/) | Roadmaps that go stale in PowerPoint |
| [User Story Generator](mini-saas-products/user-story-gen/) | Writing good user stories is hard |
| [Accessibility Checker](mini-saas-products/accessibility-checker/) | Accessibility tested only at the end |
| [Privacy Policy Generator](mini-saas-products/privacy-policy-gen/) | Generic, costly legal templates |
| [Portfolio Dashboard](mini-saas-products/portfolio-dashboard/) | Seeing all 60 projects in one view |

---

# 🔬 Portfolio 2 - Going Pro *(15 / 60, building daily)*

*Deeper builds for the routine work of real data/ML engineering teams: CDC loaders, contracts, MLOps, LLMOps, governance.*

## 🏗️ Data Engineering Pro *(10 / 10 ✅)*

| Project | The pain it kills |
|---------|-------------------|
| [Incremental / CDC Loader](data-engineering-pro/incremental-loader/) | Full reloads that take hours every night |
| [Airflow DAG Generator](data-engineering-pro/airflow-dag-gen/) | Repetitive DAG boilerplate |
| [Parquet Partitioner](data-engineering-pro/parquet-partitioner/) | A data lake that's just a pile of CSVs |
| [Data Contract Validator](data-engineering-pro/data-contract-validator/) | Producers breaking schemas without warning |
| [Backfill Planner](data-engineering-pro/backfill-planner/) | Error-prone manual backfill scripts |
| [Streaming Window Aggregator](data-engineering-pro/streaming-aggregator/) | Can't compute rolling metrics live |
| [Dedup & Survivorship Pipeline](data-engineering-pro/dedup-pipeline/) | Duplicate records everywhere |
| [API Pagination Extractor](data-engineering-pro/api-paginator/) | Hand-coding paginated API pulls every time |
| [Column-Level Lineage Parser](data-engineering-pro/column-lineage/) | No column-level lineage |
| [Pipeline SLA Monitor](data-engineering-pro/pipeline-sla-monitor/) | Missing delivery SLAs silently |

## 🧠 ML Engineering Toolkit *(5 / 10, in progress)*

| Project | The pain it kills |
|---------|-------------------|
| [Feature Engineering Pipeline](ml-engineering-toolkit/feature-factory/) | Rewriting feature code every project |
| [Train/Eval Leaderboard](ml-engineering-toolkit/train-eval-harness/) | Ad-hoc, un-reproducible model comparison |
| [Hyperparameter Tuner](ml-engineering-toolkit/hyperparam-tuner/) | Grid-searching by hand |
| [Class Imbalance Toolkit](ml-engineering-toolkit/imbalance-toolkit/) | A fraud model that ignores the minority class |
| [Model Card Generator](ml-engineering-toolkit/model-card-gen/) | Models shipped with zero documentation |

*Up next: calibration, leakage detection, model registry → then LLMOps, data governance, analytics engineering, and a data science cookbook. [See the full 120-build plan →](PORTFOLIO-2-PLAN.md)*

---

<div align="center">

### Every build ships with

**📓 A runnable notebook** (Colab + Binder) · **🖥️ A working app** (Streamlit / CLI / FastAPI) · **🐳 A Dockerfile** · **✅ CI** · **📖 A README with business impact**

**[Browse the live platform →](https://phoebefu6.github.io/phoebe-the-builder/)** &nbsp;·&nbsp; **[Track the streak →](TRACKER.md)** &nbsp;·&nbsp; **[The full plan →](PORTFOLIO-2-PLAN.md)**

*Built in public by Phoebe Fu. One problem, one product, every day.*

</div>
