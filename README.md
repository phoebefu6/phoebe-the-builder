# phoebe-the-builder

I see problems as opportunities. I build solutions to solve them. I ship products.

---

## Data Infrastructure

| Project | What it solves |
|---------|---------------|
| [CSV to PostgreSQL Loader](data-infra-toolkit/csv-loader/) | Automates messy data imports into production databases |
| [Schema Diff Tool](data-infra-toolkit/schema-diff/) | Catches breaking migration changes before they hit prod |
| [API to Warehouse Connector](data-infra-toolkit/api-warehouse-connector/) | Pulls data from SaaS APIs into your warehouse on a schedule |
| [Data Freshness Monitor](data-infra-toolkit/data-freshness-monitor/) | Alerts when dashboard data goes stale |
| [PII Detector & Masker](data-infra-toolkit/pii-detector/) | Finds and masks sensitive data before it leaks to staging |
| [ERD Generator](data-infra-toolkit/erd-generator/) | Generates database diagrams from SQL — no manual drawing |
| [dbt Model Generator](data-infra-toolkit/dbt-model-generator/) | Scaffolds dbt models from source schemas |
| [Data Lineage Visualizer](data-infra-toolkit/data-lineage-viz/) | Maps what breaks downstream when you change a table |
| [GX Config Generator](data-infra-toolkit/gx-config-generator/) | Auto-generates data validation rules from profiling |
| [DB Health Dashboard](data-infra-toolkit/db-health-dashboard/) | Real-time database performance visibility |

## Operations & Automation

| Project | What it solves |
|---------|---------------|
| [CSV Cleaner](automation-suite/csv-cleaner/) | Standardizes messy data exports in one command |
| [JSON Schema Validator](automation-suite/json-validator/) | Catches broken API payloads before they cause incidents |
| [Auto-README Generator](automation-suite/auto-readme/) | Generates documentation for undocumented repos |
| [Cron Job Monitor](automation-suite/cron-monitor/) | Detects silent batch job failures before users notice |
| [Env Variable Checker](automation-suite/env-checker/) | Prevents deployments from failing due to missing config |
| [Log Parser & Alerter](automation-suite/log-parser/) | Surfaces errors from logs without manual grep |
| [SQL Query Formatter](automation-suite/sql-formatter/) | Turns spaghetti SQL into readable, consistent queries |
| [Markdown to PDF Report](automation-suite/md-to-pdf/) | Converts markdown into polished PDF reports |
| [Duplicate File Finder](automation-suite/duplicate-finder/) | Identifies wasted storage from duplicate files |
| [Daily Standup Bot](automation-suite/standup-bot/) | Collects and organizes standup updates across teams |

## Analytics & Insights

| Project | What it solves |
|---------|---------------|
| [Auto-EDA Dashboard](analytics-accelerator/auto-eda/) | Profiles any dataset in seconds, not hours |
| [KPI Tracker](analytics-accelerator/kpi-tracker/) | Replaces the Monday morning "what are our numbers?" email |
| [A/B Test Calculator](analytics-accelerator/ab-test-calc/) | Statistical significance without the guesswork |
| [Survey Analyzer](analytics-accelerator/survey-analyzer/) | Processes survey results in minutes, not days |
| [Customer Segmentation](analytics-accelerator/customer-segments/) | Discovers natural customer groups from behavioral data |
| [Churn Predictor](analytics-accelerator/churn-predictor/) | Flags at-risk customers before they leave |
| [Sales Forecast](analytics-accelerator/sales-forecast/) | Replaces spreadsheet forecasts with time-series models |
| [Data Quality Scorecard](analytics-accelerator/data-quality-scorecard/) | Quantifies how bad (or good) your data actually is |
| [Funnel Analyzer](analytics-accelerator/funnel-analyzer/) | Shows exactly where users drop off |
| [Cohort Analysis](analytics-accelerator/cohort-analysis/) | Retention analysis that used to take 2 days |

## Document & Knowledge Tools

| Project | What it solves |
|---------|---------------|
| [PDF Q&A Bot](document-intelligence/pdf-qa-bot/) | Ask questions about long documents in plain language |
| [Meeting Summarizer](document-intelligence/meeting-summarizer/) | Turns meetings into actionable summaries and next steps |
| [Contract Extractor](document-intelligence/contract-extractor/) | Pulls key clauses from contracts in seconds |
| [FAQ Generator](document-intelligence/faq-generator/) | Builds FAQ pages from existing documentation |
| [Semantic Search](document-intelligence/semantic-search/) | Finds what you mean, not just what you typed |
| [Resume Screener](document-intelligence/resume-screener/) | Scores resumes against job requirements at scale |
| [Competitive Intel](document-intelligence/competitive-intel/) | Summarizes competitor activity from public sources |
| [Data Dictionary Generator](document-intelligence/data-dict-gen/) | Documents database columns automatically |
| [Compliance Checker](document-intelligence/compliance-checker/) | Validates documents against policy requirements |
| [Knowledge Base Builder](document-intelligence/knowledge-base/) | Captures institutional knowledge before it walks out the door |

## Intelligent Agents

| Project | What it solves |
|---------|---------------|
| [Email Draft Agent](ai-agent-workshop/email-agent/) | Drafts contextual email responses |
| [Code Review Agent](ai-agent-workshop/code-review-agent/) | Reviews PRs so they don't sit for days |
| [Pipeline Monitor Agent](ai-agent-workshop/pipeline-monitor-agent/) | Watches data pipelines and alerts on failures |
| [Ticket Router](ai-agent-workshop/ticket-router/) | Routes support tickets to the right team automatically |
| [Onboarding Agent](ai-agent-workshop/onboarding-agent/) | Guides new hires through setup and ramp-up |
| [Incident Response Agent](ai-agent-workshop/incident-agent/) | Executes runbooks when incidents fire |
| [Report Generation Agent](ai-agent-workshop/report-agent/) | Assembles recurring reports from multiple data sources |
| [Slack Q&A Agent](ai-agent-workshop/slack-qa-agent/) | Answers repeated questions from team knowledge |
| [Model Drift Detector](ai-agent-workshop/model-drift-detector/) | Catches silent model degradation in production |
| [Agent Eval Dashboard](ai-agent-workshop/agent-eval-dashboard/) | Measures whether agents are actually performing |

## Product & Strategy Tools

| Project | What it solves |
|---------|---------------|
| [Idea Validator](mini-saas-products/idea-validator/) | Tests business ideas against market signals before building |
| [Feedback Analyzer](mini-saas-products/feedback-analyzer/) | Extracts themes from thousands of user reviews |
| [Feature Prioritizer](mini-saas-products/feature-prioritizer/) | Ranks features by impact and effort with data, not opinions |
| [Retro Generator](mini-saas-products/retro-generator/) | Structures sprint retrospectives for actionable outcomes |
| [OKR Tracker](mini-saas-products/okr-tracker/) | Keeps objectives visible and progress honest |
| [Roadmap Visualizer](mini-saas-products/roadmap-viz/) | Interactive roadmaps that stay current |
| [User Story Generator](mini-saas-products/user-story-gen/) | Writes well-structured user stories with acceptance criteria |
| [Accessibility Checker](mini-saas-products/accessibility-checker/) | Catches accessibility issues early in development |
| [Privacy Policy Generator](mini-saas-products/privacy-policy-gen/) | Generates compliant privacy policies from app descriptions |
| [Portfolio Dashboard](mini-saas-products/portfolio-dashboard/) | Live overview of everything in this repo |
