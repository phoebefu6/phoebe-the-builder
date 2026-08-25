<!-- phoebe header -->

[![Open the live site](https://img.shields.io/badge/%E2%96%B6%20open%20the%20live%20site-1f6feb?style=for-the-badge)](https://phoebefu6.github.io/phoebe-the-builder/)
[![Star this repo](https://img.shields.io/github/stars/phoebefu6/phoebe-the-builder?style=for-the-badge&label=star%20this%20repo&color=444444)](https://github.com/phoebefu6/phoebe-the-builder/stargazers)

### ▶︎ [Open the live site →](https://phoebefu6.github.io/phoebe-the-builder/)

Free and open. Every build links to its source.

<!-- /phoebe header -->

# phoebe-the-builder

Small, exact tools for the parts of data work that quietly go wrong.

Each one started with something breaking in a real pipeline: a metric that disagreed with itself, a parser that read the same bytes two ways, a model that looked fine until it shipped. Every tool ships with source, a runnable Colab/Binder notebook, a working app, a Dockerfile and CI.

**[Browse them by what you are trying to do →](https://phoebefu6.github.io/phoebe-the-builder/)**


---

## Move data in

*It lives somewhere else and it has to land here, on a schedule, without losing rows.*

| Tool | The problem it was built for |
|------|------------------------------|
| [API Pagination Extractor](data-engineering-pro/api-paginator/) | Paginated API pulls are hand-coded each time - one generic extractor speaks offset, page, cursor, and link-follow, with 429 retry/backoff built in. |
| [API to Warehouse Connector](data-infra-toolkit/api-warehouse-connector/) | Every Monday, someone on the team downloads CSVs from five different SaaS tools. This replaces that. |
| [Backfill Planner](data-engineering-pro/backfill-planner/) | Backfills are error-prone manual scripts - chunk the date range, persist every status, retry the flaky, quarantine the broken, and resume after a crash… |
| [CSV to PostgreSQL Loader](data-infra-toolkit/csv-loader/) | Loading data into our database is a manual nightmare - copy-paste into SQL, fight with column types, pray nothing breaks. |
| [CSV/JSON to Parquet Partitioner](data-engineering-pro/parquet-partitioner/) | Our data lake is a pile of CSVs - convert flat files into hive-partitioned, compressed parquet and prove the query speedup with a stopwatch. |
| [Incremental / CDC Loader](data-engineering-pro/incremental-loader/) | Full reloads take hours every night - load only what changed with watermark extraction, key upsert, and soft-delete tombstones. |
| [Streaming Window Aggregator](data-engineering-pro/streaming-aggregator/) | We can't compute rolling metrics live - event-time windows with watermarks in ~100 lines, with the same semantics Flink and Spark charge a cluster for. |


---

## Make raw values usable

*The bytes arrived. What they MEAN is a decision, and every layer decides differently.*

| Tool | The problem it was built for |
|------|------------------------------|
| [A Field Name Is an Identity, Not a String](automation-suite/header-casing/) | headers["X-Request-Id"] is the wrong shape, and not for style reasons. |
| [A File Has No Lines In It, a Splitter Makes Them](data-engineering-pro/line-ending-detector/) | A file has no lines in it. It has bytes. Lines are produced by a splitter, and every runtime ships a different one. |
| [A Numeric String Does Not Contain a Number, a Reader Assigns One](data-engineering-pro/number-parser-locale/) | A numeric string does not contain a number. It contains characters. |
| [A Percentage Column Is an Apportionment](analytics-engineering-bi/percent-recomputer/) | Round three equal rows to one decimal place and the column reads 99.9%. Every row is correct. |
| [A String Does Not Contain a Boolean, a Reader Assigns One](data-engineering-pro/boolean-parser/) | A string does not contain a boolean. |
| [Allocate a Total Instead of Rounding Its Rows](data-engineering-pro/currency-rounder/) | A rounding function returns a number. It cannot return the fact that the rows no longer add up. |
| [Byte-Accurate Fixed-Width Parsing with a Pre-Flight Audit](data-engineering-pro/fixed-width-parser/) | Reading a fixed-width file is one line: pd.read_fwf(path, colspecs=...). |
| [CSV Cleaner](automation-suite/csv-cleaner/) | Turn a messy CSV export into a tidy, analysis-ready file - with an auditable report of every change. |
| [Dedup & Survivorship Pipeline](data-engineering-pro/dedup-pipeline/) | Duplicate records everywhere - normalize the variants, cluster by match keys, and merge each cluster into a golden record with field-level survivorship… |
| [Duplicate File Finder](automation-suite/duplicate-finder/) | Find byte-identical files hiding under different names - efficiently - and reclaim the space. |
| [Enumerate Every Viable Dialect Instead of Returning One](data-engineering-pro/csv-dialect-sniffer/) | csv.Sniffer().sniff() returns a Dialect or raises csv.Error. It has no third answer. |
| [JSON Flattener](data-engineering-pro/json-flattener/) | Every API response is nested. Every warehouse table is flat. |
| [Local-to-UTC Resolution with the Undecidable Cases Named](data-engineering-pro/timezone-normalizer/) | Normalising a local timestamp to UTC is one line: ts.replace(tzinfo=ZoneInfo(zone)).astimezone(utc). |
| [Lossless SQL Type Inference from Raw Text](data-engineering-pro/type-inferencer/) | Everything imported as string. |
| [ORDER BY Name Is a Collation, Not an Order](data-engineering-pro/sort-order-drift/) | ORDER BY name reads like a total order on a column. It is not. |
| [Render a GFM Table and Report What It Cannot Carry](automation-suite/markdown-tabler/) | Turning a DataFrame into a markdown table is one line. |
| [Report Every Reading a Conforming Parser Could Return](automation-suite/duration-parser/) | parse(text) -> timedelta cannot answer the question people actually have about a duration string, which is not "how long is this" but "does the parser… |
| [Report the Collisions a Sanitiser Creates](automation-suite/filename-sanitiser/) | sanitise(name) -> str is the wrong shape, and not because of which characters are on the deny-list. |
| [Report the Collisions a Slugifier Creates](automation-suite/slug-collider/) | slugify("Node.js at scale") and slugify("NodeJS at scale") both return nodejs-at-scale. Neither call is wrong. |
| [Truncate to N Does Not Name an Operation](data-engineering-pro/unicode-width-truncator/) | "Truncate to 20" does not name an operation. |


---

## Prove it is right

*Something changed upstream. Find out before a dashboard does.*

| Tool | The problem it was built for |
|------|------------------------------|
| [Column Anomaly Detector](data-quality-governance/anomaly-detector/) | Bad values slip into columns and nobody notices until a dashboard breaks - scan every column and get triaged, explainable alerts before they do damage. |
| [Data Contract Validator](data-engineering-pro/data-contract-validator/) | Producers break schemas without warning - put a YAML contract in CI so breaking changes block the merge and violating data never lands. |
| [Data Quality Scorecard](analytics-accelerator/data-quality-scorecard/) | Stop guessing how bad your data is - get a 0-100 score, a letter grade, and a ranked list of what to fix first. |
| [Dataset Snapshot Diff](data-quality-governance/data-diff/) | Between yesterday's snapshot and today's, what actually changed? |
| [dbt Test Generator](analytics-engineering-bi/dbt-test-gen/) | No tests on our models - this profiles a table and auto-generates a paste-ready dbt schema.yml. |
| [DQ Rules Engine](data-quality-governance/dq-rules-engine/) | Our data-quality rules live in a wiki nobody enforces - by the time a bad batch is caught, it's already in a report. |
| [Great Expectations Config Generator](data-infra-toolkit/gx-config-generator/) | Setting up data validation is tedious - so most teams skip it until something breaks in prod. |
| [JSON Schema Validator](automation-suite/json-validator/) | A FastAPI microservice that validates JSON payloads against a schema and returns every error at once - so broken API contracts surface before they… |
| [Null Heatmap & Missingness Report](data-quality-governance/null-heatmap/) | df.isna().sum() answers the wrong question. |
| [Schema Diff Tool](data-infra-toolkit/schema-diff/) | Database migrations break things silently - catch schema changes before they hit production. |
| [Schema Registry](data-quality-governance/schema-registry/) | A producer changes a column type and three downstream jobs break silently - we find out from the dashboard, not before. |
| [Source-to-Target Reconciliation](data-quality-governance/reconciliation-checker/) | We copied a table from the source system into the warehouse and can't prove it matched - row counts drift and values silently differ. |


---

## Control who sees what

*Who owns this column, who may read it, and can you show an auditor?*

| Tool | The problem it was built for |
|------|------------------------------|
| [Business Glossary Manager](data-quality-governance/business-glossary/) | "Active user" means five different things to five teams, so every metric argument starts from zero - this gives each business term one owned, versioned… |
| [Column-Level Lineage Parser](data-engineering-pro/column-lineage/) | Table-level lineage tells you model B reads model A. |
| [Compliance Checker](document-intelligence/compliance-checker/) | Run a document against a policy ruleset before the auditor does. |
| [Consent & Purpose Tracker](data-quality-governance/consent-tracker/) | We process personal data for purposes the user never agreed to - and can't prove otherwise when asked. |
| [Data Access Auditor](data-quality-governance/access-auditor/) | "Who can see the PII table?" takes a week of digging, so over-privileged and stale access piles up unnoticed - ingest your access grants and get… |
| [Data Dictionary Generator](document-intelligence/data-dict-gen/) | Point it at a table, get a documented data dictionary - types, PII flags, descriptions. |
| [Data Lineage Visualizer](data-infra-toolkit/data-lineage-viz/) | Paste SQL, see what breaks when you change a table. |
| [Data Retention Enforcer](data-quality-governance/retention-enforcer/) | We keep everything forever - storage bloats and we are one audit away from a finding for holding personal data past its purpose. |
| [ERD Generator from SQL](data-infra-toolkit/erd-generator/) | "We don't have database diagrams" - paste your CREATE TABLE statements and get an instant ER diagram. |
| [Lightweight Data Catalog](data-quality-governance/data-catalog/) | Nobody knows what tables and columns exist, what they mean, or who owns them - so every analysis starts by pinging three people on Slack. |
| [Model Card Generator](ml-engineering-toolkit/model-card-gen/) | "No model documentation." - a model with no card is a black box: nobody knows what it's for, what it trained on, or where it breaks. |
| [PII Detector and Masker](data-infra-toolkit/pii-detector/) | "We accidentally share sensitive data in staging" - scan CSV data for PII and mask it before it leaves production. |
| [Policy-Driven PII Redactor](data-quality-governance/pii-redactor/) | "We masked the PII, so the extract is safe to share." Two things are usually wrong with that. |
| [Privacy Policy Generator](mini-saas-products/privacy-policy-gen/) | Legal templates cost money and are still generic - answer a short questionnaire and get a GDPR/CCPA-aware policy draft plus a checklist of the… |
| [Subject Access Request Extractor](data-quality-governance/dsar-extractor/) | A subject access request arrives with one email address. Finding the rows is easy. |


---

## Know when it breaks

*The pipeline fails at 3am. The question is whether anyone finds out before the meeting.*

| Tool | The problem it was built for |
|------|------------------------------|
| [Cron Job Monitor](automation-suite/cron-monitor/) | A dead-man's-switch for scheduled jobs - heartbeat in, email out when a job goes silent. |
| [Data Freshness Monitor](data-infra-toolkit/data-freshness-monitor/) | Dashboards show stale data and nobody notices - this tool catches it first. |
| [Data Pipeline Monitor Agent](ai-agent-workshop/pipeline-monitor-agent/) | "We discover pipeline failures from angry users" - an agent that catches failures, slow runs, row-count drops, and silent staleness before anyone… |
| [Database Health Dashboard](data-infra-toolkit/db-health-dashboard/) | One screen of database visibility - cache hits, latency, locks, replication lag - scored red/amber/green so problems surface before users notice. |
| [Environment Variable Checker](automation-suite/env-checker/) | A pre-flight check for environment variables - declare what your app needs, validate it before you ship, and fail the deploy if anything's missing or… |
| [Incident Response Agent](ai-agent-workshop/incident-agent/) | Incident playbooks exist but nobody follows them - this agent executes the runbook step-by-step and escalates when a remediation step fails. |
| [Log Parser and Alerter](automation-suite/log-parser/) | Stop grepping logs by hand. |
| [Metric Anomaly Alerter](analytics-engineering-bi/metric-alerting/) | We notice metric drops late - this watches a metric and fires the day it moves, with three independent detectors. |
| [Model Drift Detector](ai-agent-workshop/model-drift-detector/) | Models degrade silently in production - this watches a live window against the training reference via PSI and alerts before accuracy quietly rots. |
| [Pipeline SLA Monitor](data-engineering-pro/pipeline-sla-monitor/) | A pipeline that lands late, runs slow, goes stale, or arrives thin usually breaks a consumer before anyone on the data team notices. |


---

## Find out what is in it

*A new dataset landed. Two hours of profiling before you can say anything about it.*

| Tool | The problem it was built for |
|------|------------------------------|
| [Auto-EDA Dashboard](analytics-accelerator/auto-eda/) | Drop in a CSV, get the whole first-look in seconds - shape, missingness, dtypes, distributions, correlations, and the quality problems worth a human's… |
| [Correlation & Multicollinearity Explorer](data-science-cookbook/correlation-explorer/) | "Which features relate?" - a correlation heatmap plus VIF to catch the redundant, collinear features that quietly wreck a model. |
| [Customer Segmentation Tool](analytics-accelerator/customer-segments/) | Find your natural customer groups with KMeans - auto-picks the number of segments, names each one, and shows how distinct they really are. |
| [Outlier Explainer](data-science-cookbook/outlier-explainer/) | "Which rows are weird, and why?" - Isolation Forest finds the anomalies; per-feature z-scores explain each one. |
| [PCA/UMAP Explorer](data-science-cookbook/dim-reducer/) | "Too many features to see" - project high-dimensional data down to 2D and actually look at its structure. |
| [Survey Results Analyzer](analytics-accelerator/survey-analyzer/) | Turn a raw survey export into a type-aware summary in seconds - NPS, Likert scores, choice breakdowns, and open-text sentiment. |
| [Topic Modeling Tool](data-science-cookbook/topic-modeler/) | "What are these documents about?" - discover the themes in a text pile automatically, no labels required. |
| [User Feedback Analyzer](mini-saas-products/feedback-analyzer/) | We have 10K reviews and no insights - this turns raw feedback into a sentiment split, ranked complaints, ranked praises, and the one thing to fix first. |


---

## Turn it into a number people act on

*One metric, three dashboards, three answers. Define it once and serve it.*

| Tool | The problem it was built for |
|------|------------------------------|
| [Cohort Analysis Tool](analytics-accelerator/cohort-analysis/) | "Retention analysis takes our analyst 2 days" - turn a raw event log into a cohort retention heatmap in seconds. |
| [Dashboard Spec Generator](analytics-engineering-bi/dashboard-spec/) | Dashboards built from vague asks - this reads the data's shape and recommends the right charts, with the reasoning. |
| [Full Portfolio Dashboard](mini-saas-products/portfolio-dashboard/) | Showcase all 60 projects in one view - the portfolio's final build is the view of the portfolio itself. |
| [Funnel Analyzer](analytics-accelerator/funnel-analyzer/) | See exactly where users drop off - biggest leak called out, step-by-step conversion, and a segment comparison. |
| [Inline SVG Trend Marks That Report Their Own Scale](analytics-engineering-bi/sparkline-gen/) | Adding a sparkline column to a table is one line of pandas. |
| [KPI Tracker](analytics-accelerator/kpi-tracker/) | Execs ask for the same metrics every Monday - this turns a metric time series into the answer automatically. |
| [KPI Tree / Driver Decomposition](analytics-engineering-bi/kpi-tree/) | "Why did revenue move?" - this splits a KPI's change across its drivers exactly, with no residual. |
| [Metric Catalog & Ownership](analytics-engineering-bi/metric-catalog/) | Which metrics exist, and who owns them? |
| [Metrics Layer / Semantic Definitions](analytics-engineering-bi/metrics-layer/) | Every dashboard computes revenue differently - a metrics layer makes one governed definition the single source of truth. |
| [Natural Language to SQL](analytics-engineering-bi/nl-to-sql/) | Non-analysts can't query the warehouse - NL→SQL lets them ask in English, behind guardrails that keep it read-only and in-schema. |
| [Period-over-Period Metric Diff](analytics-engineering-bi/metric-diff/) | "Conversion is up 5% this week!" - but is it a real shift or just noise? |
| [Pivot Narrator](analytics-engineering-bi/pivot-narrator/) | Nobody reads the pivot table. |
| [Sales Forecast Dashboard](analytics-accelerator/sales-forecast/) | Get sales forecasts out of someone's spreadsheet - a reproducible forecast with a confidence band and an honest accuracy check. |
| [Scheduled Report Sender](analytics-engineering-bi/report-scheduler/) | Weekly reports sent by hand - this defines a report once on a cron and previews every scheduled send (a dry run that never emails). |
| [Self-Serve Data Explorer](analytics-engineering-bi/self-serve-explorer/) | Analysts get pinged for every number - a self-serve explorer lets anyone pivot, aggregate, and filter without SQL or a ticket. |
| [SQL Optimizer Advisor](analytics-engineering-bi/query-optimizer/) | Slow queries, no idea why? This lints SQL for the anti-patterns that quietly force full table scans. |


---

## Decide what is actually true

*The line went up. Whether that means anything is a separate question.*

| Tool | The problem it was built for |
|------|------------------------------|
| [A/B Test Calculator](analytics-accelerator/ab-test-calc/) | Stop eyeballing significance. |
| [AIC Ranking Plus an Absolute Goodness-of-Fit Test](data-science-cookbook/distribution-fitter/) | Fitting ten distributions and ranking them by AIC is four lines of scipy. |
| [Crosstab & Chi-Square Tool](data-science-cookbook/crosstab-chi2/) | "Compare groups in survey data" - contingency table, chi-square test, effect size, and the exact cells driving the difference. |
| [Sample Size & Power Calculator](data-science-cookbook/sample-size-calc/) | "How many users do we need?" gets asked at the start of every experiment and answered by vibes. |
| [Statistical Test Advisor](data-science-cookbook/stat-test-advisor/) | "Which statistical test do I use?" - describe the data, get the right test (parametric or not), and run it. |


---

## Learn from it

*Fit something, then find out honestly whether it is better than doing nothing.*

| Tool | The problem it was built for |
|------|------------------------------|
| [Baseline Model Ladder](ml-engineering-toolkit/baseline-model/) | "We got 0.87 AUC." Compared to what? |
| [Batch Scoring Service](ml-engineering-toolkit/batch-scorer/) | Scoring new data shouldn't be a manual copy-paste ritual. |
| [Churn Predictor](analytics-accelerator/churn-predictor/) | Spot at-risk customers before they leave - ranked risk scores, churn drivers, and an honest model-quality check. |
| [Class Imbalance Toolkit](ml-engineering-toolkit/imbalance-toolkit/) | "Our fraud model ignores the minority class." - a classifier that predicts "not fraud" every time scores 98% accuracy and catches zero fraud. |
| [Classification Threshold Explorer](ml-engineering-toolkit/threshold-explorer/) | A model outputs a probability. A decision needs a cutoff. |
| [Data Leakage Detector](ml-engineering-toolkit/leakage-detector/) | The model scores 0.99 in cross-validation and falls apart in production. That gap is usually leakage. |
| [Feature Engineering Pipeline](ml-engineering-toolkit/feature-factory/) | Every new ML project you rewrite the same impute / scale / one-hot boilerplate, get the column lists slightly wrong, and leak test statistics into… |
| [Feature Importance Explainer](ml-engineering-toolkit/feature-importance/) | "Stakeholders don't trust the model." One importance number is easy to doubt - three that agree is evidence. |
| [General Time Series Forecaster](data-science-cookbook/ts-forecaster/) | "Forecast any metric" - Holt-Winters (trend + seasonality) in pure numpy, with honest backtest accuracy. |
| [Hyperparameter Tuner](ml-engineering-toolkit/hyperparam-tuner/) | We grid-search by hand - nudge n_estimators, re-run a cell, keep whatever looked better. |
| [Mini Model Registry](ml-engineering-toolkit/model-registry/) | "Which model is actually in production?" should have a one-word answer. |
| [Monotone WOE Binning with a Defensible IV](ml-engineering-toolkit/feature-binner/) | Binning a feature and printing its Information Value takes twenty lines. |
| [Probability Calibration Checker](ml-engineering-toolkit/calibration-checker/) | A "0.9" from your model should mean it happens 90% of the time. Often it doesn't. |
| [Recommendation Engine](data-science-cookbook/recommender/) | "We have no 'you may also like'" - item-item collaborative filtering, with an explanation for every recommendation. |
| [Text Classification Trainer](data-science-cookbook/text-classifier/) | "Auto-tag our support tickets" - train a TF-IDF + logistic-regression classifier and route new text by category. |
| [Train/Eval Leaderboard](ml-engineering-toolkit/train-eval-harness/) | Model selection is ad-hoc - train a few classifiers, eyeball one accuracy number, ship whichever looked best on one lucky split. |


---

## Get answers out of documents

*The answer is in a 200-page PDF nobody will read.*

| Tool | The problem it was built for |
|------|------------------------------|
| [Chunking Strategy Tester](llmops-genai-platform/chunk-optimizer/) | Bad chunks = bad answers. Prove which chunking strategy retrieves best - on evidence, not vibes. |
| [Competitive Intel Summarizer](document-intelligence/competitive-intel/) | Turn scattered competitor notes into one comparison matrix + strategic takeaways. |
| [Contract Clause Extractor](document-intelligence/contract-extractor/) | Turn a 3-day legal read into a 30-second clause + risk triage. |
| [FAQ Generator from Docs](document-intelligence/faq-generator/) | Stop answering the same question daily - turn your docs into a ready FAQ. |
| [JSON Schema Inference from LLM Outputs](llmops-genai-platform/schema-from-samples/) | Your LLM pipeline returns JSON. Nothing checks it. json.loads is not a contract. |
| [Knowledge Base Builder](document-intelligence/knowledge-base/) | Turn scattered docs into a searchable, cited knowledge base - before the expert leaves. |
| [Meeting Notes Summarizer](document-intelligence/meeting-summarizer/) | "Meetings produce no actionable output" - paste the transcript, get TL;DR + decisions + action items + open questions. |
| [Named Entity Extractor](data-science-cookbook/ner-extractor/) | "Pull the names, orgs, money, and dates out of this text" - structured entities, no heavy model download. |
| [Near-Duplicate Finder for a RAG Corpus](llmops-genai-platform/embedding-dedup/) | A duplicate in a vector index does not cost you storage. It costs you the answer. |
| [PDF Q&A Bot](document-intelligence/pdf-qa-bot/) | "Nobody reads the 200-page policy manual" - upload it, ask questions, get cited answers. |
| [Resume Screener](document-intelligence/resume-screener/) | Rank a stack of resumes by skills fit - a triage aid, not a hiring decision. |
| [Semantic Search Engine](document-intelligence/semantic-search/) | Search by meaning, not just exact words - what Ctrl+F can't do. |
| [Structured Output Enforcer](llmops-genai-platform/structured-extractor/) | Stop your pipeline from crashing when the LLM returns almost-JSON. |


---

## Check the AI is any good

*It sounds right. Sounding right is not a measurement.*

| Tool | The problem it was built for |
|------|------------------------------|
| [Agent Evaluation Dashboard](ai-agent-workshop/agent-eval-dashboard/) | "We don't know if our agents are actually good." Run a fixed eval suite against two agent versions and get pass rate, quality, latency, and per-category… |
| [Few-Shot Example Selector](llmops-genai-platform/fewshot-selector/) | Stop shipping the same static few-shot examples for every input - pick the ones that actually resemble the query. |
| [Hallucination Detector](llmops-genai-platform/hallucination-checker/) | Flag the claims your RAG answer makes that the retrieved context never supported. |
| [LLM Cost & Token Tracker](llmops-genai-platform/llm-cost-tracker/) | No more surprise API bills - log every call, roll spend up, watch the budget. |
| [LLM Guardrail Filter](llmops-genai-platform/llm-guardrails/) | Model outputs unsafe / off-topic text? Put cheap deterministic checks around it. |
| [LLM Model Router](llmops-genai-platform/llm-router/) | Stop paying frontier-model prices for "what's the capital of France?" |
| [Prompt Linter](llmops-genai-platform/prompt-linter/) | Prompts are the only part of an LLM system that ships with no review gate. |
| [Prompt Registry & Versioning](llmops-genai-platform/prompt-registry/) | Prompts get pasted into random .py files and Slack threads - no history, no diff, no rollback. |
| [RAG Evaluation Harness](llmops-genai-platform/rag-eval/) | You changed the chunking or swapped the embedding model. Did RAG get better or worse? |
| [Semantic Response Cache](llmops-genai-platform/semantic-cache/) | Stop paying for repeated similar queries - reuse an answer when the meaning matches. |
| [Token & Cost Estimator](llmops-genai-platform/token-cost-estimator/) | Price an LLM feature before you build it - not after the invoice lands. |


---

## Make it run itself

*The task is small, correct, and done by hand every single week.*

| Tool | The problem it was built for |
|------|------------------------------|
| [Accessibility Checker](mini-saas-products/accessibility-checker/) | We don't test for accessibility until the end - paste HTML and get WCAG-referenced findings, severity levels, and a 0-100 score in seconds. |
| [Airflow DAG Generator](data-engineering-pro/airflow-dag-gen/) | Writing DAG boilerplate is repetitive - describe the pipeline in ~30 lines of YAML and get a validated, ready-to-deploy Airflow DAG. |
| [Auto-README Generator](automation-suite/auto-readme/) | Scan any repo into a structured profile and generate a README - by template offline, or with Claude for richer prose. |
| [Code Review Agent](ai-agent-workshop/code-review-agent/) | "PRs sit unreviewed for days" - an agent that gives every PR an instant first pass so reviewers focus on logic, not nitpicks. |
| [Customer Ticket Router](ai-agent-workshop/ticket-router/) | "Tickets go to the wrong team 40% of the time" - an agent that routes tickets by weighted keyword signal, and escalates to human triage instead of… |
| [Daily Standup Bot](automation-suite/standup-bot/) | One place for the team's standup. |
| [dbt Model Generator](data-infra-toolkit/dbt-model-generator/) | Writing dbt staging models from scratch is repetitive boilerplate - paste your SQL DDL and get production-ready dbt artifacts instantly. |
| [Email Draft Agent](ai-agent-workshop/email-agent/) | "I spend 2 hours daily on email" - an agent that drafts replies, checks its own work against the original ask, and revises before handing it back. |
| [Markdown to PDF Report](automation-suite/md-to-pdf/) | Write the weekly report in Markdown, get a branded, print-ready PDF - cover header, styled tables, page numbers. |
| [Onboarding Checklist Agent](ai-agent-workshop/onboarding-agent/) | "New hires miss steps, ramp-up takes 3 months" - a multi-step agent that resolves onboarding step dependencies and flags what's overdue or blocked… |
| [Read a Cron Line the Way Cron Reads It](automation-suite/cron-explainer/) | 0 0 13 5 is the line somebody writes for Friday the 13th. |
| [Report Generation Agent](ai-agent-workshop/report-agent/) | Monthly reports take 2 days of copy-paste - a team of role-specialized agents turns a table of metrics into a ready-to-send report in seconds. |
| [Report the Arrival Process a Backoff Creates](automation-suite/retry-schedule/) | backoff(attempt) returns a number of seconds. That is the whole interface, and it is the wrong shape for the problem. |
| [Slack Q&A Agent](ai-agent-workshop/slack-qa-agent/) | People ask the same questions in #general every day - this agent answers them from a knowledge base via RAG, and escalates anything it isn't confident… |
| [SQL Query Formatter](automation-suite/sql-formatter/) | A FastAPI microservice that turns unreadable SQL into a clean house style - and lints it for danger (DELETE with no WHERE) and style (SELECT , implicit… |


---

## Choose, and be able to defend it

*The number is on the screen. Someone still has to decide, and later justify it.*

| Tool | The problem it was built for |
|------|------------------------------|
| [A Decision Log Is an Instrument, and an Instrument Has a Scoring Rule](mini-saas-products/decision-log/) | A decision log without a prediction attached is a diary: it records what was chosen and can never say whether choosing it was any good. |
| [Feature Prioritization Tool](mini-saas-products/feature-prioritizer/) | We argue about priorities without data - RICE turns "I feel strongly about X" into a rankable, defensible number. |
| [OKR Tracker and Advisor](mini-saas-products/okr-tracker/) | OKRs get set and forgotten - this tracks progress against pace and flags which key results are behind while there's still time to act. |
| [Product Roadmap Visualizer](mini-saas-products/roadmap-viz/) | Roadmaps are PowerPoints that go stale - generate the roadmap from data instead: a Gantt-style timeline, grouped by lane, colored by status, with a… |
| [Sprint Retrospective Generator](mini-saas-products/retro-generator/) | Retros are unstructured and repetitive - feed in the sprint numbers and get grounded observations and concrete action items in your team's format. |
| [Startup Idea Validator](mini-saas-products/idea-validator/) | We build before we validate - this scores an idea across five dimensions, builds a Lean Canvas, and hands you the cheapest experiment to test your… |
| [The Exercise Works, the Matrix That Scores It Cannot Rank](mini-saas-products/pre-mortem/) | A pre-mortem is cheap and it works: assume the project already failed, then write down why. |
| [User Story Generator](mini-saas-products/user-story-gen/) | Writing good user stories takes practice - paste raw feature ideas and get INVEST-scored stories with Given/When/Then acceptance criteria. |


---

### The spine underneath

The tools plug into a control plane built alongside them: identity, permissions, an audit log, a connector layer and orchestration. The [architecture notes](https://phoebefu6.github.io/phoebe-the-builder/wiki/02-architecture.html) walk the design decisions.

*Built in public by Phoebe Fu.*
