# Portfolio 2 — Routine Data/AI Task Coverage (Days 61-120)

Second 60-project arc. Goal: cover the **routine, recurring tasks** a data/AI team does daily that
Portfolio 1 (Days 1-60) did not — deep data engineering, full ML lifecycle, LLMOps, governance
depth, analytics engineering, and a data-science method cookbook.

Same build rules as Portfolio 1 (see the `daily-fde-build` skill): README + demo.ipynb (Colab/Binder
badges, pre-rendered) + app + requirements + Dockerfile + CI + `.gitignore`. `from __future__ import
annotations` first line of every `.py`. Notebook self-contained, one matplotlib/seaborn chart, "Try
Your Own" section. Commit, push, tick TRACKER.md, rebuild the homepage.

## Product-line folder mapping

| Days | Month | Folder |
|------|-------|--------|
| 61-70 | 7 — Data Engineering Pro | `data-engineering-pro/` |
| 71-80 | 8 — ML Engineering Toolkit | `ml-engineering-toolkit/` |
| 81-90 | 9 — LLMOps & GenAI Platform | `llmops-genai-platform/` |
| 91-100 | 10 — Data Quality & Governance Suite | `data-quality-governance/` |
| 101-110 | 11 — Analytics Engineering & BI | `analytics-engineering-bi/` |
| 111-120 | 12 — Data Science Cookbook | `data-science-cookbook/` |

Colab/Binder badge patterns (same repo):
- Colab: `https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/<folder>/<slug>/demo.ipynb`
- Binder: `https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=<folder>/<slug>/demo.ipynb`

---

## Month 7: Data Engineering Pro (`data-engineering-pro/`)
Learning: Airflow, Kafka/streaming concepts, data contracts, incremental patterns

| Day | Slug | Project | Pain Point | Tech |
|-----|------|---------|------------|------|
| 61 | incremental-loader | Incremental / CDC Loader | "Full reloads take hours every night" | watermark + upsert (pandas/SQL) |
| 62 | airflow-dag-gen | Airflow DAG Generator | "Writing DAG boilerplate is repetitive" | Jinja templates + YAML config |
| 63 | parquet-partitioner | CSV/JSON to Parquet Partitioner | "Our data lake is a pile of CSVs" | pyarrow partitioned writes |
| 64 | data-contract-validator | Data Contract Validator | "Producers break schemas without warning" | schema contract + CI-style check |
| 65 | backfill-planner | Backfill Planner | "Backfills are error-prone manual scripts" | date-range chunker + status track |
| 66 | streaming-aggregator | Streaming Window Aggregator | "We can't compute rolling metrics live" | tumbling/sliding windows |
| 67 | dedup-pipeline | Dedup & Survivorship Pipeline | "Duplicate records everywhere" | key dedup + survivorship rules |
| 68 | api-paginator | API Pagination Extractor | "Paginated API pulls hand-coded each time" | generic cursor/offset paginator |
| 69 | column-lineage | Column-Level Lineage Parser | "No column-level lineage" | sqlparse + dependency graph |
| 70 | pipeline-sla-monitor | Pipeline SLA Monitor | "We miss delivery SLAs silently" | expected-by-time checks + alert |

## Month 8: ML Engineering Toolkit (`ml-engineering-toolkit/`)
Learning: sklearn pipelines, Optuna, SHAP, model documentation, MLOps

| Day | Slug | Project | Pain Point | Tech |
|-----|------|---------|------------|------|
| 71 | feature-factory | Feature Engineering Pipeline | "We rewrite feature code every project" | sklearn ColumnTransformer builder |
| 72 | train-eval-harness | Train/Eval Leaderboard | "Model comparison is ad-hoc" | cross-val leaderboard |
| 73 | hyperparam-tuner | Hyperparameter Tuner | "We grid-search by hand" | Optuna / random search |
| 74 | imbalance-toolkit | Class Imbalance Toolkit | "Our fraud model ignores the minority class" | SMOTE / class-weights compare |
| 75 | model-card-gen | Model Card Generator | "No model documentation" | auto model-card markdown |
| 76 | feature-importance | Feature Importance Explainer | "Stakeholders don't trust the model" | SHAP / permutation importance |
| 77 | batch-scorer | Batch Scoring Service | "Scoring new data is manual" | load model + score CSV |
| 78 | calibration-checker | Probability Calibration Checker | "Our probabilities are meaningless" | reliability curve + Brier |
| 79 | leakage-detector | Data Leakage Detector | "Great CV, terrible in prod" | target/train-test leak checks |
| 80 | model-registry | Mini Model Registry | "We lose track of model versions" | versioned store + metadata |

## Month 9: LLMOps & GenAI Platform (`llmops-genai-platform/`)
Learning: RAG evaluation, guardrails, prompt ops, cost control

| Day | Slug | Project | Pain Point | Tech |
|-----|------|---------|------------|------|
| 81 | prompt-registry | Prompt Registry & Versioning | "Prompts scattered across the codebase" | versioned prompt store + diff |
| 82 | rag-eval | RAG Evaluation Harness | "We don't know if RAG retrieves right" | recall@k / faithfulness scoring |
| 83 | chunk-optimizer | Chunking Strategy Tester | "Bad chunks = bad answers" | compare chunk sizes on retrieval |
| 84 | llm-guardrails | LLM Guardrail Filter | "Model outputs unsafe/off-topic text" | input/output rule filters |
| 85 | llm-cost-tracker | LLM Cost & Token Tracker | "Surprise API bills" | per-call token + cost log |
| 86 | semantic-cache | Semantic Response Cache | "We pay for repeated similar queries" | embedding similarity cache |
| 87 | structured-extractor | Structured Output Enforcer | "LLM JSON breaks our parser" | schema validate + retry |
| 88 | hallucination-checker | Hallucination Detector | "The model makes facts up" | grounding / citation check |
| 89 | fewshot-selector | Few-Shot Example Selector | "Static examples underperform" | nearest-example (embedding) picker |
| 90 | llm-router | LLM Model Router | "We use an expensive model for easy tasks" | complexity-based routing |

## Month 10: Data Quality & Governance Suite (`data-quality-governance/`)
Learning: Great Expectations, catalogs, GDPR/CCPA, access control

| Day | Slug | Project | Pain Point | Tech |
|-----|------|---------|------------|------|
| 91 | anomaly-detector | Column Anomaly Detector | "Bad values slip into prod" | z-score / IQR outliers |
| 92 | dq-rules-engine | DQ Rules Engine | "Quality checks are one-off SQL" | declarative rule runner |
| 93 | data-catalog | Lightweight Data Catalog | "Nobody can find datasets" | searchable table metadata |
| 94 | business-glossary | Business Glossary Manager | "Everyone defines 'active user' differently" | term store + owners |
| 95 | access-auditor | Data Access Auditor | "We don't know who can see PII" | grant scan + report |
| 96 | retention-enforcer | Data Retention Enforcer | "We keep data past policy" | age-based flag/purge plan |
| 97 | consent-tracker | Consent & Purpose Tracker | "GDPR purpose tracking is manual" | consent registry |
| 98 | schema-registry | Schema Registry | "No single source for schemas" | versioned schema store |
| 99 | reconciliation-checker | Source-to-Target Reconciliation | "Do source and warehouse match?" | row/sum reconciliation |
| 100 | data-diff | Dataset Snapshot Diff | "What changed between two snapshots?" | key-level diff |

## Month 11: Analytics Engineering & BI (`analytics-engineering-bi/`)
Learning: dbt, semantic/metrics layer, NL→SQL

| Day | Slug | Project | Pain Point | Tech |
|-----|------|---------|------------|------|
| 101 | metrics-layer | Metrics Layer / Semantic Definitions | "Every dashboard computes revenue differently" | YAML metric store |
| 102 | nl-to-sql | Natural Language to SQL | "Non-analysts can't query" | Claude NL→SQL + guardrails |
| 103 | dbt-test-gen | dbt Test Generator | "No tests on our models" | auto schema/data tests |
| 104 | metric-alerting | Metric Anomaly Alerter | "We notice metric drops late" | threshold + trend alert |
| 105 | self-serve-explorer | Self-Serve Data Explorer | "Analysts get pinged for every number" | pivot/aggregate UI |
| 106 | report-scheduler | Scheduled Report Sender | "Weekly reports sent by hand" | cron + email digest |
| 107 | kpi-tree | KPI Tree / Driver Decomposition | "Why did revenue move?" | metric driver decomposition |
| 108 | query-optimizer | SQL Optimizer Advisor | "Slow queries, no idea why" | heuristic rewrite suggestions |
| 109 | dashboard-spec | Dashboard Spec Generator | "Dashboards built from vague asks" | chart recommender from data |
| 110 | metric-catalog | Metric Catalog & Ownership | "Which metrics exist + who owns them?" | metric registry |

## Month 12: Data Science Cookbook (`data-science-cookbook/`)
Learning: statistics, NLP, recommenders, time series, dimensionality reduction

| Day | Slug | Project | Pain Point | Tech |
|-----|------|---------|------------|------|
| 111 | stat-test-advisor | Statistical Test Advisor | "Which test do I use?" | decision tree + run test (scipy) |
| 112 | correlation-explorer | Correlation & Multicollinearity Explorer | "Which features relate?" | corr matrix + VIF |
| 113 | dim-reducer | PCA/UMAP Explorer | "Too many features to see" | PCA/UMAP + scatter |
| 114 | topic-modeler | Topic Modeling Tool | "What are these docs about?" | LDA / BERTopic |
| 115 | ner-extractor | Named Entity Extractor | "Pull names/orgs from text" | spaCy / Claude NER |
| 116 | text-classifier | Text Classification Trainer | "Auto-tag support tickets" | TF-IDF + logistic regression |
| 117 | recommender | Recommendation Engine | "No 'you may also like'" | item-item collaborative filtering |
| 118 | ts-forecaster | General Time Series Forecaster | "Forecast any metric" | Prophet / ETS wrapper |
| 119 | outlier-explainer | Outlier Explainer | "Which rows are weird, and why?" | isolation forest + reason attribution |
| 120 | crosstab-chi2 | Crosstab & Chi-Square Tool | "Compare groups in survey data" | contingency table + chi-square |

---

## Coverage check — full data/AI lifecycle

`ingest → transform → orchestrate → quality → govern → model → tune → serve → monitor → LLMOps → analytics → DS methods`

Portfolio 1 = applied breadth (60 shipped mini-products). Portfolio 2 = **routine operational depth** across
the stack. Together, 120 builds spanning every recurring task a data/AI team runs.
