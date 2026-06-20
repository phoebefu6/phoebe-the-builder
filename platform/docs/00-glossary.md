# 00 - Glossary (plain language, for a data person)

> Every scary platform word, explained with a data analogy you already know.
> If a term isn't here when we use it, that's a bug - tell Claude to add it.

---

### Platform / "the shell"
The one application that holds all our tools. Think of it like **a single Power BI
workspace** that, instead of just dashboards, can hold dashboards *and* notebooks
*and* models *and* AI apps - all behind one login. The 60 mini-tools become "apps"
that live inside this shell.

### Control plane
The "front desk + security + directory" of the building. It does **not** do the
data work itself - it decides *who gets in*, *what they're allowed to touch*, and
*keeps a record*. The actual tools (the SQL runner, the dashboard) are the
"offices" behind the front desk. We build the front desk; we rent the offices
(open-source tools).

> Why we care: the front desk is the part that's hard to copy and worth selling.
> Anyone can run a SQL engine; not everyone can govern who runs it.

### Gateway
The single front door. Every request from a user goes *through* the gateway first.
It checks your badge before passing you to the tool you asked for. In our build,
the gateway is one small web service (FastAPI). Like the **one turnstile** everyone
must tap into - no sneaking in a side window.

### Authentication ("authn") = *who are you?*
Proving identity. Logging in with email + password is authentication. Same idea as
signing into your data warehouse. Answers: **"Are you really Phoebe?"**

### Authorization ("authz") = *what are you allowed to do?*
After we know who you are, what can you touch? You're Phoebe (authenticated), but
are you allowed to *delete* the production dataset? Answers: **"Phoebe is allowed
to view dashboards but not edit the model registry."**

> authn = identity. authz = permission. Two different checks, in that order.

### RBAC (Role-Based Access Control)
A tidy way to do authorization. Instead of setting permissions per person (chaos at
scale), you give people **roles**, and roles carry permissions. Exactly like
**database roles / groups** you already know:
- `analyst` → can run SQL, view dashboards
- `data_scientist` → all of analyst + train/register models
- `ai_engineer` → all of that + deploy LLM apps
- `admin` (you) → everything + manage users

Add a new hire? Give them a role, done. This is how we "govern access to
dashboards, data, models, LLMs" - each app declares which role it needs.

### Session / Token (JWT)
After you log in, the gateway hands you a **temporary wristband** so you don't
re-enter your password on every click. A **JWT** (JSON Web Token) is that wristband:
a small signed string your browser sends back with each request. Signed = we can
tell if someone forged it. It expires, like a day-pass. Same role as a warehouse
session, just for the web.

### Connector / Connection layer
One shared place that knows *how to reach* each data source (Postgres, S3, BigQuery,
Snowflake) - the host, and the secret password/key. Instead of every tool having its
own copy of credentials scattered in 60 `.env` files, tools ask the connector layer
"give me the orders database" and it hands back a ready connection. Like a **single
shared connections.json** for the whole company, but safe.

### Secret / Secrets management
A "secret" = any password, API key, or token. **Secrets management** = keeping them
out of code and in one guarded place (env vars, a vault). Rule we never break:
**secrets never get written into a file we commit to git.**

### Audit log
A permanent, append-only diary: *who did what, when*. "Phoebe viewed the revenue
dashboard at 10:04." "Sam ran a query against `customers` at 10:06." You can't edit
it, only add to it. This is the single most valuable thing to an enterprise buyer in
a regulated industry - it's how they prove governance to auditors. Like an
**immutable query-history table** for the whole platform.

### App registry
A simple list (a `apps.yaml` file) of every tool mounted in the shell: its name, its
web address inside the platform, and **which role it requires**. Adding a new daily
build = adding a few lines here. The gateway reads this list to know what exists and
who's allowed in. Like a **directory board in the lobby** ("Floor 3: Churn Model -
Data Scientists only").

### Orchestration
Coordinating multi-step work so steps run in the right order, on schedule, and retry
on failure. "Pull data → clean it → train model → publish dashboard," every night at
2am. We will **not** build this ourselves - we'll plug in **Apache Airflow** (mature
open source). Our platform *governs and surfaces* Airflow; it doesn't replace it.

### Apache Airflow
The most popular open-source **orchestration** tool. It runs scheduled data
pipelines (called DAGs). We leverage it instead of reinventing ETL scheduling. You
already think in pipelines - Airflow is the engine that runs them on a clock.

### DAG (Directed Acyclic Graph)
Airflow's word for "a pipeline." Just a set of steps with arrows showing order, where
arrows never loop back (acyclic = no cycles). "Extract → Transform → Load" is a
3-step DAG. Don't overthink the name - it means **"a pipeline drawn as boxes and
arrows."**

### Open source / OSS
Free, public, community-maintained software we're allowed to use and build on
(Airflow, DuckDB, Trino). Our strategy: **mount OSS for the commodity work, build
only the governance shell ourselves.** Don't rebuild what thousands of engineers
already perfected.

### DuckDB / Trino
Open-source **SQL engines** - they run SQL queries over data. DuckDB = lightweight,
runs in-process (great for one analyst on a file). Trino = distributed (great for
querying huge data across sources). These are "offices we rent" - the SQL-running
capability we don't build ourselves.

### FastAPI / Streamlit
Two Python tools we already use. **FastAPI** = build web services/APIs (our gateway).
**Streamlit** = build data apps/dashboards fast (many of our 60 apps). You've seen
both in the portfolio already.

### Reverse proxy
A doorman that takes your request and quietly forwards it to the right office behind
the scenes, so you only ever talk to the front door. Our gateway acts as a light
reverse proxy: you hit the platform, it routes you to the correct app after checking
your role. (We'll build the simplest possible version.)

### MVP (Minimum Viable Product)
The smallest version that actually works end-to-end and proves the idea. Our spine
MVP = "log in → see only the apps your role allows → click one → it checks your role
→ every action is logged." Tiny, but real. We grow from there.

---

*Add terms as we meet them. A glossary is never finished - it's a living map of what
we've learned.*
