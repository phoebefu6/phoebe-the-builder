# Schema Diff Tool

> Database migrations break things silently — catch schema changes before they hit production.

## Business Impact
- **Before:** Developers run migrations without knowing what columns changed, types shifted, or tables disappeared. Breakages surface as runtime errors in production.
- **After:** Paste old and new schema SQL, get an instant diff with severity ratings (breaking/warning/info) and migration risk assessment.
- **Estimated ROI:** 2-3 hours/week saved on migration review + fewer production incidents from silent schema drift.

## Tech Stack
- Python 3.9+
- Streamlit (interactive UI)
- Regex-based SQL DDL parser (zero external dependencies for core logic)
- Docker

## Demo
The app shows two side-by-side SQL editors (old vs new schema). Click "Compare Schemas" to get:
- Summary metrics: total changes, breaking, warnings, info
- Migration risk level (HIGH / MEDIUM / LOW)
- Detailed change list with severity badges
- Table-level overview showing NEW, DROPPED, or change counts

Sample schemas included — toggle "Load sample schemas" to try it instantly.

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What It Detects
| Change Type | Severity | Example |
|-------------|----------|---------|
| Table dropped | Breaking | `products` table removed |
| Column dropped | Breaking | `users.status` column removed |
| NOT NULL added (no default) | Warning | `name` changed from nullable to NOT NULL |
| Type changed | Warning | `VARCHAR(255)` → `VARCHAR(500)` |
| Primary key changed | Breaking | PK columns reordered or changed |
| Table added | Info | New `payments` table |
| Column added (nullable) | Info | New `orders.currency` column |
| Default changed | Info | Default value updated |

## Learning Connection
Built while studying **Data Engineer Career Track** on DS365 and **IBM Data Engineering Certificate**.
Applies: SQL DDL parsing, schema management, migration safety, database design patterns.

## Impact Note
- **Who benefits:** Database engineers, backend developers, DevOps teams running migrations
- **Potential risks:** Parser handles standard CREATE TABLE syntax; complex DDL (partitions, stored procedures, DB-specific extensions) may need manual review
