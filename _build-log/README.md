# Build log

Internal chronology and planning notes. Nothing here is part of the showcase -
the catalog is organised by the job somebody arrived with, not by build order.

| file | what it is |
|------|------------|
| `PORTFOLIO-2-PLAN.md` | the Days 61-120 plan, kept for provenance. Moved here from the repo root on 2026-08-25 because a plan document was the second file a visitor met on the landing page. Fully built; nothing reads it. |

`TRACKER.md` deliberately stays in the repo root. It is the source of truth that
`one-data-platform/homepage/build_site.py` and the shipped
`mini-saas-products/portfolio-dashboard` tool both read by relative path, and
moving it would mean editing a shipped build to gain one line in a file listing.
