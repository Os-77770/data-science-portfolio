# Talking Points — dbt Analytics Engineering Project

Private study notes for me: likely interview questions on dbt, with crisp
answers tied to what this project actually does.

## The basics

**What is dbt?**
A transformation tool: it runs SQL `SELECT` statements in the warehouse
and materializes them as tables or views. dbt handles the dependency
graph, build order, testing and documentation. It does the "T" in ELT —
extraction and loading happen with other tools; here `src/load_raw.py`
plays the loader role.

**Why model in layers (staging → intermediate → marts)?**
Each layer has one job. Staging is 1:1 with sources — typing and renaming
only, so source quirks live in exactly one place. Intermediate holds
reusable joins and derived fields (in my project: billing joined to
customer and plan, tenure band, signup quarter). Marts are the
business-facing aggregates people actually query. If a definition changes,
I change it in one model instead of hunting through every report.

**What is a materialization?**
How dbt persists a model: `view`, `table`, `incremental`, or `ephemeral`.
I used views for staging and intermediate (cheap, always fresh, nobody
queries them directly) and tables for the marts (queried by dashboards,
worth the storage and build time). I set this per layer in
`dbt_project.yml` rather than repeating config in every model.

## Testing and quality

**What tests does the project run?**
22 data tests, all passing under `dbt build`: `unique` and `not_null` on
keys, `accepted_values` on contract type and internet service,
`relationships` tests so subscriptions and billing rows can never point
at a customer that does not exist, plus one singular test — a hand-written
SQL query asserting the billing grain (one row per customer per month)
that must return zero rows.

**What is the difference between a generic test and a singular test?**
Generic tests are reusable checks declared in YAML (`unique`,
`not_null`, `accepted_values`, `relationships`, or custom ones I define
once and apply anywhere). A singular test is one specific SQL query for
one specific assertion — like my billing-grain check. Generic first;
singular when the assertion is genuinely one-off.

**What happens when a test fails mid-build?**
dbt stops building downstream models that depend on the failed model, so
bad data does not silently flow into the marts. That fail-fast behaviour
is the point of testing during `build` rather than after.

## Concepts interviewers probe

**What are sources and refs?**
A `source()` declares an external table dbt does not build (my raw
landing tables); a `ref()` points at another model. Using them is what
lets dbt build the DAG — it knows `mart_churn_segments` must come after
`stg_customers`, `stg_subscriptions` and `stg_monthly_billing` without me
specifying an order anywhere.

**What is the DAG here?**
raw sources → 3 staging models → `int_customer_monthly` (and staging
again) → 2 marts. I can inspect it with `dbt docs generate` / the
lineage graph; the README also draws it in text.

**Why land raw data untyped (all strings)?**
Because that is what real ingestion looks like, and it forces a clean
separation: if a source format changes, only the staging model changes.
Casting early in the raw layer would bake assumptions into data I can no
longer re-derive.

**Snapshots vs incremental models?**
Incremental models process only new/changed rows (right for huge event
tables); dbt snapshots track slowly changing dimensions over time with
`valid_from`/`valid_to`. My dataset is a small full snapshot, so full
table refreshes are honest and simplest — I would say exactly that if
asked why there is no incremental model, and explain when I would add one
(billing events at a scale where a full rebuild hurts).

**What did the marts find?**
Overall churn 30.2%. Month-to-month customers in their first 6 months
churn at 62.35%; two-year customers past 49 months at 3.69%. I also flag
the snapshot caveat: churned customers' final billing month is the
snapshot month, so I do not present monthly churn as a time series —
knowing what your data cannot say is part of the job.
