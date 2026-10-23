# Talking Points — PostgreSQL Customer Analytics Project

Private study notes for me: likely interview questions on PostgreSQL,
with crisp answers tied to what this project actually does.

## Schema and loading

**Why this schema?**
A small star design. `customers` and `subscriptions` are dimension-style
tables (who the customer is, what they bought); `monthly_billing` is the
fact table at one-row-per-customer-per-month grain. The grain is
enforced by the composite primary key `(customer_id, billing_month)` —
the database itself rejects duplicate months instead of me hoping the
loader behaves. Foreign keys from billing and subscriptions to customers
mean orphan rows are impossible to insert.

**Why `NUMERIC` for money instead of `FLOAT`/`DOUBLE`?**
Floats are binary approximations — sums drift by cents. `NUMERIC(10,2)`
is exact decimal arithmetic, which is why my reconciliation check can
demand the billed total match the source to the penny
($14,483,269.68 on both sides).

**How did you load 241k rows?**
psycopg's `COPY` support, streaming the CSV. Row-by-row `INSERT` pays a
round trip and statement overhead per row; `COPY` is the bulk path and
loads the fact table in about a second. After loading I ran a checks
script: row counts vs the CSVs, orphan-FK counts, total billed, churn
rate, and a per-customer "billing rows = tenure" consistency check.

## Indexes and EXPLAIN

**Which indexes did you add, and why those?**
Only ones matching real access patterns: `customers(signup_month)` for
cohort grouping, `monthly_billing(billing_month)` for revenue scans,
`subscriptions(contract_type)` for segmentation, and
`monthly_billing(customer_id, months_since_signup)` for per-customer
history. Indexes are not free — they cost write time and storage — so
each one has a query behind it.

**Show me you read an EXPLAIN plan.**
For one customer's billing history, `EXPLAIN ANALYZE` picks an Index
Scan on the composite PK with an Index Cond on `customer_id`, ~0.2 ms.
If I saw a Seq Scan on the 241k-row table for a selective lookup, that
is the signal an index is missing (or the planner decided the table is
small enough that a scan is cheaper — on tiny tables that is normal and
fine).

**When would the planner ignore your index?**
When the predicate is not selective (returning half the table), when the
query wraps the column in a function, or when statistics are stale and
it misestimates. First checks: selectivity, `ANALYZE` freshness, and
whether the plan changes with the actual parameter values.

## SQL techniques in the queries

**CTE vs subquery?**
Same results; I use CTEs (`WITH ...`) because the pipeline reads
top-down: shape segments first, then aggregate, then rank. In
PostgreSQL 12+, a CTE is inlined like a subquery unless it is referenced
twice or I force `MATERIALIZED`, so there is no automatic performance
penalty for the readability.

**Window functions vs GROUP BY?**
`GROUP BY` collapses rows; window functions keep the rows and compute
across a frame. Query 03 needs both worlds: monthly revenue is a
`GROUP BY`, then `LAG()` gives month-over-month change, `SUM() OVER`
the running total, and `AVG() OVER (ROWS BETWEEN 2 PRECEDING AND CURRENT
ROW)` the 3-month moving average — none of which a plain aggregate can
express without self-joins. `ROW_NUMBER() OVER (PARTITION BY
contract_type ORDER BY lifetime_billed DESC)` gets the top 5 customers
per segment in one pass.

**`RANK()` vs `DENSE_RANK()` vs `ROW_NUMBER()`?**
All three number rows inside a partition. `ROW_NUMBER` never ties.
`RANK` ties share a number and then skips (1, 2, 2, 4). `DENSE_RANK`
ties share and does not skip (1, 2, 2, 3). I used `RANK` for churn
ranking inside contracts in query 02 and `ROW_NUMBER` where I needed
exactly five rows per segment in query 03.

**How do you do conditional aggregation?**
`SUM(CASE WHEN churned THEN 1 ELSE 0 END)` counts churned customers in
the same pass as `COUNT(*)` counts everyone, so a churn rate is one
scan: `100.0 * AVG(CASE WHEN churned THEN 1 ELSE 0 END)`. (PostgreSQL
also has the cleaner `COUNT(*) FILTER (WHERE churned)` syntax, which I
use in other queries.)

## The judgement question

**Why is there no retention curve / survival analysis?**
Because the dataset is a snapshot: tenure is an attribute measured at
the snapshot date, and there are no churn event timestamps. A
Kaplan-Meier curve built on it produces a plausible-looking chart that
contradicts the segment numbers — I checked. What the data does support
is cohort comparison by signup quarter, with the tenure confound stated
plainly. Knowing when the data cannot answer the question as asked is
part of the job; the fix is event-level data (signup and churn
timestamps), and I can describe exactly the table I would want.
