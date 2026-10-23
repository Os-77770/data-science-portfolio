-- Singular test: the billing grain is one row per customer per month.
-- Returns one row per duplicated (customer_id, billing_month) pair;
-- the test passes when this query returns nothing.

select
    customer_id,
    billing_month,
    count(*) as row_count
from {{ ref('stg_monthly_billing') }}
group by customer_id, billing_month
having count(*) > 1
