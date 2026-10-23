-- Staging: one row per customer per billed month, typed and renamed.

with source as (
    select * from {{ source('warehouse', 'raw_monthly_billing') }}
)

select
    customer_id,
    cast(billing_month as date)        as billing_month,
    cast(months_since_signup as integer) as months_since_signup,
    cast(amount as double)              as amount
from source
