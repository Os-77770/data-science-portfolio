-- Staging: one row per customer, typed and renamed.
-- Raw values arrive as strings (see _sources.yml); this layer is the only
-- place that knows that, so downstream models work with real types.

with source as (
    select * from {{ source('warehouse', 'raw_customers') }}
)

select
    customer_id,
    cast(senior_citizen as integer) = 1     as is_senior_citizen,
    cast(paperless_billing as integer) = 1  as has_paperless_billing,
    payment_method,
    cast(tenure_months as integer)          as tenure_months,
    cast(support_calls as integer)          as support_calls,
    cast(late_payments as integer)          as late_payments,
    cast(signup_month as date)              as signup_month,
    cast(churned as integer) = 1            as is_churned
from source
