-- Staging: one row per subscription (one per customer in this dataset),
-- typed and renamed.

with source as (
    select * from {{ source('warehouse', 'raw_subscriptions') }}
)

select
    subscription_id,
    customer_id,
    contract_type,
    internet_service,
    cast(monthly_charges as double) as monthly_charges,
    cast(total_charges as double)   as total_charges
from source
