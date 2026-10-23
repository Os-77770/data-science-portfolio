-- Intermediate: billing facts enriched with everything an analyst needs
-- about the customer and their plan, plus reusable derived fields
-- (tenure band, signup quarter). Business logic that more than one mart
-- needs lives here, not copied into each mart.

with billing as (
    select * from {{ ref('stg_monthly_billing') }}
),
customers as (
    select * from {{ ref('stg_customers') }}
),
subscriptions as (
    select * from {{ ref('stg_subscriptions') }}
)

select
    b.customer_id,
    b.billing_month,
    b.months_since_signup,
    b.amount,
    c.signup_month,
    date_trunc('quarter', c.signup_month)::date as signup_quarter,
    c.tenure_months,
    case
        when c.tenure_months <= 6  then '0-6 months'
        when c.tenure_months <= 12 then '7-12 months'
        when c.tenure_months <= 24 then '13-24 months'
        when c.tenure_months <= 48 then '25-48 months'
        else '49+ months'
    end as tenure_band,
    c.support_calls,
    c.late_payments,
    c.is_churned,
    s.contract_type,
    s.internet_service,
    s.monthly_charges
from billing b
inner join customers c on c.customer_id = b.customer_id
inner join subscriptions s on s.customer_id = b.customer_id
