-- Mart: churn outcomes by contract type and tenure band.
-- One row per segment, ready for a dashboard or a retention-team report.

with customers as (
    select * from {{ ref('stg_customers') }}
),
subscriptions as (
    select * from {{ ref('stg_subscriptions') }}
),
billing_totals as (
    select
        customer_id,
        sum(amount) as lifetime_billed,
        count(*)    as months_billed
    from {{ ref('stg_monthly_billing') }}
    group by customer_id
),
segmented as (
    select
        s.contract_type,
        case
            when c.tenure_months <= 6  then '0-6 months'
            when c.tenure_months <= 12 then '7-12 months'
            when c.tenure_months <= 24 then '13-24 months'
            when c.tenure_months <= 48 then '25-48 months'
            else '49+ months'
        end as tenure_band,
        c.customer_id,
        c.is_churned,
        s.monthly_charges,
        coalesce(b.lifetime_billed, 0) as lifetime_billed
    from customers c
    inner join subscriptions s on s.customer_id = c.customer_id
    left join billing_totals b on b.customer_id = c.customer_id
)

select
    contract_type,
    tenure_band,
    count(*)                                             as customers,
    sum(case when is_churned then 1 else 0 end)          as churned_customers,
    round(100.0 * avg(case when is_churned then 1 else 0 end), 2)
                                                         as churn_rate_pct,
    round(avg(monthly_charges), 2)                       as avg_monthly_charges,
    round(sum(lifetime_billed), 2)                       as total_lifetime_billed
from segmented
group by contract_type, tenure_band
order by contract_type, churn_rate_pct desc
