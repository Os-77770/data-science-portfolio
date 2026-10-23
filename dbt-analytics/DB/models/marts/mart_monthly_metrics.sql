-- Mart: the monthly business scorecard — revenue, active customer base,
-- new signups, and churn events per calendar month.
-- A churn event lands in the month of the customer's last billing row.

with monthly as (
    select * from {{ ref('int_customer_monthly') }}
),
revenue as (
    select
        billing_month,
        count(distinct customer_id) as active_customers,
        round(sum(amount), 2)       as revenue
    from monthly
    group by billing_month
),
signups as (
    select
        signup_month as billing_month,
        count(*)     as new_signups
    from {{ ref('stg_customers') }}
    group by signup_month
),
last_billing as (
    select
        customer_id,
        max(billing_month) as last_billing_month,
        max(case when is_churned then 1 else 0 end) as is_churned
    from monthly
    group by customer_id
),
churns as (
    select
        last_billing_month as billing_month,
        count(*)           as churned_customers
    from last_billing
    where is_churned = 1
    group by last_billing_month
)

select
    r.billing_month,
    r.active_customers,
    r.revenue,
    coalesce(s.new_signups, 0)       as new_signups,
    coalesce(ch.churned_customers, 0) as churned_customers
from revenue r
left join signups s on s.billing_month = r.billing_month
left join churns ch on ch.billing_month = r.billing_month
order by r.billing_month
