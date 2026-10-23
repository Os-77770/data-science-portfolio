-- 03_window_functions.sql
-- Window-function showcase over monthly revenue and customer value.
--
-- Part A: monthly revenue with month-over-month change (LAG), a running
--         total, and a 3-month moving average.
-- Part B: the top 5 customers by lifetime billed amount inside each
--         contract segment (ROW_NUMBER over a partition).

-- Part A -------------------------------------------------------------------
WITH monthly_revenue AS (
    SELECT
        billing_month,
        COUNT(DISTINCT customer_id) AS active_customers,
        SUM(amount)                 AS revenue
    FROM monthly_billing
    GROUP BY billing_month
)
SELECT
    billing_month,
    active_customers,
    ROUND(revenue, 2)                                                   AS revenue,
    ROUND(revenue - LAG(revenue) OVER (ORDER BY billing_month), 2)     AS mom_change,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY billing_month))
          / LAG(revenue) OVER (ORDER BY billing_month), 2)             AS mom_change_pct,
    ROUND(SUM(revenue) OVER (ORDER BY billing_month), 2)                AS running_total_revenue,
    ROUND(AVG(revenue) OVER (
        ORDER BY billing_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)                   AS revenue_3mo_avg
FROM monthly_revenue
ORDER BY billing_month;

-- Part B -------------------------------------------------------------------
WITH customer_value AS (
    SELECT
        s.contract_type,
        b.customer_id,
        SUM(b.amount)       AS lifetime_billed,
        COUNT(*)            AS months_billed,
        MAX(b.billing_month) AS last_billing_month
    FROM monthly_billing b
    JOIN subscriptions s ON s.customer_id = b.customer_id
    GROUP BY s.contract_type, b.customer_id
),
ranked AS (
    SELECT
        contract_type,
        customer_id,
        ROUND(lifetime_billed, 2) AS lifetime_billed,
        months_billed,
        ROW_NUMBER() OVER (
            PARTITION BY contract_type ORDER BY lifetime_billed DESC
        ) AS value_rank,
        ROUND(100.0 * lifetime_billed
              / SUM(lifetime_billed) OVER (PARTITION BY contract_type), 3)
            AS pct_of_segment_revenue
    FROM customer_value
)
SELECT *
FROM ranked
WHERE value_rank <= 5
ORDER BY contract_type, value_rank;
