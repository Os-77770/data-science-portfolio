-- 02_churn_by_segment.sql
-- Churn rate by contract type and by tenure band, with the contract-level
-- summary ranked against the overall churn rate.
-- Uses: CTEs, CASE-based banding, conditional aggregation, RANK().

WITH customer_segments AS (
    SELECT
        c.customer_id,
        s.contract_type,
        CASE
            WHEN c.tenure_months <= 6  THEN '0-6 months'
            WHEN c.tenure_months <= 12 THEN '7-12 months'
            WHEN c.tenure_months <= 24 THEN '13-24 months'
            WHEN c.tenure_months <= 48 THEN '25-48 months'
            ELSE '49+ months'
        END AS tenure_band,
        c.churned
    FROM customers c
    JOIN subscriptions s ON s.customer_id = c.customer_id
),
segment_stats AS (
    SELECT
        contract_type,
        tenure_band,
        COUNT(*)                                        AS customers,
        SUM(CASE WHEN churned THEN 1 ELSE 0 END)        AS churned_customers,
        ROUND(100.0 * AVG(CASE WHEN churned THEN 1 ELSE 0 END), 2) AS churn_rate_pct
    FROM customer_segments
    GROUP BY contract_type, tenure_band
),
overall AS (
    SELECT AVG(CASE WHEN churned THEN 1 ELSE 0 END) AS overall_churn
    FROM customers
)
SELECT
    st.contract_type,
    st.tenure_band,
    st.customers,
    st.churned_customers,
    st.churn_rate_pct,
    ROUND(100.0 * o.overall_churn, 2)                   AS overall_churn_pct,
    RANK() OVER (
        PARTITION BY st.contract_type
        ORDER BY st.churn_rate_pct DESC
    )                                                   AS churn_rank_within_contract
FROM segment_stats st
CROSS JOIN overall o
ORDER BY st.contract_type, st.churn_rate_pct DESC;
