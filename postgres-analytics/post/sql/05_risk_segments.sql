-- 05_risk_segments.sql
-- Behavioural risk segmentation: how churn varies with support calls and
-- late payments, and which currently-active customers sit in the highest
-- risk buckets. Uses NTILE to bucket customers by their monthly charge
-- inside the analysis, conditional aggregation, and a CTE pipeline.

WITH behaviour AS (
    SELECT
        c.customer_id,
        c.churned,
        CASE WHEN c.support_calls = 0 THEN '0'
             WHEN c.support_calls <= 2 THEN '1-2'
             WHEN c.support_calls <= 4 THEN '3-4'
             ELSE '5+' END AS support_call_band,
        CASE WHEN c.late_payments = 0 THEN '0'
             WHEN c.late_payments = 1 THEN '1'
             WHEN c.late_payments = 2 THEN '2'
             ELSE '3+' END AS late_payment_band,
        c.support_calls,
        c.late_payments
    FROM customers c
)
SELECT
    support_call_band,
    late_payment_band,
    COUNT(*) AS customers,
    ROUND(100.0 * AVG(CASE WHEN churned THEN 1 ELSE 0 END), 2) AS churn_rate_pct
FROM behaviour
GROUP BY support_call_band, late_payment_band
ORDER BY churn_rate_pct DESC;

-- Active (not yet churned) customers in the riskiest behaviour bucket:
-- 3+ support calls AND 2+ late payments, highest monthly charge first.
SELECT
    c.customer_id,
    s.contract_type,
    s.internet_service,
    c.tenure_months,
    c.support_calls,
    c.late_payments,
    s.monthly_charges,
    NTILE(4) OVER (ORDER BY s.monthly_charges) AS charge_quartile
FROM customers c
JOIN subscriptions s ON s.customer_id = c.customer_id
WHERE NOT c.churned
  AND c.support_calls >= 3
  AND c.late_payments >= 2
ORDER BY s.monthly_charges DESC
LIMIT 15;
