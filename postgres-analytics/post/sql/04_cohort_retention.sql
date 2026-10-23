-- 04_cohort_retention.sql
-- Cohort analysis: customers grouped by the calendar quarter they signed
-- up in, scored on size, churn outcome and lifetime value.
--
-- An honest note on method: this dataset is a SNAPSHOT (one row per
-- customer, tenure and churn observed at 2024-12-01), not an event log.
-- A classic retention triangle or Kaplan-Meier curve needs churn EVENT
-- dates, which a snapshot does not contain -- tenure here is an attribute
-- correlated with churn, not a churn timestamp. So instead of faking a
-- retention curve, this query does the cohort analysis the data actually
-- supports, and states its confound: newer cohorts have higher churn
-- rates largely because they are YOUNGER, and young customers churn more
-- (see query 02). Reading cohort quality without that caveat would
-- mislead.
--
-- Part A: one row per signup-quarter cohort.
-- Part B: the same cohorts split by contract type, to show the contract
-- effect is present inside every cohort, not just between them.

-- Part A -------------------------------------------------------------------
SELECT
    DATE_TRUNC('quarter', c.signup_month)::date AS cohort_quarter,
    COUNT(*)                                    AS cohort_customers,
    SUM(CASE WHEN NOT c.churned THEN 1 ELSE 0 END) AS retained_customers,
    ROUND(100.0 * AVG(CASE WHEN c.churned THEN 1 ELSE 0 END), 2)
                                                AS churn_rate_pct,
    ROUND(AVG(c.tenure_months), 1)              AS avg_tenure_months,
    ROUND(SUM(b.lifetime_billed), 2)            AS cohort_lifetime_revenue,
    ROUND(AVG(b.lifetime_billed), 2)            AS revenue_per_customer
FROM customers c
JOIN (
    SELECT customer_id, SUM(amount) AS lifetime_billed
    FROM monthly_billing
    GROUP BY customer_id
) b ON b.customer_id = c.customer_id
GROUP BY cohort_quarter
ORDER BY cohort_quarter;

-- Part B -------------------------------------------------------------------
SELECT
    DATE_TRUNC('quarter', c.signup_month)::date AS cohort_quarter,
    s.contract_type,
    COUNT(*)                                    AS customers,
    ROUND(100.0 * AVG(CASE WHEN c.churned THEN 1 ELSE 0 END), 2)
                                                AS churn_rate_pct
FROM customers c
JOIN subscriptions s ON s.customer_id = c.customer_id
GROUP BY cohort_quarter, s.contract_type
ORDER BY cohort_quarter, s.contract_type;
