-- 01_schema.sql
-- Warehouse schema for the PostgreSQL Customer Analytics project.
-- A small star-style design: customers and subscriptions describe who the
-- customers are and what plan they bought; monthly_billing is the fact
-- table with one row per customer per active month.

DROP TABLE IF EXISTS monthly_billing;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id       TEXT PRIMARY KEY,
    senior_citizen    BOOLEAN      NOT NULL,
    paperless_billing BOOLEAN      NOT NULL,
    payment_method    TEXT         NOT NULL,
    tenure_months     INTEGER      NOT NULL CHECK (tenure_months >= 1),
    support_calls     INTEGER      NOT NULL CHECK (support_calls >= 0),
    late_payments     INTEGER      NOT NULL CHECK (late_payments >= 0),
    signup_month      DATE         NOT NULL,
    churned           BOOLEAN      NOT NULL
);

CREATE TABLE subscriptions (
    subscription_id TEXT PRIMARY KEY,
    customer_id     TEXT          NOT NULL REFERENCES customers (customer_id),
    contract_type   TEXT          NOT NULL,
    internet_service TEXT         NOT NULL,
    monthly_charges NUMERIC(10,2) NOT NULL CHECK (monthly_charges >= 0),
    total_charges   NUMERIC(12,2) NOT NULL CHECK (total_charges >= 0)
);

CREATE TABLE monthly_billing (
    customer_id         TEXT          NOT NULL REFERENCES customers (customer_id),
    billing_month       DATE          NOT NULL,
    months_since_signup INTEGER       NOT NULL CHECK (months_since_signup >= 0),
    amount              NUMERIC(10,2) NOT NULL CHECK (amount >= 0),
    PRIMARY KEY (customer_id, billing_month)
);

-- Indexes for the access patterns the analysis queries actually use:
-- cohort/retention work groups by signup month and customer age in months,
-- revenue work scans by calendar month, segmentation joins on customer.
CREATE INDEX idx_customers_signup_month   ON customers (signup_month);
CREATE INDEX idx_subscriptions_customer   ON subscriptions (customer_id);
CREATE INDEX idx_subscriptions_contract   ON subscriptions (contract_type);
CREATE INDEX idx_billing_month            ON monthly_billing (billing_month);
CREATE INDEX idx_billing_customer_month   ON monthly_billing (customer_id, months_since_signup);
