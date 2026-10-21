"""
ETL + SQL analysis.

1. Loads data/customers.csv into a SQLite database (data/churn.db) as the
   raw table `customers_raw`.
2. Builds a cleaned analytics table `customers`:
      - trims whitespace in text columns
      - adds derived columns: tenure_band, high_value (monthly_charges >= 80)
      - drops rows with missing customer_id / churn (defensive cleaning)
3. Runs five documented business queries and prints the results.

Run src/generate_data.py first.
"""

import sqlite3
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "data" / "customers.csv"
DB = BASE / "data" / "churn.db"

QUERIES = {
    "Q1 - Churn rate by contract type": """
        SELECT contract_type,
               COUNT(*)                          AS customers,
               SUM(churn)                        AS churned,
               ROUND(100.0 * AVG(churn), 2)      AS churn_rate_pct
        FROM customers
        GROUP BY contract_type
        ORDER BY churn_rate_pct DESC;
    """,
    "Q2 - Churn rate by tenure band": """
        SELECT tenure_band,
               COUNT(*)                          AS customers,
               SUM(churn)                        AS churned,
               ROUND(100.0 * AVG(churn), 2)      AS churn_rate_pct
        FROM customers
        GROUP BY tenure_band
        ORDER BY MIN(tenure_months);
    """,
    "Q3 - Avg monthly charges: churned vs retained": """
        SELECT churn,
               COUNT(*)                          AS customers,
               ROUND(AVG(monthly_charges), 2)    AS avg_monthly_charges,
               ROUND(AVG(total_charges), 2)      AS avg_total_charges,
               ROUND(AVG(tenure_months), 1)      AS avg_tenure_months
        FROM customers
        GROUP BY churn;
    """,
    "Q4 - Support-call impact on churn": """
        SELECT CASE WHEN support_calls = 0 THEN '0 calls'
                    WHEN support_calls <= 2 THEN '1-2 calls'
                    ELSE '3+ calls' END           AS support_group,
               COUNT(*)                          AS customers,
               ROUND(100.0 * AVG(churn), 2)      AS churn_rate_pct
        FROM customers
        GROUP BY support_group
        ORDER BY MIN(support_calls);
    """,
    "Q5 - Late-payment impact on churn": """
        SELECT CASE WHEN late_payments = 0 THEN '0 late payments'
                    WHEN late_payments = 1 THEN '1 late payment'
                    ELSE '2+ late payments' END  AS late_group,
               COUNT(*)                          AS customers,
               ROUND(100.0 * AVG(churn), 2)      AS churn_rate_pct
        FROM customers
        GROUP BY late_group
        ORDER BY MIN(late_payments);
    """,
}


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"{CSV} not found - run src/generate_data.py first.")

    df = pd.read_csv(CSV)

    conn = sqlite3.connect(DB)
    try:
        # --- Load (raw) ---------------------------------------------------
        df.to_sql("customers_raw", conn, if_exists="replace", index=False)

        # --- Transform: cleaned analytics table ---------------------------
        conn.executescript("""
            DROP TABLE IF EXISTS customers;
            CREATE TABLE customers AS
            SELECT
                customer_id,
                tenure_months,
                monthly_charges,
                total_charges,
                TRIM(contract_type)    AS contract_type,
                TRIM(payment_method)   AS payment_method,
                TRIM(internet_service) AS internet_service,
                support_calls,
                late_payments,
                paperless_billing,
                senior_citizen,
                churn,
                CASE
                    WHEN tenure_months <= 12 THEN '0-12 months'
                    WHEN tenure_months <= 24 THEN '13-24 months'
                    WHEN tenure_months <= 48 THEN '25-48 months'
                    ELSE '49+ months'
                END AS tenure_band,
                CASE WHEN monthly_charges >= 80 THEN 1 ELSE 0 END AS high_value
            FROM customers_raw
            WHERE customer_id IS NOT NULL AND churn IS NOT NULL;
        """)
        conn.commit()

        n = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        print(f"Loaded {n:,} cleaned rows into {DB} (table: customers)\n")

        # --- Analysis queries ----------------------------------------------
        for title, sql in QUERIES.items():
            print("=" * 72)
            print(title)
            print("=" * 72)
            result = pd.read_sql_query(sql, conn)
            print(result.to_string(index=False))
            print()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
