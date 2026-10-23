"""
Create the database, build the schema, and load the warehouse CSVs.

Run order:
    python src/generate_data.py     # data/customers.csv
    python src/build_tables.py      # data/*_warehouse/subscriptions/billing CSVs
    python src/setup_db.py          # schema + load into PostgreSQL

Loading uses psycopg's COPY support (copy from a file object), which is
orders of magnitude faster than row-by-row INSERTs for the ~170k-row
billing fact table.
"""

import subprocess
from pathlib import Path

import psycopg

from pgutil import DB_NAME, DB_USER, DB_PASSWORD, ensure_server

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SQL = ROOT / "sql"


def admin_psql(sql: str) -> str:
    """Run SQL as the postgres superuser via peer auth, SQL fed on stdin."""
    result = subprocess.run(
        ["su", "postgres", "-c", "psql -v ON_ERROR_STOP=1 -tA"],
        input=sql, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def bootstrap_role_and_db() -> None:
    """Create the login role and database if they do not exist yet.

    Runs psql as the postgres OS user (Debian/Ubuntu peer auth) because the
    application role cannot create itself.
    """
    if admin_psql(f"SELECT 1 FROM pg_roles WHERE rolname='{DB_USER}'") != "1":
        admin_psql(f"CREATE ROLE {DB_USER} LOGIN PASSWORD '{DB_PASSWORD}'")
    if admin_psql(f"SELECT 1 FROM pg_database WHERE datname='{DB_NAME}'") != "1":
        admin_psql(f"CREATE DATABASE {DB_NAME} OWNER {DB_USER}")


def main() -> None:
    ensure_server()
    bootstrap_role_and_db()

    schema_sql = (SQL / "01_schema.sql").read_text()

    loads = [
        ("customers", DATA / "customers_warehouse.csv",
         ["customer_id", "senior_citizen", "paperless_billing", "payment_method",
          "tenure_months", "support_calls", "late_payments", "signup_month", "churned"]),
        ("subscriptions", DATA / "subscriptions.csv",
         ["subscription_id", "customer_id", "contract_type", "internet_service",
          "monthly_charges", "total_charges"]),
        ("monthly_billing", DATA / "monthly_billing.csv",
         ["customer_id", "billing_month", "months_since_signup", "amount"]),
    ]

    dsn = (f"host=localhost port=5432 dbname={DB_NAME} "
           f"user={DB_USER} password={DB_PASSWORD}")
    with psycopg.connect(dsn) as conn:
        conn.execute(schema_sql)
        for table, path, cols in loads:
            with conn.cursor() as cur:
                with cur.copy(
                    f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, HEADER true)"
                ) as copy:
                    with open(path, "rb") as f:
                        while chunk := f.read(1 << 20):
                            copy.write(chunk)
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table:16s} loaded: {n:,} rows (from {path.name})")
        conn.commit()
    print("Schema + load complete.")


if __name__ == "__main__":
    main()
