"""
Load the warehouse CSVs into DuckDB as raw source tables.

In a warehouse setup these raw tables are what an ingestion tool would
land in the database: untyped, uncleaned, as close to the source files as
possible. All casting, renaming and cleaning happens downstream in the
dbt staging models — that separation is the point of the staging layer.

Creates (in warehouse.duckdb, schema "main"):
    raw_customers, raw_subscriptions, raw_monthly_billing

Run order:
    python src/generate_data.py   # data/customers.csv
    python src/build_tables.py    # data/*_warehouse/subscriptions/billing CSVs
    python src/load_raw.py        # raw_* tables in warehouse.duckdb
    dbt build                     # models + tests
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB_PATH = ROOT / "warehouse.duckdb"

TABLES = {
    "raw_customers": DATA / "customers_warehouse.csv",
    "raw_subscriptions": DATA / "subscriptions.csv",
    "raw_monthly_billing": DATA / "monthly_billing.csv",
}


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    for table, path in TABLES.items():
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(
            f"CREATE TABLE {table} AS "
            f"SELECT * FROM read_csv('{path}', header = true, all_varchar = true)"
        )
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table:20s} loaded: {n:,} rows (from {path.name})")
    con.close()
    print(f"Raw layer written to {DB_PATH.name}")


if __name__ == "__main__":
    main()
