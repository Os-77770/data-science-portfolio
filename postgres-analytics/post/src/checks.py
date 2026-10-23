"""
Reconciliation checks between the source CSVs and the PostgreSQL tables.

I do not trust a load I have not reconciled, so every check compares the
database against an independently computed value from the source files:

  1. Row counts: customers / subscriptions / monthly_billing vs CSVs.
  2. Referential integrity: no orphan subscriptions or billing rows.
  3. Money: total billed in Postgres vs the sum in monthly_billing.csv.
  4. Outcome: churn rate in Postgres vs churn rate in customers.csv.
  5. Consistency: each customer's billing-row count equals their tenure,
     and each customer's max months_since_signup equals tenure - 1.
"""

import pandas as pd
from pathlib import Path

from pgutil import connect

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FAILURES = []


def check(label: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        FAILURES.append(label)


def main() -> None:
    src_customers = pd.read_csv(DATA / "customers.csv")
    src_subs = pd.read_csv(DATA / "subscriptions.csv")
    src_billing = pd.read_csv(DATA / "monthly_billing.csv")

    with connect() as conn:
        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("customers", "subscriptions", "monthly_billing")
        }
        check("row count customers",
              counts["customers"] == len(src_customers),
              f"db={counts['customers']:,} csv={len(src_customers):,}")
        check("row count subscriptions",
              counts["subscriptions"] == len(src_subs),
              f"db={counts['subscriptions']:,} csv={len(src_subs):,}")
        check("row count monthly_billing",
              counts["monthly_billing"] == len(src_billing),
              f"db={counts['monthly_billing']:,} csv={len(src_billing):,}")

        orphans_subs = conn.execute(
            "SELECT COUNT(*) FROM subscriptions s "
            "LEFT JOIN customers c ON c.customer_id = s.customer_id "
            "WHERE c.customer_id IS NULL").fetchone()[0]
        orphans_bill = conn.execute(
            "SELECT COUNT(*) FROM monthly_billing b "
            "LEFT JOIN customers c ON c.customer_id = b.customer_id "
            "WHERE c.customer_id IS NULL").fetchone()[0]
        check("referential integrity", orphans_subs == 0 and orphans_bill == 0,
              f"orphan subscriptions={orphans_subs}, orphan billing rows={orphans_bill}")

        db_total = float(conn.execute("SELECT SUM(amount) FROM monthly_billing").fetchone()[0])
        csv_total = float(src_billing["amount"].sum())
        check("total billed reconciles", abs(db_total - csv_total) < 0.01,
              f"db=${db_total:,.2f} csv=${csv_total:,.2f}")

        db_churn = float(conn.execute(
            "SELECT AVG(CASE WHEN churned THEN 1 ELSE 0 END) FROM customers").fetchone()[0])
        csv_churn = float(src_customers["churn"].mean())
        check("churn rate reconciles", abs(db_churn - csv_churn) < 1e-9,
              f"db={db_churn:.4%} csv={csv_churn:.4%}")

        bad_tenure = conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT b.customer_id FROM monthly_billing b "
            "  JOIN customers c ON c.customer_id = b.customer_id "
            "  GROUP BY b.customer_id, c.tenure_months "
            "  HAVING COUNT(*) <> c.tenure_months "
            "     OR MAX(b.months_since_signup) <> c.tenure_months - 1"
            ") t").fetchone()[0]
        check("billing rows match tenure", bad_tenure == 0,
              f"customers with mismatched billing history: {bad_tenure}")

    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
    print("All reconciliation checks passed.")


if __name__ == "__main__":
    main()
