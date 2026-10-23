"""
Build the warehouse tables (as CSVs) from the flat synthetic customer file.

The churn project works with one flat table: one row per customer. For this
project I reshape that same data into a small star-style warehouse:

  customers        one row per customer (attributes + outcome)
  subscriptions    one row per customer (their plan / contract)
  monthly_billing  one row per customer per month they were active

Derivation rules (all deterministic, seed 7):

- The dataset is a snapshot as of 2024-12-01. A customer's signup month is
  the snapshot month minus (tenure_months - 1) months, so their billing
  history runs from signup month through 2024-12.
- A churned customer's last billing month is their churn month.
- Each month's billed amount is the customer's monthly charge with a small
  +/-3% variation, the same convention the original generator used for
  total charges.

Inputs : data/customers.csv      (from src/generate_data.py)
Outputs: data/customers_warehouse.csv, data/subscriptions.csv,
         data/monthly_billing.csv
"""

import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

SEED = 7
SNAPSHOT = date(2024, 12, 1)

DATA = Path(__file__).resolve().parent.parent / "data"


def shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    """Shift a (year, month) pair by delta months."""
    idx = (year * 12 + (month - 1)) + delta
    return idx // 12, idx % 12 + 1


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA / "customers.csv")

    # ---- customers -------------------------------------------------------
    signup_years, signup_months = [], []
    for tenure in df["tenure_months"]:
        y, m = shift_month(SNAPSHOT.year, SNAPSHOT.month, -(int(tenure) - 1))
        signup_years.append(y)
        signup_months.append(m)

    customers = pd.DataFrame({
        "customer_id": df["customer_id"],
        "senior_citizen": df["senior_citizen"],
        "paperless_billing": df["paperless_billing"],
        "payment_method": df["payment_method"],
        "tenure_months": df["tenure_months"],
        "support_calls": df["support_calls"],
        "late_payments": df["late_payments"],
        "signup_month": [date(y, m, 1) for y, m in zip(signup_years, signup_months)],
        "churned": df["churn"],
    })

    # ---- subscriptions ---------------------------------------------------
    subscriptions = pd.DataFrame({
        "subscription_id": [f"SUB-{i:05d}" for i in range(1, len(df) + 1)],
        "customer_id": df["customer_id"],
        "contract_type": df["contract_type"],
        "internet_service": df["internet_service"],
        "monthly_charges": df["monthly_charges"],
        "total_charges": df["total_charges"],
    })

    # ---- monthly_billing --------------------------------------------------
    tenures = df["tenure_months"].to_numpy()
    offsets = np.arange(tenures.max())
    active = offsets[None, :] < tenures[:, None]          # (n_customers, max_tenure)
    cust_idx, month_offsets = np.nonzero(active)

    base_charges = df["monthly_charges"].to_numpy()[cust_idx]
    amounts = np.round(base_charges * rng.normal(1.0, 0.03, size=len(cust_idx)), 2)

    bill_years = np.empty(len(cust_idx), dtype=int)
    bill_months = np.empty(len(cust_idx), dtype=int)
    s_years = np.array(signup_years)
    s_months = np.array(signup_months)
    for k in range(len(cust_idx)):
        y, m = shift_month(int(s_years[cust_idx[k]]), int(s_months[cust_idx[k]]),
                           int(month_offsets[k]))
        bill_years[k], bill_months[k] = y, m

    monthly_billing = pd.DataFrame({
        "customer_id": df["customer_id"].to_numpy()[cust_idx],
        "billing_month": [date(y, m, 1) for y, m in zip(bill_years, bill_months)],
        "months_since_signup": month_offsets,
        "amount": amounts,
    }).sort_values(["customer_id", "billing_month"]).reset_index(drop=True)

    customers.to_csv(DATA / "customers_warehouse.csv", index=False)
    subscriptions.to_csv(DATA / "subscriptions.csv", index=False)
    monthly_billing.to_csv(DATA / "monthly_billing.csv", index=False)

    print(f"customers:        {len(customers):,} rows")
    print(f"subscriptions:    {len(subscriptions):,} rows")
    print(f"monthly_billing:  {len(monthly_billing):,} rows")
    print(f"billing months:   {monthly_billing['billing_month'].min()} .. "
          f"{monthly_billing['billing_month'].max()}")
    print(f"total billed:     ${monthly_billing['amount'].sum():,.2f}")


if __name__ == "__main__":
    main()
