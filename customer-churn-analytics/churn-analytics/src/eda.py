"""
Exploratory data analysis - generates charts into outputs/:

  1. churn_by_contract.png        - churn rate by contract type (bar)
  2. churn_by_tenure.png          - churn rate by tenure band (bar)
  3. monthly_charges_by_churn.png - monthly-charge distribution by churn (hist)
  4. correlation_heatmap.png      - correlation heatmap of numeric features

Reads data/customers.csv (run src/generate_data.py first).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "data" / "customers.csv"
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
CHURN_COLORS = ["#4C72B0", "#C44E52"]


def main() -> None:
    df = pd.read_csv(CSV)
    df["tenure_band"] = pd.cut(
        df["tenure_months"], bins=[0, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
    )

    # 1 - Churn rate by contract type --------------------------------------
    by_contract = df.groupby("contract_type")["churn"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    by_contract.plot(kind="bar", ax=ax, color=CHURN_COLORS[1], edgecolor="black", linewidth=0.4)
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract type")
    ax.set_ylabel("Churn rate")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", padding=3,
                     labels=[f"{v * 100:.1f}%" for v in by_contract.values])
    ax.set_ylim(0, by_contract.max() * 1.2)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(OUT / "churn_by_contract.png", dpi=150)
    plt.close(fig)

    # 2 - Churn rate by tenure band ----------------------------------------
    by_tenure = df.groupby("tenure_band", observed=True)["churn"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    by_tenure.plot(kind="bar", ax=ax, color=CHURN_COLORS[0], edgecolor="black", linewidth=0.4)
    ax.set_title("Churn Rate by Tenure Band")
    ax.set_xlabel("Tenure band")
    ax.set_ylabel("Churn rate")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", padding=3,
                     labels=[f"{v * 100:.1f}%" for v in by_tenure.values])
    ax.set_ylim(0, by_tenure.max() * 1.2)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(OUT / "churn_by_tenure.png", dpi=150)
    plt.close(fig)

    # 3 - Monthly charges distribution by churn -----------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, color in [(0, CHURN_COLORS[0]), (1, CHURN_COLORS[1])]:
        ax.hist(df.loc[df["churn"] == label, "monthly_charges"], bins=30,
                alpha=0.65, label="Retained" if label == 0 else "Churned",
                color=color, edgecolor="black", linewidth=0.3)
    ax.set_title("Monthly Charges Distribution by Churn")
    ax.set_xlabel("Monthly charges ($)")
    ax.set_ylabel("Customers")
    ax.legend(title="Outcome")
    plt.tight_layout()
    fig.savefig(OUT / "monthly_charges_by_churn.png", dpi=150)
    plt.close(fig)

    # 4 - Correlation heatmap ------------------------------------------------
    numeric = df[["tenure_months", "monthly_charges", "total_charges",
                  "support_calls", "late_payments", "paperless_billing",
                  "senior_citizen", "churn"]]
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap - Numeric Features & Churn")
    plt.tight_layout()
    fig.savefig(OUT / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # Console summary ---------------------------------------------------------
    print("Charts written to", OUT)
    print("\nChurn rate by contract type:")
    print((by_contract * 100).round(2).to_string())
    print("\nChurn rate by tenure band:")
    print((by_tenure * 100).round(2).to_string())
    print("\nCorrelations with churn:")
    print(corr["churn"].drop("churn").sort_values(key=abs, ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
