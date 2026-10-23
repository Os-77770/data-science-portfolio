"""
Run every analysis query in sql/ against PostgreSQL and save the results.

For each .sql file this:
  - splits the file into its statements (comment lines are stripped),
  - executes each statement with psycopg,
  - prints the result table to the console,
  - saves it as a CSV in outputs/ (e.g. outputs/02_churn_by_segment_1.csv).

It also renders three charts from the saved results and writes a plain
text summary of the headline numbers to outputs/analysis_summary.txt.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from pgutil import connect

ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = ROOT / "sql"
OUT = ROOT / "outputs"

QUERY_FILES = [
    "02_churn_by_segment.sql",
    "03_window_functions.sql",
    "04_cohort_retention.sql",
    "05_risk_segments.sql",
]


def split_statements(sql_text: str) -> list[str]:
    lines = [ln for ln in sql_text.splitlines()
             if not ln.strip().startswith("--")]
    body = "\n".join(lines)
    return [stmt.strip() for stmt in body.split(";") if stmt.strip()]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    results: dict[str, list[pd.DataFrame]] = {}

    with connect() as conn:
        for fname in QUERY_FILES:
            text = (SQL_DIR / fname).read_text()
            frames = []
            for i, stmt in enumerate(split_statements(text), start=1):
                cur = conn.execute(stmt)
                cols = [d.name for d in cur.description]
                frame = pd.DataFrame(cur.fetchall(), columns=cols)
                frames.append(frame)
                out_path = OUT / f"{Path(fname).stem}_{i}.csv"
                frame.to_csv(out_path, index=False)
                print(f"\n===== {fname} — statement {i} "
                      f"({len(frame)} rows) -> {out_path.name} =====")
                with pd.option_context("display.width", 200,
                                       "display.max_columns", 30):
                    print(frame.head(30).to_string(index=False))
            results[fname] = frames

    # ---- charts -----------------------------------------------------------
    seg = results["02_churn_by_segment.sql"][0]
    by_contract = (seg.groupby("contract_type")
                   .apply(lambda g: pd.Series({
                       "churn_rate_pct": (g["churned_customers"].sum()
                                          / g["customers"].sum() * 100)}),
                          include_groups=False)
                   .reset_index())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_contract["contract_type"], by_contract["churn_rate_pct"],
           color=["#c0392b", "#e67e22", "#27ae60"])
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate by contract type (PostgreSQL query 02)")
    for x, y in zip(by_contract["contract_type"], by_contract["churn_rate_pct"]):
        ax.annotate(f"{y:.1f}%", (x, y), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "churn_by_contract.png", dpi=150)
    plt.close(fig)

    rev = results["03_window_functions.sql"][0]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pd.to_datetime(rev["billing_month"]), rev["revenue"], color="#2471a3")
    ax.set_ylabel("Revenue ($)")
    ax.set_title("Monthly billed revenue (PostgreSQL query 03)")
    fig.tight_layout()
    fig.savefig(OUT / "monthly_revenue.png", dpi=150)
    plt.close(fig)

    coh = results["04_cohort_retention.sql"][0]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pd.to_datetime(coh["cohort_quarter"]), coh["churn_rate_pct"],
            marker="o", color="#c0392b")
    ax.set_xlabel("Signup quarter (cohort)")
    ax.set_ylabel("Share of cohort churned by snapshot (%)")
    ax.set_title("Churn rate by signup-quarter cohort (PostgreSQL query 04)")
    fig.tight_layout()
    fig.savefig(OUT / "cohort_churn.png", dpi=150)
    plt.close(fig)

    # ---- headline summary --------------------------------------------------
    overall = float(seg["overall_churn_pct"].iloc[0])
    total_rev = float(rev["revenue"].sum())
    peak = rev.loc[rev["revenue"].idxmax()]
    worst = seg.sort_values("churn_rate_pct", ascending=False).iloc[0]
    risk = results["05_risk_segments.sql"][0]
    worst_risk = risk.iloc[0]
    summary = [
        "PostgreSQL Customer Analytics — headline results",
        "=" * 52,
        f"Overall churn rate:            {overall:.2f}%",
        f"Worst segment:                 {worst['contract_type']} / "
        f"{worst['tenure_band']} at {worst['churn_rate_pct']}%",
        f"Total billed revenue analysed: ${total_rev:,.2f} across "
        f"{len(rev)} months",
        f"Peak revenue month:            {peak['billing_month']} "
        f"(${float(peak['revenue']):,.2f}, "
        f"{int(peak['active_customers'])} active customers)",
        f"Riskiest behaviour bucket:     {worst_risk['support_call_band']} "
        f"support calls + {worst_risk['late_payment_band']} late payments "
        f"-> {worst_risk['churn_rate_pct']}% churn "
        f"({int(worst_risk['customers'])} customers)",
    ]
    text = "\n".join(summary)
    (OUT / "analysis_summary.txt").write_text(text + "\n")
    print("\n" + text)
    print(f"\nCharts + CSVs written to {OUT}")


if __name__ == "__main__":
    main()
