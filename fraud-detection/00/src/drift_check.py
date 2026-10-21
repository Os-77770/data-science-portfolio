"""Data-drift check: is tomorrow's traffic still like training data?

Simulates a "new batch" of transactions with a shifted amount
distribution (amount_shift = 1.35 — e.g. inflation, a new market, or a
changed customer mix) and a slightly different fraud rate, then compares
it with the training data using the Population Stability Index (PSI) on
transaction amount, the standard first-line drift metric in production
ML monitoring:

    PSI < 0.10   no significant shift
    0.10 - 0.25  moderate shift — investigate
    > 0.25       significant shift — retrain / recalibrate

Run:  python src/drift_check.py
Out:  outputs/drift_report.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "outputs" / "drift_report.txt"
NEW_SEED = 777
AMOUNT_SHIFT = 1.35
NEW_FRAUD_RATE = 0.026


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    quantiles = np.unique(quantiles)
    e_perc = np.histogram(expected, quantiles)[0] / len(expected)
    a_perc = np.histogram(actual, quantiles)[0] / len(actual)
    e_perc = np.where(e_perc == 0, 0.0001, e_perc)
    a_perc = np.where(a_perc == 0, 0.0001, a_perc)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def verdict(value: float) -> str:
    if value < 0.10:
        return "no significant shift"
    if value < 0.25:
        return "moderate shift - investigate"
    return "SIGNIFICANT shift - retrain / recalibrate"


def main() -> None:
    train = pd.read_csv(PROJECT_ROOT / "data" / "transactions.csv")
    new = generate(n=20_000, seed=NEW_SEED, amount_shift=AMOUNT_SHIFT,
                   fraud_rate=NEW_FRAUD_RATE)

    amount_psi = psi(train["amount"].to_numpy(), new["amount"].to_numpy())
    velocity_psi = psi(train["transactions_last_24h"].to_numpy(),
                       new["transactions_last_24h"].to_numpy())
    lines = [
        "Fraud Detection - Drift Report",
        "=" * 70,
        f"Reference (training) rows: {len(train):,} | "
        f"New batch rows: {len(new):,}",
        f"Simulation: new batch generated with amount_shift="
        f"{AMOUNT_SHIFT} (all amounts x{AMOUNT_SHIFT}) and fraud "
        f"rate {NEW_FRAUD_RATE:.1%}.",
        "",
        f"Amount PSI:            {amount_psi:.4f}  -> {verdict(amount_psi)}",
        f"Velocity PSI:          {velocity_psi:.4f}  -> {verdict(velocity_psi)}",
        "",
        f"Amount mean:   train ${train['amount'].mean():,.2f} -> "
        f"new ${new['amount'].mean():,.2f}",
        f"Amount median: train ${train['amount'].median():,.2f} -> "
        f"new ${new['amount'].median():,.2f}",
        f"Fraud rate:    train {train['is_fraud'].mean():.2%} -> "
        f"new {new['is_fraud'].mean():.2%} "
        f"(label drift this large would also move the optimal "
        f"threshold — re-run src/evaluate.py on fresh labelled data).",
        "",
        "Interpretation: the amount feature has drifted exactly as "
        "simulated, so a production monitor would page the ML owner "
        "to check whether the cause is benign (inflation, new market) "
        "or an attack pattern, and to schedule recalibration / "
        "retraining if the shift persists.",
    ]
    report = "\n".join(lines)
    print(report)
    REPORT_PATH.write_text(report + "\n")


if __name__ == "__main__":
    sys.exit(main())
