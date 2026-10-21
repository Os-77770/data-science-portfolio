"""Offline batch scoring: what would the alert queue have looked like?

Scores a sample of the dataset through the saved model at the
operating threshold and summarises alert volume and (since the
synthetic data is labelled) how many of the alerts were real fraud —
the numbers a fraud-ops manager would plan staffing around.

Run:  python src/batch_score.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_ROWS = 5_000


def main() -> None:
    bundle = joblib.load(PROJECT_ROOT / "models" / "model.joblib")
    threshold = float(json.loads(
        (PROJECT_ROOT / "models" / "threshold.json").read_text())["threshold"])

    df = pd.read_csv(PROJECT_ROOT / "data" / "transactions.csv").head(N_ROWS)
    proba = bundle["model"].predict_proba(build_features(df))[:, 1]
    alerts = proba >= threshold

    lines = [
        "Fraud Detection - Batch Scoring Summary",
        "=" * 60,
        f"Rows scored:            {len(df):,}",
        f"Model:                  {bundle['model_name']}",
        f"Operating threshold:    {threshold:.2f}",
        f"Alerts raised:          {int(alerts.sum()):,} "
        f"({alerts.mean():.2%} of traffic)",
        f"True fraud in sample:   {int(df['is_fraud'].sum()):,} "
        f"({df['is_fraud'].mean():.2%})",
        f"Fraud caught (recall):  "
        f"{df.loc[alerts, 'is_fraud'].sum() / max(df['is_fraud'].sum(), 1):.1%}",
        f"Alert precision:        {df.loc[alerts, 'is_fraud'].mean():.1%}",
        f"Est. review workload:   {int(alerts.sum()):,} alerts x $10 "
        f"= ${int(alerts.sum()) * 10:,}",
    ]
    report = "\n".join(lines)
    print(report)
    (PROJECT_ROOT / "outputs" / "batch_score_summary.txt").write_text(report + "\n")


if __name__ == "__main__":
    sys.exit(main())
