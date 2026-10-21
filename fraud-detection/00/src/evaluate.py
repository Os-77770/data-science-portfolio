"""Threshold selection with an explicit business-cost model.

A fraud model is only useful at a chosen operating threshold. This
script sweeps thresholds over the held-out test predictions from
src/train.py and prices each one:

    cost  =  sum(amount of missed fraud)          (false negatives)
           + $10 x number of flagged legit txns   (false positives:
                                                      analyst review cost)

The cheapest threshold wins and is written to models/threshold.json
(the API reads it at startup). Sweep table, chosen-threshold test
metrics, a confusion-matrix plot and a precision-recall curve are
saved to outputs/.

Run:  python src/evaluate.py     (after src/train.py)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,  # noqa: E402
                             accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDS_PATH = PROJECT_ROOT / "outputs" / "test_predictions.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "evaluation_report.txt"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"
FP_REVIEW_COST = 10.0


def main() -> None:
    preds = pd.read_csv(PREDS_PATH)
    y = preds["is_fraud"].to_numpy()
    proba = preds["fraud_probability"].to_numpy()
    amounts = preds["amount"].to_numpy()

    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sweep = []
    for t in thresholds:
        flagged = proba >= t
        fn_mask = (y == 1) & ~flagged
        fp_mask = (y == 0) & flagged
        sweep.append({
            "threshold": t,
            "precision": precision_score(y, flagged, zero_division=0),
            "recall": recall_score(y, flagged, zero_division=0),
            "alerts": int(flagged.sum()),
            "false_positives": int(fp_mask.sum()),
            "missed_fraud": int(fn_mask.sum()),
            "fn_cost": float(amounts[fn_mask].sum()),
            "fp_cost": float(fp_mask.sum() * FP_REVIEW_COST),
            "total_cost": float(amounts[fn_mask].sum()
                                + fp_mask.sum() * FP_REVIEW_COST),
        })
    sweep_df = pd.DataFrame(sweep)
    best = sweep_df.loc[sweep_df["total_cost"].idxmin()]
    chosen = float(best["threshold"])
    print(f"Chosen threshold: {chosen:.2f} "
          f"(total cost ${best['total_cost']:,.0f} on test set)")

    pred = (proba >= chosen).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    final = {
        "threshold": chosen,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "total_cost": float(best["total_cost"]),
        "baseline_cost_all_fraud_missed": float(amounts[y == 1].sum()),
    }
    final["savings_vs_no_model"] = (1 - final["total_cost"]
                                    / final["baseline_cost_all_fraud_missed"])

    # --- plots --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["legit", "fraud"], cmap="Blues", ax=ax,
        colorbar=False)
    ax.set_title(f"Confusion matrix @ threshold {chosen:.2f}")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "outputs" / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y, proba, ax=ax)
    ax.set_title("Precision-Recall curve (best model, test set)")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "outputs" / "precision_recall_curve.png", dpi=120)
    plt.close(fig)

    with open(REPORT_PATH, "w") as fh:
        fh.write("Fraud Detection - Threshold & Cost Evaluation\n")
        fh.write("=" * 78 + "\n\n")
        fh.write(f"Cost model: false negative = full transaction amount "
                 f"lost; false positive = ${FP_REVIEW_COST:.0f} analyst "
                 f"review.\nBaseline (no model): all fraud missed = "
                 f"${final['baseline_cost_all_fraud_missed']:,.2f} on the "
                 f"test set.\n\n")
        fh.write("Threshold sweep:\n")
        fh.write(sweep_df.round(4).to_string(index=False))
        fh.write(f"\n\nChosen threshold: {chosen:.2f}\n")
        fh.write(f"Test-set results at {chosen:.2f}: "
                 f"accuracy {final['accuracy']:.4f} | precision "
                 f"{final['precision']:.4f} | recall {final['recall']:.4f} "
                 f"| F1 {final['f1']:.4f}\n")
        fh.write(f"Confusion matrix: TP {final['tp']} | FP {final['fp']} | "
                 f"TN {final['tn']} | FN {final['fn']}\n")
        fh.write(f"Total cost ${final['total_cost']:,.2f} -> "
                 f"{final['savings_vs_no_model']:.1%} cheaper than "
                 f"doing nothing.\n")
    with open(THRESHOLD_PATH, "w") as fh:
        json.dump({"threshold": chosen}, fh, indent=2)
    print(f"Saved report -> {REPORT_PATH}")
    print(f"Saved threshold -> {THRESHOLD_PATH}")


if __name__ == "__main__":
    sys.exit(main())
