"""Train and compare three fraud classifiers.

Models: logistic regression, random forest, HistGradientBoosting.
Protocol: stratified 80/20 split -> 3-fold cross-validation on the
training set (primary selection metric: PR-AUC, because with a ~1.8%
positive rate ROC-AUC flatters models that rank the huge negative
class well) -> retrain on full training set -> score the held-out test
set. The best model by CV PR-AUC is saved to models/model.joblib along
with its per-transaction test probabilities (used by src/evaluate.py
for the cost-based threshold analysis).

Class imbalance is handled with class_weight='balanced' so the models
pay proportionally more attention to the rare fraud class.

Run:  python src/train.py        (after src/generate_data.py)
Out:  models/model.joblib, outputs/training_metrics.txt,
      outputs/test_predictions.csv
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_COLUMNS, build_features  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "transactions.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "outputs" / "training_metrics.txt"
PREDS_PATH = PROJECT_ROOT / "outputs" / "test_predictions.csv"

RANDOM_STATE = 42


def make_models() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=None,
            class_weight="balanced", random_state=RANDOM_STATE),
    }


def metrics_at(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = build_features(df)
    y = df["is_fraud"].to_numpy()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    print(f"Train {len(X_train):,} | Test {len(X_test):,} | "
          f"fraud rate {y.mean():.2%}")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, model in make_models().items():
        cv_proba = cross_val_predict(model, X_train, y_train, cv=cv,
                                     method="predict_proba")[:, 1]
        cv_pr_auc = average_precision_score(y_train, cv_proba)
        cv_roc = roc_auc_score(y_train, cv_proba)

        model.fit(X_train, y_train)
        test_proba = model.predict_proba(X_test)[:, 1]
        m = metrics_at(y_test, test_proba)
        m.update({"model": name, "cv_pr_auc": cv_pr_auc, "cv_roc_auc": cv_roc})
        rows.append(m)
        fitted[name] = (model, test_proba)
        print(f"{name:26s} CV PR-AUC {cv_pr_auc:.4f} | test PR-AUC "
              f"{m['pr_auc']:.4f} | test ROC-AUC {m['roc_auc']:.4f} | "
              f"recall@0.5 {m['recall']:.3f} | precision@0.5 {m['precision']:.3f}")

    results = pd.DataFrame(rows).set_index("model")
    best_name = results["cv_pr_auc"].idxmax()
    best_model, best_proba = fitted[best_name]
    print(f"\nBest model by CV PR-AUC: {best_name}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "model_name": best_name,
                 "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)

    pd.DataFrame({
        "transaction_id": df.loc[idx_test, "transaction_id"].to_numpy(),
        "amount": df.loc[idx_test, "amount"].to_numpy(),
        "is_fraud": y_test,
        "fraud_probability": best_proba,
    }).to_csv(PREDS_PATH, index=False)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as fh:
        fh.write("Fraud Detection - Training Results\n")
        fh.write("=" * 78 + "\n\n")
        fh.write(f"Dataset: {len(df):,} synthetic transactions, "
                 f"fraud rate {y.mean():.2%}\n")
        fh.write("Split: stratified 80/20 (random_state=42). "
                 "Model selection: 3-fold CV PR-AUC on the training set.\n")
        fh.write("Test-set metrics below use the default 0.50 threshold; "
                 "src/evaluate.py selects the operating threshold.\n\n")
        fh.write(results[["cv_pr_auc", "cv_roc_auc", "accuracy", "precision",
                          "recall", "f1", "roc_auc", "pr_auc"]]
                 .round(4).to_string())
        fh.write(f"\n\nBest model (CV PR-AUC): {best_name}\n")
        fh.write(f"Saved to: {MODEL_PATH}\n")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    sys.exit(main())
