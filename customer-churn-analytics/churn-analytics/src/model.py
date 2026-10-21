"""
Churn prediction models.

Trains two classifiers on the customer data:
  1. Logistic Regression (interpretable baseline)
  2. Random Forest (non-linear ensemble)

Pipeline:
  - stratified 80/20 train/test split
  - numeric features standardised (for logistic regression), categorical
    features one-hot encoded
  - class_weight='balanced' to handle the churn/retained imbalance
  - 5-fold cross-validation (ROC-AUC) on the training set
  - final metrics on the held-out test set: accuracy, precision, recall, F1,
    ROC-AUC

Outputs:
  outputs/model_metrics.txt       - metrics table + CV scores
  outputs/feature_importance.png  - random-forest feature importance chart

Reads data/customers.csv (run src/generate_data.py first).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "data" / "customers.csv"
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

NUMERIC = ["tenure_months", "monthly_charges", "total_charges",
           "support_calls", "late_payments"]
BINARY = ["paperless_billing", "senior_citizen"]
CATEGORICAL = ["contract_type", "payment_method", "internet_service"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
TARGET = "churn"


def build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC + BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(name, pipe, X_test, y_test, cv_auc):
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "CV ROC-AUC (mean)": round(cv_auc, 4),
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
    }


def main() -> None:
    df = pd.read_csv(CSV)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Churn rate - train: {y_train.mean():.1%}, test: {y_test.mean():.1%}\n")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted = {}
    for name, model in models.items():
        pipe = build_pipeline(model)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        rows.append(evaluate(name, pipe, X_test, y_test, cv_scores.mean()))
        print(f"{name}: CV ROC-AUC = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    metrics = pd.DataFrame(rows).set_index("Model")
    print("\nTest-set metrics:")
    print(metrics.to_string())

    # --- Feature importance (random forest) ---------------------------------
    rf_pipe = fitted["Random Forest"]
    feature_names = rf_pipe.named_steps["pre"].get_feature_names_out()
    importances = rf_pipe.named_steps["model"].feature_importances_
    fi = (pd.DataFrame({"feature": feature_names, "importance": importances})
          .sort_values("importance", ascending=True))

    print("\nTop features (random forest):")
    print(fi.tail(8).iloc[::-1].to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(fi["feature"], fi["importance"], color="#4C72B0",
            edgecolor="black", linewidth=0.4)
    ax.set_title("Random Forest - Feature Importance for Churn Prediction")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)

    # --- Save metrics ---------------------------------------------------------
    with open(OUT / "model_metrics.txt", "w") as f:
        f.write("Customer Churn Prediction - Model Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {len(df):,} customers (synthetic), "
                f"overall churn rate {y.mean():.1%}\n")
        f.write(f"Split: 80/20 stratified (train {len(X_train):,}, test {len(X_test):,})\n")
        f.write("Class imbalance handled with class_weight='balanced'.\n")
        f.write("Validation: 5-fold stratified cross-validation (ROC-AUC) on train.\n\n")
        f.write(metrics.to_string())
        f.write("\n\nTop random-forest features:\n")
        f.write(fi.tail(8).iloc[::-1].to_string(index=False))
        f.write("\n")

    print(f"\nSaved {OUT / 'model_metrics.txt'} and {OUT / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
