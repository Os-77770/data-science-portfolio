"""Smoke tests for the trained model + API scoring logic.

Runnable with pytest OR plain python (python tests/test_model_smoke.py).
Requires models/model.joblib to exist (run src/train.py first).
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from api import FraudScorer  # noqa: E402
from features import FEATURE_COLUMNS, build_features  # noqa: E402

LEGIT = {
    "transaction_id": "txn_smoke_legit", "timestamp": "2026-08-15 13:05:00",
    "user_id": "u10001", "amount": 42.50, "merchant_category": "groceries",
    "payment_channel": "card_present", "device_type": "chip",
    "country": "US", "transactions_last_24h": 1, "avg_amount_last_30d": 48.0,
}
FRAUDY = {
    "transaction_id": "txn_smoke_fraud", "timestamp": "2026-08-15 03:12:00",
    "user_id": "u10001", "amount": 940.0, "merchant_category": "electronics",
    "payment_channel": "online", "device_type": "web_browser",
    "country": "RU", "transactions_last_24h": 8, "avg_amount_last_30d": 48.0,
}


def test_model_loads_and_predicts_valid_probability():
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    assert bundle["feature_columns"] == FEATURE_COLUMNS
    X = build_features(pd.DataFrame([LEGIT]))
    proba = float(bundle["model"].predict_proba(X)[0, 1])
    assert 0.0 <= proba <= 1.0


def test_scorer_flags_fraud_pattern_over_legit_pattern():
    scorer = FraudScorer()
    legit = scorer.predict_one(dict(LEGIT))
    fraud = scorer.predict_one(dict(FRAUDY))
    assert legit["decision"] in ("approve", "flag")
    assert fraud["fraud_probability"] > legit["fraud_probability"]
    assert fraud["fraud_probability"] > scorer.threshold
    assert fraud["decision"] == "flag"
    assert 0.0 <= legit["fraud_probability"] <= 1.0


if __name__ == "__main__":
    test_model_loads_and_predicts_valid_probability()
    test_scorer_flags_fraud_pattern_over_legit_pattern()
    print("test_model_smoke.py: all tests passed")
