"""Feature-engineering tests (runnable with pytest OR plain python)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import FEATURE_COLUMNS, build_features  # noqa: E402


def sample(**overrides):
    row = {
        "transaction_id": "txn_t1", "timestamp": "2026-08-15 03:30:00",
        "user_id": "u10001", "amount": 500.0,
        "merchant_category": "electronics", "payment_channel": "online",
        "device_type": "web_browser", "country": "RU",
        "transactions_last_24h": 4, "avg_amount_last_30d": 50.0,
        "is_fraud": 1,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_schema_and_shape():
    X = build_features(sample())
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape[0] == 1
    assert not X.isna().any().any()


def test_amount_ratio_and_night_flag():
    X = build_features(sample())
    # $500 vs a $50 personal average -> ratio must be exactly 10
    assert abs(X.loc[0, "amount_ratio_user"] - 10.0) < 1e-9
    # 03:30 is a night transaction
    assert X.loc[0, "is_night"] == 1.0
    # one-hot encodings pick the right buckets
    assert X.loc[0, "channel_online"] == 1.0
    assert X.loc[0, "country_RU"] == 1.0
    assert X.loc[0, "merchant_electronics"] == 1.0


def test_unknown_category_is_safe():
    X = build_features(sample(country="XX", merchant_category="moon_rocks"))
    # unknown country falls into OTHER; unknown merchant -> all zeros
    assert X.loc[0, "country_OTHER"] == 1.0
    merchant_cols = [c for c in FEATURE_COLUMNS if c.startswith("merchant_")]
    assert X.loc[0, merchant_cols].sum() == 0.0


if __name__ == "__main__":
    test_schema_and_shape()
    test_amount_ratio_and_night_flag()
    test_unknown_category_is_safe()
    print("test_features.py: all 3 tests passed")
