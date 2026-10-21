"""Shared feature engineering for training AND serving.

This module is deliberately the *single* place where raw transactions
become model features. `src/train.py` uses it to build the training
matrix and `src/api.py` uses the exact same functions to build the
feature vector for one live transaction. Sharing one implementation is
the standard defence against train/serve skew — the silent bug where a
model is trained on subtly different features from the ones it sees in
production and quietly underperforms.
"""
import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = ["groceries", "gas_station", "restaurant", "retail",
                       "utilities", "entertainment", "pharmacy", "travel",
                       "electronics", "online_marketplace"]
PAYMENT_CHANNELS = ["card_present", "online", "atm"]
DEVICE_TYPES = ["chip", "contactless", "mobile_wallet", "web_browser",
                "atm_keypad"]
COUNTRIES = ["US", "CA", "GB", "MX", "DE", "BR", "NG", "RU", "CN", "IN",
             "TR", "ZA", "OTHER"]

NUMERIC_FEATURES = [
    "log_amount",          # log1p(amount) — tames the long right tail
    "amount_ratio_user",   # amount / user's own 30-day average
    "amount_dev_user",     # amount - user's 30-day average ($)
    "velocity_24h",        # transactions by this user in the last 24 h
    "velocity_log",        # log1p of the same
    "hour_sin", "hour_cos",  # cyclical encoding of hour-of-day
    "is_night",            # 00:00–05:59 flag
    "amount_x_velocity",   # interaction: big amount AND high velocity
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + [f"merchant_{c}" for c in MERCHANT_CATEGORIES]
    + [f"channel_{c}" for c in PAYMENT_CHANNELS]
    + [f"device_{c}" for c in DEVICE_TYPES]
    + [f"country_{c}" for c in COUNTRIES]
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw transactions into the model feature matrix.

    Accepts a DataFrame with the raw schema produced by
    src/generate_data.py (one row is fine — the API passes exactly one).
    Returns a DataFrame with FEATURE_COLUMNS, in that fixed order.
    """
    out = pd.DataFrame(index=df.index)
    amount = df["amount"].astype(float)
    avg30 = df["avg_amount_last_30d"].astype(float).clip(lower=1.0)
    velocity = df["transactions_last_24h"].astype(float)

    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour.astype(float)

    out["log_amount"] = np.log1p(amount)
    out["amount_ratio_user"] = amount / avg30
    out["amount_dev_user"] = amount - avg30
    out["velocity_24h"] = velocity
    out["velocity_log"] = np.log1p(velocity)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["is_night"] = (hour < 6).astype(float)
    out["amount_x_velocity"] = out["log_amount"] * out["velocity_log"]

    for col, cats, prefix in [
        ("merchant_category", MERCHANT_CATEGORIES, "merchant"),
        ("payment_channel", PAYMENT_CHANNELS, "channel"),
        ("device_type", DEVICE_TYPES, "device"),
    ]:
        values = df[col].astype(str)
        for cat in cats:
            out[f"{prefix}_{cat}"] = (values == cat).astype(float)

    country = df["country"].astype(str).where(
        df["country"].astype(str).isin(COUNTRIES), "OTHER")
    for cat in COUNTRIES:
        out[f"country_{cat}"] = (country == cat).astype(float)

    return out[FEATURE_COLUMNS].astype(float)
