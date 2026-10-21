"""Synthetic payment-transaction generator.

The dataset is 100% SYNTHETIC. It is produced by a seeded random generator
(seed = 42) for portfolio / educational purposes only — no real customer
data is involved. Fraud rows are injected with behavioural patterns that
are well documented in the fraud literature (amount anomalies vs. the
user's own history, odd hours, channel / country changes, velocity
bursts), so a model has real signal to learn without using anyone's data.

Rolling features (transactions_last_24h, avg_amount_last_30d) are computed
causally: when a transaction is created it only "sees" that user's
*earlier* transactions, exactly like a production feature store would.
That keeps the dataset free of target leakage by construction.

Run:  python src/generate_data.py
Out:  data/transactions.csv
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "transactions.csv"

SEED = 42
N_TRANSACTIONS = 100_000
N_USERS = 4_000
FRAUD_RATE = 0.018           # ~1.8 % of transactions are fraudulent
START = pd.Timestamp("2026-07-01")
SPAN_DAYS = 90

MERCHANTS = ["groceries", "gas_station", "restaurant", "retail",
             "utilities", "entertainment", "pharmacy", "travel",
             "electronics", "online_marketplace"]
MERCHANT_P = [0.20, 0.13, 0.15, 0.16, 0.09, 0.09, 0.07, 0.03, 0.04, 0.04]
CHANNELS = ["card_present", "online", "atm"]
DEVICES = ["chip", "contactless", "mobile_wallet", "web_browser", "atm_keypad"]
HOME_COUNTRIES = ["US", "CA", "GB", "MX", "DE"]
HOME_P = [0.88, 0.04, 0.03, 0.03, 0.02]
FOREIGN = ["BR", "NG", "RU", "CN", "IN", "TR", "ZA"]

# Hour-of-day weights for legitimate spending (peaks at lunch / evening)
LEGIT_HOUR_P = np.array([2, 1, 1, 1, 1, 2, 4, 6, 7, 7, 7, 7,
                         6, 6, 6, 6, 6, 6, 5, 5, 4, 4, 3, 2], dtype=float)
LEGIT_HOUR_P /= LEGIT_HOUR_P.sum()
FRAUD_HOUR_P = np.array([8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 6], dtype=float)
FRAUD_HOUR_P /= FRAUD_HOUR_P.sum()


def build_users(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [f"u{10000 + i}" for i in range(N_USERS)],
        "home_country": rng.choice(HOME_COUNTRIES, size=N_USERS, p=HOME_P),
        # each user's typical ticket size (lognormal around ~$55)
        "base_amount": rng.lognormal(mean=np.log(55), sigma=0.55, size=N_USERS),
    })


def generate(n: int = N_TRANSACTIONS, seed: int = SEED,
             amount_shift: float = 1.0,
             fraud_rate: float = FRAUD_RATE) -> pd.DataFrame:
    """Generate `n` synthetic transactions.

    amount_shift multiplies every amount (used by the drift check to
    simulate a distribution shift); fraud_rate overrides the base rate.
    """
    rng = np.random.default_rng(seed)
    users = build_users(rng)
    base_amount = users["base_amount"].to_numpy()
    home_country = users["home_country"].to_numpy()

    user_idx = rng.integers(0, N_USERS, size=n)
    is_fraud = rng.random(n) < fraud_rate

    # --- timestamps: random instant in the window, fraud skewed to night
    offsets = rng.random(n) * SPAN_DAYS * 24 * 3600
    hours = np.where(
        is_fraud,
        rng.choice(24, size=n, p=FRAUD_HOUR_P),
        rng.choice(24, size=n, p=LEGIT_HOUR_P),
    )
    timestamps = (START + pd.to_timedelta(offsets, unit="s")).floor("s")
    timestamps = timestamps + pd.to_timedelta(
        (hours - timestamps.hour) % 24, unit="h")

    # --- amounts: legit ~ user's own distribution, fraud 3-8x inflated
    legit_amt = rng.lognormal(mean=0.0, sigma=0.45, size=n) * base_amount[user_idx]
    fraud_amt = legit_amt * rng.uniform(3.0, 8.0, size=n)
    amounts = np.where(is_fraud, fraud_amt, legit_amt) * amount_shift
    amounts = np.clip(amounts, 0.50, 20_000).round(2)

    # --- categorical fields (fraud skews online / foreign / web browser)
    merchants = rng.choice(MERCHANTS, size=n, p=MERCHANT_P)
    channels = np.where(
        is_fraud,
        rng.choice(CHANNELS, size=n, p=[0.25, 0.70, 0.05]),
        rng.choice(CHANNELS, size=n, p=[0.62, 0.30, 0.08]),
    )
    devices = np.where(
        channels == "online",
        rng.choice(["web_browser", "mobile_wallet"], size=n, p=[0.7, 0.3]),
        np.where(channels == "atm", "atm_keypad",
                 rng.choice(["chip", "contactless", "mobile_wallet"],
                            size=n, p=[0.5, 0.3, 0.2])),
    )
    countries = np.where(
        is_fraud & (rng.random(n) < 0.45),
        rng.choice(FOREIGN, size=n),
        home_country[user_idx],
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "user_id": users["user_id"].to_numpy()[user_idx],
        "amount": amounts,
        "merchant_category": merchants,
        "payment_channel": channels,
        "device_type": devices,
        "country": countries,
        "is_fraud": is_fraud.astype(int),
    }).sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # --- causal rolling features per user -------------------------------
    vel_24h, avg_30d = [], []
    for _, grp in df.groupby("user_id", sort=False):
        times, amts = deque(), deque()          # (ts, amount) windows
        all_times, all_amts = [], []
        for ts, amt in zip(grp["timestamp"], grp["amount"]):
            while times and (ts - times[0]).total_seconds() > 24 * 3600:
                times.popleft()
            while amts and (ts - amts[0][0]).total_seconds() > 30 * 24 * 3600:
                amts.popleft()
            vel_24h.append(len(times))
            avg_30d.append(float(np.mean([a for _, a in amts])) if amts else float(amt))
            times.append(ts)
            amts.append((ts, amt))
            all_times.append(ts)
            all_amts.append(amt)
    df["transactions_last_24h"] = vel_24h
    df["avg_amount_last_30d"] = np.round(avg_30d, 2)

    # fraud bursts: a stolen card is usually hit several times in a day
    burst = (df["is_fraud"] == 1) & (rng.random(len(df)) < 0.55)
    df.loc[burst, "transactions_last_24h"] += rng.integers(3, 9, size=burst.sum())

    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id",
              [f"txn_{1_000_000 + i}" for i in range(len(df))])
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df[["transaction_id", "timestamp", "user_id", "amount",
               "merchant_category", "payment_channel", "device_type",
               "country", "transactions_last_24h", "avg_amount_last_30d",
               "is_fraud"]]


def main() -> None:
    df = generate()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    rate = df["is_fraud"].mean()
    print(f"Wrote {len(df):,} rows -> {DATA_PATH}")
    print(f"Fraud rows: {df['is_fraud'].sum():,}  (rate {rate:.2%})")
    print(f"Amount: median ${df['amount'].median():.2f}, "
          f"mean ${df['amount'].mean():.2f}, max ${df['amount'].max():.2f}")


if __name__ == "__main__":
    sys.exit(main())
