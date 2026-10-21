"""Real-time fraud-scoring API (FastAPI).

Loads the trained model (models/model.joblib) and the cost-chosen
operating threshold (models/threshold.json), and scores one transaction
per request through the *same* src/features.py pipeline used in
training — so serving can never drift away from training features.

Run:   uvicorn src.api:app --port 8000     (from the project root)
Call:  curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{...}'
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"


class Transaction(BaseModel):
    transaction_id: str = "txn_live"
    timestamp: str                      # "YYYY-MM-DD HH:MM:SS"
    user_id: str = "u0"
    amount: float
    merchant_category: str
    payment_channel: str                # card_present / online / atm
    device_type: str
    country: str
    transactions_last_24h: int
    avg_amount_last_30d: float


class FraudScorer:
    """Loads the model bundle once and scores single transactions."""

    def __init__(self, model_path: Path = MODEL_PATH,
                 threshold_path: Path = THRESHOLD_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.model_name = bundle["model_name"]
        self.threshold = 0.5
        if threshold_path.exists():
            self.threshold = float(json.loads(
                threshold_path.read_text())["threshold"])

    def predict_one(self, txn: dict) -> dict:
        df = pd.DataFrame([txn])
        X = build_features(df)
        proba = float(self.model.predict_proba(X)[0, 1])
        return {
            "transaction_id": txn.get("transaction_id", "txn_live"),
            "fraud_probability": round(proba, 4),
            "threshold": self.threshold,
            "decision": "flag" if proba >= self.threshold else "approve",
            "model": self.model_name,
        }


scorer = FraudScorer()
app = FastAPI(title="Fraud Detection API", version="1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": scorer.model_name,
            "threshold": scorer.threshold}


@app.post("/predict")
def predict(txn: Transaction) -> dict:
    return scorer.predict_one(txn.model_dump())
