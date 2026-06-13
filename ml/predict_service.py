"""
predict_service.py
------------------
A small Flask service that loads the trained model and exposes a /predict
endpoint. The Node/Express backend calls this; it is kept separate from the
web API on purpose -- in real systems the ML model is usually served as its
own process so it can be scaled, swapped, or retrained independently.

Run:  python predict_service.py   (serves on http://localhost:5001)
"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model.joblib"

LABEL_NAMES = {0: "genuine", 1: "avoidable", 2: "fraudulent"}

app = Flask(__name__)
CORS(app)  # allow the Node backend / local frontend to call this in dev

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "model.joblib not found. Run `python train.py` before starting the service."
    )
model = joblib.load(MODEL_PATH)


def recommend_action(label_name: str, fraud_prob: float) -> str:
    """
    Turn the model output into a business decision.

    This is intentionally simple and rule-based on top of the model's
    probability -- it's the kind of thing a product team would tune. The
    model scores risk; this function decides what to *do* with that score.
    """
    if label_name == "fraudulent" or fraud_prob >= 0.60:
        return "route_to_agent"          # high risk -> human review
    if label_name == "avoidable":
        return "wait_for_item"           # refund once the item is back
    return "instant_refund"              # low risk, genuine -> refund now


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    """
    Expects JSON like:
    {
      "order_value": 240.0,
      "product_category": "electronics",
      "customer_return_rate": 0.7,
      "days_since_purchase": 1,
      "prior_orders": 4,
      "return_reason": "not_as_described",
      "is_high_value": 1
    }
    """
    payload = request.get_json(force=True)

    required = [
        "order_value", "product_category", "customer_return_rate",
        "days_since_purchase", "prior_orders", "return_reason", "is_high_value",
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    X = pd.DataFrame([{k: payload[k] for k in required}])

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    label_name = LABEL_NAMES[pred]
    fraud_prob = float(proba[2])  # index 2 == fraudulent

    return jsonify({
        "label": label_name,
        "probabilities": {LABEL_NAMES[i]: round(float(p), 3) for i, p in enumerate(proba)},
        "fraud_score": round(fraud_prob, 3),
        "recommended_action": recommend_action(label_name, fraud_prob),
    })


if __name__ == "__main__":
    app.run(port=5001, debug=True)
