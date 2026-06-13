"""
train.py
--------
Trains and evaluates the return-classification model.

This is the ML core of the project. It walks through a standard supervised
classification pipeline, and is written to be *read*, not just run:

  1. Load data
  2. Build a preprocessing pipeline (scale numerics, one-hot encode categoricals)
  3. Train two models and compare them:
       - Logistic Regression  (simple, interpretable baseline)
       - Random Forest        (stronger, handles non-linearity + interactions)
  4. Evaluate with metrics that matter for *imbalanced* data
       (accuracy alone is misleading when fraud is only 10% of cases)
  5. Inspect feature importance (what drives the prediction)
  6. Persist the winning pipeline to model.joblib for the API to serve

Run:  python train.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
DATA_PATH = HERE / "data" / "returns.csv"
MODEL_PATH = HERE / "model.joblib"
METRICS_PATH = HERE / "metrics.json"

LABEL_NAMES = {0: "genuine", 1: "avoidable", 2: "fraudulent"}

# Which raw columns are numeric vs. categorical. The preprocessing pipeline
# treats them differently, so we declare them explicitly.
NUMERIC_FEATURES = [
    "order_value",
    "customer_return_rate",
    "days_since_purchase",
    "prior_orders",
    "is_high_value",
]
CATEGORICAL_FEATURES = ["product_category", "return_reason"]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python generate_data.py` first."
        )
    return pd.read_csv(DATA_PATH)


def build_preprocessor() -> ColumnTransformer:
    """
    Numeric features get standardized (mean 0, variance 1) -- important for
    Logistic Regression, harmless for trees. Categorical features get one-hot
    encoded so the model sees them as separate binary columns rather than
    meaningless integer codes.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(name, model, X_test, y_test) -> dict:
    """Print a full report and return a small dict of headline metrics."""
    preds = model.predict(X_test)

    # macro-F1 weights each class equally, so the rare fraud class still
    # counts -- unlike plain accuracy, which a "predict everything genuine"
    # model could game.
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"Macro F1: {macro_f1:.3f}")
    print("\nPer-class report:")
    print(classification_report(
        y_test, preds, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    ))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test, preds))

    return {"macro_f1": round(float(macro_f1), 4)}


def main():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["label"]

    # Stratified split keeps the class proportions identical in train and test,
    # which matters a lot when one class is rare.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    # --- Model 1: Logistic Regression baseline ---------------------------
    # class_weight="balanced" tells it to pay more attention to the rare
    # fraud class instead of being swamped by the genuine majority.
    logreg = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    logreg.fit(X_train, y_train)
    logreg_metrics = evaluate("Logistic Regression (baseline)", logreg, X_test, y_test)

    # --- Model 2: Random Forest ------------------------------------------
    rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf.fit(X_train, y_train)
    rf_metrics = evaluate("Random Forest", rf, X_test, y_test)

    # --- Pick the winner by macro-F1 -------------------------------------
    if rf_metrics["macro_f1"] >= logreg_metrics["macro_f1"]:
        best_name, best_model, best_metrics = "random_forest", rf, rf_metrics
    else:
        best_name, best_model, best_metrics = "logistic_regression", logreg, logreg_metrics

    print(f"\nSelected model: {best_name} (macro F1 = {best_metrics['macro_f1']})")

    # --- Feature importance (Random Forest only) -------------------------
    # Shows which signals drive predictions -- great talking point and a
    # sanity check that the model learned sensible patterns.
    if best_name == "random_forest":
        feat_names = (
            NUMERIC_FEATURES
            + list(best_model.named_steps["prep"]
                   .named_transformers_["cat"]
                   .get_feature_names_out(CATEGORICAL_FEATURES))
        )
        importances = best_model.named_steps["clf"].feature_importances_
        top = sorted(zip(feat_names, importances), key=lambda t: t[1], reverse=True)[:8]
        print("\nTop feature importances:")
        for name, imp in top:
            print(f"  {name:30s} {imp:.3f}")

    # --- Persist for the API ---------------------------------------------
    joblib.dump(best_model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "selected_model": best_name,
            "logistic_regression": logreg_metrics,
            "random_forest": rf_metrics,
        }, f, indent=2)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
