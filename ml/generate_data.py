"""
generate_data.py
----------------
Creates a synthetic but *realistic* dataset of product-return requests.

Why synthetic data?
  Real return-fraud datasets are proprietary. Rather than depend on a Kaggle
  download that might disappear, we generate data whose statistical structure
  mimics the real problem: most returns are genuine, a smaller share are
  "avoidable" (buyer's remorse, wrong size), and a small minority are
  fraudulent (wardrobing, empty-box, serial returners).

  The fraud signal is deliberately *learnable but noisy* -- fraud correlates
  with high historical return rates, high-value items, and very fast or very
  slow return timing, but no single feature is a giveaway. That is what makes
  it a meaningful ML problem rather than a lookup table.

Output: data/returns.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)  # fixed seed -> reproducible dataset

N = 8000  # number of return requests to generate

# Class balance: genuine is the majority class, fraud is rare (realistic).
# Labels: 0 = genuine, 1 = avoidable, 2 = fraudulent
CLASS_PROBS = [0.62, 0.28, 0.10]

PRODUCT_CATEGORIES = [
    "electronics", "clothing", "home", "beauty", "toys", "books", "sports"
]
# Average price per category (used to draw a plausible order value)
CATEGORY_PRICE = {
    "electronics": 220, "clothing": 45, "home": 80, "beauty": 30,
    "toys": 35, "books": 18, "sports": 95,
}

RETURN_REASONS = [
    "defective", "not_as_described", "wrong_item", "changed_mind",
    "better_price_found", "no_longer_needed", "damaged_in_transit",
]


def _draw_row(label: int) -> dict:
    """Draw one synthetic return request conditioned on its true label."""
    category = RNG.choice(PRODUCT_CATEGORIES)
    base_price = CATEGORY_PRICE[category]

    if label == 0:  # genuine
        order_value = max(5, RNG.normal(base_price, base_price * 0.3))
        customer_return_rate = np.clip(RNG.beta(2, 8), 0, 1)      # mostly low
        days_since_purchase = RNG.integers(1, 30)
        prior_orders = RNG.integers(3, 60)
        reason = RNG.choice(
            ["defective", "damaged_in_transit", "wrong_item", "not_as_described"],
            p=[0.35, 0.25, 0.2, 0.2],
        )
        is_high_value = order_value > base_price * 1.5

    elif label == 1:  # avoidable (remorse / sizing -- genuine but preventable)
        order_value = max(5, RNG.normal(base_price, base_price * 0.4))
        customer_return_rate = np.clip(RNG.beta(3, 6), 0, 1)      # medium
        days_since_purchase = RNG.integers(1, 25)
        prior_orders = RNG.integers(2, 40)
        reason = RNG.choice(
            ["changed_mind", "no_longer_needed", "better_price_found", "not_as_described"],
            p=[0.4, 0.3, 0.2, 0.1],
        )
        is_high_value = order_value > base_price * 1.5

    else:  # fraudulent
        # Fraud skews toward high-value items, high historical return rates,
        # and timing that is either suspiciously fast or near the deadline.
        order_value = max(5, RNG.normal(base_price * 1.8, base_price * 0.5))
        customer_return_rate = np.clip(RNG.beta(6, 3), 0, 1)      # high
        days_since_purchase = RNG.choice(
            [RNG.integers(0, 2), RNG.integers(25, 45)]             # fast or late
        )
        prior_orders = RNG.integers(1, 25)
        reason = RNG.choice(
            ["not_as_described", "wrong_item", "changed_mind", "defective"],
            p=[0.35, 0.3, 0.2, 0.15],
        )
        is_high_value = order_value > base_price * 1.5

    return {
        "order_value": round(float(order_value), 2),
        "product_category": category,
        "customer_return_rate": round(float(customer_return_rate), 3),
        "days_since_purchase": int(days_since_purchase),
        "prior_orders": int(prior_orders),
        "return_reason": reason,
        "is_high_value": int(is_high_value),
        "label": label,
    }


def main():
    labels = RNG.choice([0, 1, 2], size=N, p=CLASS_PROBS)
    rows = [_draw_row(int(l)) for l in labels]
    df = pd.DataFrame(rows)

    # Add a little label noise so the model can't hit 100% -- mirrors reality.
    flip_idx = RNG.choice(df.index, size=int(0.03 * N), replace=False)
    df.loc[flip_idx, "label"] = RNG.choice([0, 1, 2], size=len(flip_idx))

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "returns.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
    print("\nClass distribution:")
    print(df["label"].value_counts().sort_index()
          .rename({0: "genuine", 1: "avoidable", 2: "fraudulent"}))


if __name__ == "__main__":
    main()
