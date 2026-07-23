from pathlib import Path

import numpy as np
import pandas as pd

USER_DATA_DIR = Path("data/user_dumps")

CATEGORY_BASELINES = {
    "Market": 28.0,
    "Transport": 14.0,
    "Utilities": 11.0,
    "Restaurants": 19.0,
    "Entertainment": 12.0,
    "Shopping": 17.0,
}


def ensure_demo_data(user_id: str = "demo") -> Path:
    """Create deterministic synthetic finance data for the public demo."""

    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = USER_DATA_DIR / f"{user_id}.csv"

    if output_path.exists():
        return output_path

    rng = np.random.default_rng(42)
    dates = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=180,
        freq="D",
    )

    rows: list[dict[str, object]] = []

    for current_date in dates:
        is_weekend = current_date.dayofweek >= 5

        for category, baseline in CATEGORY_BASELINES.items():
            multiplier = 1.0

            if is_weekend and category in {
                "Restaurants",
                "Entertainment",
                "Shopping",
            }:
                multiplier = 1.35

            monthly_pattern = 1 + 0.10 * np.sin(2 * np.pi * current_date.day / 30)

            amount = rng.normal(
                baseline * multiplier * monthly_pattern,
                baseline * 0.20,
            )

            rows.append(
                {
                    "date": current_date,
                    "category": category,
                    "amount": round(max(float(amount), 0.0), 2),
                }
            )

    demo_df = pd.DataFrame(rows)
    demo_df.to_csv(output_path, index=False)

    return output_path
