from pathlib import Path

import joblib
import pandas as pd

from core.feature_engineering import create_features


def _recent_average_forecast(
    category_df: pd.DataFrame,
    n_days_ahead: int,
) -> float:
    recent_average = float(category_df["y"].tail(7).mean())

    if pd.isna(recent_average):
        return 0.0

    return round(max(recent_average * n_days_ahead, 0.0), 2)


def forecast_with_personalization(
    user_id: str,
    category: str,
    df: pd.DataFrame,
    n_days_ahead: int = 30,
    global_model_dir: str = "models",
    user_model_dir_root: str = "user_models",
) -> tuple[float | None, str]:
    """
    Produce a total forecast for the requested period.

    Uses a global model and an optional user-specific residual model.
    Falls back to a recent-average baseline when model inference is not
    available.
    """

    safe_category = category.lower().replace("/", "_").replace(" ", "_")

    category_df = df[df["category"].astype(str).str.lower() == category.lower()].copy()

    if category_df.empty:
        return None, "No data available for this category"

    category_df = (
        category_df.groupby("date", as_index=False)["amount"]
        .sum()
        .rename(columns={"date": "ds", "amount": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )

    category_df["ds"] = pd.to_datetime(
        category_df["ds"],
        errors="coerce",
    )

    model_path = Path(global_model_dir) / f"{safe_category}_xgb.pkl"

    if not model_path.exists():
        return (
            _recent_average_forecast(category_df, n_days_ahead),
            "Recent-average baseline",
        )

    featured_df = create_features(category_df)

    if featured_df.empty:
        return (
            _recent_average_forecast(category_df, n_days_ahead),
            "Recent-average baseline",
        )

    feature_row = featured_df.drop(columns=["ds", "y"]).iloc[[-1]]

    global_model = joblib.load(model_path)
    daily_forecast = float(global_model.predict(feature_row)[0])

    residual_path = Path(user_model_dir_root) / user_id / f"{safe_category}_residual.pkl"

    model_description = "Global XGBoost"

    if residual_path.exists():
        residual_model = joblib.load(residual_path)
        residual = float(residual_model.predict(feature_row)[0])
        daily_forecast += residual
        model_description = "Global XGBoost + personalized residual"

    total_forecast = round(
        max(daily_forecast, 0.0) * n_days_ahead,
        2,
    )

    return total_forecast, model_description
