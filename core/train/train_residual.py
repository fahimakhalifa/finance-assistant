from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from core.feature_engineering import create_features


def train_residuals_for_user(
    user_id: str,
    user_csv_path: str,
    global_model_dir: str = "models",
    output_dir_root: str = "user_models",
) -> list[dict[str, float | str]]:
    source_df = pd.read_csv(
        user_csv_path,
        parse_dates=["date"],
    )

    output_directory = Path(output_dir_root) / user_id
    output_directory.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float | str]] = []

    for category in sorted(source_df["category"].dropna().unique()):
        safe_category = str(category).lower().replace("/", "_").replace(" ", "_")

        global_model_path = Path(global_model_dir) / f"{safe_category}_xgb.pkl"

        if not global_model_path.exists():
            continue

        category_df = source_df[source_df["category"] == category].copy()

        category_df = (
            category_df.groupby("date", as_index=False)["amount"]
            .sum()
            .rename(columns={"date": "ds", "amount": "y"})
            .sort_values("ds")
            .reset_index(drop=True)
        )

        featured_df = create_features(category_df)

        if len(featured_df) < 20:
            continue

        x = featured_df.drop(columns=["ds", "y"])
        actual = featured_df["y"]

        global_model = joblib.load(global_model_path)
        base_predictions = global_model.predict(x)

        residual_target = actual - base_predictions

        split_index = int(len(featured_df) * 0.8)

        if split_index <= 0 or split_index >= len(featured_df):
            continue

        x_train = x.iloc[:split_index]
        x_validation = x.iloc[split_index:]

        y_train = residual_target.iloc[:split_index]
        y_validation = residual_target.iloc[split_index:]

        residual_model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

        residual_model.fit(
            x_train,
            y_train,
        )

        validation_predictions = residual_model.predict(x_validation)

        validation_mae = mean_absolute_error(
            y_validation,
            validation_predictions,
        )

        model_path = output_directory / f"{safe_category}_residual.pkl"

        joblib.dump(
            residual_model,
            model_path,
        )

        results.append(
            {
                "category": str(category),
                "validation_mae": round(
                    float(validation_mae),
                    4,
                ),
            }
        )

    return results
