from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from core.feature_engineering import create_features


def train_global_models(
    data_path: str,
    output_dir: str = "models",
    metrics_path: str = "reports/model_metrics.csv",
) -> list[dict[str, float | str]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(data_path, parse_dates=["date"])

    required_columns = {
        "user_id",
        "date",
        "category",
        "amount",
    }

    missing_columns = required_columns.difference(source_df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing columns: {missing}")

    results: list[dict[str, float | str]] = []

    for category in sorted(source_df["category"].dropna().unique()):
        category_df = source_df[source_df["category"] == category].copy()

        category_df = (
            category_df.groupby(
                ["user_id", "date"],
                as_index=False,
            )["amount"]
            .sum()
            .rename(columns={"date": "ds", "amount": "y"})
        )

        featured_df = create_features(
            category_df,
            group_col="user_id",
        )

        if len(featured_df) < 20:
            print(f"Skipping {category}: insufficient featured rows")
            continue

        unique_dates = sorted(featured_df["ds"].unique())
        split_index = max(1, int(len(unique_dates) * 0.8))
        split_index = min(split_index, len(unique_dates) - 1)
        split_date = unique_dates[split_index]

        train_df = featured_df[featured_df["ds"] < split_date]
        test_df = featured_df[featured_df["ds"] >= split_date]

        if train_df.empty or test_df.empty:
            print(f"Skipping {category}: invalid chronological split")
            continue

        excluded_columns = ["user_id", "ds", "y"]

        x_train = train_df.drop(columns=excluded_columns)
        y_train = train_df["y"]

        x_test = test_df.drop(columns=excluded_columns)
        y_test = test_df["y"]

        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(x_train, y_train)

        predictions = model.predict(x_test)

        model_mae = mean_absolute_error(
            y_test,
            predictions,
        )

        baseline_mae = mean_absolute_error(
            y_test,
            test_df["lag_1"],
        )

        safe_category = str(category).lower().replace("/", "_").replace(" ", "_")

        model_file = output_path / f"{safe_category}_xgb.pkl"
        joblib.dump(model, model_file)

        result = {
            "category": str(category),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "model_mae": round(float(model_mae), 4),
            "baseline_mae": round(float(baseline_mae), 4),
        }

        results.append(result)

        print(f"{category}: model MAE={model_mae:.2f}, baseline MAE={baseline_mae:.2f}")

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(
        metrics_file,
        index=False,
    )

    return results
