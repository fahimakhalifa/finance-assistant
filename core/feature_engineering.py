from collections.abc import Hashable

import pandas as pd

LAGS = (1, 2, 3, 7, 14, 30)
ROLLING_WINDOWS = (7, 14, 30)


def create_features(
    df: pd.DataFrame,
    group_col: Hashable | None = None,
) -> pd.DataFrame:
    """
    Create leakage-safe time-series features.

    Target-derived features use only observations that occurred before
    the row being predicted.
    """

    required_columns = {"ds", "y"}

    if group_col is not None:
        required_columns.add(group_col)

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(str(column) for column in missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    featured = df.copy()

    featured["ds"] = pd.to_datetime(featured["ds"], errors="coerce")
    featured["y"] = pd.to_numeric(featured["y"], errors="coerce")

    sort_columns = [group_col, "ds"] if group_col is not None else ["ds"]
    featured = featured.sort_values(sort_columns).reset_index(drop=True)

    featured["dayofweek"] = featured["ds"].dt.dayofweek
    featured["quarter"] = featured["ds"].dt.quarter
    featured["month"] = featured["ds"].dt.month
    featured["dayofmonth"] = featured["ds"].dt.day
    featured["weekofyear"] = featured["ds"].dt.isocalendar().week.astype(int)
    featured["is_weekend"] = featured["dayofweek"].isin([5, 6]).astype(int)

    if group_col is None:
        for lag in LAGS:
            featured[f"lag_{lag}"] = featured["y"].shift(lag)

        history = featured["y"].shift(1)

        for window in ROLLING_WINDOWS:
            featured[f"rolling_{window}_mean"] = history.rolling(
                window=window,
                min_periods=window,
            ).mean()

        featured["diff_1"] = featured["y"].shift(1) - featured["y"].shift(2)
        featured["diff_7"] = featured["y"].shift(1) - featured["y"].shift(8)

    else:
        grouped_target = featured.groupby(
            group_col,
            sort=False,
        )["y"]

        for lag in LAGS:
            featured[f"lag_{lag}"] = grouped_target.shift(lag)

        for window in ROLLING_WINDOWS:
            featured[f"rolling_{window}_mean"] = grouped_target.transform(
                lambda series, rolling_window=window: (
                    series.shift(1)
                    .rolling(
                        window=rolling_window,
                        min_periods=rolling_window,
                    )
                    .mean()
                )
            )

        featured["diff_1"] = grouped_target.transform(
            lambda series: series.shift(1) - series.shift(2)
        )

        featured["diff_7"] = grouped_target.transform(
            lambda series: series.shift(1) - series.shift(8)
        )

    feature_columns = [
        column for column in featured.columns if column not in {"ds", "y", group_col}
    ]

    return featured.dropna(subset=feature_columns).reset_index(drop=True)
