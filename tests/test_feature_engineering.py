import pandas as pd

from core.feature_engineering import create_features


def test_current_target_does_not_change_historical_features():
    dates = pd.date_range("2026-01-01", periods=40)

    original = pd.DataFrame(
        {
            "ds": dates,
            "y": range(1, 41),
        }
    )

    modified = original.copy()
    modified.loc[39, "y"] = 100_000

    original_features = create_features(original).iloc[-1]
    modified_features = create_features(modified).iloc[-1]

    feature_columns = [column for column in original_features.index if column not in {"ds", "y"}]

    pd.testing.assert_series_equal(
        original_features[feature_columns],
        modified_features[feature_columns],
        check_names=False,
    )


def test_group_features_do_not_cross_user_boundaries():
    dates = pd.date_range("2026-01-01", periods=40)

    user_a = pd.DataFrame(
        {
            "user_id": "a",
            "ds": dates,
            "y": range(1, 41),
        }
    )

    user_b = pd.DataFrame(
        {
            "user_id": "b",
            "ds": dates,
            "y": range(100, 140),
        }
    )

    featured = create_features(
        pd.concat([user_a, user_b], ignore_index=True),
        group_col="user_id",
    )

    last_user_b_row = featured[featured["user_id"] == "b"].iloc[-1]

    assert last_user_b_row["lag_1"] == 138
