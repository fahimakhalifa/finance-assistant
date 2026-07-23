import pandas as pd

from core.model_loader import forecast_with_personalization


def test_missing_model_returns_period_total(tmp_path):
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10),
            "category": ["Food"] * 10,
            "amount": list(range(1, 11)),
        }
    )

    forecast, model_description = forecast_with_personalization(
        user_id="demo",
        category="Food",
        df=df,
        n_days_ahead=30,
        global_model_dir=str(tmp_path),
    )

    expected_daily_average = sum(range(4, 11)) / 7
    expected_total = round(expected_daily_average * 30, 2)

    assert forecast == expected_total
    assert model_description == "Recent-average baseline"
