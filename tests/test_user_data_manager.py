import pandas as pd

import app.user_data_manager as manager


def test_user_data_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        manager,
        "USER_DATA_DIR",
        str(tmp_path / "users"),
    )

    monkeypatch.setattr(
        manager,
        "SETTINGS_DIR",
        str(tmp_path / "settings"),
    )

    manager.os.makedirs(manager.USER_DATA_DIR, exist_ok=True)
    manager.os.makedirs(manager.SETTINGS_DIR, exist_ok=True)

    original_df = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "category": ["Transport"],
            "amount": [25.0],
        }
    )

    manager.save_user_data("demo", original_df)

    loaded_df = manager.load_user_data("demo")

    assert len(loaded_df) == 1
    assert loaded_df.iloc[0]["category"] == "Transport"

    settings = {
        "monthly_salary": 5000,
        "default_forecast_days": 30,
        "category_budgets": {"Transport": 500},
        "monthly_savings_goal": 1000,
    }

    manager.save_user_settings("demo", settings)

    assert manager.load_user_settings("demo") == settings
