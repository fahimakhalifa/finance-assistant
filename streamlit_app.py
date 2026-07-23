import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.budget_tab import render_budget_tab
from app.chatbot_tab import render_chatbot_tab
from app.demo_data import ensure_demo_data
from app.forecast_tab import render_forecast_tab
from app.overview_tab import render_overview_tab
from app.transactions_tab import render_transactions_tab
from app.user_data_manager import (
    load_user_data,
    load_user_settings,
    save_user_settings,
)

load_dotenv()

st.set_page_config(
    page_title="Finance Intelligence Assistant",
    page_icon="📊",
    layout="wide",
)

USER_ID = "demo"

ensure_demo_data(USER_ID)

df = load_user_data(USER_ID)

if df.empty:
    st.error("The synthetic demo dataset could not be loaded.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

df = df.dropna(subset=["date", "category", "amount"])
df = df.sort_values("date").reset_index(drop=True)

settings = load_user_settings(USER_ID)

salary_key = f"{USER_ID}_monthly_salary"
days_key = f"{USER_ID}_default_forecast_days"
budgets_key = f"{USER_ID}_category_budgets"
savings_key = f"{USER_ID}_monthly_savings_goal"

categories = sorted(df["category"].astype(str).unique())

st.session_state.setdefault(
    salary_key,
    int(settings.get("monthly_salary", 5000)),
)

st.session_state.setdefault(
    days_key,
    int(settings.get("default_forecast_days", 30)),
)

st.session_state.setdefault(
    budgets_key,
    settings.get(
        "category_budgets",
        {category: 700.0 for category in categories},
    ),
)

st.session_state.setdefault(
    savings_key,
    int(settings.get("monthly_savings_goal", 1000)),
)

df["month"] = df["date"].dt.to_period("M")
latest_month = df["month"].max()
month_df = df[df["month"] == latest_month]

st.title("📊 Finance Intelligence Assistant")

st.caption(
    "Portfolio demonstration using synthetic financial data. "
    "This application does not provide professional financial advice."
)

with st.sidebar:
    st.markdown("### Portfolio Demo")
    st.write("User: Demo User")
    st.info(
        "All displayed transactions are synthetic. "
        "Do not enter real financial or personally identifiable information."
    )

overview_tab, forecast_tab, budget_tab, transactions_tab, assistant_tab = st.tabs(
    [
        "📊 Overview",
        "🔮 Forecast",
        "💼 Budget",
        "📒 Transactions",
        "🤖 Assistant",
    ]
)

with overview_tab:
    render_overview_tab(df, USER_ID)

with forecast_tab:
    render_forecast_tab(df, USER_ID, categories)

with budget_tab:
    render_budget_tab(
        df,
        USER_ID,
        categories,
        month_df,
    )

with transactions_tab:
    render_transactions_tab(USER_ID)

with assistant_tab:
    render_chatbot_tab(USER_ID)

save_user_settings(
    USER_ID,
    {
        "monthly_salary": int(st.session_state[salary_key]),
        "default_forecast_days": int(st.session_state[days_key]),
        "category_budgets": st.session_state[budgets_key],
        "monthly_savings_goal": int(st.session_state[savings_key]),
    },
)
