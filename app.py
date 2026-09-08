"""
app.py — Rossmann Store Sales Forecast

Loads the model + preprocessors trained in rossmann_sales_forecast.ipynb
(xgb_model.pkl, scaler.pkl, encoder.pkl — all in this same folder) and
predicts daily Sales for a chosen store/date/promo scenario.

Run with:
    streamlit run app.py
"""

import pickle

import pandas as pd
import streamlit as st

NUMERIC_COLS = [
    "Store", "Promo", "SchoolHoliday", "CompetitionDistance", "CompetitionOpen",
    "Promo2", "Promo2Open", "IsPromoMonth", "Day", "Month", "Year", "WeekOfYear",
]
CATEGORICAL_COLS = ["DayOfWeek", "StateHoliday", "StoreType", "Assortment"]


@st.cache_resource
def load_artifacts():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, scaler, encoder


def predict_sales(features: dict, model, scaler, encoder) -> float:
    df = pd.DataFrame([features])
    max_distance = df["CompetitionDistance"].max()
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(max_distance)
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    encoded_cols = list(encoder.get_feature_names_out(CATEGORICAL_COLS))
    df[encoded_cols] = encoder.transform(df[CATEGORICAL_COLS])
    X = df[NUMERIC_COLS + encoded_cols]
    return float(model.predict(X)[0])


st.set_page_config(page_title="Rossmann Sales Forecast", page_icon="📈")
st.title("📈 Rossmann Store Sales Forecast")
st.caption("XGBoost regressor trained on the Kaggle Rossmann Store Sales dataset.")

try:
    model, scaler, encoder = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run rossmann_sales_forecast.ipynb first to generate xgb_model.pkl, scaler.pkl and encoder.pkl.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    store = st.number_input("Store ID", min_value=1, max_value=1115, value=1)
    date = st.date_input("Date", value=pd.Timestamp("2015-07-15"))
    store_type = st.selectbox("Store type", ["a", "b", "c", "d"])
    assortment = st.selectbox("Assortment", ["a", "b", "c"])
    state_holiday = st.selectbox("State holiday", ["0", "a", "b", "c"], format_func=lambda x: "None" if x == "0" else x)
    school_holiday = st.checkbox("School holiday")
with col2:
    promo = st.checkbox("Running promo today", value=True)
    promo2 = st.checkbox("Enrolled in Promo2 (recurring promo)")
    promo2_open_months = st.number_input("Months Promo2 has run", min_value=0, value=0, disabled=not promo2)
    is_promo_month = st.checkbox("Current month is inside the Promo2 interval", disabled=not promo2)
    competition_distance = st.number_input("Distance to nearest competitor (m)", min_value=0.0, value=1270.0)
    competition_open_months = st.number_input("Months competitor has been open", min_value=0, value=24)

if st.button("Predict sales", type="primary"):
    features = {
        "Store": store,
        "DayOfWeek": date.isoweekday(),
        "Promo": int(promo),
        "StateHoliday": state_holiday,
        "SchoolHoliday": int(school_holiday),
        "StoreType": store_type,
        "Assortment": assortment,
        "CompetitionDistance": competition_distance,
        "CompetitionOpen": competition_open_months,
        "Day": date.day,
        "Month": date.month,
        "Year": date.year,
        "WeekOfYear": int(pd.Timestamp(date).isocalendar().week),
        "Promo2": int(promo2),
        "Promo2Open": promo2_open_months if promo2 else 0,
        "IsPromoMonth": int(is_promo_month) if promo2 else 0,
    }
    prediction = predict_sales(features, model, scaler, encoder)
    st.metric("Predicted Sales", f"€{prediction:,.0f}")
