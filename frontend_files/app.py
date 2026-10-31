# ============================================================
# FreightRates Rate Forecast - Streamlit frontend
# ============================================================
import datetime as dt
import os

import pandas as pd
import requests
import streamlit as st

# Backend address: container name on the shared Docker network (overridable via env var)
BACKEND_URL = os.getenv("BACKEND_URL", "http://freight-backend:7860")

ORIGINS = {"USLAX": "Los Angeles", "USLGB": "Long Beach", "USOAK": "Oakland", "USSEA": "Seattle",
           "USTIW": "Tacoma", "USNYC": "New York / New Jersey", "USSAV": "Savannah",
           "USHOU": "Houston", "USCHS": "Charleston"}
DESTINATIONS = {"INNSA": "Nhava Sheva (JNPT)", "INMUN": "Mundra", "INMAA": "Chennai", "INPAV": "Pipavav",
                "INCCU": "Kolkata", "INTKD": "Tughlakabad ICD (Delhi)", "CNSHA": "Shanghai", "CNNGB": "Ningbo",
                "CNYTN": "Yantian (Shenzhen)", "CNTAO": "Qingdao", "CNTXG": "Tianjin Xingang", "CNXMN": "Xiamen"}
CONTAINERS = {"ST20": "20' standard", "ST40": "40' standard", "HC40": "40' high cube"}

st.set_page_config(page_title="Freight Rate Forecast", page_icon="🚢", layout="centered")
st.title("🚢 Freight Rate Forecast")
st.caption("Forecast next week's ocean rate for a US → India / China lane.")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

# ---------------- Online (single) inference ----------------
with tab_single:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Lane")
        origin = st.selectbox("Origin port", list(ORIGINS), format_func=lambda c: f"{c} - {ORIGINS[c]}")
        dest = st.selectbox("Destination", list(DESTINATIONS), format_func=lambda c: f"{c} - {DESTINATIONS[c]}")
        container = st.selectbox("Container", list(CONTAINERS), index=1, format_func=lambda c: f"{c} - {CONTAINERS[c]}")
        today = dt.date.today()
        next_monday = today + dt.timedelta(days=7 - today.weekday())
        week = st.date_input("Week to forecast (Monday)", value=next_monday)
    with col2:
        st.subheader("Recent weekly prices (USD)")
        p1 = st.number_input("1 week before", min_value=1.0, value=1750.0, step=5.0)
        p2 = st.number_input("2 weeks before", min_value=1.0, value=1730.0, step=5.0)
        p3 = st.number_input("3 weeks before", min_value=1.0, value=1760.0, step=5.0)
        p4 = st.number_input("4 weeks before", min_value=1.0, value=1720.0, step=5.0)

    if st.button("Forecast rate", type="primary"):
        payload = {"origin_locode": origin, "dest_locode": dest, "container_type": container,
                   "rate_date": str(week), "price_week_1": p1, "price_week_2": p2,
                   "price_week_3": p3, "price_week_4": p4}
        try:
            resp = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            if resp.status_code == 200:
                out = resp.json()
                st.success(f"Forecast: **USD {out['Predicted_Price']:,.2f}** ({out['Price_Unit']})")
                st.write(f"That is **{out['Predicted_Move_Pct']:+.2f}%** vs the 4-week average of USD {out['Price_Avg_4w']:,.2f}.")
                if out.get("Market_Data_Stale"):
                    st.warning(f"Market index data ends {out['Market_Data_Through']}; the latest week was used.")
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend at {BACKEND_URL}: {e}")

# ---------------- Batch inference ----------------
with tab_batch:
    st.write("Upload a CSV with columns: origin_locode, dest_locode, container_type, rate_date, "
             "price_week_1, price_week_2, price_week_3, price_week_4.")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.dataframe(batch_df.head())
        if st.button("Forecast all rows", type="primary"):
            try:
                resp = requests.post(f"{BACKEND_URL}/v1/predictbatch",
                                     files={"file": uploaded.getvalue()}, timeout=120)
                if resp.status_code == 200:
                    preds = resp.json()
                    batch_df["Predicted_Price"] = [preds[str(i)] for i in range(len(batch_df))]
                    st.success(f"Forecast {len(batch_df)} rows.")
                    st.dataframe(batch_df)
                    st.download_button("Download forecasts", batch_df.to_csv(index=False),
                                       file_name="freight_rate_forecasts.csv", mime="text/csv")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend at {BACKEND_URL}: {e}")
