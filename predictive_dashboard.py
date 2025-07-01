import streamlit as st
import pandas as pd
import joblib

st.title("Revenue Prediction Dashboard")

model = joblib.load("revenue_model.pkl")

units = st.slider("Units Sold", 10, 100)
region = st.selectbox("Region", ["North", "South", "East", "West"])
product = st.selectbox("Product", ["Widget", "Gadget", "Tool", "Device"])

input_df = pd.DataFrame({
    "units_sold": [units],
    "region": [region],
    "product": [product]
})

predicted_revenue = model.predict(input_df)[0]
st.metric("Predicted Revenue", f"${predicted_revenue:,.2f}")
