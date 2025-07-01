import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

st.title("Revenue Prediction Dashboard")

# Load dataset
df = pd.read_csv("sales.csv")

# Features and target
X = df[["units_sold", "region", "product"]]
y = df["revenue"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(), ["region", "product"])
], remainder="passthrough")

# Model pipeline
model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])
model.fit(X, y)

# User input
units = st.slider("Units Sold", 10, 100)
region = st.selectbox("Region", df["region"].unique())
product = st.selectbox("Product", df["product"].unique())

# Predict
input_df = pd.DataFrame({
    "units_sold": [units],
    "region": [region],
    "product": [product]
})
predicted_revenue = model.predict(input_df)[0]

# Show result
st.metric("Predicted Revenue", f"${predicted_revenue:,.2f}")
