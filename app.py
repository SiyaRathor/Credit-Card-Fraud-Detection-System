import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("model_columns.pkl")

st.title("Credit Card Fraud Detection System")

st.write("Enter transaction details to check if it is Fraud or Legitimate.")

# Create empty dataframe with correct columns
input_data = pd.DataFrame(columns=columns)
input_data.loc[0] = 0

# Important fields for user input
amount = st.number_input("Transaction Amount", min_value=0.0)
time = st.number_input("Transaction Time", min_value=0.0)

# Fill values
if "Amount" in input_data.columns:
    input_data["Amount"] = amount

if "Time" in input_data.columns:
    input_data["Time"] = time

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction button
if st.button("Check Transaction"):
    
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = (prob > 0.8).astype(int)

    st.write("Fraud Probability:", prob)

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("Transaction is Legitimate")