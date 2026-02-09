import streamlit as st
import pickle
import pandas as pd
import os

# Load model with error handling
try:
    if not os.path.exists("fraud_detection_model.pkl"):
        st.error("Model file not found! Please run the analysis notebook first to generate 'fraud_detection_model.pkl'")
        st.stop()
    with open("fraud_detection_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Fraud Detection Prediction App")
st.markdown("Enter the details of the transaction to predict if it is fraudulent or not.")
st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=1000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

if st.button("Predict"):
    try:
        # Create input dataframe with correct column order
        input_data = pd.DataFrame([{
            "type": transaction_type,
            "amount": float(amount),
            "oldbalanceOrg": float(oldbalanceOrg),
            "newbalanceOrig": float(newbalanceOrig),
            "oldbalanceDest": float(oldbalanceDest),
            "newbalanceDest": float(newbalanceDest)
        }])
        
        prediction = model.predict(input_data)[0]
        st.subheader(f"Prediction: '{int(prediction)}'")
        
        if prediction == 1:
            st.error("The transaction is predicted to be FRAUDULENT.")
        else:
            st.success("The transaction is predicted to be LEGITIMATE.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")