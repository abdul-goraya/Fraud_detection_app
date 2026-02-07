#  Fraud Detection ML App  
A machine learning–powered web application that predicts whether a financial transaction is **fraudulent** or **legitimate**, built using **Python, Scikit-Learn, and Streamlit**.

## Project Overview
This project demonstrates an end-to-end implementation of a fraud detection system.  
It includes:

- Data preprocessing  
- Exploratory data analysis (EDA)  
- Feature engineering  
- Machine learning model training  
- A Streamlit web app for real-time predictions  

Fraud detection is a highly imbalanced and sensitive domain, and this project focuses on building an interpretable and efficient baseline model.

---

## Machine Learning Model

### Model Used: Logistic Regression
The model is trained using a **Scikit-Learn Pipeline**, which includes:

- **OneHotEncoder** for categorical features (e.g., `type`)
- **Scaling & preprocessing** of numeric features
- **Class imbalance handling** using `class_weight="balanced"`
- **Feature engineering**, including:
  - `balanceDiffOrig = oldbalanceOrg - newbalanceOrig`
  - `balanceDiffDest = oldbalanceDest - newbalanceDest`

The final trained model is stored as:
fraud_detection_model.pkl
---

## Dataset
The dataset used is:
AIML Dataset.csv (from Kaggle)

### Key columns:
- `type` – Transaction type (CASH_OUT, TRANSFER, etc.)  
- `amount` – Transaction amount  
- `oldbalanceOrg`, `newbalanceOrig` – Sender balances  
- `oldbalanceDest`, `newbalanceDest` – Receiver balances  
- `isFraud` – Target label

The dataset contains a highly **imbalanced** distribution of fraudulent vs legitimate transactions, which is handled directly in the model pipeline.

---

## Streamlit Web App

The Streamlit app (`fraud_detection.py`) allows users to:

- Select transaction type  
- Enter amount and balances  
- Send these inputs into the ML model  
- Receive a prediction:
  - **0 → Legitimate Transaction**  
  - **1 → Fraudulent Transaction**

The app loads the model like this:

```python
model = joblib.load("fraud_detection_model.pkl")
prediction = model.predict(input_data)[0]
