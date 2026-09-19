import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page setup
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")

st.title("💰 Loan Approval Predictor")
st.write(
    "Enter applicant details below to predict whether a loan would be approved, "
    "based on a logistic regression model trained on income, credit score, "
    "loan amount, years employed, and debt-to-income ratio."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income ($)", min_value=1, max_value=1_000_000, value=50000, step=1000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)

with col2:
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=500_000, value=20000, step=1000)
    years_employed = st.number_input("Years Employed", min_value=0, max_value=50, value=5, step=1)

st.divider()

if st.button("Predict Approval", type="primary", use_container_width=True):
    # Calculate the debt-to-income ratio internally — same as during training
    debt_to_income_ratio = loan_amount / income

    # Column order must match training: income, credit_score, loan_amount, years_employed, debt_to_income_ratio
    input_data = np.array([[income, credit_score, loan_amount, years_employed, debt_to_income_ratio]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    confidence = probability[1] if prediction else probability[0]

    if prediction:
        st.success(f"✅ Loan Approved — Confidence: {confidence * 100:.1f}%")
    else:
        st.error(f"❌ Loan Not Approved — Confidence: {confidence * 100:.1f}%")

    st.caption("Model: Logistic Regression | Trained on 2000 loan applications")
    st.caption(f"Debt-to-income ratio: {debt_to_income_ratio:.2f}")

    with st.expander("What drives this decision?"):
        st.write(
            "Based on the trained model's coefficients, **credit score** has the "
            "strongest influence on approval, followed by **debt-to-income ratio** "
            "and **income**. A larger **loan amount** reduces approval odds, while "
            "**years employed** has a smaller positive effect."
        )

st.divider()
st.caption(
    "Built as part of an AI/ML engineering portfolio project — logistic regression "
    "with data leakage detection, feature engineering (debt-to-income ratio), and "
    "model interpretability via coefficient analysis."
)