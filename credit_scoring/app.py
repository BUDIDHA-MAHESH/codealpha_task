import streamlit as st
import joblib
import numpy as np


model = joblib.load('models/credit_model.pkl')

st.title("Credit Scoring Prediction App")

st.write("Enter the following details to predict the credit risk:")


age = st.slider("Age", 18, 100, 30)
job = st.selectbox("Job (0 = unemployed, 1 = unskilled, 2 = skilled, 3 = highly skilled)", [0, 1, 2, 3])
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.slider("Loan Duration (months)", 4, 72, 24)


sex = st.selectbox("Sex", ["male", "female"])
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_account = st.selectbox("Saving Account", ["little", "moderate", "quite rich", "rich", "none"])
checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "none"])
purpose = st.selectbox("Purpose", ["radio/TV", "education", "car", "furniture/equipment", "business", "domestic appliances"])


def encode_input():
    input_data = {
        'Age': age,
        'Job': job,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Sex_male': 1 if sex == 'male' else 0,
        'Housing_own': 1 if housing == 'own' else 0,
        'Housing_rent': 1 if housing == 'rent' else 0,
        'Saving accounts_little': 1 if saving_account == 'little' else 0,
        'Saving accounts_moderate': 1 if saving_account == 'moderate' else 0,
        'Saving accounts_quite rich': 1 if saving_account == 'quite rich' else 0,
        'Saving accounts_rich': 1 if saving_account == 'rich' else 0,
        'Checking account_little': 1 if checking_account == 'little' else 0,
        'Checking account_moderate': 1 if checking_account == 'moderate' else 0,
        'Checking account_rich': 1 if checking_account == 'rich' else 0,
        'Purpose_education': 1 if purpose == 'education' else 0,
        'Purpose_furniture/equipment': 1 if purpose == 'furniture/equipment' else 0,
        'Purpose_radio/TV': 1 if purpose == 'radio/TV' else 0,
        'Purpose_car': 1 if purpose == 'car' else 0,
        'Purpose_business': 1 if purpose == 'business' else 0,
        'Purpose_domestic appliances': 1 if purpose == 'domestic appliances' else 0,
    }

    
    expected_cols = model.feature_names_in_
    input_vector = [input_data.get(col, 0) for col in expected_cols]
    return np.array(input_vector).reshape(1, -1)

if st.button("Predict"):
    features = encode_input()
    prediction = model.predict(features)
    result = "Good Credit Risk" if prediction[0] else "Bad Credit Risk"
    st.success(f"Prediction: {result}")
