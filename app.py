import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature names
model = joblib.load("best_model.pkl")
feature_names = joblib.load("model_features.pkl")

# Page configuration
st.set_page_config(page_title="💼 Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction App")

# Create input fields for each feature
st.header("📝 Enter Employee Details:")

# Dictionary to collect user inputs
user_input = {}

# Define input options for categorical fields
workclass_options = ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Others']
marital_status_options = ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']
occupation_options = ['Adm-clerical', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners',
                      'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty',
                      'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving', 'Others']
relationship_options = ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
race_options = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
gender_options = ['Female', 'Male']
native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Others']

# Input widgets
user_input['age'] = st.number_input("Age", min_value=17, max_value=90, value=30)
user_input['workclass'] = st.selectbox("Workclass", workclass_options)
user_input['fnlwgt'] = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=150000)
user_input['marital-status'] = st.selectbox("Marital Status", marital_status_options)
user_input['occupation'] = st.selectbox("Occupation", occupation_options)
user_input['relationship'] = st.selectbox("Relationship", relationship_options)
user_input['race'] = st.selectbox("Race", race_options)
user_input['gender'] = st.selectbox("Gender", gender_options)
user_input['capital-gain'] = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
user_input['capital-loss'] = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
user_input['hours-per-week'] = st.slider("Hours per Week", min_value=1, max_value=99, value=40)
user_input['native-country'] = st.selectbox("Native Country", native_country_options)
user_input['educational-num'] = st.slider("Educational Number", min_value=1, max_value=20, value=10)

# When user clicks predict
if st.button("🔮 Predict Salary Class"):
    input_df = pd.DataFrame([user_input])

    # Encode categorical variables (same encoding order as training)
    label_enc_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    for col in label_enc_cols:
        input_df[col] = pd.factorize(input_df[col])[0]  # Fast encoding for consistency

    # Reorder columns to match training
    input_df = input_df.reindex(columns=feature_names)

    # Predict
    prediction = model.predict(input_df)[0]

    # Show result
    st.success(f"🧾 Predicted Salary Class: **{prediction}**")

