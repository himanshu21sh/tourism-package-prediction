import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# -------------------------------
# Load Model from Hugging Face
# -------------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="himanshu21sh/tourism-package-prediction",
        filename="best_model_v1.joblib"
    )
    return joblib.load(model_path)

model = load_model()

# -------------------------------
# App Title
# -------------------------------
st.title("🌴 Tourism Package Prediction App")
st.write("Predict whether a customer will purchase the Wellness Tourism Package.")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", 18, 90, 30)
type_of_contact = st.sidebar.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
occupation = st.sidebar.selectbox("Occupation", ['Salaried', 'Freelancer', 'Small Business', 'Large Business'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
number_of_person_visiting = st.sidebar.number_input("Number of Persons Visiting", 1, 10, 1)
preferred_property_star = st.sidebar.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.sidebar.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
number_of_trips = st.sidebar.number_input("Number of Trips Annually", 0, 50, 1)
passport = st.sidebar.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
own_car = st.sidebar.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
number_of_children_visiting = st.sidebar.number_input("Number of Children Visiting", 0, 5, 0)
designation = st.sidebar.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director', 'CEO'])
monthly_income = st.sidebar.number_input("Monthly Income", 10000, 200000, 50000)

st.sidebar.header("Customer Interaction Data")

pitch_satisfaction_score = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
product_pitched = st.sidebar.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
number_of_followups = st.sidebar.number_input("Number of Follow-ups", 0, 10, 2)
duration_of_pitch = st.sidebar.number_input("Duration of Pitch (minutes)", 1, 60, 10)

# -------------------------------
# Prepare Input Data
# -------------------------------
input_data = pd.DataFrame({
    'Age': [age],
    'TypeofContact': [type_of_contact],
    'CityTier': [city_tier],
    'Occupation': [occupation],
    'Gender': [gender],
    'NumberOfPersonVisiting': [number_of_person_visiting],
    'PreferredPropertyStar': [preferred_property_star],
    'MaritalStatus': [marital_status],
    'NumberOfTrips': [number_of_trips],
    'Passport': [passport],
    'OwnCar': [own_car],
    'NumberOfChildrenVisiting': [number_of_children_visiting],
    'Designation': [designation],
    'MonthlyIncome': [monthly_income],
    'PitchSatisfactionScore': [pitch_satisfaction_score],
    'ProductPitched': [product_pitched],
    'NumberOfFollowups': [number_of_followups],
    'DurationOfPitch': [duration_of_pitch]
})

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔍 Predict Purchase"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        result = "✅ Will Purchase Package"
    else:
        result = "❌ Will Not Purchase Package"

    st.subheader("Prediction Result:")
    st.success(result)

    st.subheader("Confidence Score:")
    st.info(f"{round(probability * 100, 2)}% likelihood of purchase")
