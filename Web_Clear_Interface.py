# ==========================
# Segment 0: Page Config
# ==========================
import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# ==========================
# Segment 1: Load Model & Scaler
# ==========================
model = joblib.load("25RP20515_model.joblib")
scaler = joblib.load("25RP20515_scaler.joblib")

# ==========================
# Sidebar
# ==========================
st.sidebar.title("📌 App Information")
st.sidebar.info(
    """
    **Project:** Diabetes Prediction  
    **Model:** Machine Learning Classifier  
    **Student ID:** 25RP20515  
    **Framework:** Streamlit  
    """
)

# ==========================
# App Title
# ==========================
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>
        🩺 Diabetes Prediction System
    </h1>
    <p style='text-align: center; font-size: 16px;'>
        Predict the risk of diabetes using patient health indicators
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================
# User Input Form
# ==========================
st.subheader("🧾 Enter Patient Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("🩸 Glucose Level", min_value=0)
        blood_pressure = st.number_input("💓 Blood Pressure", min_value=0)
        skin_thickness = st.number_input("📏 Skin Thickness", min_value=0)

    with col2:
        insulin = st.number_input("💉 Insulin Level", min_value=0)
        bmi = st.number_input("⚖️ BMI", min_value=0.0, format="%.2f")
        dpf = st.number_input("📊 Diabetes Pedigree Function", min_value=0.0, format="%.3f")
        age = st.number_input("🎂 Age", min_value=1, max_value=120)

    submit = st.form_submit_button("🔍 Predict")

# ==========================
# Prediction Output
# ==========================
if submit:
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.markdown("---")

    if prediction[0] == 1:
        st.error("⚠️ **High Risk of Diabetes**")
    else:
        st.success("✅ **Low Risk of Diabetes**")
