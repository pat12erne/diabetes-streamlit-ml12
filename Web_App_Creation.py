# ==========================
# Segment 0: Page Config and Styling
# ==========================
import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS for background, fonts, spacing
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
        min-height: 100vh;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
    }
    .sidebar .sidebar-content {
        background: #f7f9fc;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4B8BBE;
        color: white;
        font-weight: 600;
        font-size: 16px;
        border-radius: 8px;
        padding: 8px 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3a6d99;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# Segment 1: Load Model & Scaler
# ==========================
model = joblib.load("25RP20515_model.joblib")
scaler = joblib.load("25RP20515_scaler.joblib")

# ==========================
# Sidebar Info with styling
# ==========================
st.sidebar.title("📌 App Information")
st.sidebar.markdown(
    """
    <div style='color:#4B8BBE; font-weight:600; font-size:14px; line-height:1.6;'>
    **Project:** Diabetes Prediction<br>
    **Model:** Machine Learning Classifier<br>
    **Student ID:** 25RP20515<br>
    **Framework:** Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================
# App Title with styling
# ==========================
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE; font-weight: 900; text-shadow: 1px 1px 2px #888; margin-bottom: 5px;'>
        🩺 Diabetes Prediction System
    </h1>
    <p style='text-align: center; font-size: 16px; color: #555; margin-top: 0;'>
        Predict the risk of diabetes using patient health indicators
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================
# User Input Form with tooltips
# ==========================
st.subheader("🧾 Enter Patient Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "🤰 Pregnancies", min_value=0, max_value=20, value=1,
            help="Number of times the patient has been pregnant"
        )
        glucose = st.number_input(
            "🩸 Glucose Level", min_value=0,
            help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test"
        )
        blood_pressure = st.number_input(
            "💓 Blood Pressure", min_value=0,
            help="Diastolic blood pressure (mm Hg)"
        )
        skin_thickness = st.number_input(
            "📏 Skin Thickness", min_value=0,
            help="Triceps skin fold thickness (mm)"
        )

    with col2:
        insulin = st.number_input(
            "💉 Insulin Level", min_value=0,
            help="2-Hour serum insulin (mu U/ml)"
        )
        bmi = st.number_input(
            "⚖️ BMI", min_value=0.0, format="%.2f",
            help="Body mass index (weight in kg/(height in m)^2)"
        )
        dpf = st.number_input(
            "📊 Diabetes Pedigree Function", min_value=0.0, format="%.3f",
            help="Diabetes pedigree function (genetic influence)"
        )
        age = st.number_input(
            "🎂 Age", min_value=1, max_value=120,
            help="Age of the patient (years)"
        )

    submit = st.form_submit_button("🔍 Predict")

st.markdown("---")

# ==========================
# Prediction Output with styled boxes
# ==========================
if submit:
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.markdown(
            """
            <div style='background-color:#FFBABA; color:#D8000C; padding:20px; border-radius:10px; font-weight:bold; font-size:22px; text-align:center;'>
                ⚠️ High Risk of Diabetes
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style='background-color:#DFF2BF; color:#4F8A10; padding:20px; border-radius:10px; font-weight:bold; font-size:22px; text-align:center;'>
                ✅ Low Risk of Diabetes
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==========================
# Footer
# ==========================
st.markdown("---")
st.markdown(
    """
    <p style='text-align:center; color:#888; font-size:12px;'>
        © 2025 25RP20515 — Diabetes Prediction Project
    </p>
    """,
    unsafe_allow_html=True,
)
