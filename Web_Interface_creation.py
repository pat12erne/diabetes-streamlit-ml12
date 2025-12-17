# ==========================
# Segment 1: Import Libraries
# ==========================
import streamlit as st
import joblib
import numpy as np

# ==========================
# Segment 2: Load Model & Scaler
# ==========================
model = joblib.load("25RP20515_model.joblib")
scaler = joblib.load("25RP20515_scaler.joblib")

# ==========================
# Segment 3: App Title
# ==========================
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>
        🩺 25RP20515 Diabetes Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================
# Segment 4: User Input
# ==========================
st.subheader("Enter Patient Details:")

pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("🩸 Glucose Level", min_value=0)
blood_pressure = st.number_input("💓 Blood Pressure", min_value=0)
skin_thickness = st.number_input("📏 Skin Thickness", min_value=0)
insulin = st.number_input("💉 Insulin Level", min_value=0)
bmi = st.number_input("⚖️ BMI", min_value=0.0, format="%.2f")
dpf = st.number_input("📊 Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("🎂 Age", min_value=1, max_value=120)

st.markdown("---")

# ==========================
# Segment 5: Prediction
# ==========================
if st.button("🔍 Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
