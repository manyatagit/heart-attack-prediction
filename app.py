import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# Load model
model = joblib.load('heart_attack_model (1).pkl')

st.set_page_config(page_title="Heart Attack Risk Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: #d62828;'>💓 Heart Attack Risk Prediction 💓</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Enter patient details below to assess the risk.</p>", unsafe_allow_html=True)

# Space above inputs
st.markdown("---")

# Use a wider layout and custom spacing
col1, col2 = st.columns([1,1])  # Wider inputs, narrow gap  # Equal width columns
st.markdown("""
<style>
    .block-container {
        padding-left: 150px !important;
        padding-right: 150px !important;
    }
    .element-container:has(.stSlider) {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

with col1:
    st.markdown("### Patient Info")
    age = st.slider("Age", 20, 100, 50, help="Patient's age in years")
    gender = st.selectbox("Gender", ["Female", "Male"], help="Male = 1, Female = 0")
    chest_pain = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0 = Typical Angina, 3 = Asymptomatic")
    restingBP = st.number_input("Resting Blood Pressure", 80, 200, 120, help="In mm Hg")
    chol = st.number_input("Serum Cholesterol", 100, 600, 200, help="In mg/dl")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], help="1 = Yes, 0 = No")

with col2:
    st.markdown("### Medical Indicators")
    rest_ecg = st.selectbox("Resting ECG", [0, 1, 2], help="0 = Normal, 2 = Hypertrophy")
    max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150, help="Maximum HR during exercise")
    exercise_angina = st.selectbox("Exercise-Induced Angina", [0, 1], help="1 = Yes, 0 = No")
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, help="ST depression induced by exercise")
    slope = st.selectbox("Slope of ST segment", [0, 1, 2], help="0 = Upsloping, 2 = Downsloping")
    major_vessels = st.slider("Number of Major Vessels", 0, 4, 0, help="Colored by fluoroscopy")
    thal = st.selectbox("Thalassemia", [0, 1, 2], help="0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect")

# Add space between form and prediction
st.markdown("###")

# Prepare input
features = np.array([
    age,
    1 if gender == "Male" else 0,
    chest_pain,
    restingBP,
    chol,
    fasting_bs,
    rest_ecg,
    max_hr,
    exercise_angina,
    oldpeak,
    slope,
    major_vessels,
    thal
]).reshape(1, -1)

# Centered prediction button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

if predict_clicked:
    probability = model.predict_proba(features)[0][1] * 100

    if probability > 50:
        st.error(f"⚠️ High risk of Heart Disease detected! ({probability:.1f}%)")
    else:
        st.success(f"✅ No significant risk detected. ({probability:.1f}%)")

    # Risk gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
        number={'suffix': "%"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if probability > 50 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "#b7e4c7"},
                {'range': [50, 75], 'color': "#ffe066"},
                {'range': [75, 100], 'color': "#f03e3e"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': probability}
        }
    ))
    fig.update_layout(title="Predicted Risk Level")
    st.plotly_chart(fig)
