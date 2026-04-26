import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CDSS - Diabetes Prediction",
    layout="wide"
)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


st.markdown("""
<style>

/* background */
.stApp {
    background-color: #f4fbf6;
}

/* top bar */
.topbar {
    background: #a5d6a7;
    padding: 15px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    margin-bottom: 20px;
}

.logo {
    background: #1b5e20;
    color: white;
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: bold;
    margin-right: 15px;
}

.title {
    font-size: 20px;
    font-weight: 600;
    color: #1b5e20;
}

/* card style */
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
}

/* inputs spacing */
input {
    border-radius: 6px !important;
}

/* button */
.stButton > button {
    width: 100%;
    background-color: #66bb6a;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px;
}

.stButton > button:hover {
    background-color: #43a047;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="topbar">
    <div class="logo">CDSS</div>
    <div class="title">Clinical Decision Support System</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Patient Data Entry")

    col1, col2 = st.columns(2)

    with col1:
        preg = st.number_input("Pregnancies", value=0.0)
        bp = st.number_input("Blood Pressure", value=0.0)
        insulin = st.number_input("Insulin", value=0.0)
        dpf = st.number_input("Diabetes Pedigree", value=0.0)

    with col2:
        glucose = st.number_input("Glucose", value=0.0)
        skin = st.number_input("Skin Thickness", value=0.0)
        bmi = st.number_input("BMI", value=0.0)
        age = st.number_input("Age", value=0.0)

    submit = st.button("Run Diagnosis")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Analysis Output")

    if submit:

        input_data = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
            columns=[
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age"
            ])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        if probability[1] > 0.7:
            risk = "High Risk"
        elif probability[1] > 0.4:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"

        st.markdown(f"### Diagnosis: **{result}**")
        st.write(f"**Risk Level:** {risk}")

        st.write("### Probability Scores")
        st.write(f"- Non-Diabetic: {round(probability[0], 2)}")
        st.write(f"- Diabetic: {round(probability[1], 2)}")

        fig, ax = plt.subplots()

        ax.bar(
            ["Non-Diabetic", "Diabetic"],
            probability,
            color=["#81c784", "#e57373"]
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Diabetes Prediction Probability")

        for i, v in enumerate(probability):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center')

        st.pyplot(fig)

    else:
        st.write("Enter patient data to generate prediction.")

    st.markdown('</div>', unsafe_allow_html=True)