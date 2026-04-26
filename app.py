from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

import matplotlib
matplotlib.use("Agg")  # prevents GUI crash in Flask

import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ---------------- INPUT FROM FORM ----------------
        preg = float(request.form["preg"])
        glucose = float(request.form["glucose"])
        bp = float(request.form["bp"])
        skin = float(request.form["skin"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        dpf = float(request.form["dpf"])
        age = float(request.form["age"])

        # ---------------- FIX: FEATURE NAMES ISSUE ----------------
        input_data = pd.DataFrame([[
            preg, glucose, bp, skin, insulin, bmi, dpf, age
        ]], columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ])

        # ---------------- SCALE INPUT ----------------
        input_scaled = scaler.transform(input_data)

        # ---------------- PREDICTION ----------------
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        # ---------------- RISK LEVEL ----------------
        if probability[1] > 0.7:
            risk = "High Risk"
        elif probability[1] > 0.4:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"

        # ---------------- GRAPH GENERATION ----------------
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

        # Convert plot to image
        img = io.BytesIO()
        plt.savefig(img, format="png", bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # ---------------- RETURN RESULT ----------------
        return render_template(
            "index.html",
            prediction_text=result,
            risk_level=risk,
            prob_non=round(probability[0], 2),
            prob_diab=round(probability[1], 2),
            plot_url=plot_url
        )

    except Exception as e:
        return f"Error: {str(e)}"


# Optional: fix favicon 404 (safe)
@app.route("/favicon.ico")
def favicon():
    return "", 204


if __name__ == "__main__":
    app.run(debug=True)