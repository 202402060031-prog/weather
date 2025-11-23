from flask import Flask, render_template, request
import numpy as np
import pickle
import datetime as dt

app = Flask(_name_)

# Load model + label encoder
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form inputs
        temperature = float(request.form["value1"])      # temp_max
        humidity = float(request.form["value2"])         # used to estimate precipitation
        wind_speed = float(request.form["value3"])       # wind
        pressure = float(request.form["value4"])         # used to estimate temp_min

        # --------------------------
        # Convert UI inputs → model features
        # --------------------------

        # Model Feature 1: date_ordinal → use today's date
        today = dt.datetime.now()
        date_ordinal = today.toordinal()

        # Model Feature 2: precipitation (approx from humidity)
   precipitation = humidity / 100.0

        # Model Feature 3: temp_max (direct from UI)
        temp_max = temperature

        # Model Feature 4: temp_min (estimate: temp_max - 5)
        temp_min = temperature - 5

        # Model Feature 5: wind (direct from UI)
        wind = wind_speed

        # Final feature array
        features = np.array([[date_ordinal, precipitation, temp_max, temp_min, wind]])

        # Predict encoded class
        pred_encoded = model.predict(features)[0]

        # Decode to label (sun, rain, drizzle, fog, snow)
        prediction = le.inverse_transform([pred_encoded])[0]

        return render_template("index.html", result=prediction)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if name == "main":
    app.run(debug=True)