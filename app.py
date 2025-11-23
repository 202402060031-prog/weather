from flask import Flask, render_template, request
import numpy as np
import pickle
import datetime as dt

app = Flask(__name__)

# Load model and label encoder
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
        pressure = float(request.form["value4"])         # estimate for temp_min
        
        # ----- Convert inputs to model features -----

        # Feature 1: today's date as ordinal
        today = dt.datetime.now()
        date_ordinal = today.toordinal()

        # Feature 2: precipitation approx
        precipitation = humidity / 100.0

        # Feature 3: temp_max
        temp_max = temperature

        # Feature 4: temp_min (estimate)
        temp_min = temperature - 5

        # Feature 5: wind
        wind = wind_speed

        # Prepare input array
        features = np.array([[date_ordinal, precipitation, temp_max, temp_min, wind]])

        # Predict encoded class
        pred_encoded = model.predict(features)[0]

        # Decode to actual class (sun, rain, fog, snow, drizzle…)
        prediction = le.inverse_transform([pred_encoded])[0]

        return render_template("index.html", result=prediction)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
