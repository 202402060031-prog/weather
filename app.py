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
        temperature = float(request.form["value1"])
        humidity = float(request.form["value2"])
        wind_speed = float(request.form["value3"])
        pressure = float(request.form["value4"])

        # Convert date to ordinal
        today = dt.datetime.now()
        date_ordinal = today.toordinal()

        # precipitation from humidity
        precipitation = humidity / 100.0

        # temp_max and temp_min
        temp_max = temperature
        temp_min = temperature - 5

        # wind
        wind = wind_speed

        features = np.array([[date_ordinal, precipitation, temp_max, temp_min, wind]])

        pred_encoded = model.predict(features)[0]
        prediction = le.inverse_transform([pred_encoded])[0]

        return render_template("index.html", result=prediction)

    except Exception as e:
        return render_template("index.html", result="Error: " + str(e))

if __name__ == "__main__":
    app.run()
