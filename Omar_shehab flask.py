from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# load model
model = pickle.load(open("rf_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # extract model inputs
            features = [
                float(request.form["no_of_adults"]),
                float(request.form["no_of_children"]),
                float(request.form["no_of_weekend_nights"]),
                float(request.form["no_of_week_nights"]),
                float(request.form["type_of_meal"]),
                float(request.form["car_parking_space"]),
                float(request.form["room_type"]),
                float(request.form["lead_time"]),
                float(request.form["market_segment_type"]),
                float(request.form["repeated"]),
                float(request.form["p_c"]),
                float(request.form["p_not_c"]),
                float(request.form["average_price"]),
                float(request.form["special_requests"]),
                float(request.form["year_of_reservation"]),
                float(request.form["month_of_reservation"]),
                float(request.form["day_of_reservation"]),
            ]

            #
            features = np.array(features).reshape(1, -1)

            # predict using model predection
            prediction = model.predict(features)[0]
            prediction = "Canceled" if prediction == 1 else "Not Canceled"
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
