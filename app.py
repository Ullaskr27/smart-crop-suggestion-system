
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained machine learning model
try:
    model = joblib.load("crop_model.pkl")
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    app.logger.error("Failed to load crop_model.pkl: %s", e)


# Health check endpoint
@app.route("/api/health", methods=["GET"])
def health():
    if model_loaded:
        return jsonify({
            "status": "healthy",
            "model_loaded": True
        }), 200
    return jsonify({
        "status": "degraded",
        "model_loaded": False
    }), 503


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Crop prediction page
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        # Get farmer input
        nitrogen = request.form.get("nitrogen")
        phosphorus = request.form.get("phosphorus")
        potassium = request.form.get("potassium")
        ph = request.form.get("ph")
        temperature = request.form.get("temperature")
        humidity = request.form.get("humidity")
        rainfall = request.form.get("rainfall")

        # Validate nitrogen
        try:
            nitrogen_value = float(nitrogen)
            if nitrogen_value < 0:
                return "Error: Nitrogen cannot be negative."
        except (ValueError, TypeError):
            return "Error: Please enter a valid nitrogen value."

        # Validate phosphorus
        try:
            phosphorus_value = float(phosphorus)
            if phosphorus_value < 0:
                return "Error: Phosphorus cannot be negative."
        except (ValueError, TypeError):
            return "Error: Please enter a valid phosphorus value."

        # Validate potassium
        try:
            potassium_value = float(potassium)
            if potassium_value < 0:
                return "Error: Potassium cannot be negative."
        except (ValueError, TypeError):
            return "Error: Please enter a valid potassium value."

        # Validate soil pH
        try:
            ph_value = float(ph)
            if ph_value < 0 or ph_value > 14:
                return "Error: Soil pH must be between 0 and 14."
        except (ValueError, TypeError):
            return "Error: Please enter a valid soil pH."

        # Validate temperature
        try:
            temperature_value = float(temperature)
            if temperature_value < -50 or temperature_value > 60:
                return "Error: Temperature must be between -50 and 60°C."
        except (ValueError, TypeError):
            return "Error: Please enter a valid temperature."

        # Validate humidity
        try:
            humidity_value = float(humidity)
            if humidity_value < 0 or humidity_value > 100:
                return "Error: Humidity must be between 0 and 100."
        except (ValueError, TypeError):
            return "Error: Please enter valid humidity."

        # Validate rainfall
        try:
            rainfall_value = float(rainfall)
            if rainfall_value < 0:
                return "Error: Rainfall cannot be negative."
        except (ValueError, TypeError):
            return "Error: Please enter a valid rainfall value."

        # Prepare input for the ML model
        input_data = pd.DataFrame(
            [[
                nitrogen_value,
                phosphorus_value,
                potassium_value,
                temperature_value,
                humidity_value,
                ph_value,
                rainfall_value
            ]],
            columns=[
                "N",
                "P",
                "K",
                "temperature",
                "humidity",
                "ph",
                "rainfall"
            ]
        )

        # Predict suitable crop
        predicted_crop = model.predict(input_data)[0]

        # Display results page
        return render_template(
            "result.html",
            nitrogen=nitrogen_value,
            phosphorus=phosphorus_value,
            potassium=potassium_value,
            ph=ph_value,
            temperature=temperature_value,
            humidity=humidity_value,
            rainfall=rainfall_value,
            crop=predicted_crop.title()
        )

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)
