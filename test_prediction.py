
import joblib
import pandas as pd

# Load the saved ML model
model = joblib.load("crop_model.pkl")

print("Model loaded successfully!")

# Example farmer input
input_data = pd.DataFrame(
    [[90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536]],
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

# Predict crop
prediction = model.predict(input_data)

print("\nRecommended Crop:", prediction[0].title())
