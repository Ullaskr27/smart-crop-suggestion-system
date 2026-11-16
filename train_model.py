import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Separate input features
X = data.drop("label", axis=1)

# Separate target/output
y = data["label"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display dataset sizes
print("Total records:", len(data))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
print("\nTraining the model...")

model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Save the trained model
joblib.dump(model, "crop_model.pkl")

print("\nModel saved successfully!")