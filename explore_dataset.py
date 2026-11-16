
import pandas as pd

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Display first 5 records
print("FIRST 5 ROWS:")
print(data.head())

# Display dataset information
print("\nDATASET INFORMATION:")
data.info()

# Display dataset dimensions
print("\nDATASET SHAPE:")
print(data.shape)

# Display crop names
print("\nAVAILABLE CROPS:")
print(data["label"].unique())

# Check missing values
print("\nMISSING VALUES:")
print(data.isnull().sum())

# Check duplicate records
print("\nDUPLICATE RECORDS:")
print(data.duplicated().sum())

# Count records for each crop
print("\nCROP DISTRIBUTION:")
print(data["label"].value_counts())
