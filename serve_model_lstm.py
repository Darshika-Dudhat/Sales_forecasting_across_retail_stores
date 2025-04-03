import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load Model from MLflow 
run_id = "62f6781901b54abfb3ef707ef63af951"  # The correct Run ID from the UI
model_uri = f"runs:/{run_id}/lstm_model"
loaded_model = mlflow.keras.load_model(model_uri)

# Load Test Data (CSV File)
test_file_path = "./rossmann-store-sales/test_data.csv"
test_data = pd.read_csv(test_file_path, parse_dates=["Date"], index_col="Date")

# Print available columns
print("Test Data Columns:", test_data.columns)

# Drop 'Sales' column if it exists in the test data (we predict 'Sales')
test_data = test_data.drop(columns=["Sales", "Id"], errors="ignore")

# List of feature columns from the train model
feature_columns = ["Store","Open","Promo", "DayOfWeek", "StateHoliday", "SchoolHoliday", "Year", "Month", "Day", "WeekOfYear", "IsHoliday"]

# Check if all required feature columns are in the test data
missing_cols = [col for col in feature_columns if col not in test_data.columns]
if missing_cols:
    raise ValueError(f"Missing required feature columns in test data: {missing_cols}")

print("Test Data Columns after removing:", test_data.columns)
# Encode categorical variables (e.g., StateHoliday)
label_encoders = {}
for col in test_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    test_data[col] = le.fit_transform(test_data[col])
    label_encoders[col] = le  # Store encoder for later use

# Scale data using the same scaler from training
scaler = MinMaxScaler(feature_range=(-1, 1))

# Fit the scaler on the training data, then transform the test data
scaled_test_data = scaler.fit_transform(test_data)

# Print the shape of scaled_test_data
print(f"Scaled Test Data Shape: {scaled_test_data.shape}")

# Check if the data has the expected number of features
expected_features = len(feature_columns)
if scaled_test_data.shape[1] != expected_features:
    raise ValueError(f"Mismatch in number of features. Expected {expected_features} features, but got {scaled_test_data.shape[1]}.")


# Reshape for LSTM (Last 42 Days)
time_steps = 42
X_test_reshaped = scaled_test_data[-time_steps:].reshape(1, time_steps, expected_features)

# Make Predictions
predicted_sales_scaled = loaded_model.predict(X_test_reshaped)

# Convert Predictions Back to Original Scale
predicted_sales = scaler.inverse_transform(predicted_sales_scaled)

# Print Result
print(f"📊 Predicted Sales: {predicted_sales.flatten()[0]}")