import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model from MLflow
model_uri = "./rossmann-store-sales/lstm_model_final.keras"  # Path to your trained LSTM model saved as .h5
model = load_model(model_uri)

# Title of the web app
st.title("Store Sales Prediction Dashboard")

# Collecting User Inputs
st.header("Enter Key Inputs")

store_id = st.number_input("Store ID", min_value=1, step=1)
date = st.date_input("Select Date")
open_status = st.selectbox("Is Store Open? (1=Yes, 0=No)", [1, 0])
promo = st.selectbox("Promo Active? (1=Yes, 0=No)", [1, 0])
state_holiday = st.selectbox("State Holiday Type", ["0", "a", "b", "c"])
school_holiday = st.selectbox("School Holiday? (1=Yes, 0=No)", [1, 0])

# Auto-calculate date-based features
date_parsed = pd.to_datetime(date)
day_of_week = date_parsed.dayofweek
month = date_parsed.month
day = date_parsed.day
is_holiday = 0  # Set default as 0; You can use a holiday dataset to update this
month_end = int(date_parsed.is_month_end)

# Create DataFrame for prediction
manual_input_df = pd.DataFrame([{
    "Store": store_id,
    "DayOfWeek": day_of_week,
    "Open": open_status,
    "Promo": promo,
    "StateHoliday": state_holiday,
    "SchoolHoliday": school_holiday,
    "Year": date_parsed.year,
    "Month": month,
    "Day": day,
    "WeekOfYear": date_parsed.isocalendar()[1],  # Extract week number
    "IsHoliday": is_holiday,
    "Weekday": 1 if date_parsed.weekday() < 5 else 0,
    "MonthStart": 1 if date_parsed.day <= 10 else 0,
    "MonthMid": 1 if 11 <= date_parsed.day <= 20 else 0,
    "MonthEnd": month_end
}])

# Make Predictions
if st.button("Predict Sales"):
    # Define and fit the scaler on your training data
    scaler = StandardScaler()
    
    # Assuming 'df' is the data you want to scale
    numerical_cols = manual_input_df.select_dtypes(include=['int64', 'float64']).columns
    scaled_features = scaler.fit_transform(manual_input_df[numerical_cols])
    
    # Get the correct number of time steps for LSTM
    time_steps = 45  # Assuming 30 time steps
    samples = scaled_features.shape[0] // time_steps  # Calculate number of samples based on time_steps
    
    # Now reshape your data accordingly (ensure we use only complete samples)
    df_scaled_reshaped = scaled_features[:samples * time_steps].reshape(samples, time_steps, 1)
    
    # Predict sales
    predicted_sales = model.predict(df_scaled_reshaped)

    # Reverse scaling
    predicted_sales_original = scaler.inverse_transform(predicted_sales)

    # Display the predicted sales
    st.subheader(f"Predicted Sales for Store {store_id} on {date}:")
    st.write(f"Predicted Sales: **{predicted_sales_original.flatten()[0]:.2f}**")

    # Visualization
    st.subheader("Sales Prediction Visualization")
    fig, ax = plt.subplots()
    ax.plot([date_parsed], predicted_sales_original.flatten(), marker='o', linestyle='-', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Predicted Sales')
    ax.set_title(f'Sales Prediction for Store {store_id}')
    st.pyplot(fig)

# File Upload for Bulk Prediction
uploaded_file = st.file_uploader("Upload CSV file for bulk prediction", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocess the DataFrame similarly to the manual inputs
    df["Date"] = pd.to_datetime(df["Date"])
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["IsHoliday"] = 0  # Set as default (can be updated with a holiday dataset)
    
    scaler = StandardScaler()
    
    # Scale features (make sure to only scale numerical columns)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df_scaled = scaler.fit_transform(df[numerical_cols])  # Only scale the numerical features
    
    # Replace the scaled numerical columns back into the original DataFrame
    df[numerical_cols] = df_scaled
    time_steps = 45 
    samples = df_scaled.shape[0] // time_steps  # This ensures we use the entire dataset
    
    # Trim the data to ensure we have a multiple of time_steps
    df_scaled_trimmed = df_scaled[:samples * time_steps]  # Remove excess data points
    
    # Now reshape the data to (samples, time_steps, 1)
    df_scaled_reshaped = df_scaled_trimmed.reshape(samples, time_steps, 1)


    # Predict sales for bulk data
    predictions = model.predict(df_scaled_reshaped)

    # Reverse scaling for predictions
    predictions_original = scaler.inverse_transform(predictions)

    # Add predictions to DataFrame
    df["Predicted Sales"] = predictions_original

    # Display predictions
    st.subheader("Bulk Prediction Results")
    st.write(df)

    # Download the predictions as CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")
