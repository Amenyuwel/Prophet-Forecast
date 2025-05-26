# train_model.py
import os
import pandas as pd
from prophet import Prophet
import sys
from config import DATA_DIR # Import DATA_DIR from config

# This MODEL_NAME is used for file paths and error messages specific to this script run.
CURRENT_MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "sales"

# os.makedirs(DATA_DIR, exist_ok=True) # DATA_DIR is created by config.py

data_filepath = os.path.join(DATA_DIR, f"{CURRENT_MODEL_NAME}_data.csv")
forecast_filepath = os.path.join(DATA_DIR, f"{CURRENT_MODEL_NAME}_forecast.csv")

try:
    df = pd.read_csv(data_filepath) 
    if 'ds' not in df.columns:
        print(f"Error: 'ds' column not found in {data_filepath} for model '{CURRENT_MODEL_NAME}'.")
        exit(1)
    
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df.dropna(subset=['ds'], inplace=True) 
    df['ds'] = df['ds'].dt.tz_localize(None)

except FileNotFoundError:
    print(f"Error: Data file {data_filepath} not found for model '{CURRENT_MODEL_NAME}'. Please generate it first.")
    exit(1)
except Exception as e:
    print(f"Error reading or processing data file {data_filepath} for model '{CURRENT_MODEL_NAME}': {str(e)}")
    exit(1)

if df.empty or len(df) < 20: # Increased min for stability
    print(f"Error: Not enough data ({len(df)} points) in {data_filepath} to train the model for '{CURRENT_MODEL_NAME}'. Needs at least 20 data points after processing.")
    exit(1)
if 'y' not in df.columns:
    print(f"Error: 'y' column not found in {data_filepath} for model '{CURRENT_MODEL_NAME}'.")
    exit(1)

try:
    model = Prophet()
    model.fit(df[['ds', 'y']].copy())
    future = model.make_future_dataframe(periods=30) 
    forecast = model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_filepath, index=False)
    print(f"Success: Saved 30-day forecast for '{CURRENT_MODEL_NAME}' to {forecast_filepath}")
except Exception as e:
    print(f"Error during model training or prediction for '{CURRENT_MODEL_NAME}': {str(e)}")
    exit(1)
