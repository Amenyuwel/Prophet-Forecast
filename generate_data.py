import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

DATA_DIR = "prophet_data"
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "sales"

os.makedirs(DATA_DIR, exist_ok=True)
output_filepath = os.path.join(DATA_DIR, f"{MODEL_NAME}_data.csv")

periods = 365
np.random.seed(hash(MODEL_NAME) % (2**32 - 1))

# Ensure date_range is timezone-naive from the start
date_range = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods, freq='D')

y_values = None

if MODEL_NAME == "sales":
    trend = np.linspace(100, 250, periods)
    base_weekly_seasonality = np.array([1.0 + 0.2 * (1 if day.weekday() < 5 else -1) for day in date_range])
    base_monthly_seasonality = []
    for day in date_range:
        if day.month in [11, 12, 1]: base_monthly_seasonality.append(1.2)
        elif day.month in [6, 7, 8]: base_monthly_seasonality.append(0.9)
        else: base_monthly_seasonality.append(1.0)
    base_monthly_seasonality = np.array(base_monthly_seasonality)
    weekly_effect = np.array([1.0 + 0.4 if day.weekday() >= 5 else 0.9 for day in date_range]) 
    monthly_effect = base_monthly_seasonality * 1.1 
    base_values = trend * weekly_effect * monthly_effect
    noise = np.random.normal(loc=0, scale=20, size=periods)
    y_values = np.clip(base_values + noise, 30, None).round().astype(int)
elif MODEL_NAME == "part_stock_log":
    trend = np.linspace(-15, 5, periods) 
    weekly_effect = np.array([-2 if day.weekday() < 5 else 1 for day in date_range])
    base_values = trend + weekly_effect
    noise = np.random.normal(loc=0, scale=5, size=periods)
    y_values = np.round(base_values + noise).astype(int)
elif MODEL_NAME == "product_stocks":
    trend = np.linspace(200, 50, periods) 
    replenishment_spikes = np.zeros(periods)
    for i in range(0, periods, 90): 
        replenishment_spikes[i:i+3] = np.random.randint(100, 150)
    base_values = trend + replenishment_spikes
    noise = np.random.normal(loc=0, scale=10, size=periods)
    y_values = np.clip(base_values + noise, 10, None).round().astype(int)
elif MODEL_NAME == "service_request_counts":
    trend = np.linspace(5, 25, periods) 
    weekly_effect = np.array([1.2 if day.weekday() < 5 else 0.7 for day in date_range]) 
    base_values = trend * weekly_effect
    noise = np.random.normal(loc=0, scale=3, size=periods)
    y_values = np.clip(base_values + noise, 0, None).round().astype(int) 
else:
    print(f"Warning: Data generation not specifically defined for model '{MODEL_NAME}'. Using generic pattern.")
    trend = np.linspace(50, 100, periods)
    base_weekly_seasonality = np.array([1.0 + 0.2 * (1 if day.weekday() < 5 else -1) for day in date_range])
    base_values = trend * base_weekly_seasonality
    noise = np.random.normal(loc=0, scale=5, size=periods)
    y_values = np.clip(base_values + noise, 1, None).round().astype(int)

now_iso = datetime.utcnow().isoformat()

df = pd.DataFrame({
    'ds': date_range, # date_range is already timezone-naive
    'y': y_values,
    'created_at': now_iso,
    'updated_at': now_iso
})

# Just to be absolutely sure, though date_range should be naive.
df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

df.to_csv(output_filepath, index=False)
print(f"Success: Generated {output_filepath} for model '{MODEL_NAME}' with {periods} days of data.")
