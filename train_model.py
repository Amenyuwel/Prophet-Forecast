# train_model.py
from prophet import Prophet
import pandas as pd

df = pd.read_csv("sales_data.csv")

model = Prophet()
model.fit(df)

# Forecast 3 months (approx. 90 days) ahead
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Save forecast
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("forecast_data.csv", index=False)
print("✅ Saved forecast to forecast_data.csv")
