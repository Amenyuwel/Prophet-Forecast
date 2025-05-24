from prophet import Prophet
import pandas as pd

def update_forecast_with_new_data(new_data_df):
    # Load static + existing data
    static_df = pd.read_csv("sales_data.csv")
    
    # Append new data and remove duplicates
    combined_df = pd.concat([static_df, new_data_df]).drop_duplicates(subset='ds').sort_values('ds')
    combined_df.to_csv("sales_data.csv", index=False)

    # Retrain model
    model = Prophet()
    model.fit(combined_df)

    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("forecast_data.csv", index=False)

    print("✅ Forecast updated with new data")
