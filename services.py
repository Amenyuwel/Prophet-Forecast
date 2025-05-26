import os
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
from pocketbase import PocketBase

# Import configurations from config.py
from config import (
    DATA_DIR, 
    HISTORICAL_WINDOW_DAYS, 
    POCKETBASE_URL, 
    POCKETBASE_COLLECTION_CONFIG
)

def get_data_filepath(model_name):
    return os.path.join(DATA_DIR, f"{model_name}_data.csv")

def get_forecast_filepath(model_name):
    return os.path.join(DATA_DIR, f"{model_name}_forecast.csv")

def _process_and_save_historical_data(model_name: str, new_data_df: pd.DataFrame):
    """
    Helper function to combine new data with existing, apply rolling window, and save.
    Returns the processed DataFrame.
    """
    historical_data_filepath = get_data_filepath(model_name)
    now_iso = datetime.utcnow().isoformat()

    # Ensure 'ds' is datetime and timezone naive for new_data_df
    if 'ds' in new_data_df.columns:
        new_data_df['ds'] = pd.to_datetime(new_data_df['ds']).dt.tz_localize(None)
    
    if 'created_at' not in new_data_df.columns:
        new_data_df['created_at'] = now_iso
    # Always set/update 'updated_at' for new entries or fetched data
    new_data_df['updated_at'] = now_iso

    historical_df_columns = ['ds', 'y', 'created_at', 'updated_at']
    existing_df = pd.DataFrame(columns=historical_df_columns)

    if os.path.exists(historical_data_filepath):
        try:
            existing_df = pd.read_csv(historical_data_filepath)
            for col in ['ds', 'created_at', 'updated_at']: # Ensure date columns are parsed
                if col in existing_df.columns:
                    existing_df[col] = pd.to_datetime(existing_df[col], errors='coerce')
            if 'ds' in existing_df.columns:
                existing_df['ds'] = existing_df['ds'].dt.tz_localize(None)
        except Exception as e:
            print(f"Warning: Could not read or parse existing data file {historical_data_filepath}: {str(e)}. Proceeding as if empty.")
            existing_df = pd.DataFrame(columns=historical_df_columns)

    # Ensure consistent columns for concatenation
    for col in historical_df_columns:
        if col not in new_data_df.columns:
            new_data_df[col] = pd.NaT if 'date' in col or 'ds' in col else (0.0 if col == 'y' else None)
        if col not in existing_df.columns:
            existing_df[col] = pd.NaT if 'date' in col or 'ds' in col else (0.0 if col == 'y' else None)
    
    combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
    
    # Ensure 'ds' and 'updated_at' are datetime for sorting and dropping duplicates
    combined_df['ds'] = pd.to_datetime(combined_df['ds'], errors='coerce').dt.tz_localize(None)
    combined_df['updated_at'] = pd.to_datetime(combined_df['updated_at'], errors='coerce')
    combined_df.dropna(subset=['ds'], inplace=True) # Remove rows where 'ds' is NaT

    combined_df.sort_values(by=['ds', 'updated_at'], ascending=[True, True], inplace=True)
    combined_df.drop_duplicates(subset=['ds'], keep='last', inplace=True)
    
    combined_df.sort_values('ds', ascending=True, inplace=True)
    if len(combined_df) > HISTORICAL_WINDOW_DAYS:
        combined_df = combined_df.tail(HISTORICAL_WINDOW_DAYS)
    
    combined_df.reset_index(drop=True, inplace=True)
    
    # Select only relevant columns for saving
    combined_df_to_save = combined_df[historical_df_columns]
    combined_df_to_save.to_csv(historical_data_filepath, index=False)
    return combined_df_to_save

def _train_and_save_forecast(model_name: str, historical_df: pd.DataFrame):
    """Helper function to train Prophet model and save forecast."""
    forecast_filepath = get_forecast_filepath(model_name)

    if historical_df.empty or len(historical_df) < 20: # Min data points for stable Prophet
        message = f"Not enough data ({len(historical_df)} points) to train model '{model_name}'. Data saved to {get_data_filepath(model_name)}."
        if os.path.exists(forecast_filepath):
            try:
                os.remove(forecast_filepath)
                message += f" Old forecast file {forecast_filepath} removed."
            except OSError as e:
                print(f"Warning: Could not remove old forecast file {forecast_filepath}: {e}")
        return False, message

    try:
        model = Prophet()
        model.fit(historical_df[['ds', 'y']].copy()) # Use .copy()
        future = model.make_future_dataframe(periods=30) # Forecast 30 days
        forecast = model.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_filepath, index=False)
        return True, f"Forecast for model '{model_name}' (re)trained and saved successfully."
    except Exception as e:
        return False, f"Error during model training or prediction for '{model_name}': {str(e)}"

def update_forecast_manually(model_name: str, new_data_list: list):
    """
    Updates forecast by manually providing a list of new data points.
    Applies rolling window.
    """
    if not new_data_list or not isinstance(new_data_list, list):
        return False, "Invalid input: Expecting a list of new data points."

    try:
        new_data_df = pd.DataFrame(new_data_list)
        if 'ds' not in new_data_df.columns or 'y' not in new_data_df.columns:
             return False, "New data must contain 'ds' and 'y' columns."
        new_data_df['ds'] = pd.to_datetime(new_data_df['ds'], errors='coerce')
        new_data_df['y'] = pd.to_numeric(new_data_df['y'], errors='coerce')
        new_data_df.dropna(subset=['ds', 'y'], inplace=True)
    except Exception as e:
        return False, f"Failed to parse new data: {str(e)}"

    if new_data_df.empty:
        return True, "No valid new data points provided after parsing. No update performed."

    processed_historical_df = _process_and_save_historical_data(model_name, new_data_df)
    return _train_and_save_forecast(model_name, processed_historical_df)


def fetch_data_for_month_from_pb(pb_client: PocketBase, model_name: str, target_month_date: datetime.date):
    if model_name not in POCKETBASE_COLLECTION_CONFIG:
        print(f"Error: PocketBase configuration not found for model '{model_name}'.")
        return pd.DataFrame()

    config = POCKETBASE_COLLECTION_CONFIG[model_name]
    collection_name = config["collection_name"]
    ds_field = config["ds_field"]
    y_field = config.get("y_field")
    aggregation_method = config.get("aggregation_method")

    year, month = target_month_date.year, target_month_date.month
    month_start_dt = datetime(year, month, 1)
    month_end_dt = (datetime(year, month + 1, 1) - timedelta(microseconds=1)) if month < 12 else datetime(year, month, 31, 23, 59, 59, 999999)
    
    pb_filter = f"{ds_field} >= '{month_start_dt.strftime('%Y-%m-%d %H:%M:%S')}' && {ds_field} <= '{month_end_dt.strftime('%Y-%m-%d %H:%M:%S')}'"
    if config.get("filter_field") and config.get("filter_value"):
        pb_filter += f" && {config['filter_field']} = '{config['filter_value']}'"

    print(f"Fetching from PB collection '{collection_name}' with filter: {pb_filter}")
    
    try:
        records = pb_client.collection(collection_name).get_full_list(query_params={"filter": pb_filter})
    except Exception as e:
        print(f"Error fetching data from PocketBase for {model_name}: {e}")
        return pd.DataFrame()

    if not records:
        print(f"No records found in PocketBase for {model_name} for {target_month_date.strftime('%Y-%m')}.")
        return pd.DataFrame()

    new_data_points = []
    for record in records:
        try:
            record_ds_val = getattr(record, ds_field)
            ds_date = pd.to_datetime(str(record_ds_val).split(" ")[0]).date() # Get date part

            if aggregation_method == "count":
                new_data_points.append({'ds': ds_date, 'y': 1})
            else:
                record_y_val = float(getattr(record, y_field))
                new_data_points.append({'ds': ds_date, 'y': record_y_val})
        except Exception as e:
            print(f"Error processing record {record.id} for {model_name}: {e}")
    
    if not new_data_points: return pd.DataFrame()
    new_df = pd.DataFrame(new_data_points)
    new_df['ds'] = pd.to_datetime(new_df['ds'])

    if aggregation_method == "count":
        new_df = new_df.groupby('ds').size().reset_index(name='y')
    elif aggregation_method == "sum": # Example for summing daily values
        new_df = new_df.groupby('ds')['y'].sum().reset_index()
        
    return new_df


def update_and_retrain_model_from_db(model_name: str, fetch_target_month: datetime.date):
    if not POCKETBASE_URL:
        return False, "PocketBase URL not configured."
    try:
        pb_client = PocketBase(POCKETBASE_URL)
        # Add authentication if needed:
        # pb_client.admins.auth_with_password(os.getenv("PB_ADMIN_EMAIL"), os.getenv("PB_ADMIN_PASSWORD"))
    except Exception as e:
        return False, f"Failed to initialize PocketBase client: {e}"

    print(f"Attempting to fetch data for model '{model_name}' for month: {fetch_target_month.strftime('%Y-%m')}")
    new_data_df = fetch_data_for_month_from_pb(pb_client, model_name, fetch_target_month)

    if new_data_df.empty:
        return True, f"No new data fetched from PocketBase for '{model_name}' for {fetch_target_month.strftime('%Y-%m')}. Forecast not updated based on new DB data."

    processed_historical_df = _process_and_save_historical_data(model_name, new_data_df)
    success, message = _train_and_save_forecast(model_name, processed_historical_df)
    
    if success:
        return True, f"DB Update: {message} (used data for {fetch_target_month.strftime('%Y-%m')})"
    else:
        return False, f"DB Update: {message}"