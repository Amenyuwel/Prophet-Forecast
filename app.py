import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Import from local modules
from config import POCKETBASE_COLLECTION_CONFIG # For validation
import services # Import the services module

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

@app.route('/forecast/<model_name>')
def get_forecast_route(model_name):
    forecast_filepath = services.get_forecast_filepath(model_name)
    try:
        df = pd.read_csv(forecast_filepath) 
        # Ensure 'ds' is string for JSON, Prophet output is usually fine
        df['ds'] = pd.to_datetime(df['ds']).dt.strftime('%Y-%m-%d')
        return df.to_json(orient="records")
    except FileNotFoundError:
        return jsonify({"error": f"Forecast for model '{model_name}' not found."}), 404
    except Exception as e:
        return jsonify({"error": f"Error reading forecast data for '{model_name}': {str(e)}"}), 500

@app.route('/historical_data/<model_name>')
def get_historical_data_route(model_name):
    data_filepath = services.get_data_filepath(model_name)
    try:
        df = pd.read_csv(data_filepath)
        # Ensure date columns are strings for JSON
        for col in ['ds', 'created_at', 'updated_at']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ') # ISO format
        return df.to_json(orient="records")
    except FileNotFoundError:
        return jsonify({"error": f"Historical data for model '{model_name}' not found."}), 404
    except Exception as e:
        return jsonify({"error": f"Error reading historical data for '{model_name}': {str(e)}"}), 500

@app.route('/update_forecast/<string:model_name>', methods=['POST'])
def manual_update_forecast_route(model_name):
    if model_name not in POCKETBASE_COLLECTION_CONFIG: # Basic validation
        return jsonify({"error": f"Model '{model_name}' is not a configured model."}), 404
        
    new_data_list = request.get_json()
    if not isinstance(new_data_list, list):
        return jsonify({"error": "JSON payload must be a list of data points."}), 400
    
    success, message = services.update_forecast_manually(model_name, new_data_list)
    
    if success:
        return jsonify({"message": message}), 200
    else:
        return jsonify({"error": message}), 500

@app.route('/trigger_monthly_update/<string:model_name>', methods=['POST'])
def trigger_monthly_update_route(model_name):
    if model_name not in POCKETBASE_COLLECTION_CONFIG:
        return jsonify({"error": f"Model '{model_name}' not configured for PocketBase."}), 404

    json_data = request.get_json()
    if not json_data or 'year' not in json_data or 'month' not in json_data:
        return jsonify({"error": "Please provide 'year' and 'month' (1-12) for the data to fetch."}), 400

    try:
        year = int(json_data['year'])
        month = int(json_data['month'])
        fetch_target_month_date = datetime(year, month, 1).date()
    except ValueError:
        return jsonify({"error": "Invalid year or month provided."}), 400

    success, message = services.update_and_retrain_model_from_db(model_name, fetch_target_month_date)

    if success:
        return jsonify({"message": message}), 200
    else:
        return jsonify({"error": message}), 500

if __name__ == "__main__":
    # Ensure DATA_DIR exists (config.py already does this, but good for direct run)
    os.makedirs(services.DATA_DIR, exist_ok=True) 
    app.run(debug=True)
