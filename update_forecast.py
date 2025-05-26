import sys
import json

from services import update_forecast_manually # Import the service function

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python update_forecast.py <model_name> <json_data_list_string_or_filepath>")
        print("Example (JSON string): python update_forecast.py sales \"[{'ds': '2025-06-01', 'y': 100}]\"")
        print("Example (Filepath): python update_forecast.py sales ./new_sales_data.json")
        sys.exit(1)

    model_name_arg = sys.argv[1]
    data_arg = sys.argv[2]
    
    new_data = []
    try:
        # Try to interpret as JSON string first
        new_data = json.loads(data_arg.replace("'", "\"")) # Allow single quotes in CLI
    except json.JSONDecodeError:
        # If not a JSON string, try to interpret as a filepath
        try:
            with open(data_arg, 'r') as f:
                new_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Data argument '{data_arg}' is not a valid JSON string or an existing file.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: File '{data_arg}' does not contain valid JSON.")
            sys.exit(1)
    
    if not isinstance(new_data, list):
        print("Error: Parsed data is not a list.")
        sys.exit(1)

    print(f"Attempting to update model '{model_name_arg}' with {len(new_data)} new data points...")
    success, message = update_forecast_manually(model_name_arg, new_data)

    if success:
        print(f"Success: {message}")
    else:
        print(f"Error: {message}")
