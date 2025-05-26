import subprocess
import os
import sys # Import the sys module

DATA_DIR = "prophet_data"
TARGET_MODELS = ["sales", "part_stock_log", "product_stocks", "service_request_counts"]

def run_script_checked(script_name, model_name_arg=None):
    script_display_name = f"{script_name} for model '{model_name_arg}'" if model_name_arg else script_name
    print(f"Running: {script_display_name} ...")
    
    # Use sys.executable to ensure the correct Python interpreter is used
    command = [sys.executable, script_name] 
    if model_name_arg:
        command.append(model_name_arg)
    
    try:
        # Added encoding='utf-8' for better cross-platform text handling
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        print(result.stdout)
        if result.stderr:
            print(f"stderr from {script_display_name}:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_display_name}:")
        print(f"Return code: {e.returncode}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: Script '{script_name}' not found, or Python interpreter '{sys.executable}' not found.")
        return False
    except Exception as e:
        print(f"Error: An unexpected error occurred while running {script_display_name}: {str(e)}")
        return False

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Starting: Initializing data and models for: {', '.join(TARGET_MODELS)}...")

    for model_name in TARGET_MODELS:
        print(f"\n--- Processing model: {model_name} ---")
        data_file = os.path.join(DATA_DIR, f"{model_name}_data.csv")
        forecast_file = os.path.join(DATA_DIR, f"{model_name}_forecast.csv")

        if not os.path.exists(data_file):
            print(f"'{data_file}' not found. Generating initial data...")
            if not run_script_checked("generate_data.py", model_name):
                print(f"Warning: Failed to generate initial data for '{model_name}'. This model might not work correctly.")
        else:
            print(f"'{data_file}' already exists. Skipping data generation for '{model_name}'.")

        if os.path.exists(data_file):
            if not os.path.exists(forecast_file):
                print(f"'{forecast_file}' not found. Training initial model for '{model_name}'...")
                if not run_script_checked("train_model.py", model_name):
                     print(f"Warning: Failed to train initial model for '{model_name}'.")
            else:
                print(f"'{forecast_file}' already exists for '{model_name}'. Skipping initial training.")
                print(f"   To retrain, delete '{forecast_file}' and run this script again, or use the API for '{model_name}'.")
        else:
            print(f"Warning: Cannot train initial model for '{model_name}' as '{data_file}' does not exist.")

    print("\n-----------------------------------------------------")
    print("Success: Initialization complete for all models (if applicable).")
    print("Warning: Starting Flask app. Press Ctrl+C to stop.")
    print("-----------------------------------------------------")
    
    try:
        app_script_path = os.path.join(os.path.dirname(__file__), "app.py")
        if not os.path.exists(app_script_path):
             app_script_path = "app.py" 

        # Use sys.executable for the Flask app as well
        subprocess.run([sys.executable, app_script_path], check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error: Flask app exited with error: {e}")
    except KeyboardInterrupt:
        print("\nSuccess: Flask app stopped by user.")
    except FileNotFoundError:
        print(f"Error: Script 'app.py' not found, or Python interpreter '{sys.executable}' not found.")
    except Exception as e:
        print(f"Error: An unexpected error occurred while starting the Flask app: {str(e)}")
