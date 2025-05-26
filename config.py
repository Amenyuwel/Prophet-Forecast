import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DATA_DIR = "prophet_data"
HISTORICAL_WINDOW_DAYS = 365  # Define the rolling window size
POCKETBASE_URL = os.getenv("NEXT_PUBLIC_POCKETBASE_URL")

# PocketBase Collection Configuration
POCKETBASE_COLLECTION_CONFIG = {
    "sales": {
        "collection_name": "sales_records",
        "ds_field": "sale_date",
        "y_field": "amount",
    },
    "part_stock_log": {
        "collection_name": "inventory_movements",
        "ds_field": "movement_date",
        "y_field": "quantity_change",
        # Example: if you need to sum daily changes for part_stock_log
        "aggregation_method": "sum" 
    },
    "product_stocks": {
        "collection_name": "daily_product_stock_levels",
        "ds_field": "record_date",
        "y_field": "stock_level",
    },
    "service_request_counts": {
        "collection_name": "service_tickets",
        "ds_field": "created",
        "y_field": None,  # y is a count, so no specific field needed for value
        "aggregation_method": "count"  # Special handling for counting records
    }
    # Add other models here
}

# Ensure DATA_DIR exists when this module is loaded
os.makedirs(DATA_DIR, exist_ok=True)