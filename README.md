# Prophet Forecast

A Flask-based time series forecasting application using Facebook Prophet that supports multiple data models and integrates with PocketBase for data management.

## Overview

This application provides automated time series forecasting capabilities for various business metrics including sales, inventory movements, stock levels, and service request counts. It features both automated data generation for testing and real-world data integration via PocketBase.

## Features

- **Multiple Forecast Models**: Pre-configured models for different business scenarios
- **PocketBase Integration**: Fetch real data from PocketBase collections
- **REST API**: Easy-to-use endpoints for forecast updates and retraining
- **Automated Initialization**: One-command setup for all models
- **Manual Data Updates**: Support for manual data input via API
- **Monthly Retraining**: Scheduled model updates with fresh data

## Supported Models

The application supports four pre-configured forecast models:

1. **Sales** (`sales`): Revenue forecasting with seasonal patterns
2. **Part Stock Log** (`part_stock_log`): Inventory movement tracking
3. **Product Stocks** (`product_stocks`): Stock level monitoring
4. **Service Request Counts** (`service_request_counts`): Service ticket volume prediction

## Project Structure

```
Prophet-Forecast/
├── app.py                 # Flask application with API endpoints
├── config.py             # Configuration and PocketBase settings
├── generate_data.py      # Synthetic data generation for testing
├── train_model.py        # Prophet model training logic
├── run_all.py           # Main initialization script
├── services.py          # Business logic and data processing
├── prophet_data/        # Directory for data files and models
└── .env                 # Environment variables (create this)
```

## Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Prophet-Forecast
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy prophet flask python-dotenv requests
   ```

3. **Create environment file:**
   Create a `.env` file in the project root:
   ```env
   NEXT_PUBLIC_POCKETBASE_URL=http://your-pocketbase-url.com
   ```

4. **Initialize all models:**
   ```bash
   python run_all.py
   ```

This will:
- Generate sample data for all models (if not exists)
- Train initial Prophet models
- Start the Flask application

## Usage

### Starting the Application

```bash
python run_all.py
```

The Flask app will start on `http://localhost:5000` by default.

### API Endpoints

#### Manual Forecast Update
```http
POST /update_forecast/<model_name>
Content-Type: application/json

[
    {"ds": "2025-01-01", "y": 150},
    {"ds": "2025-01-02", "y": 160}
]
```

#### Monthly Model Retraining
```http
POST /trigger_monthly_update/<model_name>
Content-Type: application/json

{
    "year": 2025,
    "month": 1
}
```

### Model Names

Use these model names in API endpoints:
- `sales`
- `part_stock_log`
- `product_stocks`
- `service_request_counts`

### Generating Data

Generate sample data for a specific model:
```bash
python generate_data.py <model_name>
```

### Training Models

Train a specific model:
```bash
python train_model.py <model_name>
```

## Configuration

### PocketBase Collections

Configure your PocketBase collections in [`config.py`](config.py):

```python
POCKETBASE_COLLECTION_CONFIG = {
    "model_name": {
        "collection_name": "your_collection",
        "ds_field": "date_field",
        "y_field": "value_field",
        "aggregation_method": "sum"  # optional
    }
}
```

### Data Directory

By default, all data files and trained models are stored in the `prophet_data/` directory. This can be modified in [`config.py`](config.py).

## Data Generation Patterns

The [`generate_data.py`](generate_data.py) script creates realistic synthetic data with:

- **Sales**: Seasonal trends with weekend/holiday effects
- **Part Stock Log**: Inventory movement patterns with workday cycles
- **Product Stocks**: Declining trends with periodic replenishment spikes
- **Service Requests**: Increasing trends with weekday/weekend variations

## File Outputs

For each model, the system generates:
- `{model_name}_data.csv`: Training data
- `{model_name}_forecast.csv`: Prophet forecast results
- `{model_name}_model.pkl`: Serialized Prophet model

## Error Handling

- Models validate against configured collections
- API endpoints return appropriate HTTP status codes
- Failed operations provide descriptive error messages
- Missing data files trigger warnings with recovery suggestions

## Development

### Adding New Models

1. Add configuration to [`config.py`](config.py)
2. Add data generation logic to [`generate_data.py`](generate_data.py)
3. Update any model-specific logic in services
