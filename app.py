from flask import Flask, jsonify
from flask_cors import CORS  # ✅ correct import

app = Flask(__name__)
CORS(app)  # ✅ enables CORS for all origins

@app.route('/forecast')
def forecast():
    # Load forecast data from file or generate dynamically
    import pandas as pd
    df = pd.read_csv("forecast_data.csv")
    return df.to_json(orient="records")

if __name__ == "__main__":
    app.run(debug=True)
