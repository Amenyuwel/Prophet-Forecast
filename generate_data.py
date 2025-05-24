import pandas as pd
import numpy as np

# Parameters
periods = 365
np.random.seed(42)

# Date range: last 365 days including today
date_range = pd.date_range(end=pd.Timestamp.today(), periods=periods)

# Create trend: gradual increase over time (e.g., 0.1 sales increase per day)
trend = np.linspace(100, 200, periods)  # starting at 100, ending at 200

# Weekly seasonality: sales higher on weekends (Sat=5, Sun=6)
weekly_seasonality = [1.0 + 0.3 if day.weekday() >= 5 else 1.0 for day in date_range]

# Yearly seasonality: simulate higher sales during certain months (e.g., Nov, Dec)
monthly_seasonality = []
for day in date_range:
    if day.month in [11, 12]:  # holiday season spike
        monthly_seasonality.append(1.5)
    elif day.month in [6, 7, 8]:  # summer slow down
        monthly_seasonality.append(0.8)
    else:
        monthly_seasonality.append(1.0)

# Combine seasonalities multiplicatively
seasonality = np.array(weekly_seasonality) * np.array(monthly_seasonality)

# Generate noise: normal distribution around 0 with std dev 15
noise = np.random.normal(loc=0, scale=15, size=periods)

# Calculate final sales
sales = trend * seasonality + noise

# Clip to avoid negative or zero sales, round to integer
sales = np.clip(sales, 20, None).round().astype(int)

# Create DataFrame
df = pd.DataFrame({
    'ds': date_range,
    'y': sales
})

df.to_csv("sales_data.csv", index=False)
print("✅ Generated more realistic sales_data.csv")
