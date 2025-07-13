import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic air pollution data
data = {
    'PM2.5': np.random.uniform(5, 300, 1000),
    'PM10': np.random.uniform(10, 400, 1000),
    'NO2': np.random.uniform(5, 100, 1000),
    'SO2': np.random.uniform(2, 80, 1000),
    'CO': np.random.uniform(0.1, 5, 1000),
    'O3': np.random.uniform(10, 180, 1000),
    'Temperature': np.random.uniform(10, 45, 1000),
    'Humidity': np.random.uniform(20, 90, 1000)
}

df = pd.DataFrame(data)

# Simulated AQI based on pollutants
df['AQI'] = (
    0.5 * df['PM2.5'] +
    0.3 * df['PM10'] +
    0.2 * df['NO2'] +
    0.1 * df['SO2'] +
    0.05 * df['CO'] +
    0.05 * df['O3'] +
    np.random.normal(0, 10, 1000)
).astype(int)

# Save it
df.to_csv("aqi_data.csv", index=False)
print("✅ Generated 'aqi_data.csv' with shape:", df.shape)
print(df.head())

