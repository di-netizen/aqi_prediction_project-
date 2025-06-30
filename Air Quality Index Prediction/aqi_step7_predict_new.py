import pandas as pd
import joblib

# Load the saved model
model = joblib.load("aqi_rf_model.pkl")

# 🔍 Simulate new sensor input (customize these values!)
new_data = pd.DataFrame([{
    'PM2.5': 180,
    'PM10': 220,
    'NO2': 50,
    'SO2': 20,
    'CO': 1.5,
    'O3': 100,
    'Temperature': 32,
    'Humidity': 55
}])

# Predict AQI
predicted_aqi = model.predict(new_data)[0]
print(f"🌫️ Predicted AQI: {predicted_aqi:.2f}")
