import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("aqi_data_cleaned.csv")

# Features and target
X = df.drop('AQI', axis=1)
y = df['AQI']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Confirm shape
print("✅ Features selected")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
