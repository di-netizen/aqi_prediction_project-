import pandas as pd

# Load dataset
df = pd.read_csv("aqi_data.csv")

# Check shape and basic info
print("📊 Dataset shape:", df.shape)
print("\n🧹 Missing values:\n", df.isnull().sum())
print("\n📈 Summary stats:\n", df.describe())

# Check data types
print("\n📋 Data types:\n", df.dtypes)

# If any missing values existed, you could handle like:
# df.fillna(method='ffill', inplace=True)  # just an example

# Save cleaned version (even if no cleaning is done)
df.to_csv("aqi_data_cleaned.csv", index=False)
print("\n✅ Saved cleaned data as 'aqi_data_cleaned.csv'")

