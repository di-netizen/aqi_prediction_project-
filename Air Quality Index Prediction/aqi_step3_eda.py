import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("aqi_data_cleaned.csv")

# Plot distribution of AQI
plt.figure(figsize=(8, 5))
sns.histplot(df['AQI'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of AQI")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Pairplot for top variables affecting AQI
top_corr = df.corr()['AQI'].abs().sort_values(ascending=False)[1:5].index.tolist()
sns.pairplot(df[top_corr + ['AQI']])
plt.suptitle("Pairplot of Top Correlated Features", y=1.02)
plt.tight_layout()
plt.show()
