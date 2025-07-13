# 🌫️ Air Quality Index (AQI) Prediction using Machine Learning

This project demonstrates a complete machine learning workflow to predict the Air Quality Index (AQI) based on key air pollutants and weather parameters. It includes synthetic data generation, data cleaning, exploratory data analysis (EDA), model training, feature importance visualization, and real-time AQI prediction.

---

## 📊 Dataset

- **Type:** Synthetic data (1,000 samples)
- **Features:**
  - PM2.5, PM10, NO2, SO2, CO, O3, Temperature, Humidity
- **Target:** AQI (calculated using a weighted pollution formula)

---

## 🧪 Project Workflow

### 1. Data Generation (`aqi_generate_data.py`)
- Simulates air pollution data using `numpy` and `pandas`
- Calculates AQI using a weighted formula
- Saves as `aqi_data.csv`

### 2. Data Cleaning (`aqi_clean_data.py`)
- Checks data shape, types, and missing values
- Saves cleaned version as `aqi_data_cleaned.csv`

### 3. Exploratory Data Analysis (`aqi_eda.py`)
- Histogram + KDE plot of AQI distribution
- Correlation heatmap
- Pairplot of most AQI-correlated features

### 4. Train-Test Split (`aqi_train_test_split.py`)
- 80/20 train-test split using `train_test_split`

### 5. Model Training & Evaluation (`aqi_train_model.py`)
- Trains both Linear Regression and Random Forest
- Evaluates using MAE, RMSE, and R² Score

### 6. Feature Importance (`aqi_feature_importance_save_model.py`)
- Trains Random Forest
- Visualizes feature importances
- Saves model as `aqi_rf_model.pkl`

### 7. Real-Time Prediction (`aqi_predict.py`)
- Loads saved model
- Predicts AQI for a new sample
- Outputs predicted AQI with confidence

---

## 📈 Model Performance

| Model               | MAE   | RMSE  | R² Score |
|--------------------|-------|-------|----------|
| Linear Regression   | ~20   | ~26   | ~0.89    |
| Random Forest       | ~10   | ~14   | ~0.96    |

Random Forest outperforms Linear Regression and is used for final deployment.

---

## 📁 Folder Structure

