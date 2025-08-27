# Pharmaceutical Sales Forecast Dashboard

## Description

The Pharmaceutical Sales Forecast Dashboard is a Streamlit-based application that provides sales forecasting for various pharmaceutical product categories. The dashboard uses two forecasting methods: Prophet for monthly forecasts and XGBoost for weekly forecasts. It offers interactive visualizations, performance metrics, and AI-generated explanations of forecast trends.

## Features

- **Dual Forecasting Methods**: Choose between Prophet (monthly) and XGBoost (weekly) forecasting models
- **Multiple Medicine Categories**: Support for 8 different pharmaceutical categories (M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06)
- **Interactive Visualizations**: Dynamic charts showing historical data and forecasts
- **Performance Metrics**: Accuracy, MAPE, RMSE, and R² metrics for model evaluation
- **AI-Generated Explanations**: Natural language explanations of forecast trends using Google's Generative AI
- **Customizable Forecast Horizon**: Adjust the forecast period based on your needs
- **Data Export**: Download forecast data for further analysis

## Installation

1. Clone the repository or download the source code

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can install dependencies individually:
   ```bash
   pip install streamlit>=1.20.0 pandas>=1.3.0 numpy>=1.20.0 plotly>=5.5.0 prophet>=1.1.0 matplotlib>=3.5.0 scikit-learn>=1.0.0 google-generativeai>=0.3.0 xgboost>=1.5.0
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Use the sidebar controls to:
   - Select the forecasting method (Prophet or XGBoost)
   - Choose a medicine category
   - Set date ranges and forecast horizon

## Forecasting Methods

### Prophet (Monthly)
- Facebook's Prophet model for time series forecasting
- Works with monthly aggregated pharmaceutical sales data
- Provides trend, seasonality, and holiday components
- Best for medium to long-term forecasting with seasonal patterns

### XGBoost (Weekly)
- Gradient boosting model for time series forecasting
- Works with weekly sales data for more granular predictions
- Uses lagged features and calendar information
- Best for short to medium-term forecasting with complex patterns

## Data Requirements

The application expects the following data files:
- `monthly_dataset.csv`: Monthly sales data for Prophet models
- `salesweekly_corrected_no_outliers.csv`: Weekly sales data for XGBoost models
- Pre-trained models in the `Prophet_models` and `XgBoost_Model` directories
