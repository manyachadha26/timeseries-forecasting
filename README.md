# DemandWave - Time Series Forecasting

A comprehensive time series demand forecasting system comparing ARIMA and Prophet models trained on real e-commerce transaction data.

##  Project Overview

This project builds and compares two forecasting approaches:
- **ARIMA** (AutoRegressive Integrated Moving Average) - Classical statistical approach
- **Prophet** (Facebook's forecasting tool) - Modern time series forecasting

**Model Performance:**
- ARIMA: MAE $9,465 | RMSE $13,655 | MAPE 37.94%
- Prophet: MAE $10,598 | RMSE $15,300 | MAPE 32.68% ✅ (Better)

##  Dataset

**Real E-Commerce Transaction Data**
- Source: Kaggle E-commerce Dataset
- Records: 541,909 transactions
- Time Period: December 2010 - December 2011
- Features: Date, Quantity, Price, Customer ID, Country
- Processed: 305 days of daily aggregated demand

**Daily Demand Statistics:**
- Average: 18,560 units/day
- Range: 2,048 - 93,979 units
- Clear seasonality and trend patterns

##  Technical Stack

- **Python 3.12**
- **Libraries:** statsmodels (ARIMA), Prophet, pandas, numpy, joblib
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib (for compatibility)

## 📁 Project Structure
├── 01_explore_data.py           # EDA and data understanding
├── 02_prepare_timeseries.py     # Data cleaning and aggregation
├── 03_train_models.py           # ARIMA & Prophet training
├── 04_flask_api.py              # REST API for predictions
├── 06_retrain_joblib.py         # Retrain with joblib serialization
├── 07_simple_forecast.py        # Interactive local forecasting tool
├── data.csv                     # Original transaction data
├── daily_demand_timeseries.csv  # Processed time series
├── arima_model.joblib           # Trained ARIMA model
├── prophet_model.joblib         # Trained Prophet model
├── requirements.txt             # Python dependencies
└── README.md                    # This file

##  Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Interactive Forecasting
```bash
python 07_simple_forecast.py
```

Follow the prompts:
- Enter days to forecast (1-365)
- Choose model: `arima`, `prophet`, or `compare`
- Results saved to CSV with timestamp

### 3. Retrain Models
```bash
python 06_retrain_joblib.py
```

## How It Works

### Data Pipeline
1. **Exploration** (`01_explore_data.py`)
   - 541,909 transaction records analyzed
   - Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
   - Data quality checks and missing value handling

2. **Preparation** (`02_prepare_timeseries.py`)
   - Remove cancelled orders (negative quantities)
   - Aggregate daily demand across all products
   - Create clean time series with 305 days of data

3. **Training** (`03_train_models.py`)
   - ARIMA(5,1,2): Captures temporal dependencies
   - Prophet: Handles seasonality and trend decomposition
   - 80% train / 20% test split

### Model Comparison

| Metric | ARIMA | Prophet |
|--------|-------|---------|
| MAE | $9,465 | $10,598 |
| RMSE | $13,655 | $15,300 |
| MAPE | 37.94% | **32.68%** |
| Strength | Stationary data | Seasonal patterns |
| Speed | Fast | Moderate |

**Prophet performs better** on this e-commerce dataset due to its ability to capture weekly and yearly seasonality patterns.

## 💡 Key Insights

1. **Seasonality Patterns**: Strong weekly patterns (weekends higher demand) and yearly holiday spikes
2. **Trend**: Gradual upward trend in demand over the year
3. **Volatility**: High variance suggests promotional impact and external factors
4. **Forecast Range**: ±15% confidence intervals for decision-making

##  Usage Examples

### Forecast 30 days with Prophet
```bash
python 07_simple_forecast.py
# Enter: 30
# Enter: prophet
```

### Compare ARIMA vs Prophet for 60 days
```bash
python 07_simple_forecast.py
# Enter: 60
# Enter: compare
```

Output CSV includes Date, Demand, and (if comparing) ARIMA, Prophet, and Difference columns.




