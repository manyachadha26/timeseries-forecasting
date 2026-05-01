import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

print("=" * 70)
print("TRAINING 3 TIME SERIES FORECASTING MODELS")
print("=" * 70)

# Load data
df = pd.read_csv('daily_demand_timeseries.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Split: 80% train, 20% test
split_idx = int(len(df) * 0.8)
train_df = df[:split_idx].copy()
test_df = df[split_idx:].copy()

print(f"\n📊 Data split:")
print(f"Training set: {len(train_df)} days ({train_df['Date'].min()} to {train_df['Date'].max()})")
print(f"Test set: {len(test_df)} days ({test_df['Date'].min()} to {test_df['Date'].max()})")

# ============== MODEL 1: ARIMA ==============
print(f"\n{'='*70}")
print("MODEL 1: ARIMA (AutoRegressive Integrated Moving Average)")
print(f"{'='*70}")

try:
    arima_model = ARIMA(train_df['Demand'], order=(5, 1, 2))
    arima_fitted = arima_model.fit()
    arima_forecast = arima_fitted.forecast(steps=len(test_df))
    
    arima_mae = mean_absolute_error(test_df['Demand'], arima_forecast)
    arima_rmse = np.sqrt(mean_squared_error(test_df['Demand'], arima_forecast))
    arima_mape = np.mean(np.abs((test_df['Demand'] - arima_forecast) / test_df['Demand'])) * 100
    
    print(f"✅ ARIMA trained successfully")
    print(f"MAE: ${arima_mae:.2f}")
    print(f"RMSE: ${arima_rmse:.2f}")
    print(f"MAPE: {arima_mape:.2f}%")
    
    # Save model
    with open('arima_model.pkl', 'wb') as f:
        pickle.dump(arima_fitted, f)
    
except Exception as e:
    print(f"❌ ARIMA error: {e}")
    arima_forecast = None

# ============== MODEL 2: PROPHET ==============
print(f"\n{'='*70}")
print("MODEL 2: PROPHET (Facebook's Forecasting Tool)")
print(f"{'='*70}")

try:
    prophet_train = train_df[['Date', 'Demand']].copy()
    prophet_train.columns = ['ds', 'y']
    
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_model.fit(prophet_train)
    
    future = pd.DataFrame({'ds': test_df['Date']})
    prophet_forecast_df = prophet_model.predict(future)
    prophet_forecast = prophet_forecast_df['yhat'].values
    
    prophet_mae = mean_absolute_error(test_df['Demand'], prophet_forecast)
    prophet_rmse = np.sqrt(mean_squared_error(test_df['Demand'], prophet_forecast))
    prophet_mape = np.mean(np.abs((test_df['Demand'] - prophet_forecast) / test_df['Demand'])) * 100
    
    print(f"✅ Prophet trained successfully")
    print(f"MAE: ${prophet_mae:.2f}")
    print(f"RMSE: ${prophet_rmse:.2f}")
    print(f"MAPE: {prophet_mape:.2f}%")
    
    # Save model
    with open('prophet_model.pkl', 'wb') as f:
        pickle.dump(prophet_model, f)
    
except Exception as e:
    print(f"❌ Prophet error: {e}")
    prophet_forecast = None


    

# ============== MODEL COMPARISON ==============
print(f"\n{'='*70}")
print("MODEL COMPARISON")
print(f"{'='*70}")

results = pd.DataFrame({
    'Model': ['ARIMA', 'Prophet', 'LSTM'],
    'MAE': [arima_mae if arima_forecast is not None else np.nan,
            prophet_mae if prophet_forecast is not None else np.nan,
            lstm_mae if lstm_forecast is not None else np.nan],
    'RMSE': [arima_rmse if arima_forecast is not None else np.nan,
             prophet_rmse if prophet_forecast is not None else np.nan,
             lstm_rmse if lstm_forecast is not None else np.nan],
    'MAPE': [arima_mape if arima_forecast is not None else np.nan,
             prophet_mape if prophet_forecast is not None else np.nan,
             lstm_mape if lstm_forecast is not None else np.nan]
})

print("\n" + results.to_string(index=False))

# Save results
results.to_csv('model_comparison_results.csv', index=False)
print(f"\n✅ Results saved to model_comparison_results.csv")