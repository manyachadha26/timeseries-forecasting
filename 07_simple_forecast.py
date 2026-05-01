import pandas as pd
import joblib
from datetime import datetime, timedelta

print("=" * 60)
print("TIME SERIES FORECASTING - ARIMA vs PROPHET")
print("=" * 60)

# Load models
arima_model = joblib.load('arima_model.joblib')
prophet_model = joblib.load('prophet_model.joblib')

# Get input
days = int(input("\nDays to forecast (1-365): ") or "30")
model_choice = input("Model (arima/prophet/compare): ").lower() or "prophet"

print(f"\n📊 Forecasting {days} days with {model_choice}...")

# Load historical data for dates
df = pd.read_csv('daily_demand_timeseries.csv')
df['Date'] = pd.to_datetime(df['Date'])
last_date = df['Date'].max()

if model_choice == 'arima':
    forecast = arima_model.forecast(steps=days)
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    results_df = pd.DataFrame({
        'Date': forecast_dates,
        'Demand': forecast.values
    })
    print(f"\n✅ ARIMA Forecast")
    print(f"Average: {forecast.mean():.0f}")
    print(f"Max: {forecast.max():.0f}")
    print(f"Min: {forecast.min():.0f}")
    
elif model_choice == 'prophet':
    future = pd.DataFrame({'ds': pd.date_range(start=last_date + timedelta(days=1), periods=days)})
    forecast_df = prophet_model.predict(future)
    results_df = pd.DataFrame({
        'Date': forecast_df['ds'],
        'Demand': forecast_df['yhat']
    })
    print(f"\n✅ Prophet Forecast")
    print(f"Average: {forecast_df['yhat'].mean():.0f}")
    print(f"Max: {forecast_df['yhat'].max():.0f}")
    print(f"Min: {forecast_df['yhat'].min():.0f}")
    
elif model_choice == 'compare':
    arima_forecast = arima_model.forecast(steps=days)
    future = pd.DataFrame({'ds': pd.date_range(start=last_date + timedelta(days=1), periods=days)})
    prophet_forecast_df = prophet_model.predict(future)
    
    results_df = pd.DataFrame({
        'Date': pd.date_range(start=last_date + timedelta(days=1), periods=days),
        'ARIMA': arima_forecast.values,
        'Prophet': prophet_forecast_df['yhat'].values
    })
    results_df['Difference'] = results_df['Prophet'] - results_df['ARIMA']
    
    print(f"\n✅ ARIMA vs Prophet Comparison")
    print(f"ARIMA Average: {arima_forecast.mean():.0f}")
    print(f"Prophet Average: {prophet_forecast_df['yhat'].mean():.0f}")

# Save to CSV
filename = f"forecast_{model_choice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(filename, index=False)
print(f"\n📁 Saved to: {filename}")

# Display results
print(f"\n📋 First 10 rows:")
print(results_df.head(10))