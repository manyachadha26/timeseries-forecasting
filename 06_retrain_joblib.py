import joblib
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import pandas as pd

df = pd.read_csv('daily_demand_timeseries.csv')
df['Date'] = pd.to_datetime(df['Date'])

split_idx = int(len(df) * 0.8)
train_df = df[:split_idx].copy()

# ARIMA
print("Training ARIMA...")
arima = ARIMA(train_df['Demand'], order=(5, 1, 2)).fit()
joblib.dump(arima, 'arima_model.joblib')
print("✅ ARIMA saved")

# Prophet
print("Training Prophet...")
prophet_train = train_df[['Date', 'Demand']].copy()
prophet_train.columns = ['ds', 'y']
prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
prophet.fit(prophet_train)
joblib.dump(prophet, 'prophet_model.joblib')
print("✅ Prophet saved")

print("\n✅ Done!")