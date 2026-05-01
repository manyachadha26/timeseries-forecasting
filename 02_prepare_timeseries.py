import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 60)
print("PREPARING TIME SERIES DATA FOR FORECASTING")
print("=" * 60)

# Load data
df = pd.read_csv('data.csv', encoding='latin-1')

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M')

# Remove cancelled orders (negative quantities)
df = df[df['Quantity'] > 0]

# Calculate daily demand (total quantity sold per day)
daily_demand = df.groupby(df['InvoiceDate'].dt.date)['Quantity'].sum().reset_index()
daily_demand.columns = ['Date', 'Demand']
daily_demand['Date'] = pd.to_datetime(daily_demand['Date'])
daily_demand = daily_demand.sort_values('Date').reset_index(drop=True)

print(f"\n✅ Daily demand time series created")
print(f"📅 Date range: {daily_demand['Date'].min()} to {daily_demand['Date'].max()}")
print(f"📊 Total days: {len(daily_demand)}")
print(f"\n📈 Demand statistics:")
print(daily_demand['Demand'].describe())

print(f"\n🔍 First 10 days:")
print(daily_demand.head(10))

# Save time series
daily_demand.to_csv('daily_demand_timeseries.csv', index=False)
print(f"\n✅ Saved to daily_demand_timeseries.csv")