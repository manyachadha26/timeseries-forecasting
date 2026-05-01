import pandas as pd
import numpy as np

print("=" * 60)
print("EXPLORING E-COMMERCE DATA")
print("=" * 60)

# Load dataset
df = pd.read_csv('data.csv', encoding='latin-1')

print(f"\n📊 Dataset shape: {df.shape}")
print(f"\n📋 Columns: {df.columns.tolist()}")
print(f"\n🔍 First few rows:")
print(df.head())

print(f"\n📈 Data types:")
print(df.dtypes)

print(f"\n📊 Missing values:")
print(df.isnull().sum())

print(f"\n💡 Basic statistics:")
print(df.describe())

if 'date' in df.columns or 'Date' in df.columns or 'timestamp' in df.columns:
    date_col = [col for col in df.columns if 'date' in col.lower()][0]
    print(f"\n📅 Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Total days: {(pd.to_datetime(df[date_col]).max() - pd.to_datetime(df[date_col]).min()).days}")