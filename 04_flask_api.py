from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

print("Loading models...")
try:
    with open('arima_model.pkl', 'rb') as f:
        arima_model = pickle.load(f)
    with open('prophet_model.pkl', 'rb') as f:
        prophet_model = pickle.load(f)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    arima_model = None
    prophet_model = None

# Load historical data for context
df = pd.read_csv('daily_demand_timeseries.csv')
df['Date'] = pd.to_datetime(df['Date'])

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'API running', 'models': ['ARIMA', 'Prophet']})

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """
    POST request with:
    {
        "days": 30,
        "model": "prophet" or "arima"
    }
    """
    try:
        data = request.get_json()
        days_ahead = int(data.get('days', 30))
        model_choice = data.get('model', 'prophet').lower()
        
        if days_ahead < 1 or days_ahead > 365:
            return jsonify({'error': 'Days must be between 1 and 365'}), 400
        
        if model_choice == 'arima':
            forecast_result = arima_model.forecast(steps=days_ahead)
            model_name = 'ARIMA'
        elif model_choice == 'prophet':
            future = pd.DataFrame({
                'ds': pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=days_ahead)
            })
            forecast_df = prophet_model.predict(future)
            forecast_result = forecast_df['yhat'].values
            model_name = 'Prophet'
        else:
            return jsonify({'error': 'Model must be "arima" or "prophet"'}), 400
        
        # Generate dates
        last_date = df['Date'].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
        
        # Prepare response
        forecasts = [
            {
                'date': str(date.date()),
                'demand': float(max(0, pred))  # Ensure non-negative
            }
            for date, pred in zip(forecast_dates, forecast_result)
        ]
        
        return jsonify({
            'status': 'success',
            'model': model_name,
            'days_forecast': days_ahead,
            'last_historical_date': str(df['Date'].max().date()),
            'last_historical_demand': int(df['Demand'].iloc[-1]),
            'forecasts': forecasts,
            'avg_forecast': float(np.mean(forecast_result)),
            'max_forecast': float(np.max(forecast_result)),
            'min_forecast': float(np.min(forecast_result))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/compare', methods=['POST'])
def compare():
    """
    Compare ARIMA vs Prophet forecasts
    POST: { "days": 30 }
    """
    try:
        data = request.get_json()
        days_ahead = int(data.get('days', 30))
        
        if days_ahead < 1 or days_ahead > 365:
            return jsonify({'error': 'Days must be between 1 and 365'}), 400
        
        # ARIMA forecast
        arima_forecast = arima_model.forecast(steps=days_ahead)
        
        # Prophet forecast
        future = pd.DataFrame({
            'ds': pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=days_ahead)
        })
        prophet_df = prophet_model.predict(future)
        prophet_forecast = prophet_df['yhat'].values
        
        # Generate dates
        last_date = df['Date'].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
        
        comparison = []
        for date, arima_val, prophet_val in zip(forecast_dates, arima_forecast, prophet_forecast):
            comparison.append({
                'date': str(date.date()),
                'arima': float(max(0, arima_val)),
                'prophet': float(max(0, prophet_val)),
                'difference': float(prophet_val - arima_val)
            })
        
        return jsonify({
            'status': 'success',
            'days_forecast': days_ahead,
            'comparison': comparison,
            'arima_avg': float(np.mean(arima_forecast)),
            'prophet_avg': float(np.mean(prophet_forecast)),
            'historical_data': {
                'days': len(df),
                'date_range': f"{str(df['Date'].min().date())} to {str(df['Date'].max().date())}",
                'avg_demand': float(df['Demand'].mean()),
                'max_demand': int(df['Demand'].max()),
                'min_demand': int(df['Demand'].min())
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
