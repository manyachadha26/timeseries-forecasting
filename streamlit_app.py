import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json

st.set_page_config(page_title="DemandWave", layout="wide")

# Header
st.title("📊 DemandWave")
st.markdown("**Time Series Demand Forecasting** - ARIMA vs Prophet")

API_URL = "https://timeseries-forecasting-qbe9.onrender.com"

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    days = st.slider("Days to Forecast", 1, 365, 30)
    model = st.selectbox("Model", ["ARIMA", "Prophet", "Compare Both"])
    submit = st.button("🚀 Forecast", use_container_width=True)

# Main area
if submit:
    with st.spinner(f"Forecasting {days} days with {model}..."):
        try:
            if model == "Compare Both":
                response = requests.post(f"{API_URL}/api/compare", json={"days": days})
                data = response.json()
                
                st.success("✅ Comparison loaded!")
                
                # Display comparison table
                comparison_df = pd.DataFrame(data['comparison'])
                st.dataframe(comparison_df, use_container_width=True)
                
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=comparison_df['date'], y=comparison_df['arima'], name='ARIMA', mode='lines'))
                fig.add_trace(go.Scatter(x=comparison_df['date'], y=comparison_df['prophet'], name='Prophet', mode='lines'))
                fig.update_layout(title="ARIMA vs Prophet", xaxis_title="Date", yaxis_title="Demand", hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                response = requests.post(f"{API_URL}/api/forecast", json={"days": days, "model": model.lower()})
                data = response.json()
                
                st.success(f"✅ {data['model']} forecast loaded!")
                
                # Stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Demand", f"{int(data['avg_forecast']):,}")
                col2.metric("Max Demand", f"{int(data['max_forecast']):,}")
                col3.metric("Min Demand", f"{int(data['min_forecast']):,}")
                
                # Table
                forecast_df = pd.DataFrame(data['forecasts'])
                st.dataframe(forecast_df, use_container_width=True)
                
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['demand'], fill='tozeroy', name='Forecast'))
                fig.update_layout(title=f"{data['model']} Forecast", xaxis_title="Date", yaxis_title="Demand")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ API Error: {str(e)}")
            st.info("Make sure the API is running at: " + API_URL)

else:
    st.info("👈 Adjust settings and click 'Forecast' to get started")