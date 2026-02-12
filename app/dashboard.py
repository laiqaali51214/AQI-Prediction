"""Streamlit dashboard for AQI predictions."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import os
import time
from pathlib import Path
import sys
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AQI Predictor Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get API URL from Streamlit secrets or environment variable (fallback)
# Streamlit secrets are accessed via st.secrets
try:
    # Try to get from Streamlit secrets (for Streamlit Cloud)
    API_URL = st.secrets.get("FASTAPI_URL", "http://localhost:8000")
except (AttributeError, FileNotFoundError, KeyError):
    # Fallback to environment variable (for local development)
    API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Add requests to imports
import requests


@st.cache_data(ttl=300)
def get_predictions(forecast_days: int = 3, latitude: float = None, longitude: float = None, max_retries: int = 2):
    """Get predictions from FastAPI service with retry logic."""
    url = f"{API_URL}/predict"
    payload = {
        "forecast_days": forecast_days
    }
    if latitude and longitude:
        payload["latitude"] = latitude
        payload["longitude"] = longitude
    
    # Retry logic for cold starts and network issues
    for attempt in range(max_retries + 1):
        try:
            # Increase timeout for predictions (model loading can take time)
            timeout = 60 if attempt == 0 else 90  # Longer timeout on retry
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout as e:
            if attempt < max_retries:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            else:
                logger.error(f"Error calling prediction API: Request timed out after {max_retries + 1} attempts")
                return None
        except requests.exceptions.RequestException as e:
            # Handle 502 Bad Gateway (Railway cold start or service down)
            is_502 = hasattr(e, 'response') and e.response is not None and e.response.status_code == 502
            is_timeout = "timeout" in str(e).lower() or isinstance(e, requests.exceptions.Timeout)
            
            if attempt < max_retries and (is_timeout or is_502):
                wait_time = 2 ** attempt
                logger.warning(f"Request error (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                if is_502:
                    logger.info(f"502 Bad Gateway - API may be starting up. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Error calling prediction API: {str(e)}")
                return None
    return None


@st.cache_data(ttl=300)
def get_available_models(max_retries: int = 2):
    """Get list of available models from API with retry logic."""
    url = f"{API_URL}/models"
    
    # Retry logic for cold starts
    for attempt in range(max_retries + 1):
        try:
            # Increase timeout for model listing (can take time on cold start)
            timeout = 30 if attempt == 0 else 45  # Longer timeout on retry
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout as e:
            if attempt < max_retries:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                logger.error(f"Error getting models: Request timed out after {max_retries + 1} attempts")
                return None
        except requests.exceptions.RequestException as e:
            # Handle 502 Bad Gateway (Railway cold start or service down)
            is_502 = hasattr(e, 'response') and e.response is not None and e.response.status_code == 502
            is_timeout = "timeout" in str(e).lower() or isinstance(e, requests.exceptions.Timeout)
            
            if attempt < max_retries and (is_timeout or is_502):
                wait_time = 2 ** attempt
                logger.warning(f"Request error (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                if is_502:
                    logger.info(f"502 Bad Gateway - API may be starting up. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Error getting models: {str(e)}")
                return None
    return None


def get_aqi_category(aqi: float) -> tuple:
    """Get AQI category and color based on AQI value."""
    if aqi <= 50:
        return "Good", "green", ""
    elif aqi <= 100:
        return "Moderate", "yellow", ""
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange", ""
    elif aqi <= 200:
        return "Unhealthy", "red", ""
    elif aqi <= 300:
        return "Very Unhealthy", "purple", ""
    else:
        return "Hazardous", "maroon", ""




def main():
    """Main dashboard function."""
    st.markdown('<h1 class="main-header">AQI Predictor Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.text(f"City: {config['city']['name']}")
        st.text(f"Location: {config['city']['latitude']:.4f}, {config['city']['longitude']:.4f}")
        
        forecast_days = st.slider("Forecast Days", 1, 7, 3)
        
        # Use configured city location
        latitude = config['city']['latitude']
        longitude = config['city']['longitude']
        
        st.header("About")
        st.info("""
        This dashboard provides real-time AQI predictions 
        for the next 3 days using machine learning models.
        """)
        
        st.header("API Status")
        try:
            # Increased timeout for health check (cold starts can be slow)
            health_response = requests.get(f"{API_URL}/health", timeout=15)
            if health_response.status_code == 200:
                st.success("API Connected")
            else:
                st.error("API Unavailable")
        except requests.exceptions.Timeout:
            st.warning("API Slow to Respond (may be cold start)")
        except:
            st.error("API Unavailable")
    
    # Check API connection (with longer timeout for cold starts)
    try:
        health = requests.get(f"{API_URL}/health", timeout=15)
        if health.status_code != 200:
            st.error("Prediction API is not available. Please ensure the FastAPI service is running.")
            st.info("If this is a Railway deployment, the API may be starting up (cold start). Try again in a moment.")
            return
    except requests.exceptions.Timeout:
        st.warning("⚠️ API is slow to respond. This may be a cold start (first request after inactivity).")
        st.info("💡 **Tip**: Railway apps can take 30-60 seconds to start. Please wait and try again.")
        st.info("The API will respond faster on subsequent requests.")
        return
    except Exception as e:
        st.error(f"Cannot connect to Prediction API: {str(e)}")
        st.info("If this is a Railway deployment, check that the API URL is correct and the service is running.")
        return
    
    # Get model info
    models_info = get_available_models()
    if models_info and isinstance(models_info, dict) and models_info.get('models'):
        st.header("Model Information")
        
        # Find and highlight the best model (lowest RMSE, excluding invalid models)
        best_model = None
        best_rmse = float('inf')
        for model in models_info['models']:
            if isinstance(model, dict):
                rmse = model.get('metrics', {}).get('rmse', float('inf'))
                # Only consider models with valid RMSE (> 0)
                if isinstance(rmse, (int, float)) and rmse > 0 and rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
        
        # Display best model prominently
        if best_model:
            st.success(f"**Best Model**: {best_model.get('name', 'Unknown')} (RMSE: {best_model.get('metrics', {}).get('rmse', 'N/A'):.2f})")
        
        # Display best model metrics in columns (not latest, but best)
        col1, col2, col3 = st.columns(3)
        with col1:
            if best_model:
                st.metric("Best Model", best_model.get('name', 'Unknown'))
            else:
                st.metric("Best Model", "N/A")
        with col2:
            if best_model and best_model.get('metrics'):
                r2 = best_model['metrics'].get('r2', 0)
                st.metric("R² Score", f"{r2:.3f}" if isinstance(r2, (int, float)) else "N/A")
            else:
                st.metric("R² Score", "N/A")
        with col3:
            if best_model and best_model.get('metrics'):
                rmse = best_model['metrics'].get('rmse', 0)
                st.metric("RMSE", f"{rmse:.2f}" if isinstance(rmse, (int, float)) else "N/A")
            else:
                st.metric("RMSE", "N/A")
    elif models_info:
        st.header("Model Information")
        st.warning("Model information is not available in the expected format.")
    
    # Get predictions
    st.header("Current Air Quality and Forecast")
    st.info("Click the button below to fetch real-time AQI data and generate predictions for the configured city.")
    
    if st.button("Get Predictions", type="primary"):
        with st.spinner("Fetching predictions..."):
            predictions_data = get_predictions(forecast_days=forecast_days, latitude=latitude, longitude=longitude)
            
            if predictions_data is None:
                st.error("Could not fetch predictions. Please check API connection.")
                return
            
            # Display model information used for prediction
            if predictions_data.get('model_name'):
                model_name = predictions_data.get('model_name', 'Unknown')
                model_metrics = predictions_data.get('model_metrics', {})
                rmse = model_metrics.get('rmse', 'N/A')
                r2 = model_metrics.get('r2', 'N/A')
                
                # Format metrics properly
                rmse_str = f"{rmse:.2f}" if isinstance(rmse, (int, float)) else str(rmse)
                r2_str = f"{r2:.3f}" if isinstance(r2, (int, float)) else str(r2)
                
                st.success(f"**Model Used for Prediction**: {model_name} | **RMSE**: {rmse_str} | **R²**: {r2_str}")
            
            # Display current AQI
            if predictions_data.get('current_aqi'):
                current_aqi = predictions_data['current_aqi']
                category, color, _ = get_aqi_category(current_aqi)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current AQI", f"{current_aqi:.0f}")
                with col2:
                    st.markdown(f"### {category}")
                with col3:
                    if current_aqi > config['dashboard']['alert_thresholds']['unhealthy']:
                        st.error("Alert: Unhealthy air quality detected!")
            
            # Display predictions
            predictions = predictions_data.get('predictions', [])
            if predictions:
                pred_df = pd.DataFrame(predictions)
                
                # Create visualization
                fig = go.Figure()
                
                for _, row in pred_df.iterrows():
                    aqi = row.get('predicted_aqi')
                    if aqi is not None:
                        category, color, _ = get_aqi_category(aqi)
                        
                        fig.add_trace(go.Bar(
                            x=[row['date']],
                            y=[aqi],
                            name=f"Day {row['day']}",
                            marker_color=color,
                            text=f"{aqi:.0f}",
                            textposition='outside',
                            hovertemplate=f"<b>{row['date']}</b><br>AQI: {aqi:.0f}<br>Category: {category}<extra></extra>"
                        ))
                
                # Set Y-axis range to show full AQI scale (0-500 EPA standard)
                max_aqi = max(pred_df['predicted_aqi'].max() if not pred_df.empty else 100, 100)
                yaxis_max = max(500, max_aqi * 1.2)  # Show up to 500, or 20% above max value
                
                fig.update_layout(
                    title=f"AQI Forecast for Next {forecast_days} Days",
                    xaxis_title="Date",
                    yaxis_title="AQI",
                    yaxis=dict(range=[0, yaxis_max], dtick=50),  # Fixed range with 50-unit ticks
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Display predictions table
                st.subheader("Forecast Details")
                display_df = pred_df.copy()
                display_df['Predicted AQI'] = display_df['predicted_aqi'].apply(lambda x: f"{x:.0f}" if x else "N/A")
                display_df = display_df[['date', 'Predicted AQI', 'category']]
                display_df.columns = ['Date', 'Predicted AQI', 'Category']
                st.dataframe(display_df, width='stretch')
                
                # Alerts
                st.subheader("Alerts")
                alerts = []
                for _, row in pred_df.iterrows():
                    aqi = row.get('predicted_aqi')
                    if aqi and aqi > config['dashboard']['alert_thresholds']['unhealthy']:
                        alerts.append(f"{row['date']}: Unhealthy AQI predicted ({aqi:.0f})")
                
                if alerts:
                    for alert in alerts:
                        st.warning(alert)
                else:
                    st.success("No alerts - air quality is expected to be within acceptable limits.")
    
    # Additional information
    with st.expander("About AQI Categories"):
        st.markdown("""
        - **Good (0-50)**: Air quality is satisfactory.
        - **Moderate (51-100)**: Acceptable for most people.
        - **Unhealthy for Sensitive Groups (101-150)**: Sensitive groups may experience health effects.
        - **Unhealthy (151-200)**: Everyone may begin to experience health effects.
        - **Very Unhealthy (201-300)**: Health alert - everyone may experience serious health effects.
        - **Hazardous (301+)**: Health warning - entire population likely affected.
        """)


if __name__ == "__main__":
    main()
