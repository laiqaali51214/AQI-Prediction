

"""
Streamlit dashboard for AQI (Air Quality Index) predictions.

This app connects to the FastAPI backend to:
- Fetch current AQI and multi-day forecasts (1–7 days)
- Display metrics for all models and highlight the best one
- Show bar charts and tables of predicted AQI by date
- Trigger alerts when AQI exceeds unhealthy thresholds
"""
import streamlit as st
import pandas as pd
import numpy as np
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

# --- Page configuration ---
st.set_page_config(
    page_title="AQI Predictor Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API base URL ---
try:
    API_URL = st.secrets.get("FASTAPI_URL", "http://localhost:8000")
except (AttributeError, FileNotFoundError, KeyError):
    API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")


# API_URL = "http://127.0.0.1:8000"



# --- Custom CSS ---
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #00BFA6;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #1E2228;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    color: #FAFAFA;
    margin-bottom: 1rem;
}
.alert-box {
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    font-weight: bold;
}
.category-good { color: #00E676; }
.category-moderate { color: #FFD600; }
.category-unhealthy { color: #FF5252; }
.category-very-unhealthy { color: #9C27B0; }
.category-hazardous { color: #800000; }
</style>
""", unsafe_allow_html=True)

# ============== Helper Functions ==============

@st.cache_data(ttl=300)
def get_predictions(forecast_days=3, latitude=None, longitude=None, max_retries=2):
    url = f"{API_URL}/predict"
    payload = {"forecast_days": forecast_days}
    if latitude and longitude:
        payload["latitude"] = latitude
        payload["longitude"] = longitude

    for attempt in range(max_retries + 1):
        try:
            timeout = 60 if attempt == 0 else 90
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.warning(f"Retry {attempt+1}: {str(e)}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Failed after {max_retries+1} attempts: {str(e)}")
                return None
    return None

@st.cache_data(ttl=300)
def get_available_models(max_retries=2):
    url = f"{API_URL}/models"
    for attempt in range(max_retries + 1):
        try:
            timeout = 30 if attempt == 0 else 45
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.warning(f"Retry {attempt+1}: {str(e)}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Failed after {max_retries+1} attempts: {str(e)}")
                return None
    return None

def get_aqi_category(aqi: float):
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "yellow"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    elif aqi <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"

def format_metric(value, decimals=2):
    """Safely format numeric metrics, else return 'N/A'."""
    return f"{value:.{decimals}f}" if isinstance(value, (int, float)) else "N/A"

# ============== Dashboard UI ==============

def main():
    st.markdown('<h1 class="main-header">AQI Predictor Dashboard</h1>', unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        st.text(f"City: {config['city']['name']}")
        st.text(f"Location: {config['city']['latitude']:.4f}, {config['city']['longitude']:.4f}")
        forecast_days = st.slider("Forecast Days", 1, 7, 3)
        latitude = config['city']['latitude']
        longitude = config['city']['longitude']

        st.header("About")
        st.info("Real-time AQI predictions using machine learning models.")

        st.header("API Status")
        try:
            health_response = requests.get(f"{API_URL}/health", timeout=15)
            if health_response.status_code == 200:
                st.success("API Connected")
            else:
                st.error("API Unavailable")
        except:
            st.error("API Unavailable")

    # --- Fetch all models ---
    models_info = get_available_models()
    best_model = None

    if models_info and isinstance(models_info, dict) and models_info.get('models'):
        st.header("All Trained Models Metrics")

        # Build metrics table
        metrics_data = []
        for model in models_info['models']:
            m_metrics = model.get('metrics', {})
            metrics_data.append({
                "Model": model.get('name', 'Unknown'),
                "RMSE": format_metric(m_metrics.get('rmse')),
                "MAE": format_metric(m_metrics.get('mae')),
                "R²": format_metric(m_metrics.get('r2'), 3),
                "MAPE (%)": format_metric(m_metrics.get('mape')),
                # "Explained Variance": format_metric(m_metrics.get('explained_variance'))
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

        # Identify best model (lowest RMSE)
        # Identify best model (lowest RMSE)
        rmse_values = []
        valid_models = []

        for m in models_info['models']:
            rmse = m.get('metrics', {}).get('rmse')
            if isinstance(rmse, (int, float)):
                rmse_values.append(rmse)
                valid_models.append(m)

        if rmse_values:
            best_idx = np.argmin(rmse_values)
            best_model = valid_models[best_idx]
            st.success(f"Best Model: {best_model.get('name')} (RMSE: {format_metric(best_model.get('metrics', {}).get('rmse'))})")


    # --- Main predictions section ---
    st.header("Current Air Quality and Forecast")
    st.info("Click the button to fetch real-time AQI predictions.")

    if st.button("Get Predictions", type="primary"):
        with st.spinner("Fetching predictions..."):
            predictions_data = get_predictions(forecast_days=forecast_days, latitude=latitude, longitude=longitude)
            if predictions_data is None:
                st.error("Could not fetch predictions. Check API connection.")
                return

            # Display model metrics safely
            model_name = predictions_data.get('model_name', 'Unknown')
            metrics = predictions_data.get('model_metrics', {})
            rmse = format_metric(metrics.get('rmse'))
            mae = format_metric(metrics.get('mae'))
            r2 = format_metric(metrics.get('r2'), 3)
            mape = format_metric(metrics.get('mape'))
            # ev = format_metric(metrics.get('explained_variance'))

            # Show metrics in cards
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("RMSE", rmse)
            col2.metric("MAE", mae)
            col3.metric("R²", r2)
            col4.metric("MAPE (%)", mape)
            # col5.metric("Explained Var", ev)
            st.markdown(f"**Model Used:** {model_name}")

            # Current AQI
            if predictions_data.get('current_aqi'):
                current_aqi = predictions_data['current_aqi']
                category, _ = get_aqi_category(current_aqi)
                col1, col2, col3 = st.columns(3)
                col1.metric("Current AQI", f"{current_aqi:.0f}")
                col2.markdown(f"### {category}")
                col3.markdown(f"<div class='alert-box'>{'⚠️ Unhealthy air quality!' if current_aqi > config['dashboard']['alert_thresholds']['unhealthy'] else ''}</div>", unsafe_allow_html=True)

            # Forecast chart
            predictions = predictions_data.get('predictions', [])
            if predictions:
                pred_df = pd.DataFrame(predictions)
                fig = go.Figure()
                for _, row in pred_df.iterrows():
                    aqi = row.get('predicted_aqi')
                    if aqi is not None:
                        _, color = get_aqi_category(aqi)
                        fig.add_trace(go.Bar(
                            x=[row['date']],
                            y=[aqi],
                            marker_color=color,
                            text=f"{aqi:.0f}",
                            textposition='outside',
                            hovertemplate=f"<b>{row['date']}</b><br>AQI: {aqi:.0f}<extra></extra>"
                        ))
                max_aqi = max(pred_df['predicted_aqi'].max(), 100)
                yaxis_max = max(500, max_aqi * 1.2)
                fig.update_layout(
                    title=f"AQI Forecast Next {forecast_days} Days",
                    xaxis_title="Date",
                    yaxis_title="AQI",
                    yaxis=dict(range=[0, yaxis_max], dtick=50),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Forecast table
                pred_df['Predicted AQI'] = pred_df['predicted_aqi'].apply(lambda x: f"{x:.0f}" if x else "N/A")
                display_df = pred_df[['date', 'Predicted AQI', 'category']]
                display_df.columns = ['Date', 'Predicted AQI', 'Category']
                st.dataframe(display_df, use_container_width=True)

                # Alerts
                st.subheader("Alerts")
                alerts = [f"{row['date']}: Unhealthy AQI ({row['predicted_aqi']:.0f})" 
                          for _, row in pred_df.iterrows() 
                          if row.get('predicted_aqi') and row['predicted_aqi'] > config['dashboard']['alert_thresholds']['unhealthy']]
                if alerts:
                    for alert in alerts:
                        st.warning(alert)
                else:
                    st.success("No alerts - air quality within acceptable limits.")

    # Collapsible reference
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
