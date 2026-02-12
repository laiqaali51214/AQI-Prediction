"""Utility functions for pipelines."""
import logging
from datetime import datetime
from typing import Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_data(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        True if valid, False otherwise
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    if df.empty:
        logger.error("DataFrame is empty")
        return False
    
    return True


def log_pipeline_step(step_name: str, status: str = "started"):
    """
    Log pipeline step execution.
    
    Args:
        step_name: Name of the pipeline step
        status: Status of the step (started, completed, failed)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {step_name} - {status.upper()}")


def format_aqi_category(aqi: float) -> str:
    """
    Format AQI value into category string.
    
    Args:
        aqi: AQI value
    
    Returns:
        Category string
    """
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"
