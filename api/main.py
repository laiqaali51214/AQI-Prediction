"""FastAPI service for AQI predictions."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.mongodb_store import MongoDBStore
from pipelines.data_fetcher import OpenMeteoDataFetcher
from pipelines.feature_engineering import FeatureEngineer
from pipelines.data_cleaning import DataCleaner
from pipelines.aqi_calculator import EPAAQICalculator
from config.settings import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AQI Prediction API",
    description="API for predicting Air Quality Index (AQI) for the next 3 days",
    version="1.0.0"
)

# CORS middleware - allow origins from environment or default to all
import os
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
if allowed_origins == ["*"]:
    # Allow all origins (for development or if not specified)
    allow_origins_list = ["*"]
else:
    # Allow specific origins from environment variable
    allow_origins_list = [origin.strip() for origin in allowed_origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
mongodb_store = MongoDBStore()
feature_engineer = FeatureEngineer()
data_cleaner = DataCleaner()
aqi_calculator = EPAAQICalculator()
data_fetcher = OpenMeteoDataFetcher()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        logger.info("Starting up AQI Prediction API...")
        # Test MongoDB connection on startup
        mongodb_store._connect()
        logger.info("MongoDB connection established")
        logger.info("API startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # Don't raise - let the app start and handle errors in endpoints


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    forecast_days: int = 3


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[Dict]
    current_aqi: Optional[float]
    model_name: str
    model_metrics: Dict
    generated_at: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AQI Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Get AQI predictions",
            "/health": "GET - Health check",
            "/models": "GET - List available models"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint - lightweight for Railway health checks."""
    try:
        # Quick MongoDB connection test (don't raise if it fails, just report)
        try:
            mongodb_store._connect()
            mongodb_status = "connected"
        except Exception as db_error:
            mongodb_status = f"disconnected: {str(db_error)}"
        
        return {
            "status": "healthy" if mongodb_status == "connected" else "degraded",
            "mongodb": mongodb_status,
            "timestamp": datetime.now().isoformat(),
            "service": "AQI Prediction API"
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/models")
async def list_models():
    """List available models, sorted by trained_at (newest first)."""
    try:
        # Get model names from MongoDB, sorted by trained_at descending
        collection = mongodb_store.db[mongodb_store.models_collection_name]
        models = list(collection.find(
            {}, 
            {'model_name': 1, 'trained_at': 1, 'metrics': 1}
        ).sort('trained_at', -1))  # Sort by trained_at descending (newest first)
        
        model_list = []
        for model in models:
            metrics = model.get('metrics', {})
            # Filter out models with invalid metrics (RMSE = 0 or missing)
            rmse = metrics.get('rmse', float('inf'))
            if isinstance(rmse, (int, float)) and rmse > 0:
                model_list.append({
                    'name': model.get('model_name'),
                    'trained_at': model.get('trained_at').isoformat() if model.get('trained_at') else None,
                    'metrics': metrics
                })
        
        # If no valid models found, include all models (for debugging)
        if not model_list:
            for model in models:
                model_list.append({
                    'name': model.get('model_name'),
                    'trained_at': model.get('trained_at').isoformat() if model.get('trained_at') else None,
                    'metrics': model.get('metrics', {})
                })
        
        return {"models": model_list}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_aqi(request: PredictionRequest):
    """
    Predict AQI for the next N days.
    
    Args:
        request: Prediction request with optional city/location
    
    Returns:
        PredictionResponse with forecasts
    """
    try:
        # Load best model (lowest RMSE)
        model, metrics, feature_names, metadata = mongodb_store.get_best_model()
        
        if model is None:
            raise HTTPException(status_code=404, detail="No trained model found. Please train a model first.")
        
        # Fetch current data
        if request.latitude and request.longitude:
            aqi_data = data_fetcher.fetch_air_quality(lat=request.latitude, lon=request.longitude)
            weather_data = data_fetcher.fetch_weather_forecast(lat=request.latitude, lon=request.longitude)
            combined_data = {**aqi_data, **weather_data}
        else:
            raw_data = data_fetcher.fetch_combined_data()
            if raw_data.empty:
                raise HTTPException(status_code=400, detail="Could not fetch current air quality data")
            combined_data = raw_data.iloc[0].to_dict()
        
        current_df = pd.DataFrame([combined_data])
        
        # Clean and engineer features
        cleaned_data = data_cleaner.clean_data(current_df)
        features = feature_engineer.engineer_features(cleaned_data, create_targets=False)
        
        # Generate predictions for each day
        predictions = []
        current_aqi = features.get('aqi', [None])[0] if 'aqi' in features.columns else None
        
        for day in range(1, request.forecast_days + 1):
            future_date = datetime.now() + timedelta(days=day)
            
            # Create features for future date
            future_features = features.iloc[0].copy()
            
            # Update time-based features
            future_features['hour'] = 12  # Noon
            future_features['day_of_week'] = future_date.weekday()
            future_features['day_of_month'] = future_date.day
            future_features['month'] = future_date.month
            future_features['quarter'] = (future_date.month - 1) // 3 + 1
            future_features['is_weekend'] = 1 if future_date.weekday() >= 5 else 0
            
            # Cyclical encoding
            future_features['hour_sin'] = np.sin(2 * np.pi * 12 / 24)
            future_features['hour_cos'] = np.cos(2 * np.pi * 12 / 24)
            future_features['day_of_week_sin'] = np.sin(2 * np.pi * future_date.weekday() / 7)
            future_features['day_of_week_cos'] = np.cos(2 * np.pi * future_date.weekday() / 7)
            future_features['month_sin'] = np.sin(2 * np.pi * future_date.month / 12)
            future_features['month_cos'] = np.cos(2 * np.pi * future_date.month / 12)
            
            # Prepare feature vector
            feature_vector = pd.DataFrame([future_features])
            
            # Select only features used in training
            available_features = [f for f in feature_names if f in feature_vector.columns]
            missing_features = [f for f in feature_names if f not in feature_vector.columns]
            
            # Fill missing features with 0
            for f in missing_features:
                feature_vector[f] = 0
            
            feature_vector = feature_vector[feature_names]
            
            # Make prediction
            try:
                predicted_aqi = model.predict(feature_vector)[0]
                predicted_aqi = max(0, predicted_aqi)  # Ensure non-negative
                
                category = aqi_calculator.get_aqi_category(predicted_aqi)
                
                predictions.append({
                    'date': future_date.strftime('%Y-%m-%d'),
                    'day': day,
                    'predicted_aqi': float(predicted_aqi),
                    'category': category
                })
            except Exception as e:
                logger.error(f"Error making prediction for day {day}: {str(e)}")
                predictions.append({
                    'date': future_date.strftime('%Y-%m-%d'),
                    'day': day,
                    'predicted_aqi': None,
                    'category': 'Unknown',
                    'error': str(e)
                })
        
        return PredictionResponse(
            predictions=predictions,
            current_aqi=float(current_aqi) if current_aqi else None,
            model_name=metadata.get('model_name', 'unknown'),
            model_metrics=metrics,
            generated_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to bind to all interfaces, but access via localhost:8000 in browser
    print("\n" + "="*70)
    print("FastAPI Server Starting...")
    print("="*70)
    print(f"Server will be available at:")
    print(f"  - http://localhost:8000")
    print(f"  - http://127.0.0.1:8000")
    print(f"  - http://0.0.0.0:8000 (bind address, use localhost in browser)")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
