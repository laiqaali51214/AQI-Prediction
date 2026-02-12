"""Backfill script to generate historical training data using Open-Meteo API."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root)) 

from pipelines.data_fetcher import OpenMeteoDataFetcher
from pipelines.feature_engineering import FeatureEngineer
from pipelines.data_cleaning import DataCleaner
from pipelines.aqi_calculator import EPAAQICalculator
from pipelines.mongodb_store import MongoDBStore
from config.settings import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _normalize_raw_data_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame to standard raw data schema with consistent columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized columns
    """
    if df.empty:
        return df
    
    # Define standard schema for raw data
    standard_columns = [
        'pm25', 'pm10', 'no2', 'o3', 'co', 'so2',
        'temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure',
        'timestamp', 'source'
    ]
    
    # Create new DataFrame with standard columns
    normalized_df = pd.DataFrame(index=df.index, columns=standard_columns)
    
    # Copy existing columns that match standard schema
    for col in standard_columns:
        if col in df.columns:
            normalized_df[col] = df[col].values
        else:
            # Fill missing columns with None
            normalized_df[col] = None
    
    # Ensure timestamp is datetime
    if 'timestamp' in normalized_df.columns:
        normalized_df['timestamp'] = pd.to_datetime(normalized_df['timestamp'], errors='coerce')
    
    return normalized_df


def _save_raw_data(raw_data: pd.DataFrame, project_root: Path):
    """
    Save raw data to local CSV file with consistent schema.
    
    Args:
        raw_data: DataFrame with raw data
        project_root: Project root directory path
    """
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize schema before saving
    normalized_data = _normalize_raw_data_schema(raw_data)
    
    # Save to single consolidated CSV file
    consolidated_file = raw_dir / "raw_data.csv"
    
    if consolidated_file.exists():
        try:
            existing_df = pd.read_csv(consolidated_file)
            # Normalize existing data to ensure schema consistency
            existing_df = _normalize_raw_data_schema(existing_df)
            
            # Combine with new data
            combined_df = pd.concat([existing_df, normalized_data], ignore_index=True)
            
            # Use DataCleaner to remove duplicates
            data_cleaner = DataCleaner()
            combined_df = data_cleaner.remove_duplicates(
                combined_df,
                subset=['timestamp'] if 'timestamp' in combined_df.columns else None
            )
            
            # Sort by timestamp if available
            if 'timestamp' in combined_df.columns:
                combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
                combined_df = combined_df.sort_values('timestamp')
            
            # Ensure schema consistency before saving
            combined_df = _normalize_raw_data_schema(combined_df)
            combined_df.to_csv(consolidated_file, index=False)
            logger.info(f"Updated raw data CSV: {consolidated_file} ({len(combined_df)} total records)")
        except Exception as e:
            logger.warning(f"Error updating raw data CSV: {str(e)}. Creating new file.")
            normalized_data.to_csv(consolidated_file, index=False)
    else:
        normalized_data.to_csv(consolidated_file, index=False)
        logger.info(f"Created raw data CSV: {consolidated_file}")

 
def _save_features(features: pd.DataFrame, project_root: Path):
    """
    Save features to local CSV file with consistent schema.
    
    Args:
        features: DataFrame with engineered features
        project_root: Project root directory path
    """
    features_dir = project_root / "data" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to single consolidated CSV file
    consolidated_file = features_dir / "features.csv"
    
    if consolidated_file.exists():
        try:
            existing_df = pd.read_csv(consolidated_file)
            
            # Get union of all columns from existing and new data
            all_columns = list(set(existing_df.columns.tolist() + features.columns.tolist()))
            
            # Normalize both DataFrames to have the same columns
            for col in all_columns:
                if col not in existing_df.columns:
                    existing_df[col] = None
                if col not in features.columns:
                    features[col] = None
            
            # Reorder columns consistently
            existing_df = existing_df[all_columns]
            features = features[all_columns]
            
            # Combine with new data
            combined_df = pd.concat([existing_df, features], ignore_index=True)
            
            # Use DataCleaner to remove duplicates
            data_cleaner = DataCleaner()
            combined_df = data_cleaner.remove_duplicates(
                combined_df,
                subset=['timestamp'] if 'timestamp' in combined_df.columns else None
            )
            
            # Sort by timestamp if available
            if 'timestamp' in combined_df.columns:
                combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
                combined_df = combined_df.sort_values('timestamp')
            
            # Save consolidated file
            combined_df.to_csv(consolidated_file, index=False)
            logger.info(f"Updated features CSV: {consolidated_file} ({len(combined_df)} total records, {len(all_columns)} columns)")
        except Exception as e:
            logger.warning(f"Error updating features CSV: {str(e)}. Creating new file.")
            features.to_csv(consolidated_file, index=False)
    else:
        features.to_csv(consolidated_file, index=False)
        logger.info(f"Created features CSV: {consolidated_file} ({len(features.columns)} columns)")


def backfill_historical_data(start_date: str, end_date: str, batch_days: int = 30):
    """
    Backfill historical data for training using Open-Meteo historical API.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        batch_days: Number of days to fetch per API call (Open-Meteo allows up to ~1 year per request)
    """
    logger.info(f"Starting backfill from {start_date} to {end_date}")
    
    fetcher = OpenMeteoDataFetcher()
    feature_engineer = FeatureEngineer()
    data_cleaner = DataCleaner()
    aqi_calculator = EPAAQICalculator()
    mongodb_store = MongoDBStore()
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Ensure end date is not today or in the future (use yesterday as maximum)
    today = datetime.now().date()
    if end.date() >= today:
        end = datetime.combine(today - timedelta(days=1), datetime.min.time())
        logger.info(f"End date adjusted to yesterday ({end.date()}) to exclude current day")
    
    # Process in batches to avoid overwhelming the API
    current_start = start
    total_records = 0
    
    while current_start <= end:
        # Calculate batch end date
        current_end = min(current_start + timedelta(days=batch_days), end)
        
        try:
            logger.info(f"Fetching historical data from {current_start.date()} to {current_end.date()}...")
            
            # Fetch historical weather data
            historical_weather = fetcher.fetch_historical_weather(current_start, current_end)
            
            if historical_weather.empty:
                logger.warning(f"No data fetched for {current_start.date()} to {current_end.date()}")
                current_start = current_end + timedelta(days=1)
                continue
            
            # Open-Meteo Air Quality API doesn't have historical endpoint
            # Estimate AQI based on weather patterns for historical data
            logger.info("Note: Historical air quality data not available. Estimating AQI from weather patterns.")
            
            # Batch process all records at once instead of one-by-one
            if historical_weather.empty:
                current_start = current_end + timedelta(days=1)
                continue
            
            # Estimate pollutant values based on weather patterns for all records at once
            temp = historical_weather['temperature'].fillna(20)
            wind = historical_weather['wind_speed'].fillna(10)
            humidity = historical_weather['humidity'].fillna(50)
            
            # Estimate PM2.5 based on weather correlations
            base_pm25 = 30 + (temp - 20) * 0.5 - (wind - 10) * 2
            base_pm25 = base_pm25.clip(10, 150)  # Clamp between 10-150
            
            # Add variation based on time of day and random factors
            hours = historical_weather['timestamp'].dt.hour
            day_variation = np.sin(2 * np.pi * hours / 24) * 5  # Diurnal variation
            random_variation = np.random.normal(0, 8, len(historical_weather))  # Random variation
            pm25 = (base_pm25 + day_variation + random_variation).clip(5, None)
            
            # Estimate other pollutants proportionally
            pm10 = (pm25 * 1.15 + np.random.normal(0, 3, len(historical_weather))).clip(0, None)
            no2 = (pm25 * 0.8 + np.random.normal(0, 5, len(historical_weather))).clip(0, None)
            o3 = (20 + (temp - 20) * 0.3 + np.random.normal(0, 5, len(historical_weather))).clip(0, None)
            
            # Create pollutant data DataFrame with proper index alignment
            num_records = len(historical_weather)
            aqi_data = pd.DataFrame({
                'pm25': pm25.values if hasattr(pm25, 'values') else pm25,
                'pm10': pm10.values if hasattr(pm10, 'values') else pm10,
                'no2': no2.values if hasattr(no2, 'values') else no2,
                'o3': o3.values if hasattr(o3, 'values') else o3,
                'co': [None] * num_records,
                'so2': [None] * num_records,
                'source': ['openmeteo_historical_estimated'] * num_records
            }, index=historical_weather.index)
            
            # Combine with weather data (avoid duplicate columns)
            # Remove 'source' from historical_weather if it exists, use our new one
            weather_data = historical_weather.drop(columns=['source'], errors='ignore')
            combined_df = pd.concat([aqi_data, weather_data], axis=1)
            
            # Ensure timestamp is set
            if 'timestamp' not in combined_df.columns:
                combined_df['timestamp'] = historical_weather['timestamp']
            
            # Save raw data to local system (batch save)
            _save_raw_data(combined_df, project_root)
            
            # Clean data (batch processing)
            cleaned_df = data_cleaner.clean_data(combined_df)
            
            # Calculate AQI if not present (batch processing)
            if 'aqi' not in cleaned_df.columns or cleaned_df['aqi'].isna().all():
                cleaned_df = aqi_calculator.calculate_aqi_from_dataframe(cleaned_df)
                if 'aqi' not in cleaned_df.columns:
                    cleaned_df['aqi'] = cleaned_df.get('aqi_calculated', None)
            
            # Engineer features (batch processing)
            features_df = feature_engineer.engineer_features(cleaned_df, create_targets=False)
            
            # Add metadata
            features_df['pipeline_run_date'] = datetime.now()
            features_df['city'] = config['city']['name']
            features_df['backfill_date'] = features_df['timestamp']
            
            # Save features to local system (batch save)
            if not features_df.empty:
                _save_features(features_df, project_root)
                
                # Store features in MongoDB (batch insert)
                mongodb_store.insert_features(features_df, metadata={
                    'backfill': True,
                    'batch_start': current_start.isoformat(),
                    'batch_end': current_end.isoformat()
                })
                total_records += len(features_df)
            
            logger.info(f"Stored {len(features_df)} records for batch {current_start.date()} to {current_end.date()}")
            
            logger.info(f"Completed batch: {current_start.date()} to {current_end.date()}")
            current_start = current_end + timedelta(days=1)
            
            # Rate limiting
            import time
            time.sleep(1)
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing batch {current_start.date()} to {current_end.date()}: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            current_start = current_end + timedelta(days=1)
            continue
    
    logger.info(f"Backfill complete! Total records stored: {total_records}")


def main():
    """Main entry point for backfill script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill historical AQI data using Open-Meteo')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--batch-days', type=int, default=30,
                       help='Number of days to fetch per API call (default: 30)')
    
    args = parser.parse_args()
    
    backfill_historical_data(args.start_date, args.end_date, args.batch_days)


if __name__ == "__main__":
    # Example: backfill last 365 days (1 year)
    # Use yesterday as end date (exclude current day)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    
    backfill_historical_data(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        batch_days=30
    )
